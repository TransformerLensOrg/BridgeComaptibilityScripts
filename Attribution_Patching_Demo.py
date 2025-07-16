# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Attribution_Patching_Demo.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# %% [markdown]
#  # Attribution Patching Demo
#  **Read [the accompanying blog post here](https://neelnanda.io/attribution-patching) for more context**
#  This is an interim research report, giving a whirlwind tour of some unpublished work I did at Anthropic (credit to the then team - Chris Olah, Catherine Olsson, Nelson Elhage and Tristan Hume for help, support, and mentorship!)
# 
#  The goal of this work is run activation patching at an industrial scale, by using gradient based attribution to approximate the technique - allow an arbitrary number of patches to be made on two forwards and a single backward pass
# 
#  I have had less time than hoped to flesh out this investigation, but am writing up a rough investigation and comparison to standard activation patching on a few tasks to give a sense of the potential of this approach, and where it works vs falls down.

# %% [markdown]
#  <b style="color: red">To use this notebook, go to Runtime > Change Runtime Type and select GPU as the hardware accelerator.</b>
# 
#  **Tips for reading this Colab:**
#  * You can run all this code for yourself!
#  * The graphs are interactive!
#  * Use the table of contents pane in the sidebar to navigate
#  * Collapse irrelevant sections with the dropdown arrows
#  * Search the page using the search in the sidebar, not CTRL+F

# %% [markdown]
#  ## Setup (Ignore)

# %%
# Janky code to do different setup when run in a Colab notebook vs VSCode
import os

# %%
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio

pio.renderers.default = "notebook_connected"

# %%
# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.notebook as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from torchtyping import TensorType as TT
from typing import List, Union, Optional, Callable
from functools import partial
import copy
import itertools
import json

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML, Markdown

# %%
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)

from transformer_lens.model_bridge import TransformerBridge

# %% [markdown]
#  Plotting helper functions from a janky personal library of plotting utils. The library is not documented and I recommend against trying to read it, just use your preferred plotting library if you want to do anything non-obvious:

# %%
from neel_plotly import line, imshow, scatter

# %%
import transformer_lens.patching as patching

# %% [markdown]
#  ## IOI Patching Setup
#  This just copies the relevant set up from Exploratory Analysis Demo, and isn't very important.

# %%
model = TransformerBridge.boot_transformers("gpt2")
model.set_use_attn_result(True)

# %%
prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]
answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]

clean_tokens = model.to_tokens(prompts)
# Swap each adjacent pair, with a hacky list comprehension
corrupted_tokens = clean_tokens[
    [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
]
print("Clean string 0", model.to_string(clean_tokens[0]))
print("Corrupted string 0", model.to_string(corrupted_tokens[0]))

answer_token_indices = torch.tensor(
    [
        [model.to_single_token(answers[i][j]) for j in range(2)]
        for i in range(len(answers))
    ],
    device=model.cfg.device,
)
print("Answer token indices", answer_token_indices)

# %%
def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()


clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
print(f"Clean logit diff: {clean_logit_diff:.4f}")

corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

# %%
CLEAN_BASELINE = clean_logit_diff
CORRUPTED_BASELINE = corrupted_logit_diff


def ioi_metric(logits, answer_token_indices=answer_token_indices):
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (
        CLEAN_BASELINE - CORRUPTED_BASELINE
    )


print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")

# %% [markdown]
#  ## Patching
#  In the following cells, we define attribution patching and use it in various ways on the model.

# %%
Metric = Callable[[TT["batch_and_pos_dims", "d_model"]], float]

# %%
filter_not_qkv_input = lambda name: "_input" not in name


def get_cache_fwd_and_bwd(model, tokens, metric):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")

    value = metric(model(tokens))
    value.backward()
    model.reset_hooks()
    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )


clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
    model, clean_tokens, ioi_metric
)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))
print("Clean Gradients Cached:", len(clean_grad_cache))
corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
    model, corrupted_tokens, ioi_metric
)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(corrupted_cache))
print("Corrupted Gradients Cached:", len(corrupted_grad_cache))

# %% [markdown]
#  ### Attention Attribution
#  The easiest thing to start with is to not even engage with the corrupted tokens/patching, but to look at the attribution of the attention patterns - that is, the linear approximation to what happens if you set each element of the attention pattern to zero. This, as it turns out, is a good proxy to what is going on with each head!
#  Note that this is *not* the same as what we will later do with patching. In particular, this does not set up a careful counterfactual! It's a good tool for what's generally going on in this problem, but does not control for eg stuff that systematically boosts John > Mary in general, stuff that says "I should activate the IOI circuit", etc. Though using logit diff as our metric *does*
#  Each element of the batch is independent and the metric is an average logit diff, so we can analyse each batch element independently here. We'll look at the first one, and then at the average across the whole batch (note - 4 prompts have indirect object before subject, 4 prompts have it the other way round, making the average pattern harder to interpret - I plot it over the first sequence of tokens as a mildly misleading reference).
#  We can compare it to the interpretability in the wild diagram, and basically instantly recover most of the circuit!

# %%
def create_attention_attr(
    clean_cache, clean_grad_cache
) -> TT["batch", "layer", "head_index", "dest", "src"]:
    attention_stack = torch.stack(
        [clean_cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0
    )
    attention_grad_stack = torch.stack(
        [clean_grad_cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0
    )
    attention_attr = attention_grad_stack * attention_stack
    attention_attr = einops.rearrange(
        attention_attr,
        "layer batch head_index dest src -> batch layer head_index dest src",
    )
    return attention_attr


attention_attr = create_attention_attr(clean_cache, clean_grad_cache)

# %%
HEAD_NAMES = [
    f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
]
HEAD_NAMES_SIGNED = [f"{name}{sign}" for name in HEAD_NAMES for sign in ["+", "-"]]
HEAD_NAMES_QKV = [
    f"{name}{act_name}" for name in HEAD_NAMES for act_name in ["Q", "K", "V"]
]
print(HEAD_NAMES[:5])
print(HEAD_NAMES_SIGNED[:5])
print(HEAD_NAMES_QKV[:5])

# %% [markdown]
#  ## Factual Knowledge Patching Example
#  Incomplete, but maybe of interest!
#  Note that I have better results with the corrupted prompt as having random words rather than Colosseum.

# %%
gpt2_xl = HookedTransformer.from_pretrained("gpt2-xl")
clean_prompt = "The Eiffel Tower is located in the city of"
clean_answer = " Paris"
# corrupted_prompt = "The red brown fox jumps is located in the city of"
corrupted_prompt = "The Colosseum is located in the city of"
corrupted_answer = " Rome"
utils.test_prompt(clean_prompt, clean_answer, gpt2_xl)
utils.test_prompt(corrupted_prompt, corrupted_answer, gpt2_xl)

# %%
clean_answer_index = gpt2_xl.to_single_token(clean_answer)
corrupted_answer_index = gpt2_xl.to_single_token(corrupted_answer)


def factual_logit_diff(logits: TT["batch", "position", "d_vocab"]):
    return logits[0, -1, clean_answer_index] - logits[0, -1, corrupted_answer_index]

# %%
clean_logits, clean_cache = gpt2_xl.run_with_cache(clean_prompt)
CLEAN_LOGIT_DIFF_FACTUAL = factual_logit_diff(clean_logits).item()
corrupted_logits, _ = gpt2_xl.run_with_cache(corrupted_prompt)
CORRUPTED_LOGIT_DIFF_FACTUAL = factual_logit_diff(corrupted_logits).item()


def factual_metric(logits: TT["batch", "position", "d_vocab"]):
    return (factual_logit_diff(logits) - CORRUPTED_LOGIT_DIFF_FACTUAL) / (
        CLEAN_LOGIT_DIFF_FACTUAL - CORRUPTED_LOGIT_DIFF_FACTUAL
    )


print("Clean logit diff:", CLEAN_LOGIT_DIFF_FACTUAL)
print("Corrupted logit diff:", CORRUPTED_LOGIT_DIFF_FACTUAL)
print("Clean Metric:", factual_metric(clean_logits))
print("Corrupted Metric:", factual_metric(corrupted_logits))

# %%
# corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(gpt2_xl, corrupted_prompt, factual_metric)

# %%
clean_tokens = gpt2_xl.to_tokens(clean_prompt)
clean_str_tokens = gpt2_xl.to_str_tokens(clean_prompt)
corrupted_tokens = gpt2_xl.to_tokens(corrupted_prompt)
corrupted_str_tokens = gpt2_xl.to_str_tokens(corrupted_prompt)
print("Clean:", clean_str_tokens)
print("Corrupted:", corrupted_str_tokens)