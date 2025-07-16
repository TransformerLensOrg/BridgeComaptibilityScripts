# %%
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio

pio.renderers.default = "png"

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

from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML

# %%
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# %% [markdown]
#  We turn automatic differentiation off, to save GPU memory, as this notebook focuses on model inference not model training.

# %%
torch.set_grad_enabled(False)

# %% [markdown]
#  Plotting helper functions from a janky personal library of plotting utils. The library is not documented and I recommend against trying to read it, just use your preferred plotting library if you want to do anything non-obvious:

# %%

# %%
import transformer_lens.patching as patching
from transformer_lens.model_bridge import TransformerBridge

# %% [markdown]
#  ## Activation Patching Setup
#  This just copies the relevant set up from Exploratory Analysis Demo, and isn't very important.

# %%
model = TransformerBridge.boot_transformers("gpt2")

# %%
prompts = ['When John and Mary went to the shops, John gave the bag to', 'When John and Mary went to the shops, Mary gave the bag to', 'When Tom and James went to the park, James gave the ball to', 'When Tom and James went to the park, Tom gave the ball to', 'When Dan and Sid went to the shops, Sid gave an apple to', 'When Dan and Sid went to the shops, Dan gave an apple to', 'After Martin and Amy went to the park, Amy gave a drink to', 'After Martin and Amy went to the park, Martin gave a drink to']
answers = [(' Mary', ' John'), (' John', ' Mary'), (' Tom', ' James'), (' James', ' Tom'), (' Dan', ' Sid'), (' Sid', ' Dan'), (' Martin', ' Amy'), (' Amy', ' Martin')]

clean_tokens = model.to_tokens(prompts)
# Swap each adjacent pair, with a hacky list comprehension
corrupted_tokens = clean_tokens[
    [(i+1 if i%2==0 else i-1) for i in range(len(clean_tokens)) ]
    ]
print("Clean string 0", model.to_string(clean_tokens[0]))
print("Corrupted string 0", model.to_string(corrupted_tokens[0]))

answer_token_indices = torch.tensor([[model.to_single_token(answers[i][j]) for j in range(2)] for i in range(len(answers))], device=model.cfg.device)
print("Answer token indices", answer_token_indices)

# %%
def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    if len(logits.shape)==3:
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
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (CLEAN_BASELINE  - CORRUPTED_BASELINE)

print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")

# %% [markdown]
#  ## Patching
#  In the following cells, we use the patching module to call activation patching utilities

# %%
# Whether to do the runs by head and by position, which are much slower
DO_SLOW_RUNS = True

# %% [markdown]
#  ### Patching Single Activation Types
#  We start by patching single types of activation
#  The general syntax is that the functions are called get_act_patch_... and take in (model, corrupted_tokens, clean_cache, patching_metric)

# %% [markdown]
#  We can patch head outputs over each head in each layer, patching on each position in turn
#  out -> q, k, v, pattern all also work, though note that pattern has output shape [layer, pos, head]
#  We reshape it to plot nicely

# %

# %% [markdown]
#  ### Patching multiple activation types
#  Some utilities are provided to patch multiple activations types *in turn*. Note that this is *not* a utility to patch multiple activations at once, it's just a useful scan to get a sense for what's going on in a model
#  By block: We patch the residual stream at the start of each block, attention output and MLP output over each layer and position

# %% [markdown]
#  ## Induction Patching
#  To show how easy it is, lets do that again with induction heads in a 2L Attention Only model
#  The input will be repeated random tokens eg BOS 1 5 8 9 2 1 5 8 9 2, and we judge the model's ability to predict the second repetition with its induction heads
#  Lets call A, B and C different (non-repeated) random sequences. We'll start with clean tokens AA and corrupted tokens AB, and see how well the model can predict the second A given the first A

# %% [markdown]
#  ### Setup

# %%
attn_only = TransformerBridge.boot_transformers("attn-only-2l") # TODO: this is one of Neel's models, does this make sense with boot?
batch = 4
seq_len = 20
rand_tokens_A = torch.randint(100, 10000, (batch, seq_len)).to(attn_only.cfg.device)
rand_tokens_B = torch.randint(100, 10000, (batch, seq_len)).to(attn_only.cfg.device)
rand_tokens_C = torch.randint(100, 10000, (batch, seq_len)).to(attn_only.cfg.device)
bos = torch.tensor([attn_only.tokenizer.bos_token_id]*batch)[:, None].to(attn_only.cfg.device)
clean_tokens_induction = torch.cat([bos, rand_tokens_A, rand_tokens_A], dim=1).to(attn_only.cfg.device)
corrupted_tokens_induction = torch.cat([bos, rand_tokens_A, rand_tokens_B], dim=1).to(attn_only.cfg.device)

# %%
clean_logits_induction, clean_cache_induction = attn_only.run_with_cache(clean_tokens_induction)
corrupted_logits_induction, corrupted_cache_induction = attn_only.run_with_cache(corrupted_tokens_induction)

# %% [markdown]
#  We define our metric as negative loss on the second half (negative loss so that higher is better)
#  This time we won't normalise our metric

# %%
def induction_loss(logits, answer_token_indices=rand_tokens_A):
    seq_len = answer_token_indices.shape[1]

    # logits: batch x seq_len x vocab_size
    # Take the logits for the answers, cut off the final element to get the predictions for all but the first element of the answers (which can't be predicted)
    final_logits = logits[:, -seq_len:-1]
    final_log_probs = final_logits.log_softmax(-1)
    return final_log_probs.gather(-1, answer_token_indices[:, 1:].unsqueeze(-1)).mean()
CLEAN_BASELINE_INDUCTION = induction_loss(clean_logits_induction).item()
print("Clean baseline:", CLEAN_BASELINE_INDUCTION)
CORRUPTED_BASELINE_INDUCTION = induction_loss(corrupted_logits_induction).item()
print("Corrupted baseline:", CORRUPTED_BASELINE_INDUCTION)
