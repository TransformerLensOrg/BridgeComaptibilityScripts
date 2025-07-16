# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Patchscopes_Generation_Demo.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# %% [markdown]
# # Patchscopes & Generation with Patching
# 
# This notebook contains a demo for Patchscopes (https://arxiv.org/pdf/2401.06102) and demonstrates how to generate multiple tokens with patching. Since there're also some applications in [Patchscopes](##Patchscopes-pipeline) that require generating multiple tokens with patching, I think it's suitable to put both of them in the same notebook. Additionally, generation with patching can be well-described using Patchscopes. Therefore, I simply implement it with the Patchscopes pipeline (see [here](##Generation-with-patching)).

# %% [markdown]
# ## Setup (Ignore)

# %%
# Janky code to do different setup when run in a Colab notebook vs VSCode
import os
import torch
from typing import List, Callable, Tuple, Union
from functools import partial
from jaxtyping import Float
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens.model_bridge import TransformerBridge

# %% [markdown]
# ## Helper Funcs
# 
# A helper function to plot logit lens

# %%
import plotly.graph_objects as go
import numpy as np

# Parameters
num_layers = 5
seq_len = 10

# Create a matrix of tokens for demonstration
tokens = np.array([["token_{}_{}".format(i, j) for j in range(seq_len)] for i in range(num_layers)])[::-1]
values = np.random.rand(num_layers, seq_len)
orig_tokens = ['Token {}'.format(i) for i in range(seq_len)]

def draw_logit_lens(num_layers, seq_len, orig_tokens, tokens, values):
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=orig_tokens,
        y=['Layer {}'.format(i) for i in range(num_layers)][::-1],
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title='Value')
    ))

    # Add text annotations
    annotations = []
    for i in range(num_layers):
        for j in range(seq_len):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=tokens[i, j],
                    showarrow=False,
                    font=dict(color='white')
                )
            )

    fig.update_layout(
        annotations=annotations,
        xaxis=dict(side='top'),
        yaxis=dict(autorange='reversed'),
        margin=dict(l=50, r=50, t=100, b=50),
        width=1000,
        height=600,
        plot_bgcolor='white'
    )

    # Show the plot
    fig.show()
# draw_logit_lens(num_layers, seq_len, orig_tokens, tokens, values)

# %% [markdown]
# ## Model Preparation

# %%
# NBVAL_IGNORE_OUTPUT
# I'm using an M2 macbook air, so I use CPU for better support
model = TransformerBridge.boot_transformers("gpt2", device="cpu")
model.eval()

# %% [markdown]
# ## Patchscopes Definition
# 
# Here we first wirte down the formal definition decribed in the paper https://arxiv.org/pdf/2401.06102.
# 
# The representations are:
# 
# source: (S, i, M, l), where S is the source prompt, i is the source position, M is the source model, and l is the source layer.
# 
# target: (T,i*,f,M*,l*), where T is the target prompt, i* is the target position, M* is the target model, l* is the target layer, and f is the mapping function that takes the original hidden states as input and output the target hidden states
# 
# By defulat, S = T, i = i*, M = M*, l = l*, f = identity function

# %% [markdown]
# ## Patchscopes Pipeline
# 
# ### Get hidden representation from the source model
# 
# 1. We first need to extract the source hidden states from model M at position i of layer l with prompt S. In TransformerLens, we can do this using run_with_cache.
# 2. Then, we map the source representation with a function f, and feed the hidden representation to the target position using a hook. Specifically, we focus on residual stream (resid_post), whereas you can manipulate more fine-grainedly with TransformerLens
# 

# %%
prompts = ["Patchscopes is a nice tool to inspect hidden representation of language model"]
input_tokens = model.to_tokens(prompts)
clean_logits, clean_cache = model.run_with_cache(input_tokens)

# %%
def get_source_representation(prompts: List[str], layer_id: int, model: HookedTransformer, pos_id: Union[int, List[int]]=None) -> torch.Tensor:
    """Get source hidden representation represented by (S, i, M, l)
    
    Args:
        - prompts (List[str]): a list of source prompts
        - layer_id (int): the layer id of the model
        - model (HookedTransformer): the source model
        - pos_id (Union[int, List[int]]): the position id(s) of the model, if None, return all positions

    Returns:
        - source_rep (torch.Tensor): the source hidden representation
    """
    input_tokens = model.to_tokens(prompts)
    _, cache = model.run_with_cache(input_tokens)
    layer_name = "blocks.{id}.hook_resid_post"
    layer_name = layer_name.format(id=layer_id)
    if pos_id is None:
        return cache[layer_name][:, :, :]
    else:
        return cache[layer_name][:, pos_id, :]

# %%
source_rep = get_source_representation(
    prompts=["Patchscopes is a nice tool to inspect hidden representation of language model"],
    layer_id=2,
    model=model,
    pos_id=5
)

# %% [markdown]
# ### Feed the representation to the target position
# 
# First we need to map the representation using mapping function f, and then feed the target representation to the target position represented by (T,i*,f,M*,l*)

# %%
# here we use an identity function for demonstration purposes
def identity_function(source_rep: torch.Tensor) -> torch.Tensor:
    return source_rep

# %%
# recall the target representation (T,i*,f,M*,l*), and we also need the hidden representation from our source model (S, i, M, l)
def feed_source_representation(source_rep: torch.Tensor, prompt: List[str], f: Callable, model: HookedTransformer, layer_id: int, pos_id: Union[int, List[int]]=None) -> ActivationCache:
    """Feed the source hidden representation to the target model
    
    Args:
        - source_rep (torch.Tensor): the source hidden representation
        - prompt (List[str]): the target prompt
        - f (Callable): the mapping function
        - model (HookedTransformer): the target model
        - layer_id (int): the layer id of the target model
        - pos_id (Union[int, List[int]]): the position id(s) of the target model, if None, return all positions
    """
    mapped_rep = f(source_rep)
    # similar to what we did for activation patching, we need to define a function to patch the hidden representation
    def resid_ablation_hook(
        value: Float[torch.Tensor, "batch pos d_resid"],
        hook: HookPoint
    ) -> Float[torch.Tensor, "batch pos d_resid"]:
        # print(f"Shape of the value tensor: {value.shape}")
        # print(f"Shape of the hidden representation at the target position: {value[:, pos_id, :].shape}")
        value[:, pos_id, :] = mapped_rep
        return value
    
    input_tokens = model.to_tokens(prompt)

    logits = model.run_with_hooks(
        input_tokens,
        return_type="logits",
        fwd_hooks=[(
            utils.get_act_name("resid_post", layer_id),
            resid_ablation_hook
            )]
        )
    
    return logits

# %%
patched_logits = feed_source_representation(
    source_rep=source_rep,
    prompt=prompts,
    pos_id=3,
    f=identity_function,
    model=model,
    layer_id=2
)

# %%
# NBVAL_IGNORE_OUTPUT
clean_logits[:, 5], patched_logits[:, 5]

# %% [markdown]
# ## Generation with Patching
# 
# In the last step, we've implemented the basic version of Patchscopes where we can only run one single forward pass. Let's now unlock the power by allowing it to generate multiple tokens!

# %%
def generate_with_patching(model: HookedTransformer, prompts: List[str], target_f: Callable, max_new_tokens: int = 50):
    temp_prompts = prompts
    input_tokens = model.to_tokens(temp_prompts)
    for _ in range(max_new_tokens):
        logits = target_f(
            prompt=temp_prompts,
        )
        next_tok = torch.argmax(logits[:, -1, :])
        input_tokens = torch.cat((input_tokens, next_tok.view(input_tokens.size(0), 1)), dim=1)
        temp_prompts = model.to_string(input_tokens)

    return model.to_string(input_tokens)[0]

# %%
prompts = ["Patchscopes is a nice tool to inspect hidden representation of language model"]
input_tokens = model.to_tokens(prompts)
target_f = partial(
    feed_source_representation,
    source_rep=source_rep,
    pos_id=-1,
    f=identity_function,
    model=model,
    layer_id=2
)
gen = generate_with_patching(model, prompts, target_f, max_new_tokens=3)
print(gen)

# %%
# Original generation
print(model.generate(prompts[0], verbose=False, max_new_tokens=50, do_sample=False))

# %% [markdown]
# ## Application Examples

# %% [markdown]
# ### Logit Lens
# 
# For Logit Lens, the configuration is l* ← L*. Here, L* is the last layer.

# %%
token_list = []
value_list = []

def identity_function(source_rep: torch.Tensor) -> torch.Tensor:
    return source_rep

for source_layer_id in range(12):
    # Prepare source representation
    source_rep = get_source_representation(
        prompts=["Patchscopes is a nice tool to inspect hidden representation of language model"],
        layer_id=source_layer_id,
        model=model,
        pos_id=None
    )

    logits = feed_source_representation(
        source_rep=source_rep,
        prompt=["Patchscopes is a nice tool to inspect hidden representation of language model"],
        f=identity_function,
        model=model,
        layer_id=11
    )
    token_list.append([model.to_string(token_id.item()) for token_id in logits.argmax(dim=-1).squeeze()])
    value_list.append([value for value in torch.max(logits.softmax(dim=-1), dim=-1)[0].detach().squeeze().numpy()])

# %%
token_list = np.array(token_list[::-1])
value_list = np.array(value_list[::-1])

# %%
num_layers = 12
seq_len = len(token_list[0])
orig_tokens = [model.to_string(token_id) for token_id in model.to_tokens(["Patchscopes is a nice tool to inspect hidden representation of language model"])[0]]


# %% [markdown]
# ### Entity Description
# 
# Entity description tries to answer "how LLMs resolve entity mentions across multiple layers. Concretely, given a subject entity name, such as “the summer Olympics of 1996”, how does the model contextualize the input tokens of the entity and at which layer is it fully resolved?"
# 
# The configuration is l* ← l, i* ← m, and it requires generating multiple tokens. Here m refers to the last position (the position of x)

# %%
 # Prepare source representation
source_rep = get_source_representation(
    prompts=["Diana, Princess of Wales"],
    layer_id=11,
    model=model,
    pos_id=-1
)

# %%
target_prompt = ["Syria: Country in the Middle East, Leonardo DiCaprio: American actor, Samsung: South Korean multinational major appliance and consumer electronics corporation, x"]
# need to calcualte an absolute position, instead of a relative position
last_pos_id = len(model.to_tokens(target_prompt)[0]) - 1
# we need to define the function that takes the generation as input
for target_layer_id in range(12):
    target_f = partial(
        feed_source_representation,
        source_rep=source_rep,
        pos_id=last_pos_id,
        f=identity_function,
        model=model,
        layer_id=target_layer_id
    )
    gen = generate_with_patching(model, target_prompt, target_f, max_new_tokens=20)
    print(f"Generation by patching layer {target_layer_id}:\n{gen}\n{'='*30}\n")

# %% [markdown]
# As we can see, maybe the early layers of gpt2-small are doing something related to entity resolution, whereas the late layers are apparently not(?)

# %% [markdown]
# ### Zero-Shot Feature Extraction
# 
# Zero-shot Feature Extraction "Consider factual and com- monsense knowledge represented as triplets (σ,ρ,ω) of a subject (e.g., “United States”), a relation (e.g., “largest city of”), and an object (e.g.,
# “New York City”). We investigate to what extent the object ω can be extracted from the last token representation of the subject σ in an arbitrary input context."
# 
# The configuration is l∗ ← j′ ∈ [1,...,L∗], i∗ ← m, T ← relation verbalization followed by x

# %%
# for a triplet (company Apple, co-founder of, Steve Jobs), we need to first make sure that the object is in the continuation
source_prompt = "Co-founder of company Apple"
model.generate(source_prompt, verbose=False, max_new_tokens=20, do_sample=False)

# %%
# Still need an aboslute position
last_pos_id = len(model.to_tokens(["Co-founder of x"])[0]) - 1
target_prompt = ["Co-founder of x"]

# Check all the combinations, you'll see that the model is able to generate "Steve Jobs" in several continuations
for source_layer_id in range(12):
    # Prepare source representation, here we can use relative position
    source_rep = get_source_representation(
        prompts=["Co-founder of company Apple"],
        layer_id=source_layer_id,
        model=model,
        pos_id=-1
    )
    for target_layer_id in range(12):
        target_f = partial(
            feed_source_representation,
            source_rep=source_rep,
            prompt=target_prompt,
            f=identity_function,
            model=model,
            pos_id=last_pos_id,
            layer_id=target_layer_id
        )
        gen = generate_with_patching(model, target_prompt, target_f, max_new_tokens=20)
        print(gen)


