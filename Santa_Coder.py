# %%
# Janky code to do different setup when run in a Colab notebook vs VSCode

# %%
# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
from tqdm import tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from torchtyping import TensorType as TT
from typing import List, Union, Optional
from jaxtyping import Float, Int
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
# import circuitsvis as cv

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from transformer_lens.model_bridge import TransformerBridge

torch.set_grad_enabled(False)

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

# %%
# load hf model
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

# %%
# Disable folding norms and folding norms and biases so that intermediate value
# in between transformer blocks can be compared
bloom = TransformerBridge.boot_transformers("bloom-560m",fold_ln=False, fold_value_biases=False, center_writing_weights=False)

# %%
text = '''
TransformerLens lets you load in 50+ different open source language models,
and exposes the internal activations of the model to you. You can cache
any internal activation in the model, and add in functions to edit, remove
or replace these activations as the model runs.
'''
input_ids = tokenizer(text, return_tensors='pt')['input_ids']
gt_logits = model(input_ids)['logits'] # ground truth logits from hf
my_logits = bloom(input_ids)
centered_gt_logits = gt_logits - gt_logits.mean(-1, keepdim=True)
mean_diff = (my_logits.cpu() - centered_gt_logits).mean()
print("avg logits difference:", mean_diff.item())
max_diff = (my_logits.cpu() - centered_gt_logits).abs().max()
print("max logits difference:", max_diff.item())

# %%
gt_cache = model(input_ids, output_hidden_states=True)['hidden_states']
_, my_cache = bloom.run_with_cache(input_ids)
use_loose_bound = False
pass_loose_bound = True
print("*"*5, "Matching hf and T-Lens residual stream in between transformer blocks", "*"*5)
for i in range(24):
    try:
        torch.testing.assert_close(my_cache['resid_pre',i], gt_cache[i].cuda())
    except:
        max_diff = (my_cache['resid_pre',i] - gt_cache[i].cuda()).abs().max()
        print(f"layer {i} \t not close, max difference: {max_diff}")
        use_loose_bound = True

if use_loose_bound:
    atol = rtol = 1e-3
    print("*"*5, f"\ttesting with atol={atol} and rtol={rtol}\t","*"*5)
    for i in range(24):
        try:
            torch.testing.assert_close(my_cache['resid_pre',i], gt_cache[i].cuda(), atol=atol, rtol=rtol)
        except:
            max_diff = (my_cache['resid_pre',i] - gt_cache[i].cuda()).abs().max()
            print(f"layer {i} \t not close, max difference: {max_diff}")
            pass_loose_bound = False

    if pass_loose_bound:
        print(f"All layers match with atol={atol} rtol={rtol}")
else: 
    print("All layers match")

# %%
my_loss = bloom(input_ids, return_type='loss')
print("T-Lens next token loss:", my_loss.item())
gt_outputs = model(input_ids, labels=input_ids)
gt_loss = gt_outputs.loss
print("HF next token loss:", gt_loss.item())
print("diff in loss (abs):", (gt_loss-my_loss).abs().item())


