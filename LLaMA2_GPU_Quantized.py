# %% [markdown]
# # LLaMA and Llama-2 in TransformerLens

# %% [markdown]
# ## Setup (skip)

# %%
# NBVAL_IGNORE_OUTPUT
# Janky code to do different setup when run in a Colab notebook vs VSCode
import os

# %%
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio
pio.renderers.default = "notebook_connected"
print(f"Using renderer: {pio.renderers.default}")

import circuitsvis as cv

# %%
# Import stuff
import torch
import tqdm.auto as tqdm
import plotly.express as px

from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from jaxtyping import Float

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer
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

# %% [markdown]
# ## Loading LLaMA

# %% [markdown]
# LLaMA weights are not available on HuggingFace, so you'll need to download and convert them
# manually:
# 
# 1. Get LLaMA weights here: https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform
# 
# 2. Convert the official weights to huggingface:
# 
# ```bash
# python src/transformers/models/llama/convert_llama_weights_to_hf.py \
#     --input_dir /path/to/downloaded/llama/weights \
#     --model_size 7B \
#     --output_dir /llama/weights/directory/
# ```
# 
# Note: this didn't work for Arthur by default (even though HF doesn't seem to show this anywhere). I
# had to change <a
# href="https://github.com/huggingface/transformers/blob/07360b6/src/transformers/models/llama/convert_llama_weights_to_hf.py#L295">this</a>
# line of my pip installed `src/transformers/models/llama/convert_llama_weights_to_hf.py` file (which
# was found at
# `/opt/conda/envs/arthurenv/lib/python3.10/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py`)
# from `input_base_path=os.path.join(args.input_dir, args.model_size),` to `input_base_path=os.path.join(args.input_dir),`
# 
# 3. Change the ```MODEL_PATH``` variable in the cell below to where the converted weights are stored.

# %%
MODEL_PATH=''

if MODEL_PATH:
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
    hf_model = LlamaForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True)

    model = TransformerBridge.boot_transformers("llama-7b", hf_model=hf_model, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer)

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.generate("The capital of Germany is", max_new_tokens=20, temperature=0)

# %% [markdown]
# ## Loading LLaMA-2
# LLaMA-2 is hosted on HuggingFace, but gated by login.
# 
# Before running the notebook, log in to HuggingFace via the cli on your machine:
# ```bash
# transformers-cli login
# ```
# This will cache your HuggingFace credentials, and enable you to download LLaMA-2.

# %% [markdown]
# ## Install additional dependenceis requred for quantization

# %%

# %% [markdown]
# ## Load quantized model

# %%

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LLAMA_2_7B_CHAT_PATH = "meta-llama/Llama-2-7b-chat-hf"
inference_dtype = torch.float32
# inference_dtype = torch.float32
# inference_dtype = torch.float16

hf_model = AutoModelForCausalLM.from_pretrained(LLAMA_2_7B_CHAT_PATH,
                                             torch_dtype=inference_dtype,
                                             device_map = "cuda:0",
                                             quantization_config=BitsAndBytesConfig(load_in_4bit=True))

tokenizer = AutoTokenizer.from_pretrained(LLAMA_2_7B_CHAT_PATH)

model = TransformerBridge.boot_transformers(LLAMA_2_7B_CHAT_PATH,
                                             hf_model=hf_model,
                                             dtype=inference_dtype,
                                             fold_ln=False,
                                             fold_value_biases=False,
                                             center_writing_weights=False,
                                             center_unembed=False,
                                             tokenizer=tokenizer)

model.generate("The capital of Germany is", max_new_tokens=2, temperature=0)



# %% [markdown]
# ### Verify GPU memory use

# %%
print("free(Gb):", torch.cuda.mem_get_info()[0]/1000000000, "total(Gb):", torch.cuda.mem_get_info()[1]/1000000000)

# %% [markdown]
# ### Compare logits with HuggingFace model

# %%
prompts = [
    "The capital of Germany is",
    "2 * 42 = ",
    "My favorite",
    "aosetuhaosuh aostud aoestuaoentsudhasuh aos tasat naostutshaosuhtnaoe usaho uaotsnhuaosntuhaosntu haouaoshat u saotheu saonuh aoesntuhaosut aosu thaosu thaoustaho usaothusaothuao sutao sutaotduaoetudet uaosthuao uaostuaoeu aostouhsaonh aosnthuaoscnuhaoshkbaoesnit haosuhaoe uasotehusntaosn.p.uo ksoentudhao ustahoeuaso usant.hsa otuhaotsi aostuhs",
]

model.eval()
hf_model.eval()
prompt_ids = [tokenizer.encode(prompt, return_tensors="pt") for prompt in prompts]
tl_logits = [model(prompt_ids).detach().cpu() for prompt_ids in tqdm(prompt_ids)]

# hf logits are really slow as it's on CPU. If you have a big/multi-GPU machine, run `hf_model = hf_model.to("cuda")` to speed this up
logits = [hf_model(prompt_ids).logits.detach().cpu() for prompt_ids in tqdm(prompt_ids)]

for i in range(len(prompts)):
    if i == 0:
        print("logits[i]", i, logits[i].dtype, logits[i])
        print("tl_logits[i]", i, tl_logits[i].dtype, tl_logits[i])
    assert torch.allclose(logits[i], tl_logits[i], atol=1e-4, rtol=1e-2)

# %% [markdown]
# ## TransformerLens Demo

# %% [markdown]
# ### Reading from hooks

# %%
llama_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
llama_tokens = model.to_tokens(llama_text)
llama_logits, llama_cache = model.run_with_cache(llama_tokens, remove_batch_dim=True)

attention_pattern = llama_cache["pattern", 0, "attn"]
llama_str_tokens = model.to_str_tokens(llama_text)

print("Layer 0 Head Attention Patterns:")

# %% [markdown]
# ### Writing to hooks

# %%
layer_to_ablate = 0
head_index_to_ablate = 31

# We define a head ablation hook
# The type annotations are NOT necessary, they're just a useful guide to the reader
#
def head_ablation_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    print(f"Shape of the value tensor: {value.shape}")
    value[:, :, head_index_to_ablate, :] = 0.
    return value

original_loss = model(llama_tokens, return_type="loss")
ablated_loss = model.run_with_hooks(
    llama_tokens,
    return_type="loss",
    fwd_hooks=[(
        utils.get_act_name("v", layer_to_ablate),
        head_ablation_hook
        )]
    )
print(f"Original Loss: {original_loss.item():.3f}")
print(f"Ablated Loss: {ablated_loss.item():.3f}")


