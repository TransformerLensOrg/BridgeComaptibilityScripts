# %%
# Janky code to do different setup when run in a Colab notebook vs VSCode
import os

# %%
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio

pio.renderers.default = "notebook_connected"

# %%
# Imports
import torch

from transformers import AutoTokenizer
from transformer_lens import HookedEncoderDecoder
from transformer_lens.model_bridge import TransformerBridge

model_name = "t5-small"
model = TransformerBridge.boot_transformers(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
torch.set_grad_enabled(False)

# %% [markdown]
# ## Basic sanity check - Model generates some tokens

# %%
prompt = "translate English to French: Hello, how are you? "
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
decoder_input_ids = torch.tensor([[model.cfg.decoder_start_token_id]]).to(input_ids.device)


while True:
    logits = model.forward(input=input_ids, one_zero_attention_mask=attention_mask, decoder_input=decoder_input_ids)
    # logits.shape == (batch_size (1), predicted_pos, vocab_size)

    token_idx = torch.argmax(logits[0, -1, :]).item()
    print("generated token: \"", tokenizer.decode(token_idx), "\", token id: ", token_idx, sep="")

    # append token to decoder_input_ids
    decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([[token_idx]]).to(input_ids.device)], dim=-1)

    # break if End-Of-Sequence token generated
    if token_idx == tokenizer.eos_token_id:
        break

print(prompt, "\n", tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True))

# %% [markdown]
# ## Model also allows strings or a list of strings as input
# The model also allows strings and a list of strings as input, not just tokens.
# Here is an example of a string as input to the forward function

# %%
single_prompt = "translate English to French: Hello, do you like apples?"
logits = model(single_prompt)
print(logits.shape)

# %% [markdown]
# And here is an example of a list of strings as input to the forward function:

# %%
prompts = [
        "translate English to German: Hello, do you like bananas?",
        "translate English to French: Hello, do you like bananas?",
        "translate English to Spanish: Hello, do you like bananas?",
    ]

logits = model(prompts)
print(logits.shape)

# %% [markdown]
# ## Text can be generated via the generate function

# %%
prompt="translate English to German: Hello, do you like bananas?"

output = model.generate(prompt, do_sample=False, max_new_tokens=20)
print(output)

# %% [markdown]
# ### visualise encoder patterns

# %%
import circuitsvis as cv
# Testing that the library works
cv.examples.hello("Neel")

# %%
prompt = "translate English to French: Hello, how are you? "
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]


logits,cache = model.run_with_cache(input=input_ids, one_zero_attention_mask=attention_mask, decoder_input=decoder_input_ids, remove_batch_dim=True)

# %%
# the usual way of indexing cache via cache["pattetn",0,"attn"] does not work
# besause it uses cache["block.0....]  indexing
# t5 is implementes as separate stack of blocks for encoder and decoder
# so indexing is cache["encoder.0.."], cache["decoder.0.."] 
# lets see what is in cache and choose the right key for encoder attention pattern on layer 0
print("\n".join(cache.keys()))

# %%
encoder_attn_pattern = cache["encoder.0.attn.hook_pattern"]
input_str_tokens = [w.lstrip("▁") for w in tokenizer.convert_ids_to_tokens(input_ids[0])]

# %%


# %% [markdown]
# ### visualise decoder pattern

# %%
decoder_str_tokens = tokenizer.convert_ids_to_tokens(decoder_input_ids[0])
decoder_str_tokens

# %%
decoder_attn_pattern = cache["decoder.0.attn.hook_pattern"]

# %% [markdown]
# ## topk tokens visualisation

# %%
# list of samples of shape (n_layers, n_tokens, n_neurons) for each sample
# i take the activations after the mlp layer
# you can also pass the activations after the attention layer (hook_attn_out),
#  after the cross attention layer (hook_cross_attn_out) or after the mlp layer (hook_mlp_out)
activations = [
    torch.stack([cache[f"decoder.{layer}.hook_mlp_out"] for layer in range(model.cfg.n_layers)]).cpu().numpy()
    ]

# list of samples of shape (n_tokens)
tokens = [decoder_str_tokens]

# if we have an arbitrary selection of layers, when change the layer labels, now just pass the layer index
layer_labels = [i for i in range(model.cfg.n_layers)]



