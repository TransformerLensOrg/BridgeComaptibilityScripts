from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.model_bridge import TransformerBridge
import torch as t

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# %%
# NBVAL_IGNORE_OUTPUT


reference_gpt2 = TransformerBridge.boot_transformers(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
    device=device,
)

# %%

# [1.1] Transformer From Scratch
# 1️⃣ UNDERSTANDING INPUTS & OUTPUTS OF A TRANSFORMER

sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])
first_vocab = sorted_vocab[0]
assert isinstance(first_vocab, tuple)
assert isinstance(first_vocab[0], str)
print(first_vocab[1])

# %%
print(reference_gpt2.to_str_tokens("Ralph"))

# %%
print(reference_gpt2.to_str_tokens(" Ralph"))

# %%

print(reference_gpt2.to_str_tokens(" ralph"))


# %%
print(reference_gpt2.to_str_tokens("ralph"))

# %%

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text)
print(tokens.shape)


# %%

logits, cache = reference_gpt2.run_with_cache(tokens, device=device)
print(logits.shape)


# %%

most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])
print(most_likely_next_tokens[-1])



# %%
# 2️⃣ CLEAN TRANSFORMER IMPLEMENTATION

layer_0_hooks = [
    (name, tuple(tensor.shape)) for name, tensor in cache.items() if ".0." in name
]
non_layer_hooks = [
    (name, tuple(tensor.shape)) for name, tensor in cache.items() if "blocks" not in name
]


print(*sorted(non_layer_hooks, key=lambda x: x[0]), sep="\n")


# %%

print(*sorted(layer_0_hooks, key=lambda x: x[0]), sep="\n")

# %%
# NBVAL_IGNORE_OUTPUT
# [1.2] Intro to mech interp
# 2️⃣ FINDING INDUCTION HEADS

cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True, # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b", 
    seed=398,
    use_attn_result=True,
    normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer"
)
model = HookedTransformer(cfg)

# %%


text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)

print(logits.shape)

# %%
print(cache["embed"].ndim)


