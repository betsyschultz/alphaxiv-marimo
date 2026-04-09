"""
Validation test: Can we extract pre-softmax attention logits from GPT-2?
=========================================================================
Tests three things:
1. Can we hook into attention layers and get pre-softmax QK scores?
2. Can we apply ReLU attention post-hoc from those scores?
3. Does prepending a sink token break positional encoding?

Run: python3 validate_hooks.py
"""

import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("VALIDATION TEST: Pre-softmax hooks + fix feasibility")
print("=" * 60)

# Load model
print("\n[1/5] Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()

text = "The cat sat on the mat and looked around the room quietly"
inputs = tokenizer(text, return_tensors="pt")
seq_len = inputs["input_ids"].shape[1]
print(f"  Tokens: {seq_len}")
print(f"  Transformers version: ", end="")
import transformers
print(transformers.__version__)

# ─── TEST 1: Extract pre-softmax logits ───
print("\n[2/5] Testing pre-softmax hook extraction...")

pre_softmax_logits = {}
post_softmax_weights = {}

def make_hook(layer_idx):
    def hook_fn(module, args, kwargs, output):
        # In GPT2Attention, we need to intercept the attention computation
        # Let's try to recompute QK from the module's internals
        pass
    return hook_fn

# Alternative approach: recompute QK ourselves from Q, K tensors
qkv_cache = {}

def capture_qkv(layer_idx):
    """Hook the c_attn projection to capture Q, K, V before attention."""
    def hook_fn(module, input, output):
        qkv_cache[layer_idx] = output.detach()
    return hook_fn

hooks = []
for i, block in enumerate(model.transformer.h):
    h = block.attn.c_attn.register_forward_hook(capture_qkv(i))
    hooks.append(h)

with torch.no_grad():
    out = model(**inputs, output_attentions=True)

for h in hooks:
    h.remove()

# Check what we captured
if len(qkv_cache) == 0:
    print("  FAIL: No QKV tensors captured")
    sys.exit(1)

print(f"  Captured QKV from {len(qkv_cache)} layers")
print(f"  QKV shape: {qkv_cache[0].shape}")

# Split into Q, K, V and compute pre-softmax scores
n_heads = model.config.n_head
d_model = model.config.n_embd
d_head = d_model // n_heads

test_layer = 0
qkv = qkv_cache[test_layer]  # (batch, seq, 3 * d_model)
q, k, v = qkv.split(d_model, dim=-1)

# Reshape to (batch, heads, seq, d_head)
q = q.view(1, seq_len, n_heads, d_head).transpose(1, 2)
k = k.view(1, seq_len, n_heads, d_head).transpose(1, 2)

# Pre-softmax scores: Q @ K^T / sqrt(d_head)
raw_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)

# Apply causal mask
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
raw_scores = raw_scores.masked_fill(causal_mask == 0, float('-inf'))

# Compute softmax ourselves
our_softmax = torch.nn.functional.softmax(raw_scores, dim=-1)

# Compare with model's attention weights
model_attn = out.attentions[test_layer][0].detach()  # (heads, seq, seq)
our_attn = our_softmax[0]  # (heads, seq, seq)

max_diff = (model_attn - our_attn).abs().max().item()
mean_diff = (model_attn - our_attn).abs().mean().item()

print(f"\n  Verification (layer {test_layer}):")
print(f"    Max diff between our softmax and model's: {max_diff:.8f}")
print(f"    Mean diff: {mean_diff:.8f}")

if max_diff < 0.01:
    print("  PASS ✓ — We can reconstruct attention from QKV. Pre-softmax logits available.")
    presoftmax_works = True
else:
    print(f"  FAIL ✗ — Diff too large ({max_diff:.4f}). Model may use rotary/bias we're missing.")
    presoftmax_works = False

# ─── TEST 2: ReLU attention from pre-softmax scores ───
print("\n[3/5] Testing ReLU attention...")

# ReLU attention: replace softmax with ReLU + L1 normalize
relu_scores = torch.relu(raw_scores)
relu_scores = relu_scores.masked_fill(causal_mask == 0, 0)
relu_attn = relu_scores / (relu_scores.sum(dim=-1, keepdim=True) + 1e-9)

relu_attn_np = relu_attn[0].numpy()
sink_weight_relu = relu_attn_np[:, :, 0].mean(axis=1).mean()  # avg across heads and queries
sink_weight_std = model_attn.numpy()[:, :, 0].mean(axis=1).mean()

print(f"  Standard — mean attn to pos 0: {sink_weight_std:.4f}")
print(f"  ReLU     — mean attn to pos 0: {sink_weight_relu:.4f}")
print(f"  Delta: {sink_weight_relu - sink_weight_std:+.4f}")

if sink_weight_relu < sink_weight_std:
    print("  PASS ✓ — ReLU attention reduces sink weight")
    relu_works = True
else:
    print("  NOTE — ReLU didn't reduce sink (may need more tokens/layers to see effect)")
    relu_works = False

# ─── TEST 3: Sink token prepend ───
print("\n[4/5] Testing sink token prepend...")

# Prepend a zero token (token id 0) and see if model still produces valid output
sink_input_ids = torch.cat([
    torch.zeros(1, 1, dtype=torch.long),  # garbage token at position 0
    inputs["input_ids"]
], dim=1)
sink_attention_mask = torch.ones_like(sink_input_ids)

try:
    with torch.no_grad():
        sink_out = model(
            input_ids=sink_input_ids,
            attention_mask=sink_attention_mask,
            output_attentions=True
        )

    # Check: does the prepended token absorb attention?
    sink_attn_layer8 = sink_out.attentions[8][0].numpy()  # (heads, seq+1, seq+1)
    # Attention to position 0 (the prepended garbage token) from real tokens (positions 1+)
    attn_to_garbage = sink_attn_layer8[:, 1:, 0].mean()
    # Attention to position 1 (what used to be position 0) from real tokens
    attn_to_old_first = sink_attn_layer8[:, 1:, 1].mean()

    print(f"  Attn to prepended garbage token (pos 0): {attn_to_garbage:.4f}")
    print(f"  Attn to original first token (now pos 1): {attn_to_old_first:.4f}")

    if attn_to_garbage > 0.1:
        print("  PASS ✓ — Garbage token absorbs attention (acts as designed sink)")
        sink_token_works = True
    else:
        print("  PARTIAL — Garbage token doesn't absorb much. Positional shift may confuse model.")
        sink_token_works = False

    # Check output quality — does the model still produce coherent logits?
    std_logits = out.logits[0, -1, :].topk(5).indices.tolist()
    sink_logits = sink_out.logits[0, -1, :].topk(5).indices.tolist()
    std_tokens = [tokenizer.decode(t) for t in std_logits]
    sink_tokens = [tokenizer.decode(t) for t in sink_logits]
    overlap = len(set(std_logits) & set(sink_logits))

    print(f"\n  Output quality check (top-5 next token predictions):")
    print(f"    Standard:   {std_tokens}")
    print(f"    With sink:  {sink_tokens}")
    print(f"    Overlap: {overlap}/5 tokens match")

    if overlap >= 3:
        print("  PASS ✓ — Output quality preserved with prepended sink token")
    else:
        print("  WARNING — Output diverges significantly. Sink token disrupts model.")

except Exception as e:
    print(f"  FAIL ✗ — Error: {e}")
    sink_token_works = False

# ─── TEST 4: Elastic softmax with fixed offset ───
print("\n[5/5] Testing elastic softmax (fixed offset grid)...")

offsets = [0.0, -0.5, -1.0, -2.0, -5.0]
print(f"  Testing offsets: {offsets}")
print(f"  {'Offset':>8} | {'Sink Weight':>11} | {'Delta vs Std':>12}")
print("  " + "-" * 40)

elastic_results = {}
for offset in offsets:
    shifted_scores = raw_scores + offset
    shifted_scores = shifted_scores.masked_fill(causal_mask == 0, float('-inf'))
    elastic_attn = torch.nn.functional.softmax(shifted_scores, dim=-1)
    sink_w = elastic_attn[0].numpy()[:, :, 0].mean(axis=1).mean()
    elastic_results[offset] = sink_w
    delta = sink_w - sink_weight_std
    print(f"  {offset:>8.1f} | {sink_w:>10.4f} | {delta:>+11.4f}")

best_offset = min(elastic_results, key=elastic_results.get)
print(f"\n  Best offset: {best_offset} (sink weight: {elastic_results[best_offset]:.4f})")
if elastic_results[best_offset] < sink_weight_std * 0.8:
    print("  PASS ✓ — Elastic softmax reduces sink with fixed offset")
    elastic_works = True
else:
    print("  NOTE — Fixed offset doesn't strongly reduce sink. May need per-head offsets.")
    elastic_works = False

# ─── SUMMARY ───
print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
results = {
    "Pre-softmax logits extractable": presoftmax_works,
    "ReLU attention reduces sink": relu_works,
    "Sink token absorbs attention": sink_token_works,
    "Elastic softmax (fixed offset)": elastic_works,
}

for test, passed in results.items():
    status = "✓ PASS" if passed else "⚠ NEEDS WORK"
    print(f"  {status:>14}  {test}")

all_pass = all(results.values())
critical_pass = results["Pre-softmax logits extractable"]

print()
if critical_pass and all_pass:
    print("ALL CLEAR — Full 4-act spec is feasible. Build the 11-cell version.")
elif critical_pass:
    failing = [k for k, v in results.items() if not v]
    print(f"PARTIAL — Pre-softmax works (critical). Adjust: {', '.join(failing)}")
    print("Build 6-cell slim version. Expand fixes that passed validation.")
else:
    print("BLOCKED — Cannot extract pre-softmax logits. Rethink approach.")
    print("Fallback: post-hoc normalization tricks only (ESA + entropy dashboard).")
