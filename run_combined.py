"""Standalone script: compute sink + adaptive temperature combined metrics."""
import numpy as np
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

start = time.time()
print("Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()

TEXT = (
    "The history of attention mechanisms in neural networks begins with "
    "Bahdanau's 2014 paper on sequence-to-sequence models. Rather than "
    "compressing an entire input sequence into a fixed-length vector, the "
    "model learned to attend to different parts of the input at each "
    "decoding step. This idea was transformative. Vaswani's 2017 paper "
    "introduced the Transformer architecture, which replaced recurrence "
    "entirely with self-attention. Each token attends to every other token "
    "in the sequence, weighted by learned query-key dot products. The "
    "resulting architecture proved remarkably scalable. GPT, BERT, and "
    "their descendants all build on this foundation. But attention has "
    "quirks. Recent work by LeCun and collaborators at NYU discovered that "
    "certain token positions absorb a disproportionate share of attention "
    "weight across many heads and layers. These attention sinks appear to "
    "emerge from the pre-norm architecture used in most modern transformers."
)
inputs = tokenizer(TEXT, return_tensors="pt")
seq_len = inputs["input_ids"].shape[1]

n_heads = model.config.n_head
d_model = model.config.n_embd
d_head = d_model // n_heads
n_layers = model.config.n_layer

print(f"Sequence length: {seq_len} tokens, {n_layers} layers, {n_heads} heads")

# ── Standard forward pass with QKV hooks ──
qkv_cache = {}
def capture_qkv(layer_idx):
    def hook_fn(module, input, output):
        qkv_cache[layer_idx] = output.detach()
    return hook_fn

hooks = []
for i, block in enumerate(model.transformer.h):
    hooks.append(block.attn.c_attn.register_forward_hook(capture_qkv(i)))

with torch.no_grad():
    standard_out = model(**inputs, output_attentions=True)

for h in hooks:
    h.remove()

standard_attn = np.stack([layer[0].numpy() for layer in standard_out.attentions])

# Compute raw scores (standard)
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
raw_scores_all = []
for i in range(n_layers):
    q, k, _ = qkv_cache[i].split(d_model, dim=-1)
    q = q.view(1, seq_len, n_heads, d_head).transpose(1, 2)
    k = k.view(1, seq_len, n_heads, d_head).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
    scores = scores.masked_fill(causal_mask == 0, float('-inf'))
    raw_scores_all.append(scores[0].numpy())
raw_scores_all = np.stack(raw_scores_all)

# ── Sink token forward pass with QKV hooks ──
sink_ids = torch.cat([torch.zeros(1, 1, dtype=torch.long), inputs["input_ids"]], dim=1)
sink_seq_len = sink_ids.shape[1]

qkv_cache_sink = {}
def capture_qkv_sink(layer_idx):
    def hook_fn(module, input, output):
        qkv_cache_sink[layer_idx] = output.detach()
    return hook_fn

hooks_sink = []
for i, block in enumerate(model.transformer.h):
    hooks_sink.append(block.attn.c_attn.register_forward_hook(capture_qkv_sink(i)))

with torch.no_grad():
    sink_out = model(input_ids=sink_ids, output_attentions=True)

for h in hooks_sink:
    h.remove()

sink_attn_full = np.stack([layer[0].numpy() for layer in sink_out.attentions])

# Compute raw scores (sink)
causal_mask_sink = torch.tril(torch.ones(sink_seq_len, sink_seq_len)).unsqueeze(0).unsqueeze(0)
raw_scores_sink = []
for i in range(n_layers):
    q, k, _ = qkv_cache_sink[i].split(d_model, dim=-1)
    q = q.view(1, sink_seq_len, n_heads, d_head).transpose(1, 2)
    k = k.view(1, sink_seq_len, n_heads, d_head).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
    scores = scores.masked_fill(causal_mask_sink == 0, float('-inf'))
    raw_scores_sink.append(scores[0].numpy())
raw_scores_sink = np.stack(raw_scores_sink)

print(f"Forward passes done in {time.time() - start:.1f}s")

# ── Compute entropy + classify heads ──
def head_entropy(attn_4d):
    p = np.clip(attn_4d, 1e-9, 1.0)
    return -(p * np.log(p)).sum(axis=-1).mean(axis=-1)

def count_health(ent_2d):
    med = np.median(ent_2d)
    sick = int((ent_2d < med * 0.7).sum())
    diffuse = int((ent_2d > np.percentile(ent_2d, 90)).sum())
    healthy = n_layers * n_heads - sick - diffuse
    return sick, diffuse, healthy

ent_std = head_entropy(standard_attn)
median_ent = np.median(ent_std)
sick_threshold = median_ent * 0.7

# Per-head T map
T_map = np.ones((n_layers, n_heads))
for li in range(n_layers):
    for hi in range(n_heads):
        e = ent_std[li, hi]
        if e < sick_threshold:
            severity = 1.0 - (e / sick_threshold)
            T_map[li, hi] = 1.0 + severity * 2.0

# ── Adaptive-only (standard raw scores + per-head T) ──
adaptive_attn = np.zeros_like(raw_scores_all)
for li in range(n_layers):
    for hi in range(n_heads):
        t = T_map[li, hi]
        scaled = raw_scores_all[li, hi] / t
        exp = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
        adaptive_attn[li, hi] = exp / exp.sum(axis=-1, keepdims=True)

# ── Combined (sink raw scores + per-head T) ──
combined_attn = np.zeros_like(raw_scores_sink)
for li in range(n_layers):
    for hi in range(n_heads):
        t = T_map[li, hi]
        scaled = raw_scores_sink[li, hi] / t
        exp = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
        combined_attn[li, hi] = exp / exp.sum(axis=-1, keepdims=True)

# ── Metrics ──
std_waste = standard_attn[:, :, :, 0].mean() * 100
sink_only_real1 = sink_attn_full[:, :, 1:, 1].mean() * 100
adap_waste = adaptive_attn[:, :, :, 0].mean() * 100
combined_real1 = combined_attn[:, :, 1:, 1].mean() * 100
combined_garbage = combined_attn[:, :, 1:, 0].mean() * 100

# Entropy for sink-only (real tokens only)
sink_real = sink_attn_full[:, :, 1:, :]
ent_sink = head_entropy(sink_real)

ent_adaptive = head_entropy(adaptive_attn)
ent_combined = head_entropy(combined_attn[:, :, 1:, :])

std_sick, std_diff, std_healthy = count_health(ent_std)
sink_sick, sink_diff, sink_healthy = count_health(ent_sink)
adap_sick, adap_diff, adap_healthy = count_health(ent_adaptive)
comb_sick, comb_diff, comb_healthy = count_health(ent_combined)

n_treated = int((T_map > 1.0).sum())
avg_T = T_map[T_map > 1.0].mean() if n_treated > 0 else 0

# ── Print results ──
print("\n" + "=" * 72)
print("COMBINED SINK TOKEN + ADAPTIVE TEMPERATURE — RESULTS")
print("=" * 72)
print(f"\n{n_treated} of 144 heads treated (avg T={avg_T:.2f} for sick heads)")
print(f"Sick threshold: 70% of median entropy = {sick_threshold:.3f}")
print()

header = f"{'Metric':<30} {'Standard':>10} {'Sink only':>10} {'Adaptive':>10} {'Combined':>10}"
print(header)
print("-" * len(header))
print(f"{'Waste on real 1st token':<30} {std_waste:>9.1f}% {sink_only_real1:>9.1f}% {adap_waste:>9.1f}% {combined_real1:>9.1f}%")
print(f"{'Garbage token absorption':<30} {'n/a':>10} {sink_attn_full[:,:,1:,0].mean()*100:>9.1f}% {'n/a':>10} {combined_garbage:>9.1f}%")
print(f"{'Sick heads':<30} {std_sick:>10} {sink_sick:>10} {adap_sick:>10} {comb_sick:>10}")
print(f"{'Healthy heads':<30} {std_healthy:>10} {sink_healthy:>10} {adap_healthy:>10} {comb_healthy:>10}")
print(f"{'Diffuse heads':<30} {std_diff:>10} {sink_diff:>10} {adap_diff:>10} {comb_diff:>10}")

print(f"\n{'Healed vs standard':<30} {'—':>10} {sink_healthy - std_healthy:>+10} {adap_healthy - std_healthy:>+10} {comb_healthy - std_healthy:>+10}")

print(f"\nTotal time: {time.time() - start:.1f}s")
