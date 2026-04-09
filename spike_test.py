"""
Spike Test: Do attention sinks appear in GPT-2 (124M) on CPU?
===============================================================
Runtime: ~30-60 min on M-series Mac (most time is in the heatmap rendering)
Actual inference: <30 seconds

Pass criteria:
  - In layers 0-2, >50% of heads assign >15% avg weight to position 0
  - Visible as a bright vertical stripe in column 0 of attention heatmaps

Then: apply Exclusive Self Attention mask and check if sinks diminish.

Usage:
  pip install transformers torch matplotlib numpy
  python spike_test.py
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = Path(__file__).parent / "spike_results"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load model + tokenize a medium-length passage
# ---------------------------------------------------------------------------

# Try models in order of preference for sink observation
MODELS = [
    ("gpt2", "GPT-2 (124M)"),
    ("EleutherAI/pythia-70m", "Pythia 70M"),
    ("distilgpt2", "DistilGPT-2 (82M)"),
]

model = None
tokenizer = None
model_name = None

for model_id, label in MODELS:
    try:
        print(f"Trying {label}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True)
        model.eval()
        model_name = label
        print(f"Loaded {label} ✓")
        break
    except Exception as e:
        print(f"  Failed: {e}")
        continue

if model is None:
    print("ERROR: Could not load any model. Check your transformers + torch install.")
    sys.exit(1)

# ~200 tokens of coherent English — enough to see sink patterns
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
    "certain token positions — particularly the first token — absorb a "
    "disproportionate share of attention weight across many heads and "
    "layers. These attention sinks appear to emerge from the pre-norm "
    "architecture used in most modern transformers. They have practical "
    "consequences for quantization and KV-cache optimization."
)

inputs = tokenizer(TEXT, return_tensors="pt")
seq_len = inputs["input_ids"].shape[1]
tokens = [tokenizer.decode(t) for t in inputs["input_ids"][0]]
print(f"Sequence length: {seq_len} tokens")


# ---------------------------------------------------------------------------
# 2. Standard attention — check for sinks
# ---------------------------------------------------------------------------

print("\nRunning standard attention...")
with torch.no_grad():
    standard_out = model(**inputs, output_attentions=True)

# Extract attentions — handle both old tuple and new format
raw_attn = standard_out.attentions
if raw_attn is None or len(raw_attn) == 0:
    print("ERROR: Model returned no attentions. Trying with attn_implementation='eager'...")
    model = AutoModelForCausalLM.from_pretrained(
        model.config._name_or_path,
        output_attentions=True,
        attn_implementation="eager",
    )
    model.eval()
    with torch.no_grad():
        standard_out = model(**inputs, output_attentions=True)
    raw_attn = standard_out.attentions

if raw_attn is None or len(raw_attn) == 0:
    print("ERROR: Still no attentions. Cannot proceed.")
    sys.exit(1)

# attentions: tuple of (batch, heads, seq, seq) per layer
standard_attn = [layer[0].numpy() for layer in raw_attn]
n_layers = len(standard_attn)
n_heads = standard_attn[0].shape[0]
print(f"Model: {n_layers} layers, {n_heads} heads per layer")


def measure_sinks(attn_layers, label=""):
    """For each layer, compute % of heads where position 0 gets >15% avg weight."""
    results = []
    for layer_idx, attn in enumerate(attn_layers):
        # attn shape: (heads, seq, seq)
        # For each head, average attention to position 0 across all query positions
        avg_weight_to_pos0 = attn[:, :, 0].mean(axis=1)  # (heads,)
        sink_heads = (avg_weight_to_pos0 > 0.15).sum()
        pct = sink_heads / n_heads * 100
        results.append({
            "layer": layer_idx,
            "sink_heads": int(sink_heads),
            "pct": pct,
            "max_weight": float(avg_weight_to_pos0.max()),
            "mean_weight": float(avg_weight_to_pos0.mean()),
        })
    return results


print("\n--- STANDARD ATTENTION: Sink Analysis ---")
print(f"{'Layer':>5} | {'Sink Heads':>10} | {'%':>6} | {'Max Wt':>7} | {'Mean Wt':>7}")
print("-" * 50)
standard_sinks = measure_sinks(standard_attn, "standard")
for r in standard_sinks:
    flag = " <<<" if r["pct"] >= 50 else ""
    print(f"{r['layer']:>5} | {r['sink_heads']:>10} | {r['pct']:>5.1f}% | {r['max_weight']:>6.3f} | {r['mean_weight']:>6.3f}{flag}")

pass_layers = [r for r in standard_sinks[:3] if r["pct"] >= 50]
print(f"\nPASS CRITERIA: >50% sink heads in layers 0-2")
print(f"RESULT: {len(pass_layers)}/3 early layers pass → ", end="")
if len(pass_layers) >= 1:
    print("SINKS DETECTED ✓ — proceed with Plan A (ESA × sink extension)")
else:
    all_sinks = [r for r in standard_sinks if r["pct"] >= 50]
    if all_sinks:
        print(f"AMBIGUOUS — sinks in layers {[r['layer'] for r in all_sinks]} but not early layers")
    else:
        print("NO SINKS ✗ — proceed with Plan B (ESA × positional encoding)")


# ---------------------------------------------------------------------------
# 3. Exclusive Self Attention — mask out the diagonal, re-run
# ---------------------------------------------------------------------------

print("\n\nApplying Exclusive Self Attention (masking self-position)...")


def esa_hook_factory(layer_idx):
    """Create a hook that masks self-attention (diagonal) before softmax.

    GPT2Attention computes: attn_weights = torch.matmul(query, key.T) / sqrt(d)
    then applies causal mask, then softmax. We hook into the attention weights
    AFTER softmax by re-normalizing with diagonal zeroed out.

    Note: ideally we'd hook pre-softmax. For this spike test, post-softmax
    diagonal removal + renormalization is a valid approximation.
    """
    def hook(module, args, output):
        # output is (attn_output, attn_weights) or (attn_output, attn_weights, past_kv)
        attn_output = output[0]
        attn_weights = output[1]  # (batch, heads, seq, seq)

        # Zero out diagonal (self-attention)
        mask = 1.0 - torch.eye(attn_weights.shape[-1], device=attn_weights.device)
        masked_weights = attn_weights * mask

        # Renormalize so rows sum to 1
        masked_weights = masked_weights / (masked_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Recompute attention output: masked_weights @ V
        # We need V from the module internals — extract from the forward pass
        # For the spike test, we just return the modified weights for analysis
        # (we don't need correct output, just the attention pattern)

        if len(output) == 3:
            return (attn_output, masked_weights, output[2])
        return (attn_output, masked_weights)

    return hook


# Register hooks on all attention layers
hooks = []
for i, layer in enumerate(model.transformer.h):
    h = layer.attn.register_forward_hook(esa_hook_factory(i))
    hooks.append(h)

with torch.no_grad():
    esa_out = model(**inputs, output_attentions=True)

# Remove hooks
for h in hooks:
    h.remove()

esa_attn = [layer[0].numpy() for layer in esa_out.attentions]

print("\n--- EXCLUSIVE SELF ATTENTION: Sink Analysis ---")
print(f"{'Layer':>5} | {'Sink Heads':>10} | {'%':>6} | {'Max Wt':>7} | {'Mean Wt':>7}")
print("-" * 50)
esa_sinks = measure_sinks(esa_attn, "ESA")
for r in esa_sinks:
    flag = " <<<" if r["pct"] >= 50 else ""
    print(f"{r['layer']:>5} | {r['sink_heads']:>10} | {r['pct']:>5.1f}% | {r['max_weight']:>6.3f} | {r['mean_weight']:>6.3f}{flag}")


# ---------------------------------------------------------------------------
# 4. Compare: did ESA reduce sinks?
# ---------------------------------------------------------------------------

print("\n\n--- COMPARISON: Standard vs ESA ---")
print(f"{'Layer':>5} | {'Std Sink%':>9} | {'ESA Sink%':>9} | {'Delta':>7} | {'Verdict':>10}")
print("-" * 55)
for s, e in zip(standard_sinks, esa_sinks):
    delta = e["pct"] - s["pct"]
    if delta < -10:
        verdict = "REDUCED"
    elif delta > 10:
        verdict = "INCREASED"
    else:
        verdict = "UNCHANGED"
    print(f"{s['layer']:>5} | {s['pct']:>8.1f}% | {e['pct']:>8.1f}% | {delta:>+6.1f}% | {verdict:>10}")

# Summary
std_total = sum(r["mean_weight"] for r in standard_sinks)
esa_total = sum(r["mean_weight"] for r in esa_sinks)
pct_change = (esa_total - std_total) / std_total * 100

print(f"\nAggregate sink weight: standard={std_total:.3f}, ESA={esa_total:.3f} ({pct_change:+.1f}%)")
if pct_change < -20:
    print("VERDICT: ESA substantially reduces sinks → Plan A is strong")
elif pct_change < -5:
    print("VERDICT: ESA partially reduces sinks → Plan A works, frame as partial effect")
elif abs(pct_change) <= 5:
    print("VERDICT: ESA has minimal effect on sinks → pivot to Plan B")
else:
    print("VERDICT: ESA increases sink weight (unexpected) → investigate or pivot to Plan B")


# ---------------------------------------------------------------------------
# 5. Save heatmaps for visual inspection
# ---------------------------------------------------------------------------

print(f"\nGenerating heatmaps → {OUT_DIR}/")

# Plot layers 0, 5, 11 for both conditions
for layer_idx in [0, 5, n_layers - 1]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, attn, title in [
        (axes[0], standard_attn[layer_idx], f"Standard — Layer {layer_idx}"),
        (axes[1], esa_attn[layer_idx], f"Exclusive SA — Layer {layer_idx}"),
    ]:
        # Average across heads for cleaner visualization
        avg_attn = attn.mean(axis=0)  # (seq, seq)
        im = ax.imshow(avg_attn, cmap="viridis", aspect="auto")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"layer_{layer_idx}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved layer_{layer_idx}_comparison.png")

# Plot sink magnitude across all layers
fig, ax = plt.subplots(figsize=(10, 5))
layers = range(n_layers)
std_means = [r["mean_weight"] for r in standard_sinks]
esa_means = [r["mean_weight"] for r in esa_sinks]
ax.plot(layers, std_means, "o-", label="Standard", color="#e74c3c", linewidth=2)
ax.plot(layers, esa_means, "s-", label="Exclusive SA", color="#3498db", linewidth=2)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Mean attention weight to position 0", fontsize=12)
ax.set_title("Attention Sink Magnitude: Standard vs Exclusive Self-Attention", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "sink_magnitude_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved sink_magnitude_comparison.png")

print("\n✓ Spike test complete. Check spike_results/ for heatmaps.")
print("Share results with the team to decide Plan A vs Plan B.")
