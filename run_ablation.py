"""
run_ablation.py — Zero out sink heads during inference, measure perplexity impact.

Proves whether sinks are load-bearing or passengers by comparing:
1. Baseline perplexity (no intervention)
2. Perplexity with sink heads zeroed out
3. Perplexity with random heads zeroed out (control)

If sinks are load-bearing, zeroing them should hurt more than random heads.
"""

import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ── Dataset ──────────────────────────────────────────────────────────────────

class WikiTextDataset(Dataset):
    def __init__(self, split, tokenizer, max_len=256):
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join([t for t in raw["text"] if t.strip()])
        tokens = tokenizer.encode(text)
        n_chunks = len(tokens) // max_len
        self.chunks = [
            torch.tensor(tokens[i * max_len : (i + 1) * max_len], dtype=torch.long)
            for i in range(n_chunks)
        ]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


# ── Identify sink heads ─────────────────────────────────────────────────────

def identify_sink_heads(model, tokenizer, device):
    """Run forward pass on eval text, return list of (layer, head) tuples for sick heads."""
    eval_text = (
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

    inputs = tokenizer(eval_text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    # Compute per-head entropy
    n_layers = len(out.attentions)
    n_heads = out.attentions[0].shape[1]
    entropy = np.zeros((n_layers, n_heads))

    for li, layer_attn in enumerate(out.attentions):
        p = layer_attn[0].cpu().numpy().clip(1e-9, 1.0)
        entropy[li] = -(p * np.log(p)).sum(axis=-1).mean(axis=-1)

    median = np.median(entropy)
    threshold = median * 0.7

    sink_heads = []
    for li in range(n_layers):
        for hi in range(n_heads):
            if entropy[li, hi] < threshold:
                sink_heads.append((li, hi))

    return sink_heads, entropy


# ── Ablation hooks ───────────────────────────────────────────────────────────

class HeadAblationHook:
    """Zero out specific attention heads during forward pass."""

    def __init__(self, heads_to_ablate, n_heads):
        """heads_to_ablate: dict mapping layer_idx -> set of head indices."""
        self.heads_to_ablate = heads_to_ablate
        self.n_heads = n_heads
        self.handles = []

    def _make_hook(self, layer_idx):
        ablate_heads = self.heads_to_ablate.get(layer_idx, set())
        n_heads = self.n_heads

        def hook_fn(module, args):
            # Pre-hook: args[0] is c_proj INPUT = merged head outputs
            # Shape: (batch, seq, n_heads * d_head) — heads still separable
            x = args[0]
            batch, seq, total = x.shape
            d_head = total // n_heads

            reshaped = x.view(batch, seq, n_heads, d_head).clone()
            for hi in ablate_heads:
                reshaped[:, :, hi, :] = 0.0

            return (reshaped.view(batch, seq, total),) + args[1:]

        return hook_fn

    def attach(self, model):
        for li, block in enumerate(model.transformer.h):
            if li in self.heads_to_ablate:
                handle = block.attn.c_proj.register_forward_pre_hook(self._make_hook(li))
                self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ── Perplexity computation ───────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch.to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        total_loss += outputs.loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()

    return math.exp(total_loss / total_tokens)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
    model.to(device)
    model.eval()

    # Load validation data
    print("Loading WikiText-2 validation...")
    val_dataset = WikiTextDataset("validation", tokenizer, max_len=256)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=True)

    # Identify sink heads
    print("Identifying sink heads...")
    sink_heads, entropy = identify_sink_heads(model, tokenizer, device)
    n_sink = len(sink_heads)
    print(f"  Found {n_sink} sink heads")

    # Pick random heads (same count) as control — deterministic seed
    rng = np.random.RandomState(42)
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    all_heads = [(li, hi) for li in range(n_layers) for hi in range(n_heads)]
    non_sink = [h for h in all_heads if h not in sink_heads]
    random_indices = rng.choice(len(non_sink), size=n_sink, replace=False)
    random_heads = [non_sink[int(i)] for i in random_indices]

    # Also pick the "diffuse" heads (high entropy, top 10%)
    flat_ent = entropy.flatten()
    diffuse_threshold = np.percentile(flat_ent, 90)
    diffuse_heads = []
    for li in range(n_layers):
        for hi in range(n_heads):
            if entropy[li, hi] > diffuse_threshold:
                diffuse_heads.append((li, hi))

    def heads_to_dict(head_list):
        d = {}
        for li, hi in head_list:
            d.setdefault(li, set()).add(hi)
        return d

    # Run experiments
    results = {}

    # 1. Baseline
    print("\n1. Baseline (no ablation)...")
    t0 = time.time()
    baseline_ppl = compute_perplexity(model, val_loader, device)
    results["baseline"] = {"perplexity": round(baseline_ppl, 2), "heads_ablated": 0}
    print(f"   Perplexity: {baseline_ppl:.2f} ({time.time() - t0:.1f}s)")

    # 2. Ablate sink heads
    print(f"\n2. Ablate sink heads ({n_sink} heads)...")
    t0 = time.time()
    hook = HeadAblationHook(heads_to_dict(sink_heads), n_heads)
    hook.attach(model)
    sink_ppl = compute_perplexity(model, val_loader, device)
    hook.remove()
    results["sink_ablated"] = {
        "perplexity": round(sink_ppl, 2),
        "heads_ablated": n_sink,
        "heads": sink_heads,
        "delta": round(sink_ppl - baseline_ppl, 2),
    }
    print(f"   Perplexity: {sink_ppl:.2f} (+{sink_ppl - baseline_ppl:.2f}) ({time.time() - t0:.1f}s)")

    # 3. Ablate random heads (control)
    print(f"\n3. Ablate random heads ({n_sink} heads, control)...")
    t0 = time.time()
    hook = HeadAblationHook(heads_to_dict(random_heads), n_heads)
    hook.attach(model)
    random_ppl = compute_perplexity(model, val_loader, device)
    hook.remove()
    results["random_ablated"] = {
        "perplexity": round(random_ppl, 2),
        "heads_ablated": n_sink,
        "heads": random_heads,
        "delta": round(random_ppl - baseline_ppl, 2),
    }
    print(f"   Perplexity: {random_ppl:.2f} (+{random_ppl - baseline_ppl:.2f}) ({time.time() - t0:.1f}s)")

    # 4. Ablate diffuse heads
    n_diffuse = len(diffuse_heads)
    print(f"\n4. Ablate diffuse/noisy heads ({n_diffuse} heads)...")
    t0 = time.time()
    hook = HeadAblationHook(heads_to_dict(diffuse_heads), n_heads)
    hook.attach(model)
    diffuse_ppl = compute_perplexity(model, val_loader, device)
    hook.remove()
    results["diffuse_ablated"] = {
        "perplexity": round(diffuse_ppl, 2),
        "heads_ablated": n_diffuse,
        "heads": diffuse_heads,
        "delta": round(diffuse_ppl - baseline_ppl, 2),
    }
    print(f"   Perplexity: {diffuse_ppl:.2f} (+{diffuse_ppl - baseline_ppl:.2f}) ({time.time() - t0:.1f}s)")

    # Summary
    print(f"\n{'=' * 60}")
    print("ABLATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  {'Condition':<30} {'PPL':>8} {'Delta':>8} {'Heads':>6}")
    print(f"  {'-' * 52}")
    for key in ["baseline", "sink_ablated", "random_ablated", "diffuse_ablated"]:
        r = results[key]
        delta = f"+{r.get('delta', 0):.2f}" if r.get("delta", 0) > 0 else f"{r.get('delta', 0):.2f}"
        print(f"  {key:<30} {r['perplexity']:>8.2f} {delta:>8} {r['heads_ablated']:>6}")

    print(f"\n  Sink heads hurt {(sink_ppl - baseline_ppl) / (random_ppl - baseline_ppl):.1f}x more than random heads")
    print(f"  Diffuse heads hurt {(diffuse_ppl - baseline_ppl) / (random_ppl - baseline_ppl):.1f}x more than random heads")

    # Save
    # Convert sets/tuples for JSON
    for key in results:
        if "heads" in results[key]:
            results[key]["heads"] = [[int(li), int(hi)] for li, hi in results[key]["heads"]]

    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to ablation_results.json")


if __name__ == "__main__":
    main()
