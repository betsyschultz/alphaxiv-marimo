"""
run_cumulative_ablation.py — Progressive head removal across 3 sort orders.

For each ordering (sink_first, important_first, random), progressively zero out
heads at 14 sample points and measure WikiText-2 perplexity. Produces the full
cumulative ablation curve showing that sink heads can be removed cheaply while
important heads cause immediate collapse.

Output: cumulative_ablation.json
"""

import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ── Config ──────────────────────────────────────────────────────────────────

SAMPLE_POINTS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 144]

EVAL_TEXT = (
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


# ── Dataset ─────────────────────────────────────────────────────────────────

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


# ── Ablation hook ───────────────────────────────────────────────────────────

class HeadAblationHook:
    """Zero out specific attention heads via pre-hook on c_proj."""

    def __init__(self, heads_to_ablate, n_heads):
        """heads_to_ablate: dict mapping layer_idx -> set of head indices."""
        self.heads_to_ablate = heads_to_ablate
        self.n_heads = n_heads
        self.handles = []

    def _make_hook(self, layer_idx):
        ablate_heads = self.heads_to_ablate.get(layer_idx, set())
        n_heads = self.n_heads

        def hook_fn(module, args):
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


# ── Entropy ranking ─────────────────────────────────────────────────────────

def rank_heads_by_entropy(model, tokenizer, device):
    """Forward pass on eval text, return all 144 heads sorted by entropy (asc)."""
    inputs = tokenizer(EVAL_TEXT, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    n_layers = len(out.attentions)
    n_heads = out.attentions[0].shape[1]

    head_entropies = []
    for li, layer_attn in enumerate(out.attentions):
        p = layer_attn[0].cpu().numpy().clip(1e-9, 1.0)
        ent = -(p * np.log(p)).sum(axis=-1).mean(axis=-1)  # shape: (n_heads,)
        for hi in range(n_heads):
            head_entropies.append((li, hi, float(ent[hi])))

    # Sort by entropy ascending (lowest entropy = strongest sinks first)
    head_entropies.sort(key=lambda x: x[2])
    return head_entropies


# ── Perplexity ──────────────────────────────────────────────────────────────

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


def compute_perplexity_with_ablation(model, dataloader, device, heads_list, n_model_heads):
    """Attach ablation hook for given heads, compute PPL, remove hook."""
    if not heads_list:
        return compute_perplexity(model, dataloader, device)

    heads_dict = {}
    for li, hi in heads_list:
        heads_dict.setdefault(li, set()).add(hi)

    hook = HeadAblationHook(heads_dict, n_model_heads)
    hook.attach(model)
    ppl = compute_perplexity(model, dataloader, device)
    hook.remove()
    return ppl


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading GPT-2...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
    model.to(device)
    model.eval()

    n_model_heads = model.config.n_head  # 12

    print("Loading WikiText-2 validation...")
    val_dataset = WikiTextDataset("validation", tokenizer, max_len=256)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=True)
    print(f"  {len(val_dataset)} chunks")

    # Rank all 144 heads by entropy
    print("Ranking heads by entropy...")
    ranked = rank_heads_by_entropy(model, tokenizer, device)
    print(f"  Entropy range: {ranked[0][2]:.4f} (lowest) to {ranked[-1][2]:.4f} (highest)")

    # Build 3 sort orders
    sink_first = [(li, hi) for li, hi, _ in ranked]           # lowest entropy first
    important_first = [(li, hi) for li, hi, _ in reversed(ranked)]  # highest entropy first

    rng = np.random.RandomState(42)
    random_order = list(sink_first)  # copy
    rng.shuffle(random_order)

    orders = {
        "sink_first": sink_first,
        "important_first": important_first,
        "random": random_order,
    }

    # Compute baseline once (n=0)
    print("\nBaseline (0 heads ablated)...")
    t0 = time.time()
    baseline_ppl = compute_perplexity(model, val_loader, device)
    print(f"  PPL = {baseline_ppl:.2f} ({time.time() - t0:.1f}s)")

    # Run cumulative ablation for each order
    curves = {}
    total_evals = sum(1 for p in SAMPLE_POINTS if p > 0) * len(orders)
    done = 0
    overall_t0 = time.time()

    for order_name, head_order in orders.items():
        print(f"\n{'=' * 60}")
        print(f"Order: {order_name}")
        print(f"{'=' * 60}")

        curve = []
        for n in SAMPLE_POINTS:
            if n == 0:
                ppl = baseline_ppl
                print(f"  n={n:>3}: PPL = {ppl:.2f} (baseline)")
            else:
                done += 1
                heads_to_remove = head_order[:n]
                t0 = time.time()
                ppl = compute_perplexity_with_ablation(
                    model, val_loader, device, heads_to_remove, n_model_heads
                )
                elapsed = time.time() - t0
                eta_mins = (elapsed * (total_evals - done)) / 60
                print(f"  n={n:>3}: PPL = {ppl:.2f} (+{ppl - baseline_ppl:.2f}) "
                      f"[{elapsed:.1f}s, ~{eta_mins:.0f}min remaining]")

            curve.append({"n_heads": n, "perplexity": round(ppl, 2)})

        curves[order_name] = curve

    total_time = time.time() - overall_t0

    # Save
    result = {
        "sample_points": SAMPLE_POINTS,
        "curves": curves,
    }

    with open("cumulative_ablation.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done in {total_time / 60:.1f} minutes")
    print("Saved to cumulative_ablation.json")

    # Quick summary
    for name, curve in curves.items():
        final = curve[-1]["perplexity"]
        mid = next(p for p in curve if p["n_heads"] == 30)["perplexity"]
        print(f"  {name}: n=30 → {mid:.2f}, n=144 → {final:.2f}")


if __name__ == "__main__":
    main()
