"""
run_pythia_validation.py — Cross-model attention sink comparison.

Runs sink analysis on GPT-2 and Pythia-70M to prove sinks aren't
architecture-specific. Computes sink magnitude, per-head entropy,
sick head count, and WikiText-2 perplexity for each model.

Output: pythia_results.json
"""

import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


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

MODELS = {
    "gpt2": {
        "name": "gpt2",
        "display": "gpt2",
        "n_params": 124_000_000,
    },
    "pythia-70m": {
        "name": "EleutherAI/pythia-70m",
        "display": "EleutherAI/pythia-70m",
        "n_params": 70_000_000,
    },
}


# -- Dataset -----------------------------------------------------------------

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


# -- Perplexity --------------------------------------------------------------

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


# -- Sink analysis -----------------------------------------------------------

def analyze_sinks(model, tokenizer, device):
    """Forward pass with output_attentions, compute sink metrics."""
    inputs = tokenizer(EVAL_TEXT, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    n_layers = len(out.attentions)
    n_heads = out.attentions[0].shape[1]

    # Sink magnitude per layer: fraction of attention on position 0, averaged
    # across heads and query positions
    sink_mag = []
    for li, layer_attn in enumerate(out.attentions):
        # shape: (1, n_heads, seq_len, seq_len)
        attn = layer_attn[0].cpu().float().numpy()
        # Replace any NaN with 0 for robustness
        attn = np.nan_to_num(attn, nan=0.0)
        # attention to position 0 across all heads and query positions
        sink_frac = attn[:, :, 0].mean()
        sink_mag.append(float(sink_frac))

    # Per-head entropy
    entropy = np.zeros((n_layers, n_heads))
    for li, layer_attn in enumerate(out.attentions):
        p = layer_attn[0].cpu().float().numpy()
        p = np.nan_to_num(p, nan=0.0)
        p = p.clip(1e-9, 1.0)
        # p shape: (n_heads, seq_len, seq_len)
        # entropy per head: average entropy across query positions
        entropy[li] = -(p * np.log(p)).sum(axis=-1).mean(axis=-1)

    # Sick heads: entropy < 70% of median
    median_ent = np.median(entropy)
    threshold = median_ent * 0.7
    n_sick = int((entropy < threshold).sum())
    n_total = n_layers * n_heads

    # Aggregate sink stats
    sink_waste_pct = float(np.mean(sink_mag) * 100)
    peak_sink_pct = float(np.max(sink_mag) * 100)

    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "sink_waste_pct": round(sink_waste_pct, 1),
        "peak_sink_pct": round(peak_sink_pct, 1),
        "n_sick_heads": n_sick,
        "n_total_heads": n_total,
        "sick_pct": round(n_sick / n_total * 100, 1),
        "sink_magnitude_per_layer": [round(x, 4) for x in sink_mag],
        "entropy": [[round(float(entropy[li, hi]), 4) for hi in range(n_heads)] for li in range(n_layers)],
    }


# -- Main --------------------------------------------------------------------

def run_model(key, spec, device, cached_ppl=None):
    print(f"\n{'=' * 60}")
    print(f"  {spec['display']}")
    print(f"{'=' * 60}")

    t0 = time.time()

    print(f"  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(spec["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # All models need attn_implementation="eager" for output_attentions=True
    model = AutoModelForCausalLM.from_pretrained(
        spec["name"], attn_implementation="eager"
    )
    model.to(device)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Sink analysis
    print(f"  Running sink analysis...")
    t1 = time.time()
    result = analyze_sinks(model, tokenizer, device)
    print(f"  Sink analysis done in {time.time() - t1:.1f}s")
    print(f"    sink_waste_pct: {result['sink_waste_pct']}%")
    print(f"    peak_sink_pct:  {result['peak_sink_pct']}%")
    print(f"    sick heads:     {result['n_sick_heads']}/{result['n_total_heads']} ({result['sick_pct']}%)")

    # Perplexity
    if cached_ppl is not None:
        ppl = cached_ppl
        print(f"  Perplexity: {ppl:.2f} (cached from ablation_results.json)")
    else:
        print(f"  Computing WikiText-2 perplexity...")
        t2 = time.time()
        # Reload model without attn_implementation="eager" for Pythia
        # (eager causes NaN loss in GPTNeoX models)
        del model
        ppl_model = AutoModelForCausalLM.from_pretrained(spec["name"])
        ppl_model.to(device)
        ppl_model.eval()
        val_dataset = WikiTextDataset("validation", tokenizer, max_len=256)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=True)
        ppl = compute_perplexity(ppl_model, val_loader, device)
        del ppl_model
        print(f"  Perplexity: {ppl:.2f} ({time.time() - t2:.1f}s)")

    result["model_name"] = spec["display"]
    result["n_params"] = spec["n_params"]
    result["perplexity"] = round(ppl, 2)

    # Clean up to free memory before next model
    try:
        del model
    except NameError:
        pass
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Total: {time.time() - t0:.1f}s")
    return result


def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    # Use cached GPT-2 perplexity from ablation_results.json to save ~20 min
    gpt2_cached_ppl = None
    try:
        with open("ablation_results.json") as f:
            ablation = json.load(f)
        gpt2_cached_ppl = ablation["baseline"]["perplexity"]
        print(f"Using cached GPT-2 perplexity: {gpt2_cached_ppl}")
    except (FileNotFoundError, KeyError):
        print("No cached GPT-2 perplexity found, will compute from scratch")

    results = {}
    for key, spec in MODELS.items():
        cached = gpt2_cached_ppl if key == "gpt2" else None
        results[key] = run_model(key, spec, device, cached_ppl=cached)

    # Summary
    print(f"\n{'=' * 60}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25} {'GPT-2':>12} {'Pythia-70M':>12}")
    print(f"  {'-' * 49}")
    for metric in ["sink_waste_pct", "peak_sink_pct", "sick_pct", "perplexity"]:
        g = results["gpt2"][metric]
        p = results["pythia-70m"][metric]
        print(f"  {metric:<25} {g:>12} {p:>12}")

    with open("pythia_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to pythia_results.json")


if __name__ == "__main__":
    main()
