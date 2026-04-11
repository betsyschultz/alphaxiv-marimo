"""
run_lambda_sweep.py — Train at 3 alignment loss weights to prove sinks resist
regardless of gradient pressure.

Runs 500 steps each at lambda_align = 0.1, 0.5, 1.0.
If sinks hold at lambda=1.0 (where alignment loss equals LM loss), the
"signal too weak" critique is dead.

Output: lambda_sweep_results.json
"""

import json
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset

import random
import numpy as np


SEED = 42
STEPS = 500
LAMBDAS = [0.1, 0.5, 1.0]
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 5e-5
MAX_LEN = 256
LOG_EVERY = 50


def seed_everything(seed=SEED):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def compute_head_entropy(attn_weights):
    p = attn_weights.clamp(min=1e-9)
    entropy = -(p * p.log()).sum(dim=-1)
    return entropy.mean(dim=(0, 2))


def compute_alignment_loss(all_attentions, n_heads):
    total_cosine_dist = 0.0
    total_sick = 0
    all_pos0_attn = []

    for layer_attn in all_attentions:
        head_ent = compute_head_entropy(layer_attn)
        median_ent = head_ent.median()
        sick_threshold = median_ent * 0.7

        is_sick = head_ent < sick_threshold
        is_healthy = head_ent >= sick_threshold

        all_pos0_attn.append(layer_attn[:, :, :, 0].mean().item())

        n_sick = is_sick.sum().item()
        n_healthy = is_healthy.sum().item()
        if n_sick == 0 or n_healthy == 0:
            continue

        healthy_indices = torch.where(is_healthy)[0]
        with torch.no_grad():
            healthy_ref = layer_attn[:, healthy_indices, :, :].mean(dim=1, keepdim=True)

        sick_indices = torch.where(is_sick)[0]
        for si in sick_indices:
            sick_pattern = layer_attn[:, si, :, :].reshape(-1)
            ref_pattern = healthy_ref.reshape(-1)
            cos_sim = F.cosine_similarity(sick_pattern.unsqueeze(0), ref_pattern.unsqueeze(0))
            total_cosine_dist += (1.0 - cos_sim)
            total_sick += 1

    if total_sick == 0:
        return torch.tensor(0.0, device=all_attentions[0].device), 0, 0.0

    alignment_loss = total_cosine_dist / total_sick
    sink_waste_pct = sum(all_pos0_attn) / len(all_pos0_attn) * 100.0
    return alignment_loss, total_sick, sink_waste_pct


def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            total_loss += outputs.loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    return math.exp(total_loss / total_tokens)


def full_eval_metrics(model, tokenizer, device):
    """Quick sink metrics on eval text."""
    eval_text = (
        "The history of attention mechanisms in neural networks begins with "
        "Bahdanau's 2014 paper on sequence-to-sequence models. Rather than "
        "compressing an entire input sequence into a fixed-length vector, the "
        "model learned to attend to different parts of the input at each "
        "decoding step. This idea was transformative. Vaswani's 2017 paper "
        "introduced the Transformer architecture, which replaced recurrence "
        "entirely with self-attention. Each token attends to every other token "
        "in the sequence, weighted by learned query-key dot products."
    )
    inputs = tokenizer(eval_text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    n_layers = len(out.attentions)
    n_heads = out.attentions[0].shape[1]

    entropy = torch.zeros(n_layers, n_heads)
    sink_per_layer = []

    for li, layer_attn in enumerate(out.attentions):
        entropy[li] = compute_head_entropy(layer_attn).cpu()
        sink_per_layer.append(layer_attn[0, :, :, 0].mean().item() * 100)

    median = entropy.median().item()
    threshold = median * 0.7
    n_sick = int((entropy < threshold).sum().item())

    return {
        "sink_waste_pct": round(sum(sink_per_layer) / len(sink_per_layer), 2),
        "num_sick_heads": n_sick,
    }


def train_one_lambda(lam, tokenizer, train_loader, val_loader, device):
    """Train GPT-2 for STEPS optimizer steps at a given lambda_align."""
    seed_everything(SEED)

    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
    model.to(device)

    # Baseline before training
    baseline_metrics = full_eval_metrics(model, tokenizer, device)
    baseline_ppl = compute_perplexity(model, val_loader, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    warmup = STEPS // 10
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, STEPS)

    model.train()
    global_step = 0
    log_history = []
    running_lm = 0.0
    running_align = 0.0
    running_sink = 0.0
    running_sick = 0
    log_count = 0

    for batch in train_loader:
        if global_step >= STEPS:
            break

        input_ids = batch.to(device)
        outputs = model(input_ids=input_ids, labels=input_ids, output_attentions=True)

        lm_loss = outputs.loss / GRAD_ACCUM
        align_loss, num_sick, sink_pct = compute_alignment_loss(
            outputs.attentions, model.config.n_head
        )
        scaled_align = (lam * align_loss) / GRAD_ACCUM
        total_loss = lm_loss + scaled_align
        total_loss.backward()

        running_lm += lm_loss.item() * GRAD_ACCUM
        if isinstance(align_loss, torch.Tensor):
            running_align += align_loss.item()
        running_sink += sink_pct
        running_sick += num_sick
        log_count += 1

        if log_count % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                n = LOG_EVERY * GRAD_ACCUM
                entry = {
                    "step": global_step,
                    "lm_loss": round(running_lm / n, 4),
                    "align_loss": round(running_align / n, 4),
                    "sink_waste_pct": round(running_sink / n, 2),
                    "num_sick_heads": round(running_sick / n, 1),
                }
                log_history.append(entry)
                print(
                    f"    step {global_step:4d} | "
                    f"lm {entry['lm_loss']:.4f} | "
                    f"align {entry['align_loss']:.4f} | "
                    f"sink {entry['sink_waste_pct']:.1f}% | "
                    f"sick {entry['num_sick_heads']:.0f}"
                )
                running_lm = 0.0
                running_align = 0.0
                running_sink = 0.0
                running_sick = 0

    # Final eval
    final_metrics = full_eval_metrics(model, tokenizer, device)
    final_ppl = compute_perplexity(model, val_loader, device)

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    return {
        "lambda_align": lam,
        "steps": global_step,
        "baseline": {
            "perplexity": round(baseline_ppl, 2),
            **baseline_metrics,
        },
        "final": {
            "perplexity": round(final_ppl, 2),
            **final_metrics,
        },
        "log_history": log_history,
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = WikiTextDataset("train", tokenizer, max_len=MAX_LEN)
    val_dataset = WikiTextDataset("validation", tokenizer, max_len=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    print(f"Train: {len(train_dataset)} chunks, Val: {len(val_dataset)} chunks")

    results = []
    for lam in LAMBDAS:
        print(f"\n{'=' * 60}")
        print(f"TRAINING: lambda_align = {lam} ({STEPS} steps)")
        print(f"{'=' * 60}")
        t0 = time.time()
        r = train_one_lambda(lam, tokenizer, train_loader, val_loader, device)
        elapsed = time.time() - t0
        r["elapsed_seconds"] = round(elapsed, 1)
        results.append(r)

        print(f"\n  Baseline PPL: {r['baseline']['perplexity']}")
        print(f"  Final PPL:    {r['final']['perplexity']}")
        print(f"  Sink waste:   {r['baseline']['sink_waste_pct']}% → {r['final']['sink_waste_pct']}%")
        print(f"  Sick heads:   {r['baseline']['num_sick_heads']} → {r['final']['num_sick_heads']}")
        print(f"  Time: {elapsed:.0f}s")

    # Summary
    print(f"\n{'=' * 60}")
    print("LAMBDA SWEEP SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Lambda':>8} {'PPL before':>12} {'PPL after':>12} {'Sink before':>12} {'Sink after':>12} {'Sick before':>12} {'Sick after':>12}")
    print(f"  {'-' * 80}")
    for r in results:
        b, f_ = r["baseline"], r["final"]
        print(
            f"  {r['lambda_align']:>8.1f} "
            f"{b['perplexity']:>12.2f} "
            f"{f_['perplexity']:>12.2f} "
            f"{b['sink_waste_pct']:>11.1f}% "
            f"{f_['sink_waste_pct']:>11.1f}% "
            f"{b['num_sick_heads']:>12} "
            f"{f_['num_sick_heads']:>12}"
        )

    with open("lambda_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to lambda_sweep_results.json")


if __name__ == "__main__":
    main()
