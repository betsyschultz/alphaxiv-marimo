"""
train_blend.py — Finetune GPT-2 Small with auxiliary attention alignment loss.

Hypothesis: adding a training loss that encourages sick attention heads
(low-entropy, sink-dumping) to match healthy heads in the same layer
will eliminate attention sinks during actual inference.

Usage:
    python train_blend.py --epochs 3 --lambda_align 0.1 --device mps
    python train_blend.py --eval-only --checkpoint checkpoints/step_1500.pt --device mps
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset


# ── Reproducibility ──────────────────────────────────────────────────────────

SEED = 42

def seed_everything(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random, numpy as np
    random.seed(seed)
    np.seed = seed
    np.random.seed(seed)


# ── Dataset ──────────────────────────────────────────────────────────────────

class WikiTextDataset(Dataset):
    """Tokenized WikiText-2 chunked into fixed-length sequences."""

    def __init__(self, split: str, tokenizer, max_len: int = 256):
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        # Concatenate all text, then chunk into max_len blocks
        text = "\n\n".join([t for t in raw["text"] if t.strip()])
        tokens = tokenizer.encode(text)
        # Drop last incomplete chunk
        n_chunks = len(tokens) // max_len
        self.chunks = [
            torch.tensor(tokens[i * max_len : (i + 1) * max_len], dtype=torch.long)
            for i in range(n_chunks)
        ]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


# ── Attention metrics ────────────────────────────────────────────────────────

def compute_head_entropy(attn_weights):
    """
    Compute per-head entropy from attention weights.

    Args:
        attn_weights: (batch, heads, seq_q, seq_k) — post-softmax attention

    Returns:
        (heads,) tensor of entropy values averaged across batch and query positions.
    """
    p = attn_weights.clamp(min=1e-9)
    entropy = -(p * p.log()).sum(dim=-1)  # (batch, heads, seq_q)
    return entropy.mean(dim=(0, 2))  # average over batch and query positions


def classify_heads(entropy_per_head, median_entropy):
    """
    Classify heads as sick, healthy, or diffuse.

    - Sick: entropy < 70% of median (concentrated, likely sink-dumping)
    - Diffuse: entropy > 90th percentile (too spread out)
    - Healthy: everything else

    Returns:
        is_sick: bool tensor (heads,)
        is_healthy: bool tensor (heads,)
        is_diffuse: bool tensor (heads,)
    """
    sick_threshold = median_entropy * 0.7
    diffuse_threshold = torch.quantile(entropy_per_head, 0.9)

    is_sick = entropy_per_head < sick_threshold
    is_diffuse = entropy_per_head > diffuse_threshold
    is_healthy = ~is_sick & ~is_diffuse

    return is_sick, is_healthy, is_diffuse


def compute_alignment_loss(all_attentions, n_heads):
    """
    Compute the attention alignment loss across all layers.

    For each layer:
    1. Compute per-head entropy
    2. Classify heads as sick/healthy/diffuse
    3. Build a healthy reference pattern (mean of healthy heads, detached)
    4. For each sick head, compute cosine distance to the reference
    5. Average across all sick heads

    Args:
        all_attentions: tuple of (batch, heads, seq_q, seq_k) tensors, one per layer
        n_heads: number of attention heads

    Returns:
        alignment_loss: scalar tensor (mean cosine distance of sick heads to healthy reference)
        num_sick: total count of sick heads across all layers
        sink_waste_pct: mean attention to position 0 across all layers and heads
    """
    total_cosine_dist = 0.0
    total_sick = 0
    all_pos0_attn = []

    for layer_attn in all_attentions:
        # layer_attn: (batch, heads, seq_q, seq_k)
        head_ent = compute_head_entropy(layer_attn)  # (heads,)
        median_ent = head_ent.median()

        is_sick, is_healthy, is_diffuse = classify_heads(head_ent, median_ent)
        n_sick = is_sick.sum().item()
        n_healthy = is_healthy.sum().item()

        # Track sink waste: mean attention to position 0
        all_pos0_attn.append(layer_attn[:, :, :, 0].mean().item())

        if n_sick == 0 or n_healthy == 0:
            continue

        # Build healthy reference (detached — no gradient through reference)
        healthy_indices = torch.where(is_healthy)[0]
        with torch.no_grad():
            healthy_ref = layer_attn[:, healthy_indices, :, :].mean(dim=1, keepdim=True)
            # (batch, 1, seq_q, seq_k)

        # For each sick head, compute cosine distance to healthy reference
        sick_indices = torch.where(is_sick)[0]
        for si in sick_indices:
            sick_pattern = layer_attn[:, si, :, :].reshape(-1)  # flatten
            ref_pattern = healthy_ref.reshape(-1)

            # Cosine distance = 1 - cosine_similarity
            cos_sim = F.cosine_similarity(sick_pattern.unsqueeze(0), ref_pattern.unsqueeze(0))
            total_cosine_dist += (1.0 - cos_sim)
            total_sick += 1

    if total_sick == 0:
        return torch.tensor(0.0, device=all_attentions[0].device), 0, 0.0

    alignment_loss = total_cosine_dist / total_sick
    sink_waste_pct = sum(all_pos0_attn) / len(all_pos0_attn) * 100.0

    return alignment_loss, total_sick, sink_waste_pct


# ── Full evaluation ──────────────────────────────────────────────────────────

@torch.no_grad()
def full_evaluation(model, tokenizer, device, eval_text=None):
    """
    Run full attention sink diagnostics on a fixed evaluation text.

    Returns dict with:
        - sink_waste_pct: mean attention to position 0
        - num_sick_heads: count of sick heads
        - num_healthy_heads: count of healthy heads
        - num_diffuse_heads: count of diffuse heads
        - per_layer_sink: list of per-layer sink waste
        - cosine_sick_to_healthy: mean cosine sim of sick heads to best healthy neighbor
        - perplexity: perplexity on evaluation text (if provided separately)
    """
    model.eval()

    if eval_text is None:
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

    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions  # tuple of (1, heads, seq, seq)

    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]

    # Per-layer metrics
    per_layer_sink = []
    all_entropy = []  # (layers, heads)
    all_attn = []

    for layer_attn in attentions:
        # Sink waste for this layer
        sink_pct = layer_attn[0, :, :, 0].mean().item() * 100.0
        per_layer_sink.append(sink_pct)

        # Per-head entropy
        head_ent = compute_head_entropy(layer_attn)  # (heads,)
        all_entropy.append(head_ent.cpu())
        all_attn.append(layer_attn[0].cpu())  # (heads, seq, seq)

    entropy_grid = torch.stack(all_entropy)  # (layers, heads)
    attn_stack = torch.stack(all_attn)  # (layers, heads, seq, seq)

    # Global classification
    median_ent = entropy_grid.median()
    sick_threshold = median_ent * 0.7
    diffuse_threshold = torch.quantile(entropy_grid.flatten(), 0.9)

    is_sick = entropy_grid < sick_threshold
    is_diffuse = entropy_grid > diffuse_threshold
    is_healthy = ~is_sick & ~is_diffuse

    num_sick = is_sick.sum().item()
    num_diffuse = is_diffuse.sum().item()
    num_healthy = is_healthy.sum().item()

    # Overall sink waste
    sink_waste_pct = sum(per_layer_sink) / len(per_layer_sink)

    # Cosine similarity: each sick head vs best-matching healthy head in same layer
    cosine_sims = []
    for li in range(n_layers):
        healthy_idx = torch.where(is_healthy[li])[0]
        if len(healthy_idx) == 0:
            continue
        for hi in range(n_heads):
            if not is_sick[li, hi]:
                continue
            sick_flat = attn_stack[li, hi].flatten()
            best_sim = max(
                F.cosine_similarity(sick_flat.unsqueeze(0), attn_stack[li, hj].flatten().unsqueeze(0)).item()
                for hj in healthy_idx
            )
            cosine_sims.append(best_sim)

    avg_cosine = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0

    return {
        "sink_waste_pct": round(sink_waste_pct, 2),
        "num_sick_heads": num_sick,
        "num_healthy_heads": num_healthy,
        "num_diffuse_heads": num_diffuse,
        "per_layer_sink": [round(x, 2) for x in per_layer_sink],
        "cosine_sick_to_healthy": round(avg_cosine, 4),
        "entropy_grid": entropy_grid.tolist(),
    }


def compute_perplexity(model, dataloader, device):
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.to(device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


# ── Training loop ────────────────────────────────────────────────────────────

def train(args):
    seed_everything(SEED)

    # Device selection
    device = select_device(args.device)
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading GPT-2 Small...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
    model.to(device)

    # Load dataset
    print("Loading WikiText-2...")
    train_dataset = WikiTextDataset("train", tokenizer, max_len=args.max_len)
    val_dataset = WikiTextDataset("validation", tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    print(f"Train chunks: {len(train_dataset)}, Val chunks: {len(val_dataset)}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")

    # Baseline evaluation BEFORE training
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION (before finetuning)")
    print("=" * 60)

    baseline_metrics = full_evaluation(model, tokenizer, device)
    baseline_ppl = compute_perplexity(model, val_loader, device)
    baseline_metrics["perplexity"] = round(baseline_ppl, 2)

    print(f"  Perplexity:          {baseline_ppl:.2f}")
    print(f"  Sink waste:          {baseline_metrics['sink_waste_pct']:.2f}%")
    print(f"  Sick heads:          {baseline_metrics['num_sick_heads']}")
    print(f"  Healthy heads:       {baseline_metrics['num_healthy_heads']}")
    print(f"  Diffuse heads:       {baseline_metrics['num_diffuse_heads']}")
    print(f"  Cosine (sick->healthy): {baseline_metrics['cosine_sick_to_healthy']:.4f}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Checkpointing
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training
    print(f"\n{'=' * 60}")
    print(f"TRAINING — {args.epochs} epochs, lambda_align={args.lambda_align}")
    print(f"{'=' * 60}\n")

    global_step = 0
    log_history = []
    model.train()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        running_lm = 0.0
        running_align = 0.0
        running_sink = 0.0
        running_sick = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch.to(device)
            labels = input_ids.clone()

            # Forward pass with attention weights
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                output_attentions=True,
            )

            lm_loss = outputs.loss / args.grad_accum

            # Compute alignment loss
            align_loss, num_sick, sink_pct = compute_alignment_loss(
                outputs.attentions, model.config.n_head
            )
            scaled_align = (args.lambda_align * align_loss) / args.grad_accum

            total_loss = lm_loss + scaled_align
            total_loss.backward()

            # Track running metrics
            running_lm += lm_loss.item() * args.grad_accum
            if isinstance(align_loss, torch.Tensor):
                running_align += align_loss.item()
            running_sink += sink_pct
            running_sick += num_sick

            # Gradient accumulation step
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log every 100 steps
                if global_step % args.log_every == 0:
                    avg_lm = running_lm / args.log_every / args.grad_accum
                    avg_align = running_align / args.log_every / args.grad_accum
                    avg_sink = running_sink / args.log_every / args.grad_accum
                    avg_sick = running_sick / args.log_every / args.grad_accum

                    log_entry = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "lm_loss": round(avg_lm, 4),
                        "align_loss": round(avg_align, 4),
                        "total_loss": round(avg_lm + args.lambda_align * avg_align, 4),
                        "sink_waste_pct": round(avg_sink, 2),
                        "num_sick_heads": round(avg_sick, 1),
                        "lr": round(scheduler.get_last_lr()[0], 8),
                    }
                    log_history.append(log_entry)

                    print(
                        f"  step {global_step:5d} | "
                        f"lm {avg_lm:.4f} | "
                        f"align {avg_align:.4f} | "
                        f"sink {avg_sink:.1f}% | "
                        f"sick {avg_sick:.0f} | "
                        f"lr {scheduler.get_last_lr()[0]:.2e}"
                    )

                    running_lm = 0.0
                    running_align = 0.0
                    running_sink = 0.0
                    running_sick = 0

                # Checkpoint every 500 steps
                if global_step % args.save_every == 0:
                    ckpt_path = ckpt_dir / f"step_{global_step}.pt"
                    torch.save({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "args": vars(args),
                    }, ckpt_path)
                    print(f"  >> Checkpoint saved: {ckpt_path}")

        epoch_time = time.time() - epoch_start
        print(f"\n  Epoch {epoch + 1}/{args.epochs} done in {epoch_time:.0f}s")

        # Validation perplexity at end of each epoch
        val_ppl = compute_perplexity(model, val_loader, device)
        print(f"  Val perplexity: {val_ppl:.2f}")
        model.train()

    # Save final checkpoint
    final_path = ckpt_dir / "final.pt"
    torch.save({
        "step": global_step,
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
    }, final_path)
    print(f"\nFinal checkpoint saved: {final_path}")

    # Final evaluation AFTER training
    print(f"\n{'=' * 60}")
    print("FINAL EVALUATION (after finetuning)")
    print(f"{'=' * 60}")

    finetuned_metrics = full_evaluation(model, tokenizer, device)
    finetuned_ppl = compute_perplexity(model, val_loader, device)
    finetuned_metrics["perplexity"] = round(finetuned_ppl, 2)

    print(f"  Perplexity:          {finetuned_ppl:.2f}")
    print(f"  Sink waste:          {finetuned_metrics['sink_waste_pct']:.2f}%")
    print(f"  Sick heads:          {finetuned_metrics['num_sick_heads']}")
    print(f"  Healthy heads:       {finetuned_metrics['num_healthy_heads']}")
    print(f"  Diffuse heads:       {finetuned_metrics['num_diffuse_heads']}")
    print(f"  Cosine (sick->healthy): {finetuned_metrics['cosine_sick_to_healthy']:.4f}")

    # Comparison
    print(f"\n{'=' * 60}")
    print("COMPARISON: BASELINE vs FINETUNED")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25} {'Baseline':>10} {'Finetuned':>10} {'Delta':>10}")
    print(f"  {'-' * 55}")

    comparisons = [
        ("Perplexity", baseline_metrics["perplexity"], finetuned_metrics["perplexity"]),
        ("Sink waste %", baseline_metrics["sink_waste_pct"], finetuned_metrics["sink_waste_pct"]),
        ("Sick heads", baseline_metrics["num_sick_heads"], finetuned_metrics["num_sick_heads"]),
        ("Healthy heads", baseline_metrics["num_healthy_heads"], finetuned_metrics["num_healthy_heads"]),
        ("Diffuse heads", baseline_metrics["num_diffuse_heads"], finetuned_metrics["num_diffuse_heads"]),
        ("Cosine sim", baseline_metrics["cosine_sick_to_healthy"], finetuned_metrics["cosine_sick_to_healthy"]),
    ]

    for name, base, fine in comparisons:
        delta = fine - base
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<25} {base:>10} {fine:>10} {sign}{delta:>9.2f}")

    # Save results JSON
    results = {
        "args": vars(args),
        "baseline": baseline_metrics,
        "finetuned": finetuned_metrics,
        "log_history": log_history,
        "total_steps": global_step,
    }

    results_path = Path(args.results_file)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


# ── Eval-only mode ───────────────────────────────────────────────────────────

def eval_only(args):
    device = select_device(args.device)
    print(f"Using device: {device}")

    print("Loading GPT-2 Small...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    print(f"  Checkpoint step: {ckpt['step']}, epoch: {ckpt['epoch']}")
    if "args" in ckpt:
        print(f"  Training args: lambda_align={ckpt['args'].get('lambda_align', '?')}, "
              f"lr={ckpt['args'].get('lr', '?')}")

    # Baseline (untrained GPT-2) for comparison
    print("\nLoading baseline GPT-2 for comparison...")
    baseline_model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
    baseline_model.to(device)

    # Perplexity on validation set
    val_dataset = WikiTextDataset("validation", tokenizer, max_len=256)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=True)

    print("\nRunning evaluations...")

    baseline_metrics = full_evaluation(baseline_model, tokenizer, device)
    baseline_ppl = compute_perplexity(baseline_model, val_loader, device)
    baseline_metrics["perplexity"] = round(baseline_ppl, 2)

    finetuned_metrics = full_evaluation(model, tokenizer, device)
    finetuned_ppl = compute_perplexity(model, val_loader, device)
    finetuned_metrics["perplexity"] = round(finetuned_ppl, 2)

    del baseline_model  # free memory

    print(f"\n{'=' * 60}")
    print("COMPARISON: BASELINE vs FINETUNED (checkpoint)")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25} {'Baseline':>10} {'Finetuned':>10} {'Delta':>10}")
    print(f"  {'-' * 55}")

    comparisons = [
        ("Perplexity", baseline_metrics["perplexity"], finetuned_metrics["perplexity"]),
        ("Sink waste %", baseline_metrics["sink_waste_pct"], finetuned_metrics["sink_waste_pct"]),
        ("Sick heads", baseline_metrics["num_sick_heads"], finetuned_metrics["num_sick_heads"]),
        ("Healthy heads", baseline_metrics["num_healthy_heads"], finetuned_metrics["num_healthy_heads"]),
        ("Diffuse heads", baseline_metrics["num_diffuse_heads"], finetuned_metrics["num_diffuse_heads"]),
        ("Cosine sim", baseline_metrics["cosine_sick_to_healthy"], finetuned_metrics["cosine_sick_to_healthy"]),
    ]

    for name, base, fine in comparisons:
        delta = fine - base
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<25} {base:>10} {fine:>10} {sign}{delta:>9.2f}")

    # Save results
    results = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": ckpt["step"],
        "baseline": baseline_metrics,
        "finetuned": finetuned_metrics,
    }

    results_path = Path(args.results_file)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


# ── Device selection ─────────────────────────────────────────────────────────

def select_device(requested: str) -> torch.device:
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("WARNING: MPS not available, falling back to CPU")
        return torch.device("cpu")
    elif requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("WARNING: CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device("cpu")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Finetune GPT-2 with attention alignment loss")

    # Mode
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, evaluate a checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for --eval-only mode")

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=256,
                        help="Max sequence length for tokenized chunks")
    parser.add_argument("--lambda_align", type=float, default=0.1,
                        help="Weight for the attention alignment loss")

    # Logging and checkpointing
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log metrics every N optimizer steps")
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save checkpoint every N optimizer steps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--results_file", type=str, default="blend_results.json",
                        help="Output JSON file for results")

    # Device
    parser.add_argument("--device", type=str, default="mps",
                        choices=["mps", "cuda", "cpu"])

    args = parser.parse_args()

    if args.eval_only:
        if args.checkpoint is None:
            parser.error("--eval-only requires --checkpoint")
        eval_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
