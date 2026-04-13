"""
Colab script: Learned sink tokens on LLaMA-3.2-1B
Run on Google Colab with T4 GPU (free tier works).

Prerequisites:
1. Accept Meta's license: https://huggingface.co/meta-llama/Llama-3.2-1B
2. Add HF token: In Colab, go to Secrets (key icon) and add HF_TOKEN

Results save to llama_sink_results.json — download and add to notebook repo.
"""

# !pip install torch transformers datasets accelerate -q

import os
# Load HF token from environment, Colab secrets, or ~/.claude/.env
if not os.environ.get("HF_TOKEN"):
    for envpath in ["~/.claude/.env", ".env"]:
        p = os.path.expanduser(envpath)
        if os.path.exists(p):
            for line in open(p):
                if line.startswith("HF_TOKEN=") and line.strip().split("=", 1)[1]:
                    os.environ["HF_TOKEN"] = line.strip().split("=", 1)[1]
                    break

import json, math, time, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# --- Config ---
MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# --- Dataset ---
class WikiTextDataset(Dataset):
    def __init__(self, split, tokenizer, max_len=256):
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join([t for t in raw["text"] if t.strip()])
        tokens = tokenizer.encode(text)
        n = len(tokens) // max_len
        self.chunks = [torch.tensor(tokens[i*max_len:(i+1)*max_len], dtype=torch.long) for i in range(n)]
    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx): return self.chunks[idx]

# --- Load model ---
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
    attn_implementation="eager",
)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
d_model = model.config.hidden_size
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
print(f"  {n_params/1e9:.1f}B params, {n_layers}L x {n_heads}H, d={d_model}")

# --- Data ---
train_ds = WikiTextDataset("train", tokenizer, max_len=256)
val_ds = WikiTextDataset("validation", tokenizer, max_len=256)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, drop_last=True)
print(f"Train: {len(train_ds)} chunks, Val: {len(val_ds)} chunks")

# --- Eval functions ---
@torch.no_grad()
def eval_ppl_standard(model, loader):
    total_loss = total_tokens = 0
    for batch in loader:
        ids = batch.to(DEVICE)
        out = model(input_ids=ids, labels=ids)
        total_loss += out.loss.item() * ids.numel()
        total_tokens += ids.numel()
    return math.exp(total_loss / total_tokens)

@torch.no_grad()
def eval_ppl_with_embed(model, embed_vecs, loader, position="start"):
    """Evaluate with learned embeddings prepended (start) or appended (end)."""
    total_loss = total_tokens = 0
    n_tok = embed_vecs.shape[0]
    for batch in loader:
        ids = batch.to(DEVICE)
        tok_embeds = model.model.embed_tokens(ids).float()
        prefix = embed_vecs.unsqueeze(0).expand(ids.shape[0], -1, -1).float()

        if position == "start":
            inputs_embeds = torch.cat([prefix, tok_embeds], dim=1).half()
            labels = torch.cat([
                torch.full((ids.shape[0], n_tok), -100, dtype=torch.long, device=DEVICE),
                ids
            ], dim=1)
        else:
            inputs_embeds = torch.cat([tok_embeds, prefix], dim=1).half()
            labels = torch.cat([
                ids,
                torch.full((ids.shape[0], n_tok), -100, dtype=torch.long, device=DEVICE)
            ], dim=1)

        out = model(inputs_embeds=inputs_embeds, labels=labels)
        total_loss += out.loss.item() * ids.numel()
        total_tokens += ids.numel()
    return math.exp(total_loss / total_tokens)

def train_sink_tokens(model, n_tokens, train_ds, steps=2000, lr=5e-3, seed=42):
    """Train learned embeddings, model frozen."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    for p in model.parameters():
        p.requires_grad = False

    learned = torch.nn.Parameter(
        torch.randn(n_tokens, d_model, device=DEVICE, dtype=torch.float32) * 0.02
    )
    optimizer = torch.optim.Adam([learned], lr=lr)

    model.train()
    for step in range(steps):
        idx = np.random.randint(len(train_ds))
        ids = train_ds[idx].unsqueeze(0).to(DEVICE)
        tok_embeds = model.model.embed_tokens(ids).detach().float()
        prefix = learned.unsqueeze(0)
        inputs_embeds = torch.cat([prefix, tok_embeds], dim=1).half()
        labels = torch.cat([
            torch.full((1, n_tokens), -100, dtype=torch.long, device=DEVICE),
            ids
        ], dim=1)

        out = model(inputs_embeds=inputs_embeds, labels=labels)
        loss = out.loss.float()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"    step {step+1} | loss {loss.item():.4f}")

    model.eval()
    return learned.data

# --- Sink analysis (manual QKV extraction) ---
print("\n=== Sink Analysis ===")
TEXT = (
    "The history of attention mechanisms in neural networks begins with "
    "Bahdanau's 2014 paper on sequence-to-sequence models. Rather than "
    "compressing an entire input sequence into a fixed-length vector, the "
    "model learned to attend to different parts of the input at each "
    "decoding step. This idea was transformative."
)
inputs = tokenizer(TEXT, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)

with torch.no_grad():
    out = model(**inputs, output_attentions=True)

attn = torch.stack([l[0].float().cpu() for l in out.attentions]).numpy()
sink_mag = attn[:, :, :, 0].mean(axis=(1, 2))
p = np.clip(attn, 1e-9, 1.0)
entropy = -(p * np.log(p)).sum(axis=-1).mean(axis=-1)
median_ent = np.median(entropy)
n_sick = int((entropy < median_ent * 0.7).sum())
n_total = n_layers * n_heads

print(f"  Sink waste: {float(sink_mag.mean())*100:.1f}%")
print(f"  Sick heads: {n_sick}/{n_total} ({n_sick/n_total*100:.1f}%)")

# --- Baseline PPL ---
print("\n=== Baseline ===")
baseline = eval_ppl_standard(model, val_loader)
print(f"  Baseline PPL: {baseline:.4f}")

# --- Train + eval learned tokens ---
results = {
    "model": MODEL_NAME,
    "n_params": n_params,
    "n_layers": n_layers,
    "n_heads": n_heads,
    "d_model": d_model,
    "sink_waste_pct": round(float(sink_mag.mean()) * 100, 1),
    "n_sick_heads": n_sick,
    "n_total_heads": n_total,
    "sick_pct": round(n_sick / n_total * 100, 1),
    "baseline_ppl": round(baseline, 4),
    "experiments": [],
}

for n_tok in [1, 4]:
    print(f"\n=== Training {n_tok} token(s) ===")
    t0 = time.time()
    learned = train_sink_tokens(model, n_tok, train_ds, steps=2000, lr=5e-3)

    # Eval start position
    ppl_start = eval_ppl_with_embed(model, learned, val_loader, position="start")
    delta_start = (ppl_start - baseline) / baseline * 100
    print(f"  {n_tok}tok/start: PPL={ppl_start:.4f} ({delta_start:+.2f}%)")

    # Eval end position (control)
    ppl_end = eval_ppl_with_embed(model, learned, val_loader, position="end")
    delta_end = (ppl_end - baseline) / baseline * 100
    print(f"  {n_tok}tok/end:   PPL={ppl_end:.4f} ({delta_end:+.2f}%)")

    elapsed = time.time() - t0
    results["experiments"].append({
        "n_tokens": n_tok,
        "params": n_tok * d_model,
        "start_ppl": round(ppl_start, 4),
        "start_improvement_pct": round(-delta_start, 2),
        "end_ppl": round(ppl_end, 4),
        "end_improvement_pct": round(-delta_end, 2),
        "time_s": round(elapsed, 1),
    })

# --- Summary ---
print(f"\n{'='*60}")
print(f"LLAMA-3.2-1B LEARNED SINK RESULTS")
print(f"{'='*60}")
print(f"  Baseline PPL: {baseline:.2f}")
print(f"  Sink waste: {results['sink_waste_pct']}%")
print(f"  {'Config':<20} {'PPL':>8} {'Δ':>8} {'Control':>8}")
print(f"  {'-'*44}")
for e in results["experiments"]:
    print(f"  {e['n_tokens']}tok/start{'':<12} {e['start_ppl']:>8.2f} {e['start_improvement_pct']:>+7.2f}% {'':<8}")
    print(f"  {e['n_tokens']}tok/end{'':<14} {e['end_ppl']:>8.2f} {e['end_improvement_pct']:>+7.2f}% {'(control)':<8}")

with open("llama_sink_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to llama_sink_results.json")
print("Download this file and add to your notebook repo.")
