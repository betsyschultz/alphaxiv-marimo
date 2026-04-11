"""Learned sink token: can the dump target carry useful information?"""
import json, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

ds = load_dataset("wikitext", "wikitext-2-raw-v1")
all_text = "\n".join([t for t in ds["train"]["text"] if t.strip()])
tokens = tokenizer(all_text, return_tensors="pt")["input_ids"][0]
SEQ = 256
chunks = [tokens[i:i+SEQ] for i in range(0, len(tokens)-SEQ, SEQ)]

val_text = "\n".join([t for t in ds["validation"]["text"] if t.strip()])
val_tokens = tokenizer(val_text, return_tensors="pt")["input_ids"][0]
val_chunks = [val_tokens[i:i+SEQ] for i in range(0, len(val_tokens)-SEQ, SEQ)]
print(f"Train: {len(chunks)} chunks, Val: {len(val_chunks)} chunks")

def evaluate_with_prefix_id(model, prefix_id, chunks, device, n_eval=50):
    """Evaluate with a token ID prepended."""
    model.eval()
    total_loss, total_tokens = 0, 0
    sink_wastes = []
    with torch.no_grad():
        for chunk in chunks[:n_eval]:
            ids = torch.cat([torch.tensor([prefix_id]), chunk]).unsqueeze(0).to(device)
            out = model(ids, labels=ids, output_attentions=True)
            # Loss on real tokens only (skip prefix)
            logits = out.logits[:, 1:-1, :]
            labels = ids[:, 2:]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            total_loss += loss.item() * labels.shape[1]
            total_tokens += labels.shape[1]
            # Sink waste: real tokens attending to position 0
            attn = torch.stack([a[0] for a in out.attentions])  # (layers, heads, seq+1, seq+1)
            sw = attn[:, :, 1:, 0].mean().item()
            sink_wastes.append(sw)
    ppl = np.exp(total_loss / total_tokens)
    sink = np.mean(sink_wastes) * 100
    model.train()
    return ppl, sink

def evaluate_with_embed(model, embed_vec, chunks, device, n_eval=50):
    """Evaluate with a custom embedding prepended (bypass token lookup)."""
    model.eval()
    total_loss, total_tokens = 0, 0
    sink_wastes = []
    with torch.no_grad():
        for chunk in chunks[:n_eval]:
            ids = chunk.unsqueeze(0).to(device)
            # Build embeddings manually
            tok_embeds = model.transformer.wte(ids)  # (1, seq, 768)
            prefix = embed_vec.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 768)
            inputs_embeds = torch.cat([prefix, tok_embeds], dim=1)  # (1, seq+1, 768)
            
            # Create labels: -100 for prefix position, then real token ids
            labels = torch.cat([torch.tensor([[-100]]).to(device), ids], dim=1)
            
            out = model(inputs_embeds=inputs_embeds, labels=labels, output_attentions=True)
            
            # Loss only on real tokens
            total_loss += out.loss.item() * ids.shape[1]
            total_tokens += ids.shape[1]
            
            attn = torch.stack([a[0] for a in out.attentions])
            sw = attn[:, :, 1:, 0].mean().item()
            sink_wastes.append(sw)
    
    ppl = np.exp(total_loss / total_tokens)
    sink = np.mean(sink_wastes) * 100
    model.train()
    return ppl, sink

# ===== BASELINES =====
print(f"\n{'='*60}")
print("LEARNED SINK TOKEN EXPERIMENT")
print(f"{'='*60}")

# Baseline: no prefix
model.eval()
total_loss, total_tokens = 0, 0
with torch.no_grad():
    for chunk in val_chunks[:50]:
        ids = chunk.unsqueeze(0).to(device)
        out = model(ids, labels=ids)
        total_loss += out.loss.item() * ids.shape[1]
        total_tokens += ids.shape[1]
baseline_ppl = np.exp(total_loss / total_tokens)
print(f"\nNo prefix (baseline):     ppl = {baseline_ppl:.2f}")

# Zero token (token ID 0 = "!")
zero_ppl, zero_sink = evaluate_with_prefix_id(model, 0, val_chunks, device)
print(f"Zero token (ID=0):        ppl = {zero_ppl:.2f}, sink absorbs = {zero_sink:.1f}%")

# BOS-like token (using embedding approach with zeros vector)
zero_vec = torch.zeros(model.config.n_embd, device=device)
zvec_ppl, zvec_sink = evaluate_with_embed(model, zero_vec, val_chunks, device)
print(f"Zero embedding:           ppl = {zvec_ppl:.2f}, sink absorbs = {zvec_sink:.1f}%")

# ===== TRAIN LEARNED EMBEDDING =====
print(f"\nTraining learned sink embedding (768 params, model frozen)...")

for p in model.parameters():
    p.requires_grad = False

learned = torch.nn.Parameter(torch.randn(model.config.n_embd, device=device) * 0.02)
optimizer = torch.optim.Adam([learned], lr=5e-3)

STEPS = 500
model.train()
start = time.time()

for step in range(STEPS):
    idx = np.random.randint(len(chunks))
    ids = chunks[idx].unsqueeze(0).to(device)
    
    tok_embeds = model.transformer.wte(ids).detach()
    prefix = learned.unsqueeze(0).unsqueeze(0)
    inputs_embeds = torch.cat([prefix, tok_embeds], dim=1)
    labels = torch.cat([torch.tensor([[-100]]).to(device), ids], dim=1)
    
    out = model(inputs_embeds=inputs_embeds, labels=labels)
    loss = out.loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (step + 1) % 100 == 0:
        print(f"  step {step+1:4d} | loss {loss.item():.4f}")

elapsed = time.time() - start
print(f"Training done in {elapsed:.0f}s")

# Evaluate learned token
learned_ppl, learned_sink = evaluate_with_embed(model, learned.data, val_chunks, device)

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"{'Condition':<25} {'PPL':>8} {'Sink%':>8}")
print("-" * 45)
print(f"{'No prefix':<25} {baseline_ppl:>8.2f} {'n/a':>8}")
print(f"{'Zero token (ID=0)':<25} {zero_ppl:>8.2f} {zero_sink:>7.1f}%")
print(f"{'Zero embedding':<25} {zvec_ppl:>8.2f} {zvec_sink:>7.1f}%")
print(f"{'LEARNED embedding':<25} {learned_ppl:>8.2f} {learned_sink:>7.1f}%")

delta_vs_zero = zvec_ppl - learned_ppl
print(f"\nLearned vs zero embedding: {'+' if delta_vs_zero > 0 else ''}{delta_vs_zero:.2f} perplexity")
if learned_ppl < zvec_ppl - 0.5:
    print(">>> LEARNED TOKEN IMPROVES PERPLEXITY — the sink CAN carry useful information!")
elif abs(delta_vs_zero) < 0.5:
    print(">>> No meaningful difference — the model just needs a dump target.")
else:
    print(">>> Learned token is worse — interference with learned patterns.")

results = {
    "baseline_ppl": float(baseline_ppl),
    "zero_token_ppl": float(zero_ppl), "zero_token_sink": float(zero_sink),
    "zero_embed_ppl": float(zvec_ppl), "zero_embed_sink": float(zvec_sink),
    "learned_ppl": float(learned_ppl), "learned_sink": float(learned_sink),
    "delta_vs_zero": float(delta_vs_zero),
    "steps": STEPS, "time_s": float(elapsed),
}
with open("learned_sink_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to learned_sink_results.json")
