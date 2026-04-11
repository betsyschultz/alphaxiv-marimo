"""Recursive self-improvement: iterative blend-and-train with moving references."""
import json, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# Load model and data
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.train()

ds = load_dataset("wikitext", "wikitext-2-raw-v1")
all_text = "\n".join([t for t in ds["train"]["text"] if t.strip()])
tokens = tokenizer(all_text, return_tensors="pt")["input_ids"][0]
SEQ = 256
chunks = [tokens[i:i+SEQ] for i in range(0, len(tokens)-SEQ, SEQ)]
print(f"Chunks: {len(chunks)}, seq_len: {SEQ}")

val_text = "\n".join([t for t in ds["validation"]["text"] if t.strip()])
val_tokens = tokenizer(val_text, return_tensors="pt")["input_ids"][0]
val_chunks = [val_tokens[i:i+SEQ] for i in range(0, len(val_tokens)-SEQ, SEQ)]

n_l, n_h = model.config.n_layer, model.config.n_head

def evaluate(model, text_chunks, device):
    """Compute perplexity + attention metrics."""
    model.eval()
    total_loss, total_tokens = 0, 0
    # Attention metrics on first 20 chunks
    all_attn = []
    with torch.no_grad():
        for i, chunk in enumerate(text_chunks[:50]):
            ids = chunk.unsqueeze(0).to(device)
            out = model(ids, labels=ids, output_attentions=(i < 20))
            total_loss += out.loss.item() * ids.shape[1]
            total_tokens += ids.shape[1]
            if i < 20:
                attn = torch.stack([a[0].cpu() for a in out.attentions]).numpy()
                all_attn.append(attn)
    
    ppl = np.exp(total_loss / total_tokens)
    attn_stack = np.mean(all_attn, axis=0)  # avg across chunks
    
    sink_waste = attn_stack[:, :, :, 0].mean() * 100
    p = np.clip(attn_stack, 1e-9, 1.0)
    ent = -(p * np.log(p)).sum(axis=-1).mean(axis=-1)
    med = np.median(ent)
    n_sick = int((ent < med * 0.7).sum())
    n_diff = int((ent > np.percentile(ent, 90)).sum())
    n_healthy = n_l * n_h - n_sick - n_diff
    
    model.train()
    return {"ppl": ppl, "sink": sink_waste, "sick": n_sick, "healthy": n_healthy, 
            "diffuse": n_diff, "entropy": ent}

def get_healthy_refs(model, chunks, device):
    """Get current healthy head reference patterns."""
    model.eval()
    all_attn = []
    with torch.no_grad():
        for chunk in chunks[:20]:
            ids = chunk.unsqueeze(0).to(device)
            out = model(ids, output_attentions=True)
            attn = torch.stack([a[0].cpu() for a in out.attentions]).numpy()
            all_attn.append(attn)
    avg_attn = np.mean(all_attn, axis=0)
    
    p = np.clip(avg_attn, 1e-9, 1.0)
    ent = -(p * np.log(p)).sum(axis=-1).mean(axis=-1)
    med = np.median(ent)
    thresh = med * 0.7
    diff_thresh = np.percentile(ent, 90)
    
    sick = [(li,hi) for li in range(n_l) for hi in range(n_h) if ent[li,hi] < thresh]
    healthy_by_layer = {}
    for li in range(n_l):
        healthy_by_layer[li] = [hi for hi in range(n_h) if thresh <= ent[li,hi] <= diff_thresh]
    
    # Build per-layer healthy reference (as tensors for training)
    refs = {}
    for li in range(n_l):
        hlthy = healthy_by_layer[li]
        if hlthy:
            refs[li] = torch.tensor(avg_attn[li, hlthy].mean(axis=0), dtype=torch.float32)
    
    model.train()
    return refs, sick, ent

# ===== RECURSIVE TRAINING =====
N_ROUNDS = 4
STEPS_PER_ROUND = 300
LAMBDA = 0.5  # Aggressive — previous run used 0.1
LR = 3e-5
BATCH = 4
ACCUM = 4

print(f"\n{'='*60}")
print(f"RECURSIVE SELF-IMPROVEMENT: {N_ROUNDS} rounds x {STEPS_PER_ROUND} steps")
print(f"Lambda: {LAMBDA}, LR: {LR}")
print(f"{'='*60}")

# Baseline
print("\nRound 0 (baseline):")
baseline = evaluate(model, val_chunks, device)
print(f"  ppl={baseline['ppl']:.2f} sink={baseline['sink']:.1f}% sick={baseline['sick']} healthy={baseline['healthy']}")

results = [{"round": 0, **{k:v for k,v in baseline.items() if k != "entropy"}}]

for rnd in range(1, N_ROUNDS + 1):
    print(f"\n--- Round {rnd}/{N_ROUNDS} ---")
    
    # Get CURRENT healthy references (these update each round)
    refs, sick_heads, ent = get_healthy_refs(model, chunks, device)
    print(f"  Reference: {len(sick_heads)} sick heads, refs from {len(refs)} layers")
    
    if not sick_heads:
        print("  No sick heads remaining! Stopping.")
        break
    
    # Build sick head set for fast lookup
    sick_set = set(sick_heads)
    med = np.median(ent)
    thresh = med * 0.7
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model.train()
    
    step = 0
    accum_lm, accum_align = 0, 0
    optimizer.zero_grad()
    
    indices = np.random.permutation(len(chunks))
    
    for idx in indices:
        if step >= STEPS_PER_ROUND * ACCUM:
            break
            
        ids = chunks[idx].unsqueeze(0).to(device)
        out = model(ids, labels=ids, output_attentions=True)
        lm_loss = out.loss / ACCUM
        
        # Alignment loss with CURRENT references
        align_loss = torch.tensor(0.0, device=device)
        n_align = 0
        for li in range(n_l):
            if li not in refs:
                continue
            ref = refs[li].to(device)
            for hi in range(n_h):
                if (li, hi) in sick_set:
                    head_attn = out.attentions[li][0, hi]  # (seq, seq)
                    # Cosine distance
                    a = head_attn.reshape(-1)
                    b = ref.reshape(-1)[:a.shape[0]] if ref.reshape(-1).shape[0] != a.shape[0] else ref.reshape(-1)
                    if a.shape == b.shape:
                        cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
                        align_loss = align_loss + (1 - cos)
                        n_align += 1
        
        if n_align > 0:
            align_loss = align_loss / n_align / ACCUM
        
        total = lm_loss + LAMBDA * align_loss
        total.backward()
        
        accum_lm += lm_loss.item()
        accum_align += align_loss.item() if isinstance(align_loss, torch.Tensor) else 0
        
        step += 1
        if step % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            opt_step = step // ACCUM
            
            if opt_step % 50 == 0:
                print(f"  step {opt_step:4d} | lm {accum_lm:.4f} | align {accum_align:.4f}")
                accum_lm, accum_align = 0, 0
    
    # Evaluate after this round
    rnd_eval = evaluate(model, val_chunks, device)
    print(f"  Round {rnd} result: ppl={rnd_eval['ppl']:.2f} sink={rnd_eval['sink']:.1f}% sick={rnd_eval['sick']} healthy={rnd_eval['healthy']}")
    results.append({"round": rnd, **{k:v for k,v in rnd_eval.items() if k != "entropy"}})

print(f"\n{'='*60}")
print("RECURSIVE TRAINING SUMMARY")
print(f"{'='*60}")
print(f"{'Round':>6} {'PPL':>8} {'Sink%':>7} {'Sick':>6} {'Healthy':>8} {'Diffuse':>8}")
print("-" * 50)
for r in results:
    print(f"{r['round']:>6} {r['ppl']:>8.2f} {r['sink']:>7.1f} {r['sick']:>6} {r['healthy']:>8} {r['diffuse']:>8}")

# Save
with open("recursive_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to recursive_results.json")
