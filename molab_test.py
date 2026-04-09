# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def loading():
    import marimo as mo
    mo.md("## molab Platform Test\nLoading GPT-2...")
    return (mo,)


@app.cell
def load_model(mo):
    import time
    start = time.time()

    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.eval()

    text = "The cat sat on the mat and looked around the room quietly"
    inputs = tokenizer(text, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]

    elapsed = time.time() - start

    # Hook to capture QKV
    qkv_cache = {}
    def capture_qkv(layer_idx):
        def hook_fn(module, input, output):
            qkv_cache[layer_idx] = output.detach()
        return hook_fn

    hooks = []
    for i, block in enumerate(model.transformer.h):
        h = block.attn.c_attn.register_forward_hook(capture_qkv(i))
        hooks.append(h)

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    for h in hooks:
        h.remove()

    mo.md(f"""
    ### Load complete

    - **Model:** GPT-2 (124M)
    - **Tokens:** {seq_len}
    - **Load time:** {elapsed:.1f}s
    - **QKV captured:** {len(qkv_cache)} layers
    - **Attentions returned:** {len(out.attentions)} layers
    - **Torch version:** {torch.__version__}
    """)
    return model, tokenizer, inputs, out, qkv_cache, seq_len, torch, np


@app.cell
def heatmap(mo, out, np):
    import matplotlib.pyplot as plt

    attn = out.attentions[8][0].detach().numpy()
    avg_attn = attn.mean(axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(avg_attn, cmap="viridis", aspect="auto")
    ax.set_title("Layer 8 — Average Attention", fontsize=13)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    mo.md("### Attention Heatmap (Layer 8)")
    return (fig,)


@app.cell
def verify_hook(mo, qkv_cache, out, torch, seq_len):
    n_heads = 12
    d_model = 768
    d_head = d_model // n_heads

    qkv = qkv_cache[0]
    q, k, v = qkv.split(d_model, dim=-1)
    q = q.view(1, seq_len, n_heads, d_head).transpose(1, 2)
    k = k.view(1, seq_len, n_heads, d_head).transpose(1, 2)

    raw_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    raw_scores = raw_scores.masked_fill(causal_mask == 0, float('-inf'))
    our_softmax = torch.nn.functional.softmax(raw_scores, dim=-1)

    model_attn = out.attentions[0][0].detach()
    max_diff = (model_attn - our_softmax[0]).abs().max().item()

    # ReLU test
    relu_scores = torch.relu(raw_scores.masked_fill(causal_mask == 0, 0))
    relu_attn = relu_scores / (relu_scores.sum(dim=-1, keepdim=True) + 1e-9)
    std_sink = model_attn[:, :, 0].mean().item()
    relu_sink = relu_attn[0, :, :, 0].mean().item()

    status = "ALL SYSTEMS GO" if max_diff < 0.001 else "HOOK ERROR"

    mo.md(f"""
    ### Validation Results

    | Test | Result |
    |------|--------|
    | QKV reconstruction error | **{max_diff:.10f}** |
    | Standard sink weight | **{std_sink:.4f}** |
    | ReLU sink weight | **{relu_sink:.4f}** ({(relu_sink/std_sink - 1)*100:+.0f}%) |
    | **Status** | **{status}** |
    """)
    return


if __name__ == "__main__":
    app.run()
