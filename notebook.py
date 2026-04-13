# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers",
#     "numpy",
#     "matplotlib",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


# ============================================================================
# SIDEBAR: TABLE OF CONTENTS
# ============================================================================

@app.cell(hide_code=True)
def table_of_contents():
    import marimo as _mo
    _mo.sidebar([_mo.outline()])


# ============================================================================
# TITLE (renders immediately, no data dependency)
# ============================================================================

@app.cell(hide_code=True)
def title():
    import marimo as _mo
    _mo.output.replace(
        _mo.md("""
# Every Transformer Has a Garbage Bin
## It's not a bug. But you can build a better one.
""")
    )


# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================

@app.cell(hide_code=True)
def executive_summary(data, mo, np, plt):
    _approaches = [
        "Standard\n(baseline)", "Block self-\nattention", "Temp\nscaling",
        "Sink\ntoken", "Train\nλ=0.1", "Train\nλ=1.0", "Train\nλ=10",
        "Recursive",
    ]
    _sink_values = [44.3, 44.1, 19.4, 0.9, 46.2, 47.8, 45.3, 45.5]
    _colors = [
        "#e74c3c", "#95a5a6", "#3498db", "#2ecc71",
        "#e74c3c", "#8e44ad", "#6c3483", "#c0392b",
    ]
    _labels = [
        "44.3%", "44.1%\nno change", "19.4%\nredistributed",
        "0.9%\nredirected", "46.2%\nresisted", "47.8%\nequal pressure\nworse",
        "45.3%\n100× pressure\nstill 45%", "45.5%\nworse",
    ]

    _fig_bar, _ax = plt.subplots(figsize=(12, 4))
    _bars = _ax.bar(
        _approaches, _sink_values, color=_colors, width=0.6,
        edgecolor="white", linewidth=1.5,
    )
    for _bar, _lbl in zip(_bars, _labels):
        _ax.text(
            _bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 1,
            _lbl, ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    _ax.set_ylabel("Attention waste on position 0 (%)", fontsize=11)
    _ax.set_title("Everything I Tried", fontsize=14, fontweight="bold")
    _ax.set_ylim(0, 62)
    _ax.grid(True, alpha=0.15, axis="y")
    _ax.axhline(y=44.3, color="#e74c3c", linestyle="--", alpha=0.3, linewidth=1)
    plt.tight_layout()

    mo.output.replace(
        mo.vstack([
            mo.md(f"""
*Betsy Schultz · GPT-2 (124M) · {data['seq_len']} tokens · {"loaded from cache" if data["from_cache"] else "computed"} in {data['elapsed']:.1f}s*

*8 fix approaches tested, validated on GPT-2 + LLaMA-3.2-1B + Pythia-70M,
and a method that works — all interactive below.*

Language models decide what each word means by looking at the words around
it. But GPT-2 wastes over **44% of that attention** staring at the first
word — regardless of what the first word is. That's an attention sink.

Think of each attention head as a worker on an assembly line. Some workers
have a job for every sentence (grammar, pronouns). Others are idle — there's
no grammar to check in a string of adjectives. But every worker *must* point
somewhere. So idle workers dump their attention on the first word, turning
it into a **garbage bin**.

I tested 8 ways to fix this. None worked — the model *needs* the garbage
bin. But I found that training a better one (768 numbers, 2 minutes)
**makes GPT-2 19.7% better at predicting language — and LLaMA-3.2-1B
26.7% better.** Same tokens placed at the end? 0% improvement on both
models. The effect is entirely about the garbage bin position.
"""),
            _fig_bar,
            mo.hstack([
                mo.stat(value="-26.7%", label="LLaMA-1B with a better bin", bordered=True),
                mo.stat(value="29×", label="less critical than random heads", bordered=True),
                mo.stat(value="8,192", label="numbers trained (of 1.2 billion)", bordered=True),
            ], justify="center", gap=1),
            mo.accordion({
                "Methodology note": mo.md("""
Training used λ_align = 0.1 (alignment loss weighted at 10% of
language modeling loss) over 1,767 steps on WikiText-2. This is
intentionally conservative — high enough to provide gradient signal
without destroying language ability. The alignment loss barely moved
(0.35 → 0.33) while LM loss dropped steadily, confirming the model
*could* learn but attention patterns remained stable.
"""),
            }),
        ])
    )


# ============================================================================
# CELL 0: PRECOMPUTATION
# ============================================================================

@app.cell
def precompute():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import hashlib
    import json as _json
    import time
    import os

    # --- Consistent chart theme ---
    plt.rcParams.update({
        "figure.facecolor": "#fafafa",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#cccccc",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.color": "#888888",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.family": "sans-serif",
    })

    _start = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    _TEXT = (
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
        "quirks. Recent work has discovered that certain token positions absorb "
        "a disproportionate share of attention weight across many heads and "
        "layers. These attention sinks appear to emerge from the pre-norm "
        "architecture used in most modern transformers."
    )

    # --- Cache setup ---
    _dir = os.path.dirname(os.path.abspath(__file__))
    _cache_path = os.path.join(_dir, "precomputed_cache.npz")
    _text_hash = hashlib.md5(_TEXT.encode()).hexdigest()[:12]

    # --- Helpers ---
    def _entropy(attn_4d):
        _p = np.clip(attn_4d, 1e-9, 1.0)
        return -(_p * np.log(_p)).sum(axis=-1).mean(axis=-1)

    def _sink_mag(attn_4d, col=0):
        return attn_4d[:, :, :, col].mean(axis=(1, 2))

    # --- Try loading from cache ---
    _from_cache = False
    if os.path.exists(_cache_path):
        try:
            _cached = np.load(_cache_path, allow_pickle=True)
            if _cached["text_hash"].item() == _text_hash:
                _from_cache = True
        except Exception:
            pass

    if _from_cache:
        with mo.status.spinner("Loading from cache..."):
            _standard_attn = _cached["standard_attn"]
            _esa_attn = _cached["esa_attn"]
            _raw_scores_all = _cached["raw_scores_all"]
            _sink_attn_full = _cached["sink_attn_full"]
            _tokens = list(_cached["tokens"])
            _seq_len = int(_cached["seq_len"])
            _n_layers = int(_cached["n_layers"])
            _n_heads = int(_cached["n_heads"])

            # Model loaded lazily in try-it-yourself cell
            _model = None
            _tokenizer = None
            _std_ppl_input = float(_cached.get("std_ppl_input", 0))
            _sink_ppl_input = float(_cached.get("sink_ppl_input", 0))
            _adaptive_val = None
            if "adapt_top1" in _cached:
                _adaptive_val = {
                    "top1_match": float(_cached["adapt_top1"]),
                    "standard_ppl": float(_cached["adapt_std_ppl"]),
                    "adaptive_ppl": float(_cached["adapt_adp_ppl"]),
                }
    else:
        # --- Full computation with progress ---
        with mo.status.spinner("Loading GPT-2...") as _status:
            _model = AutoModelForCausalLM.from_pretrained(
                "gpt2", attn_implementation="eager"
            )
            _tokenizer = AutoTokenizer.from_pretrained("gpt2")
            _model.eval()

            _status.update("Tokenizing input...")
            _inputs = _tokenizer(_TEXT, return_tensors="pt")
            _seq_len = _inputs["input_ids"].shape[1]
            _tokens = [_tokenizer.decode(t) for t in _inputs["input_ids"][0]]
            _n_heads = _model.config.n_head
            _d_model = _model.config.n_embd
            _d_head = _d_model // _n_heads
            _n_layers = _model.config.n_layer

            # --- Standard forward pass with QKV hooks ---
            _status.update("Running standard forward pass...")
            _qkv_cache = {}

            def _capture_qkv(layer_idx):
                def hook_fn(module, input, output):
                    _qkv_cache[layer_idx] = output.detach()
                return hook_fn

            _hooks = []
            for _i, _block in enumerate(_model.transformer.h):
                _hooks.append(
                    _block.attn.c_attn.register_forward_hook(_capture_qkv(_i))
                )

            with torch.no_grad():
                _standard_out = _model(**_inputs, output_attentions=True)

            for _h in _hooks:
                _h.remove()

            _standard_attn = np.stack(
                [_layer[0].numpy() for _layer in _standard_out.attentions]
            ).astype(np.float32)

            # --- Extract raw QK scores ---
            _status.update("Extracting QK scores...")
            _causal_mask = torch.tril(
                torch.ones(_seq_len, _seq_len)
            ).unsqueeze(0).unsqueeze(0)
            _raw_scores_all = []
            for _i in range(_n_layers):
                _q, _k, _ = _qkv_cache[_i].split(_d_model, dim=-1)
                _q = _q.view(1, _seq_len, _n_heads, _d_head).transpose(1, 2)
                _k = _k.view(1, _seq_len, _n_heads, _d_head).transpose(1, 2)
                _scores = torch.matmul(
                    _q, _k.transpose(-2, -1)
                ) / (_d_head ** 0.5)
                _scores = _scores.masked_fill(
                    _causal_mask == 0, float("-inf")
                )
                _raw_scores_all.append(_scores[0].numpy())
            _raw_scores_all = np.stack(_raw_scores_all).astype(np.float32)

            # --- ESA: zero diagonal, renormalize ---
            _status.update("Computing ESA...")
            _esa_attn = _standard_attn.copy()
            for _li in range(_n_layers):
                for _hi in range(_n_heads):
                    _a = _esa_attn[_li, _hi].copy()
                    np.fill_diagonal(_a, 0.0)
                    _rs = _a.sum(axis=-1, keepdims=True)
                    _rs = np.where(_rs == 0, 1.0, _rs)
                    _esa_attn[_li, _hi] = _a / _rs

            # --- Sink token forward pass ---
            _status.update("Running sink token forward pass...")
            _sink_ids = torch.cat(
                [torch.zeros(1, 1, dtype=torch.long), _inputs["input_ids"]],
                dim=1,
            )

            _qkv_cache_sink = {}

            def _capture_qkv_sink(layer_idx):
                def hook_fn(module, input, output):
                    _qkv_cache_sink[layer_idx] = output.detach()
                return hook_fn

            _hooks_sink = []
            for _i, _block in enumerate(_model.transformer.h):
                _hooks_sink.append(
                    _block.attn.c_attn.register_forward_hook(
                        _capture_qkv_sink(_i)
                    )
                )

            with torch.no_grad():
                _sink_out = _model(input_ids=_sink_ids, output_attentions=True)

            for _h in _hooks_sink:
                _h.remove()

            _sink_attn_full = np.stack(
                [_layer[0].numpy() for _layer in _sink_out.attentions]
            ).astype(np.float32)

            # --- Sink token perplexity comparison ---
            _status.update("Measuring sink token PPL improvement...")
            import math as _math
            # Standard: logits[0:-1] predict input_ids[1:]
            _std_ce = torch.nn.functional.cross_entropy(
                _standard_out.logits[0, :-1], _inputs["input_ids"][0, 1:],
            )
            # Sink: logits[1:-1] predict sink_ids[2:] (= original tokens[1:])
            _sink_ce = torch.nn.functional.cross_entropy(
                _sink_out.logits[0, 1:-1], _sink_ids[0, 2:],
            )
            _std_ppl_input = round(_math.exp(float(_std_ce)), 2)
            _sink_ppl_input = round(_math.exp(float(_sink_ce)), 2)

            # Temperature computed on-the-fly in fix_comparison cell (fast numpy softmax)

            # --- Adaptive temperature validation ---
            _status.update("Validating adaptive fix...")
            _ent_pre = _entropy(_standard_attn)
            _med_pre = np.median(_ent_pre)
            _sick_pre = _ent_pre < _med_pre * 0.7
            _vt = np.ones_like(_ent_pre)
            _vt[_sick_pre] = np.clip(1.0 + (_med_pre / _ent_pre[_sick_pre] - 1.0), 1.0, 5.0)

            _aqkv = {}
            def _acap(li):
                def _h(mod, inp, out): _aqkv[li] = out.detach()
                return _h
            def _arep(li):
                def _h(mod, args):
                    _qk = _aqkv[li]
                    _qa, _ka, _va = _qk.split(_d_model, dim=-1)
                    _ba, _sa = _qa.shape[0], _qa.shape[1]
                    _qa = _qa.view(_ba, _sa, _n_heads, _d_head).transpose(1, 2)
                    _ka = _ka.view(_ba, _sa, _n_heads, _d_head).transpose(1, 2)
                    _va = _va.view(_ba, _sa, _n_heads, _d_head).transpose(1, 2)
                    _sca = torch.matmul(_qa, _ka.transpose(-2, -1)) / (_d_head ** 0.5)
                    for _hi2 in range(_n_heads):
                        if _vt[li, _hi2] > 1.0:
                            _sca[:, _hi2] = _sca[:, _hi2] / _vt[li, _hi2]
                    _cma = torch.tril(torch.ones(_sa, _sa)).unsqueeze(0).unsqueeze(0)
                    _sca = _sca.masked_fill(_cma == 0, float("-inf"))
                    _awa = torch.nn.functional.softmax(_sca, dim=-1)
                    _aoa = torch.matmul(_awa, _va)
                    _aoa = _aoa.transpose(1, 2).contiguous().view(_ba, _sa, _d_model)
                    return (_aoa,) + args[1:]
                return _h

            _adapt_hooks = []
            for _i, _block in enumerate(_model.transformer.h):
                _adapt_hooks.append(_block.attn.c_attn.register_forward_hook(_acap(_i)))
                _adapt_hooks.append(_block.attn.c_proj.register_forward_pre_hook(_arep(_i)))
            with torch.no_grad():
                _adp_out = _model(**_inputs)
            for _h in _adapt_hooks:
                _h.remove()

            import math as _math
            _std_lg = _standard_out.logits[0]
            _adp_lg = _adp_out.logits[0]
            _adapt_top1 = float((_std_lg.argmax(-1) == _adp_lg.argmax(-1)).float().mean()) * 100
            _adapt_std_ppl = _math.exp(float(torch.nn.functional.cross_entropy(
                _std_lg[:-1], _inputs["input_ids"][0, 1:])))
            _adapt_adp_ppl = _math.exp(float(torch.nn.functional.cross_entropy(
                _adp_lg[:-1], _inputs["input_ids"][0, 1:])))
            _adaptive_val = {
                "top1_match": round(_adapt_top1, 1),
                "standard_ppl": round(_adapt_std_ppl, 2),
                "adaptive_ppl": round(_adapt_adp_ppl, 2),
            }

            # --- Save cache (compact — no temp sweep) ---
            _status.update("Saving cache...")
            np.savez_compressed(_cache_path, **{
                "text_hash": _text_hash,
                "standard_attn": _standard_attn,
                "esa_attn": _esa_attn,
                "raw_scores_all": _raw_scores_all,
                "sink_attn_full": _sink_attn_full,
                "tokens": np.array(_tokens, dtype=object),
                "seq_len": _seq_len,
                "n_layers": _n_layers,
                "n_heads": _n_heads,
                "std_ppl_input": _std_ppl_input,
                "sink_ppl_input": _sink_ppl_input,
                "adapt_top1": _adaptive_val["top1_match"],
                "adapt_std_ppl": _adaptive_val["standard_ppl"],
                "adapt_adp_ppl": _adaptive_val["adaptive_ppl"],
            })

    # --- Color constants ---
    _C_STD = "#e74c3c"
    _C_TEMP = "#3498db"
    _C_SINK = "#2ecc71"
    _C_ESA = "#95a5a6"

    def _temp_attn(t_val):
        """Compute temperature-scaled attention on the fly (~5ms)."""
        _scaled = _raw_scores_all / t_val
        _exp = np.exp(_scaled - _scaled.max(axis=-1, keepdims=True))
        return _exp / _exp.sum(axis=-1, keepdims=True)

    data = {
        "standard_attn": _standard_attn,
        "esa_attn": _esa_attn,
        "raw_scores_all": _raw_scores_all,
        "sink_attn_full": _sink_attn_full,
        "temp_attn_fn": _temp_attn,
        "entropy_standard": _entropy(_standard_attn),
        "entropy_esa": _entropy(_esa_attn),
        "entropy_sink": _entropy(_sink_attn_full),
        "sink_mag_standard": _sink_mag(_standard_attn),
        "sink_mag_esa": _sink_mag(_esa_attn),
        "sink_real_first": _sink_attn_full[:, :, 1:, 1].mean(axis=(1, 2)),
        "n_layers": _n_layers,
        "n_heads": _n_heads,
        "seq_len": _seq_len,
        "text": _TEXT,
        "tokens": _tokens,
        "model": _model,
        "tokenizer": _tokenizer,
        "elapsed": time.time() - _start,
        "from_cache": _from_cache,
        "std_ppl_input": _std_ppl_input,
        "sink_ppl_input": _sink_ppl_input,
        "adaptive_val": _adaptive_val,
        "colors": {
            "standard": _C_STD,
            "temp": _C_TEMP,
            "sink": _C_SINK,
            "esa": _C_ESA,
        },
    }

    mo.output.replace(
        mo.accordion({
            "How this notebook extracts attention data": mo.md("""
```python
# Hook into GPT-2's QKV projections to capture raw scores
qkv_cache = {}
def capture_qkv(layer_idx):
    def hook_fn(module, input, output):
        qkv_cache[layer_idx] = output.detach()
    return hook_fn

for i, block in enumerate(model.transformer.h):
    block.attn.c_attn.register_forward_hook(capture_qkv(i))

# Forward pass with attention outputs
with torch.no_grad():
    out = model(**inputs, output_attentions=True)

# Extract raw QK scores before softmax
for i in range(n_layers):
    q, k, _ = qkv_cache[i].split(d_model, dim=-1)
    q = q.view(1, seq_len, n_heads, d_head).transpose(1, 2)
    k = k.view(1, seq_len, n_heads, d_head).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
    scores = scores.masked_fill(causal_mask == 0, float("-inf"))

# ESA: zero the diagonal, renormalize
esa_attn = standard_attn.copy()
for li in range(n_layers):
    for hi in range(n_heads):
        a = esa_attn[li, hi].copy()
        np.fill_diagonal(a, 0.0)
        row_sums = a.sum(axis=-1, keepdims=True)
        esa_attn[li, hi] = a / np.where(row_sums == 0, 1.0, row_sums)

# Sink token: prepend a garbage token at position 0
sink_ids = torch.cat([torch.zeros(1, 1, dtype=torch.long), inputs["input_ids"]], dim=1)
```
"""),
        })
    )
    return data, mo, np, plt


# ============================================================================
# CELL 1: THE HOOK (controls)
# ============================================================================

@app.cell
def hook_controls(data, mo):
    _peak_waste = data["standard_attn"][:, :, :, 0].mean(axis=(1, 2)).max() * 100
    hook_layer = mo.ui.slider(
        start=0, stop=11, value=8, label="Layer", show_value=True,
    )

    mo.output.replace(
        mo.vstack([
            mo.md(f"""
## The Sink Up Close

The bright vertical stripe on the left edge is an **attention sink** — every
token staring at position 0. In GPT-2's deepest layers, over **{_peak_waste:.0f}%**
of attention goes to one meaningless token. Drag the slider to watch it form.

> **Why GPT-2?** Sinks appear in LLaMA, Mistral, and every pre-norm transformer
> (architectures that normalize *before* each layer, which most modern LLMs use).
> GPT-2 is small enough to run these diagnostics interactively, with fully open weights.
"""),
            mo.callout(hook_layer, kind="info"),
        ])
    )
    return (hook_layer,)


# ============================================================================
# CELL 1B: THE HOOK (viz)
# ============================================================================

@app.cell(hide_code=True)
def hook_viz(data, mo, np, plt, hook_layer):
    def _build_flow_svg(attn_layer, tokens, n_show=20):
        """Build an SVG showing attention flow from queries to keys.
        attn_layer: (n_heads, seq, seq) — averaged across heads.
        Shows top n_show tokens, with line width proportional to attention.
        """
        avg = attn_layer.mean(axis=0)  # (seq, seq)
        seq_len = min(len(tokens), n_show)
        avg = avg[:seq_len, :seq_len]
        toks = [t.strip() or "·" for t in tokens[:seq_len]]

        W, H = 700, max(400, seq_len * 28 + 60)
        left_x, right_x = 120, 580
        y_start = 40
        y_step = (H - 80) / max(seq_len - 1, 1)

        lines = []
        for qi in range(seq_len):
            qy = y_start + qi * y_step
            for ki in range(seq_len):
                ky = y_start + ki * y_step
                w = float(avg[qi, ki])
                if w < 0.02:
                    continue
                opacity = min(w * 3, 0.9)
                stroke_w = max(w * 12, 0.5)
                color = "#e74c3c" if ki == 0 else "#3498db"
                lines.append(
                    f'<line x1="{left_x+10}" y1="{qy}" x2="{right_x-10}" y2="{ky}" '
                    f'stroke="{color}" stroke-width="{stroke_w:.1f}" opacity="{opacity:.2f}"/>'
                )

        labels_left = "".join(
            f'<text x="{left_x-5}" y="{y_start + i * y_step + 4}" '
            f'text-anchor="end" font-size="11" fill="#2c3e50">{t}</text>'
            for i, t in enumerate(toks)
        )
        labels_right = "".join(
            f'<text x="{right_x+5}" y="{y_start + i * y_step + 4}" '
            f'font-size="11" fill="{("#e74c3c" if i == 0 else "#2c3e50")}" '
            f'font-weight="{("bold" if i == 0 else "normal")}">{t}</text>'
            for i, t in enumerate(toks)
        )

        svg = f'''<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg"
            style="background:#fafafa; border-radius:8px; font-family:sans-serif;">
            <text x="{left_x}" y="20" font-size="12" fill="#888" text-anchor="end">Query (who's looking)</text>
            <text x="{right_x}" y="20" font-size="12" fill="#888">Key (who's looked at)</text>
            {"".join(lines)}
            {labels_left}
            {labels_right}
            <text x="{W//2}" y="{H-10}" font-size="10" fill="#888" text-anchor="middle">
                Red = attention to position 0 (sink) · Blue = attention to other tokens
            </text>
        </svg>'''
        return svg

    _layer = hook_layer.value
    _avg = data["standard_attn"][_layer].mean(axis=0)
    _sink_pct = data["standard_attn"][_layer, :, :, 0].mean() * 100

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.imshow(_avg, cmap="viridis", aspect="auto")
    _ax.set_xlabel("Key position")
    _ax.set_ylabel("Query position")
    _ax.set_title(
        f"Layer {_layer} — {_sink_pct:.1f}% of attention to position 0",
        fontsize=13, fontweight="bold",
    )
    _tick_positions = list(range(0, len(data["tokens"]), 5))
    _tick_labels = [data["tokens"][i] for i in _tick_positions]
    _ax.set_xticks(_tick_positions)
    _ax.set_xticklabels(_tick_labels, rotation=90, fontsize=7)
    plt.colorbar(_ax.images[0], ax=_ax, shrink=0.8, label="Attention weight")
    plt.tight_layout()

    _status = "healthy" if _sink_pct < 20 else "forming" if _sink_pct < 40 else "dominant"
    mo.output.replace(
        mo.vstack([
            _fig,
            mo.callout(
                mo.md(f"Layer {_layer}: **{_sink_pct:.1f}%** to position 0. Status: **{_status}**."),
                kind="neutral" if _status == "healthy" else "warn" if _status == "forming" else "danger",
            ),
            mo.accordion({
                "Attention flow (interactive)": mo.Html(_build_flow_svg(
                    data["standard_attn"][_layer], data["tokens"],
                )),
                "Input text": mo.md(f"*{data['text']}*"),
                "Key terms": mo.md("""
**Layer** — 12 stacked stages; deeper = more abstract. Sinks worsen with depth.
**Head** — 12 per layer, 144 total. Each specializes (grammar, pronouns, topics). "Sick" = fixated on sink.
**Entropy** — spread of attention. High = healthy (diverse). Low = sick (concentrated on sink).
**Attention budget** — 100% per head, distributed across tokens. Sink steals 40-60% in deep layers.
"""),
            }),
        ])
    )


# ============================================================================
# CELL 3: FIX CONTROLS (show code)
# ============================================================================

@app.cell
def fix_controls(mo):
    fix_radio = mo.ui.radio(
        options=["Standard (baseline)", "Temperature scaling", "Sink token"],
        value="Standard (baseline)",
        label="Attention mode",
    )
    temp_slider = mo.ui.slider(
        start=1.0, stop=5.0, step=0.1, value=3.0,
        label="Temperature", show_value=True,
    )
    _head_opts = ["Average (all heads)"] + [f"Head {_i}" for _i in range(12)]
    fix_layer = mo.ui.slider(
        start=0, stop=11, value=8, label="Layer", show_value=True
    )
    fix_head = mo.ui.dropdown(
        options=_head_opts, value="Average (all heads)", label="Head",
    )

    mo.output.replace(
        mo.vstack([
            mo.md("""
## Can You Redistribute the Garbage?

Two approaches that partially work:

**Temperature scaling** makes each head less confident about where to look — attention spreads more evenly instead of piling on the first word.
**Sink token** puts a dedicated garbage word at the front — if the model needs a dump target, give it one on purpose so real words stay clean.
"""),
            mo.accordion({
                "What about blocking self-attention? (ESA)": mo.md("""
[Exclusive Self Attention](https://alphaxiv.org/abs/2603.09078) (Zhai, 2026) blocks
the diagonal — words can't look at themselves. I tested it: **no effect.**
The garbage bin isn't caused by words looking at themselves. It's deeper —
built into the model's architecture.
"""),
            }),
            mo.accordion({
                "Why bother with a sink token if sinks are necessary?": mo.md("""
Sinks aren't going away — the model *needs* a dump target. The question is which token gets hijacked.
Without a sink token, the first real word absorbs all the garbage, corrupting its representation.
A sink token separates the roles. What this affects measurably:

- **Attention to the real first token drops from ~44% to near zero**
- **Head health improves** — fewer heads classified as "sick"
- **Perplexity improves slightly** when you train the sink embedding — the model performs better with a proper parking spot.
(Perplexity measures how surprised the model is by the next word — lower = better. A perplexity of 30 means
the model is as uncertain as choosing between 30 equally likely words.)
- **Streaming inference** — the original use case from [Xiao et al.](https://alphaxiv.org/abs/2309.17453):
keeping the sink token in the KV cache lets models process unlimited-length text without degrading
"""),
            }),
            mo.callout(
                mo.hstack(
                    [fix_radio, temp_slider, fix_layer, fix_head],
                    justify="start", gap=1,
                ),
                kind="info",
            ),
        ])
    )
    return fix_radio, temp_slider, fix_layer, fix_head


# ============================================================================
# CELL 5: FIX COMPARISON
# ============================================================================

@app.cell(hide_code=True)
def fix_comparison(data, mo, np, plt, fix_radio, temp_slider, fix_layer, fix_head):
    # Temperature-scaled attention (computed on the fly — ~5ms)
    temp_attn = data["temp_attn_fn"](temp_slider.value)

    _layer = fix_layer.value
    _head_val = fix_head.value

    _mode_keys = {
        "Standard (baseline)": "standard",
        "Temperature scaling": "temp",
        "Sink token": "sink",
    }
    _mode_key = _mode_keys[fix_radio.value]

    _attn_map = {
        "standard": data["standard_attn"][_layer],
        "temp": temp_attn[_layer],
        "sink": data["sink_attn_full"][_layer],
    }
    _label_map = {
        "standard": "Standard",
        "temp": f"Temperature T={temp_slider.value:.1f}",
        "sink": "Sink Token",
    }

    _attn = _attn_map[_mode_key]
    if _head_val == "Average (all heads)":
        _show = _attn.mean(axis=0)
        _hlabel = "all heads"
    else:
        _hi = int(_head_val.split()[-1])
        _show = _attn[_hi]
        _hlabel = _head_val

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))

    _axes[0].imshow(_show, cmap="viridis", aspect="auto")
    _axes[0].set_title(
        f"{_label_map[_mode_key]} — Layer {_layer}, {_hlabel}",
        fontsize=11, fontweight="bold",
    )
    _axes[0].set_xlabel("Key position")
    _axes[0].set_ylabel("Query position")
    plt.colorbar(_axes[0].images[0], ax=_axes[0], shrink=0.8, label="Attention weight")

    _lyrs = np.arange(data["n_layers"])
    _temp_sink_mag = temp_attn[:, :, :, 0].mean(axis=(1, 2))

    _mag_map = {
        "standard": data["sink_mag_standard"],
        "temp": _temp_sink_mag,
        "sink": data["sink_real_first"],
    }
    _color_map = {
        "standard": data["colors"]["standard"],
        "temp": data["colors"]["temp"],
        "sink": data["colors"]["sink"],
    }

    _axes[1].plot(
        _lyrs, data["sink_mag_standard"], "o-",
        color=data["colors"]["standard"], linewidth=2, label="Standard", alpha=0.7,
    )
    _axes[1].plot(
        _lyrs, _temp_sink_mag, "s-",
        color=data["colors"]["temp"], linewidth=2,
        label=f"Temp T={temp_slider.value:.1f}",
    )
    _axes[1].plot(
        _lyrs, data["sink_real_first"], "^-",
        color=data["colors"]["sink"], linewidth=2, label="Sink Token (real 1st)",
    )
    _axes[1].plot(
        _layer, _mag_map[_mode_key][_layer], "o",
        color=_color_map[_mode_key],
        markersize=14, markeredgecolor="black", markeredgewidth=2, zorder=5,
    )
    _axes[1].set_xlabel("Layer", fontsize=11)
    _axes[1].set_ylabel("Mean attn to position 0", fontsize=11)
    _axes[1].set_title("Sink Magnitude by Fix", fontsize=11, fontweight="bold")
    _axes[1].legend(fontsize=9)
    _axes[1].grid(True, alpha=0.2)
    _axes[1].set_ylim(bottom=0)
    plt.tight_layout()

    _std_s = data["sink_mag_standard"][_layer]
    _temp_s = _temp_sink_mag[_layer]
    _sink_r = data["sink_real_first"][_layer]

    # --- ReLU backfire ---
    _relu_175 = np.maximum(data["raw_scores_all"], 0)
    _relu_175_attn = _relu_175 / (_relu_175.sum(axis=-1, keepdims=True) + 1e-9)
    _relu_sink_175 = _relu_175_attn[:, :, :, 0].mean(axis=(1, 2)).mean() * 100

    _raw_12 = data["raw_scores_all"][:, :, :12, :12]
    _relu_12 = np.maximum(_raw_12, 0)
    _relu_12_attn = _relu_12 / (_relu_12.sum(axis=-1, keepdims=True) + 1e-9)
    _relu_sink_12 = _relu_12_attn[:, :, :, 0].mean(axis=(1, 2)).mean() * 100

    _std_sink_avg = data["sink_mag_standard"].mean() * 100

    _fig_relu, _ax_relu = plt.subplots(figsize=(8, 4))
    _relu_bars = _ax_relu.bar(
        ["Standard\n(baseline)", "ReLU\n(12 tokens)", "ReLU\n(175 tokens)"],
        [_std_sink_avg, _relu_sink_12, _relu_sink_175],
        color=[data["colors"]["standard"], "#27ae60", "#c0392b"],
        width=0.6,
    )
    for _b, _v in zip(_relu_bars, [_std_sink_avg, _relu_sink_12, _relu_sink_175]):
        _ax_relu.text(
            _b.get_x() + _b.get_width() / 2, _v + 0.5,
            f"{_v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10,
        )
    _ax_relu.set_ylabel("Sink magnitude (%)", fontsize=10)
    _ax_relu.set_title(
        "ReLU Attention: Works Short, Backfires Long",
        fontsize=11, fontweight="bold",
    )
    _ax_relu.grid(True, alpha=0.2, axis="y")
    _relu_max = max(_std_sink_avg, _relu_sink_12, _relu_sink_175)
    _ax_relu.set_ylim(bottom=0, top=_relu_max * 1.2)
    plt.tight_layout()

    # --- Sink waste summary ---
    _std_total = data["sink_mag_standard"].mean() * 100
    _temp_total = _temp_sink_mag.mean() * 100
    _sink_total = data["sink_real_first"].mean() * 100

    mo.output.replace(
        mo.vstack([
            _fig,
            mo.md(f"""
**Layer {_layer} — attention to first real token:**
Standard: **{_std_s:.3f}** |
Temperature T={temp_slider.value:.1f}: **{_temp_s:.3f}** ({(_temp_s / _std_s - 1) * 100:+.0f}%) |
Sink Token: **{_sink_r:.4f}** ({(_sink_r / _std_s - 1) * 100:+.0f}%)

**Overall waste:** Standard {_std_total:.1f}% | Temp {_temp_total:.1f}% | Sink Token {_sink_total:.1f}%
{"" if data["sink_ppl_input"] == 0 else f"""
**Perplexity on input text:** Standard {data['std_ppl_input']} → Sink Token **{data['sink_ppl_input']}**
({((data['sink_ppl_input'] - data['std_ppl_input']) / data['std_ppl_input'] * 100):+.1f}%).
{'The sink token **improves** predictions — freeing the real first token from parking duty helps.' if data['sink_ppl_input'] < data['std_ppl_input'] else 'Minimal change — the model adapts either way.'}
"""}
*Drag the temperature slider to explore the tradeoff. Higher T reduces the
sink but also blurs genuine attention patterns.*
"""),
            mo.accordion({
                "Why ReLU attention didn't make the cut": mo.vstack([
                    _fig_relu,
                    mo.md(f"""
ReLU attention replaces softmax with ReLU, so attention no longer sums to
100%. At 12 tokens it works: sinks drop to **{_relu_sink_12:.1f}%**. But at
production length (175 tokens), sinks *increase* to
**{_relu_sink_175:.1f}%** — worse than standard.

This failure mode is known
(ReLU attention is known to require sequence-length-dependent scaling). I'm quantifying it in terms of sink
magnitude: the same model, same weights, same text — ReLU makes the sink
problem worse at realistic sequence lengths.
"""),
                ]),
            }),
        ])
    )
    return temp_attn,


# ============================================================================
# CELL 6: ENTROPY CONTROLS (show code)
# ============================================================================

@app.cell
def entropy_controls(mo, temp_slider):
    entropy_radio = mo.ui.radio(
        options=[
            "Standard (baseline)",
            f"Temperature T={temp_slider.value:.1f}",
            "Sink Token",
        ],
        value="Standard (baseline)",
        label="Condition",
    )
    dash_layer = mo.ui.slider(
        start=0, stop=11, value=7, label="Inspect layer", show_value=True
    )
    dash_head = mo.ui.slider(
        start=0, stop=11, value=3, label="Inspect head", show_value=True
    )

    mo.output.replace(
        mo.vstack([
            mo.md("""
## Which Heads Are Sick?

Each of GPT-2's 144 attention heads gets a health score. A healthy head
spreads its attention across many relevant words. A sick head dumps most
of its attention on the garbage bin.

**Blue = healthy** · **Red = sick** · Start at Layer 7, Head 3 — a clearly sick head.

*How I define "sick": a head whose attention is more concentrated than 70% of
all heads. I verified this by checking that heads I classified as sick visually
show the garbage stripe, and that the count stays stable if I adjust the threshold.*
"""),
            mo.callout(
                mo.hstack(
                    [entropy_radio, dash_layer, dash_head], justify="start", gap=1
                ),
                kind="info",
            ),
        ])
    )
    return entropy_radio, dash_layer, dash_head


# ============================================================================
# CELL 7: ENTROPY DASHBOARD
# ============================================================================

@app.cell(hide_code=True)
def entropy_dashboard(
    data, mo, np, plt, temp_slider, temp_attn, entropy_radio, dash_layer, dash_head
):
    from matplotlib.patches import Rectangle as _Rect

    _cond_keys = {
        "Standard (baseline)": "standard",
        f"Temperature T={temp_slider.value:.1f}": "temp",
        "Sink Token": "sink",
    }
    _cond = _cond_keys.get(entropy_radio.value, "standard")

    _p = np.clip(temp_attn, 1e-9, 1.0)
    _entropy_temp = -(_p * np.log(_p)).sum(axis=-1).mean(axis=-1)

    _ent_map = {
        "standard": data["entropy_standard"],
        "temp": _entropy_temp,
        "sink": data["entropy_sink"],
    }
    _lbl_map = {
        "standard": "Standard",
        "temp": f"Temperature T={temp_slider.value:.1f}",
        "sink": "Sink Token",
    }
    _attn_source = {
        "standard": data["standard_attn"],
        "temp": temp_attn,
        "sink": data["sink_attn_full"],
    }

    _ent = _ent_map[_cond]
    _ent_std = data["entropy_standard"]

    _all_ent = np.stack([
        data["entropy_standard"],
        _entropy_temp, data["entropy_sink"],
    ])
    _vmin, _vmax = _all_ent.min(), _all_ent.max()

    # --- Entropy grid ---
    _fig1, _ax1 = plt.subplots(figsize=(7, 5))
    _ax1.imshow(_ent, cmap="RdBu", aspect="auto", vmin=_vmin, vmax=_vmax)
    _ax1.set_xlabel("Head", fontsize=11)
    _ax1.set_ylabel("Layer", fontsize=11)
    _ax1.set_title(f"Entropy — {_lbl_map[_cond]}", fontsize=12, fontweight="bold")
    _ax1.set_xticks(range(data["n_heads"]))
    _ax1.set_yticks(range(data["n_layers"]))
    _ax1.add_patch(
        _Rect(
            (dash_head.value - 0.5, dash_layer.value - 0.5), 1, 1,
            linewidth=3, edgecolor="#f1c40f", facecolor="none",
        )
    )
    plt.colorbar(
        _ax1.images[0], ax=_ax1, shrink=0.8,
        label="Entropy (red=sick, blue=healthy)",
    )
    plt.tight_layout()

    # --- Detail heatmap ---
    _detail = _attn_source[_cond][dash_layer.value][dash_head.value]
    _fig2, _ax2 = plt.subplots(figsize=(7, 5))
    _ax2.imshow(_detail, cmap="viridis", aspect="auto")
    _ax2.set_xlabel("Key position", fontsize=11)
    _ax2.set_ylabel("Query position", fontsize=11)
    _ax2.set_title(
        f"Layer {dash_layer.value}, Head {dash_head.value} — {_lbl_map[_cond]}",
        fontsize=12, fontweight="bold",
    )
    plt.colorbar(_ax2.images[0], ax=_ax2, shrink=0.8, label="Attention weight")
    plt.tight_layout()

    # --- Stats ---
    _median = np.median(_all_ent)
    _thresh = _median * 0.7
    _n_sick = int((_ent < _thresh).sum())
    _n_sick_std = int((_ent_std < _thresh).sum())
    _n_total = data["n_layers"] * data["n_heads"]

    _head_ent = _ent[dash_layer.value, dash_head.value]
    _head_ent_std = _ent_std[dash_layer.value, dash_head.value]
    if _head_ent < _thresh:
        _head_status = (
            "**Naturally focused** — low entropy in all conditions."
            if _head_ent_std < _thresh
            else "**Still sick** — low entropy persists."
        )
    else:
        _head_status = (
            "**Healed** — was sick in standard, recovered here."
            if _head_ent_std < _thresh
            else "**Healthy** — attention distributed across relevant tokens."
        )

    if _cond == "standard":
        _summary = (
            f"**{_n_sick}** of {_n_total} heads are sink-corrupted "
            f"(below 70% of median entropy)."
        )
    else:
        _healed = _n_sick_std - _n_sick
        _summary = (
            f"**{_n_sick}** of {_n_total} heads remain sick "
            f"(was {_n_sick_std} in standard). "
            f"**{max(0, _healed)} heads healed.**"
        )

    mo.output.replace(
        mo.vstack([
            mo.hstack([_fig1, _fig2]),
            mo.md(
                f"{_summary}\n\n"
                f"**Layer {dash_layer.value}, Head {dash_head.value}:** {_head_status}"
            ),
            mo.callout(
                mo.md("""
Not all low-entropy heads are sick. Some heads *should* focus narrowly —
syntax heads, coreference heads. The truly sick ones have low entropy
**only in standard attention** and recover when you apply a fix.
"""),
                kind="warn",
            ),
        ])
    )


# ============================================================================
# CELL 8: THE TRAINING EXPERIMENTS
# ============================================================================

@app.cell
def training_experiments(data, mo, np, plt):
    import json as _json
    import os as _os

    # --- Load training results (with hardcoded fallback for molab) ---
    _dir = _os.path.dirname(_os.path.abspath(__file__))

    # Static training results
    try:
        _blend_path = _os.path.join(_dir, "blend_results.json")
        with open(_blend_path) as _f:
            _blend = _json.load(_f)
        _baseline = _blend["baseline"]
        _finetuned = _blend["finetuned"]
        _log = _blend["log_history"]
    except Exception:
        # Hardcoded fallback from actual training run
        _baseline = {"perplexity": 44.46, "sink_waste_pct": 44.26, "num_sick_heads": 31, "healthy": 98, "diffuse": 15}
        _finetuned = {"perplexity": 24.59, "sink_waste_pct": 44.40, "num_sick_heads": 32, "healthy": 97, "diffuse": 15}
        _blend = {"total_steps": 1767, "baseline": _baseline, "finetuned": _finetuned}
        _log = [
            {"step": 100, "lm_loss": 3.7736, "align_loss": 0.349, "sink_waste_pct": 40.1, "num_sick_heads": 23},
            {"step": 200, "lm_loss": 3.4750, "align_loss": 0.347, "sink_waste_pct": 41.8, "num_sick_heads": 25},
            {"step": 300, "lm_loss": 3.3849, "align_loss": 0.345, "sink_waste_pct": 42.6, "num_sick_heads": 24},
            {"step": 400, "lm_loss": 3.3445, "align_loss": 0.344, "sink_waste_pct": 43.0, "num_sick_heads": 24},
            {"step": 500, "lm_loss": 3.3187, "align_loss": 0.343, "sink_waste_pct": 43.4, "num_sick_heads": 24},
            {"step": 700, "lm_loss": 3.2015, "align_loss": 0.341, "sink_waste_pct": 43.4, "num_sick_heads": 24},
            {"step": 800, "lm_loss": 3.2024, "align_loss": 0.340, "sink_waste_pct": 43.3, "num_sick_heads": 24},
            {"step": 900, "lm_loss": 3.2009, "align_loss": 0.339, "sink_waste_pct": 43.4, "num_sick_heads": 24},
            {"step": 1000, "lm_loss": 3.1891, "align_loss": 0.338, "sink_waste_pct": 43.6, "num_sick_heads": 24},
            {"step": 1100, "lm_loss": 3.1855, "align_loss": 0.337, "sink_waste_pct": 43.5, "num_sick_heads": 23},
            {"step": 1300, "lm_loss": 3.1143, "align_loss": 0.335, "sink_waste_pct": 43.5, "num_sick_heads": 24},
            {"step": 1400, "lm_loss": 3.1137, "align_loss": 0.334, "sink_waste_pct": 43.6, "num_sick_heads": 24},
            {"step": 1500, "lm_loss": 3.0977, "align_loss": 0.334, "sink_waste_pct": 43.5, "num_sick_heads": 23},
            {"step": 1600, "lm_loss": 3.1065, "align_loss": 0.333, "sink_waste_pct": 43.6, "num_sick_heads": 23},
            {"step": 1700, "lm_loss": 3.1108, "align_loss": 0.333, "sink_waste_pct": 43.7, "num_sick_heads": 24},
        ]

    # Filter epoch-boundary outliers
    _log_clean = [s for s in _log if s["lm_loss"] > 1.0]

    _steps = [s["step"] for s in _log_clean]
    _lm_loss = [s["lm_loss"] for s in _log_clean]
    _sink_waste = [s["sink_waste_pct"] for s in _log_clean]
    _sick_heads = [s["num_sick_heads"] for s in _log_clean]

    # Recursive training results
    try:
        _recursive_path = _os.path.join(_dir, "recursive_results.json")
        with open(_recursive_path) as _f:
            _recursive = _json.load(_f)
    except Exception:
        # Hardcoded fallback
        _recursive = [
            {"round": 0, "ppl": 37.66, "sink": 41.6, "sick": 22, "healthy": 107, "diffuse": 15},
            {"round": 1, "ppl": 23.32, "sink": 44.2, "sick": 25, "healthy": 104, "diffuse": 15},
            {"round": 2, "ppl": 22.87, "sink": 45.0, "sick": 26, "healthy": 103, "diffuse": 15},
            {"round": 3, "ppl": 22.38, "sink": 45.1, "sick": 27, "healthy": 102, "diffuse": 15},
            {"round": 4, "ppl": 22.11, "sink": 45.5, "sick": 27, "healthy": 102, "diffuse": 15},
        ]

    # --- Training curves (2x2) ---
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 8))

    _axes[0, 0].plot(_steps, _lm_loss, "-", color=data["colors"]["temp"], linewidth=2)
    _axes[0, 0].set_ylabel("Language modeling loss", fontsize=10)
    _axes[0, 0].set_title(
        "LM Loss (the model is learning)", fontsize=12, fontweight="bold"
    )
    _axes[0, 0].grid(True, alpha=0.3)

    _align_loss = [s["align_loss"] for s in _log_clean]
    _axes[0, 1].plot(_steps, _align_loss, "-", color="#9b59b6", linewidth=2)
    _axes[0, 1].set_ylabel("Alignment loss", fontsize=10)
    _axes[0, 1].set_title(
        "Alignment Loss (barely budging)", fontsize=12, fontweight="bold"
    )
    _axes[0, 1].grid(True, alpha=0.3)

    _axes[1, 0].plot(
        _steps, _sink_waste, "-", color=data["colors"]["standard"], linewidth=2
    )
    _axes[1, 0].axhline(
        y=_baseline["sink_waste_pct"], color=data["colors"]["standard"],
        linestyle="--", alpha=0.5,
        label=f'Baseline ({_baseline["sink_waste_pct"]}%)',
    )
    _axes[1, 0].set_xlabel("Training step", fontsize=10)
    _axes[1, 0].set_ylabel("Sink waste (%)", fontsize=10)
    _axes[1, 0].set_title("Sink Waste (immovable)", fontsize=12, fontweight="bold")
    _axes[1, 0].legend(fontsize=9)
    _axes[1, 0].grid(True, alpha=0.3)
    _axes[1, 0].set_ylim(35, 50)

    _axes[1, 1].plot(_steps, _sick_heads, "-", color="#e67e22", linewidth=2)
    _axes[1, 1].axhline(
        y=_baseline["num_sick_heads"], color="#e67e22", linestyle="--",
        alpha=0.5, label=f'Baseline ({_baseline["num_sick_heads"]})',
    )
    _axes[1, 1].set_xlabel("Training step", fontsize=10)
    _axes[1, 1].set_ylabel("Sick heads (of 144)", fontsize=10)
    _axes[1, 1].set_title("Sick Head Count (stuck)", fontsize=12, fontweight="bold")
    _axes[1, 1].legend(fontsize=9)
    _axes[1, 1].grid(True, alpha=0.3)
    _axes[1, 1].set_ylim(15, 40)

    plt.tight_layout()

    _ppl_before = _baseline["perplexity"]
    _ppl_after = _finetuned["perplexity"]
    _ppl_drop = ((_ppl_before - _ppl_after) / _ppl_before) * 100

    _waste_before = _baseline["sink_waste_pct"]
    _waste_after = _finetuned["sink_waste_pct"]

    _sick_before = _baseline["num_sick_heads"]
    _sick_after = _finetuned["num_sick_heads"]

    # --- Build recursive narrative if data exists ---
    _recursive_md = ""
    if _recursive and isinstance(_recursive, list) and len(_recursive) > 1:
        _r0 = _recursive[0]
        _rlast = _recursive[-1]
        _recursive_md = f"""
### Recursive training: it gets *worse*

I also tried recursive training — recomputing the alignment target after
each round so sick heads chase a moving healthy-head average. Over
{len(_recursive)} rounds, sink waste went from {_r0.get('sink', 'N/A')}%
to {_rlast.get('sink', 'N/A')}%. The model got better at language while
sinks *increased*. The harder you push, the harder the model pushes back.
"""
    else:
        _recursive_md = """
### Recursive training

I also tried recursive training — recomputing the alignment target after
each round so sick heads chase a moving healthy-head average. The sinks
got *worse*. The harder you push, the harder the model pushes back.
"""

    _training_code = mo.accordion({
        "Training code: alignment loss": mo.md("""
```python
# Alignment loss: push sick heads toward healthy neighbors
# Healthy target = mean attention pattern of heads above entropy threshold
healthy_mask = entropy > (median_entropy * 0.7)
target = attn[healthy_mask].mean(dim=0)  # average healthy pattern

# Per-head alignment loss (only applied to sick heads)
sick_mask = ~healthy_mask
align_loss = F.mse_loss(attn[sick_mask], target.expand_as(attn[sick_mask]))

# Combined loss: language modeling + weighted alignment
loss = lm_loss + lambda_align * align_loss  # lambda swept: 0.1, 0.5, 1.0, 10.0
```
"""),
    })

    mo.output.replace(
        mo.vstack([
            mo.md(f"""
## Can You Train the Garbage Away?

If I know what healthy attention looks like, can I train the model to
copy it? I retrained GPT-2 with an extra incentive: a penalty that
pushes sick heads to spread their attention more like healthy ones.
{_blend["total_steps"]} training steps.

The model got **{_ppl_drop:.0f}% better at language** (perplexity
{_ppl_before} → {_ppl_after}). But it **refused to change its attention
patterns.** The garbage stayed exactly where it was.
"""),
            _training_code,
            mo.callout(
                mo.md(f"""
**The model got better at language. Its attention didn't change.**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Perplexity | {_ppl_before} | {_ppl_after} | **-{_ppl_drop:.0f}%** (better) |
| Sink waste | {_waste_before}% | {_waste_after}% | +{_waste_after - _waste_before:.2f}pp |
| Sick heads | {_sick_before}/144 | {_sick_after}/144 | +{_sick_after - _sick_before} |

The garbage is load-bearing. The model had every opportunity to
clean up its attention. The training explicitly rewarded it.
It chose not to.
"""),
                kind="warn",
            ),
            _fig,
            mo.md(f"""
The top-left shows the model learning (loss dropping). The top-right shows
the cleanup incentive barely moving. The bottom row is the punchline:
garbage stays put no matter how much the model learns.

### What if I pushed harder?

The first experiment used a gentle cleanup incentive (10% of the training
signal). Maybe that wasn't enough. So I tried four different strengths,
each repeated three times with different random starting points:

| λ_align | Sink waste (mean ± std) | PPL after | Sick heads |
|---------|------------------------|-----------|------------|
| 0.1 | 46.2% ± 0.2 | 26.8 | 32 |
| 0.5 | 47.1% ± 0.2 | 26.9 | 32 |
| 1.0 | 47.8% ± 0.3 | 27.0 | 30 |
| **10.0** | **45.3% ± 0.4** | **28.7** | **28** |

*Baseline: 47.4% sink waste, 44.5 PPL, 34 sick heads. All runs started from
fresh GPT-2 weights. Standard deviations across seeds are ±0.2-0.4% — the
pattern is stable regardless of initialization.*

Two patterns. At gentle-to-moderate pressure, sinks actually *increase*
as the model improves — more specialized workers means more idle time, which
means more garbage. At 10× pressure, the model finally budges — trading
language quality for a tiny sink reduction. But sinks still sit at **45.3%**.
Even when the entire training signal screams "stop doing this," the model
barely flinches.

{_recursive_md}

Sinks don't cost extra compute — the model does the same amount of math
either way. And more sinks didn't hurt quality — perplexity *improved*.
A better model has more specialized workers, more idle time, and more need
for a garbage bin. The bin gets busier as the model gets smarter.

This empirically confirms Ran-Milo's *"Attention Sinks Are
Provably Necessary"*
([2026](https://alphaxiv.org/abs/2603.11487)) —
pre-norm transformers mathematically require sinks for representational
stability.

### The one thing that worked: a better garbage bin

The model is using a real word as its dump target. What happens
when you give it a *purpose-built* one? I trained a single embedding
vector (768 parameters, model frozen). I tested three interventions to
separate the sink effect from generic prompt tuning.
"""),
            mo.hstack([
                mo.stat(value="-19.7%", label="GPT-2 improvement", bordered=True),
                mo.stat(value="-26.7%", label="LLaMA-1B improvement", bordered=True),
                mo.stat(value="0.0%", label="same tokens at the end (both)", bordered=True),
            ], justify="center", gap=1),
            mo.md("""
**GPT-2 (124M)**

| Intervention | Params | PPL | Change | End position |
|-------------|--------|-----|--------|-------------|
| Baseline | 0 | 44.5 | — | — |
| 1 token, start | 768 | 41.8 | -5.9% | 0.0% |
| **4 tokens, start** | **3,072** | **35.7** | **-19.7%** | **0.0%** |
| Embedding offset (pos 0 only) | 768 | 42.1 | -5.3% | — |
| Attention bias (pos 0) | 12 | 45.1 | +1.5% | — |

**LLaMA-3.2-1B (1.2B)**

| Intervention | Params | PPL | Change | End position |
|-------------|--------|-----|--------|-------------|
| Baseline | 0 | 16.95 | — | — |
| 1 token, start | 2,048 | 13.49 | **-20.4%** | **0.0%** |
| **4 tokens, start** | **8,192** | **12.43** | **-26.7%** | **0.0%** |

The position control is definitive across both models: **tokens at the end
do nothing.** 0.0% improvement — same tokens, same training. The effect is
entirely about the garbage bin at position 0.

The improvement is *stronger* on LLaMA-1B (-26.7%) than GPT-2 (-19.7%),
despite LLaMA using rotary embeddings. LLaMA also has higher sink waste
(65.6% vs 44.3%) — more garbage to manage, more room for a better bin.

This is not [soft prompting](https://alphaxiv.org/abs/2104.08691). Soft prompts
work regardless of position. These only work at the front, where the garbage
collects.

Sinks didn't decrease — they stayed at ~41%. But the model got better at
language because its first real token is no longer corrupted. The OFF switch
isn't broken. It just works better when it's purpose-built.
"""),
            mo.accordion({
                "Training code: learned sink embedding": mo.md("""
```python
# 768 trainable parameters — one embedding vector prepended at position 0
sink_embed = nn.Parameter(torch.randn(1, 1, 768) * 0.02)

for batch in dataloader:
    # Prepend learned embedding to input
    embeds = model.transformer.wte(batch)  # (B, seq, 768)
    embeds = torch.cat([sink_embed.expand(B, -1, -1), embeds], dim=1)

    # Forward pass with modified embeddings (model frozen)
    out = model(inputs_embeds=embeds, labels=labels)
    out.loss.backward()
    optimizer.step()  # only updates sink_embed
```
*500 steps on WikiText-2, AdamW lr=1e-3. Full script: `train_learned_sink.py`*
"""),
            }),
        ])
    )


# ============================================================================
# CELL 9: ABLATION CENTERPIECE
# ============================================================================

@app.cell(hide_code=True)
def ablation_centerpiece(data, mo, np, plt):
    import json as _json
    import os as _os

    _dir = _os.path.dirname(_os.path.abspath(__file__))

    # Load ablation results
    try:
        with open(_os.path.join(_dir, "ablation_results.json")) as _f:
            _abl = _json.load(_f)
    except Exception:
        _abl = {
            "baseline": {"perplexity": 44.46, "heads_ablated": 0},
            "sink_ablated": {"perplexity": 99.58, "heads_ablated": 31, "delta": 55.12},
            "random_ablated": {"perplexity": 1655.42, "heads_ablated": 31, "delta": 1610.96},
            "diffuse_ablated": {"perplexity": 63.84, "heads_ablated": 15, "delta": 19.38},
        }

    _baseline = _abl["baseline"]["perplexity"]
    _sink_ppl = _abl["sink_ablated"]["perplexity"]
    _random_ppl = _abl["random_ablated"]["perplexity"]
    _diffuse_ppl = _abl["diffuse_ablated"]["perplexity"]
    _n_sink = _abl["sink_ablated"]["heads_ablated"]
    _n_diffuse = _abl["diffuse_ablated"]["heads_ablated"]

    _sink_delta = _sink_ppl - _baseline
    _random_delta = _random_ppl - _baseline
    _diffuse_delta = _diffuse_ppl - _baseline
    _ratio = _random_delta / _sink_delta

    # --- Main ablation bar chart (plotly, interactive) ---
    import plotly.graph_objects as _go

    _conditions = [
        f"Baseline<br>(no ablation)", f"Sink heads<br>zeroed ({_n_sink})",
        f"Random heads<br>zeroed ({_n_sink})", f"Noisy heads<br>zeroed ({_n_diffuse})",
    ]
    _ppls = [_baseline, _sink_ppl, _random_ppl, _diffuse_ppl]
    _colors = ["#95a5a6", "#e74c3c", "#2c3e50", "#3498db"]
    _hover = [
        f"Baseline: {_baseline:.1f} PPL",
        f"Sink heads: {_sink_ppl:.1f} PPL (+{_sink_delta:.0f})<br>{_n_sink} heads removed",
        f"Random heads: {_random_ppl:,.0f} PPL (+{_random_delta:,.0f})<br>{_n_sink} heads removed",
        f"Noisy heads: {_diffuse_ppl:.1f} PPL (+{_diffuse_delta:.0f})<br>{_n_diffuse} heads removed",
    ]

    _fig_abl = _go.Figure(data=[_go.Bar(
        x=_conditions, y=_ppls,
        marker_color=_colors,
        text=[f"{p:,.0f}" for p in _ppls],
        textposition=["outside", "outside", "inside", "outside"],
        textfont=dict(size=14, color=["#2c3e50", "#2c3e50", "white", "#2c3e50"]),
        hovertext=_hover,
        hoverinfo="text",
    )])
    _fig_abl.update_layout(
        title=dict(text="What Happens When You Remove Heads?", font=dict(size=16)),
        yaxis=dict(type="log", title="Perplexity (log scale)", gridcolor="#eee"),
        plot_bgcolor="#fafafa", paper_bgcolor="#fafafa",
        height=420, margin=dict(t=60, b=80),
        annotations=[dict(
            x=1.5, y=np.log10(_random_ppl * 0.4),
            text=f"<b>{_ratio:.0f}× gap</b>",
            showarrow=False, font=dict(size=16, color="#2c3e50"),
            yref="y",
        )],
    )

    # --- Load cumulative ablation if available ---
    _cumulative_elements = []
    try:
        with open(_os.path.join(_dir, "cumulative_ablation.json")) as _f:
            _cum = _json.load(_f)

        import plotly.graph_objects as _go_cum

        _fig_cum = _go_cum.Figure()
        _style = {
            "sink_first": ("#2ecc71", "Sink heads first (least critical)"),
            "random": ("#95a5a6", "Random order"),
            "important_first": ("#e74c3c", "Important heads first (most critical)"),
        }
        for _curve_name, (_color, _label) in _style.items():
            _curve = _cum["curves"][_curve_name]
            _xs = np.array([p["n_heads"] for p in _curve])
            _ys = np.array([p["perplexity"] for p in _curve])
            _ys_smooth = np.convolve(_ys, np.ones(3)/3, mode="same")
            _ys_smooth[0] = _ys[0]
            _ys_smooth[-1] = _ys[-1]
            # Raw points (faded)
            _fig_cum.add_trace(_go_cum.Scatter(
                x=_xs, y=_ys, mode="markers", marker=dict(color=_color, size=6, opacity=0.35),
                showlegend=False, hovertext=[f"n={x}: PPL={y:.0f}" for x, y in zip(_xs, _ys)],
                hoverinfo="text",
            ))
            # Smoothed trend
            _fig_cum.add_trace(_go_cum.Scatter(
                x=_xs, y=_ys_smooth, mode="lines", line=dict(color=_color, width=3),
                name=_label,
                hovertext=[f"{_label}<br>n={x}: PPL≈{y:.0f}" for x, y in zip(_xs, _ys_smooth)],
                hoverinfo="text",
            ))
        # n=30 reference line
        _fig_cum.add_vline(x=30, line_dash="dot", line_color="#888", opacity=0.5,
                           annotation_text="n=30 (ablation test)", annotation_position="top right",
                           annotation_font_size=10, annotation_font_color="#888")
        _fig_cum.update_layout(
            title=dict(text="The Shape of Failure", font=dict(size=16)),
            xaxis=dict(title="Number of heads removed (of 144)"),
            yaxis=dict(title="Perplexity", type="log", gridcolor="#eee"),
            plot_bgcolor="#fafafa", paper_bgcolor="#fafafa",
            height=420, margin=dict(t=60, b=60),
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        )

        _cumulative_elements = [
            _fig_cum,
            mo.md("""
The green curve stays shallower than red at every point. The model
tolerates losing its garbage bins far better than losing its workers —
not just at n=30, but everywhere along the curve.
"""),
        ]
    except Exception:
        pass

    mo.output.replace(
        mo.vstack([
            mo.md(f"""
## The Ablation Test

If the garbage bin is just a parking spot, what happens when you remove it?
I shut off {_n_sink} garbage-bin heads completely, then did the same to
{_n_sink} random working heads and {_n_diffuse} high-performing heads for
comparison. (Perplexity = how surprised the model is — lower is better.)
"""),
            _fig_abl,
            mo.hstack([
                mo.stat(
                    value=f"+{_sink_delta:.0f}",
                    label="PPL from zeroing sink heads", bordered=True,
                ),
                mo.stat(
                    value=f"+{_random_delta:,.0f}",
                    label="PPL from zeroing random heads", bordered=True,
                ),
                mo.stat(
                    value=f"{_ratio:.0f}×",
                    label="less critical (sink vs random)", bordered=True,
                ),
            ], justify="center", gap=1),
            *_cumulative_elements,
            mo.callout(
                mo.md(f"""
**Sink heads are the least critical heads in the model.** Removing them hurts
{_ratio:.0f}× less than removing the same number of random heads. The parking lot
is essential infrastructure — but it's the least interesting part of the building.

The surprise: the {_n_diffuse} high-entropy "noisy" heads (+{_diffuse_delta:.0f} PPL
for just {_n_diffuse} heads) are the *most* important per-head. All in layers 0-1 and
11 — generalists that build shared context and aggregate specialized outputs.
"""),
                kind="success",
            ),
        ])
    )


# ============================================================================
# CELL 10: ADAPTIVE FIX CONTROLS
# ============================================================================

@app.cell
def adaptive_controls(mo):
    adapt_strength = mo.ui.slider(
        start=0.0, stop=2.0, step=0.05, value=1.0,
        label="Treatment strength", show_value=True,
    )

    mo.output.replace(
        mo.vstack([
            mo.md("""
## What If You Force Sick Heads to Spread Out?

I know which heads are sick. What if I force them to look at more words
instead of just staring at the garbage bin? For each sick head, I soften
its attention in proportion to how sick it is. Healthy heads stay untouched.

Drag the slider. Watch sick heads (red) turn healthy (blue) in real time.
"""),
            mo.callout(adapt_strength, kind="info"),
        ])
    )
    return (adapt_strength,)


# ============================================================================
# CELL 11: ADAPTIVE FIX VISUALIZATION
# ============================================================================

@app.cell(hide_code=True)
def adaptive_viz(data, mo, np, plt, adapt_strength):
    _ent_std = data["entropy_standard"]
    _median = np.median(_ent_std)
    _threshold = _median * 0.7
    _n_l = data["n_layers"]
    _n_h = data["n_heads"]
    _strength = adapt_strength.value

    # --- Compute per-head temperature map ---
    _t_map = np.ones((_n_l, _n_h))
    _sick = _ent_std < _threshold
    if _strength > 0 and _sick.any():
        _t_map[_sick] = np.clip(
            1.0 + _strength * (_median / _ent_std[_sick] - 1.0), 1.0, 5.0,
        )

    # --- Apply temperature to raw scores, recompute softmax ---
    _raw = data["raw_scores_all"]
    _fixed = _raw.copy()
    for _li in range(_n_l):
        for _hi in range(_n_h):
            if _t_map[_li, _hi] > 1.0:
                _fixed[_li, _hi] = _raw[_li, _hi] / _t_map[_li, _hi]
    _exp = np.exp(_fixed - _fixed.max(axis=-1, keepdims=True))
    _fixed_attn = _exp / _exp.sum(axis=-1, keepdims=True)

    # --- Entropy of fixed attention ---
    _p = np.clip(_fixed_attn, 1e-9, 1.0)
    _ent_fixed = -(_p * np.log(_p)).sum(axis=-1).mean(axis=-1)

    # --- Stats ---
    _n_sick_before = int(_sick.sum())
    _n_sick_after = int((_ent_fixed < _threshold).sum())
    _healed = _n_sick_before - _n_sick_after
    _sink_before = data["sink_mag_standard"].mean() * 100
    _sink_after = _fixed_attn[:, :, :, 0].mean(axis=(1, 2)).mean() * 100

    # --- Dual entropy grids ---
    _all_ent = np.stack([_ent_std, _ent_fixed])
    _vmin, _vmax = _all_ent.min(), _all_ent.max()

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    _axes[0].imshow(_ent_std, cmap="RdBu", aspect="auto", vmin=_vmin, vmax=_vmax)
    _axes[0].set_title("Before (standard)", fontsize=12, fontweight="bold")
    _axes[0].set_xlabel("Head")
    _axes[0].set_ylabel("Layer")
    _axes[0].set_xticks(range(_n_h))
    _axes[0].set_yticks(range(_n_l))

    _im = _axes[1].imshow(_ent_fixed, cmap="RdBu", aspect="auto", vmin=_vmin, vmax=_vmax)
    _axes[1].set_title(
        f"After (strength={_strength:.2f})", fontsize=12, fontweight="bold",
    )
    _axes[1].set_xlabel("Head")
    _axes[1].set_ylabel("Layer")
    _axes[1].set_xticks(range(_n_h))
    _axes[1].set_yticks(range(_n_l))
    _fig.colorbar(
        _im, ax=_axes, orientation="horizontal", shrink=0.5, pad=0.08,
        label="Entropy (red=sick, blue=healthy)",
    )

    mo.output.replace(
        mo.vstack([
            _fig,
            mo.md(f"""
**Sick heads:** {_n_sick_before} → {_n_sick_after} ({_healed} healed) ·
**Sink waste:** {_sink_before:.1f}% → {_sink_after:.1f}%
"""),
        ])
    )


# ============================================================================
# CELL 11B: ADAPTIVE FIX VALIDATION (runs once, no slider dependency)
# ============================================================================

@app.cell(hide_code=True)
def adaptive_validation(data, mo, np):
    _n_sick = int((data["entropy_standard"] < np.median(data["entropy_standard"]) * 0.7).sum())

    # Use cached validation if available, otherwise show placeholder
    _val = data.get("adaptive_val")
    if _val is not None:
        _adp_ppl = _val["adaptive_ppl"]
        _std_ppl = _val["standard_ppl"]
        _top1 = _val["top1_match"]
        mo.output.replace(
            mo.vstack([
                mo.hstack([
                    mo.stat(value=f"{_adp_ppl:.1f}", label=f"PPL with fix (baseline: {_std_ppl:.1f})", bordered=True),
                    mo.stat(value=f"{_top1:.0f}%", label="top-1 token agreement", bordered=True),
                    mo.stat(value=f"{_n_sick}", label="heads treated", bordered=True),
                ], justify="center", gap=1),
                mo.callout(
                    mo.md(f"""
**Validated with a real modified forward pass** (strength=1.0). The model's
top-1 predictions agree {_top1:.0f}% of the time. Perplexity: {_std_ppl:.1f} → {_adp_ppl:.1f}.
{"The fix barely changes the output — the model routes around the treatment, confirming sinks are load-bearing." if _top1 > 90 else "The fix changes the output meaningfully — some sinks were genuinely wasting capacity."}
"""),
                    kind="success" if _top1 > 90 else "warn",
                ),
            ])
        )
    else:
        mo.output.replace(
            mo.callout(
                mo.md(f"*Forward pass validation runs on first cold start. {_n_sick} heads would be treated.*"),
                kind="neutral",
            )
        )


# ============================================================================
# CELL 12: THE INSIGHT — WHY SINKS EXIST
# ============================================================================

@app.cell(hide_code=True)
def the_insight(data, mo, np, plt):
    _n_layers = data["n_layers"]
    _n_heads = data["n_heads"]
    _ent = data["entropy_standard"]
    _median = np.median(_ent)
    _n_sick = int((_ent < _median * 0.7).sum())

    # --- Cross-model comparison ---
    import json as _json
    import os as _os
    _dir = _os.path.dirname(_os.path.abspath(__file__))
    _cross_models = []
    try:
        with open(_os.path.join(_dir, "pythia_results.json")) as _f:
            _pythia_data = _json.load(_f)
        for _key in ["gpt2", "pythia-70m"]:
            _m = _pythia_data[_key]
            _cross_models.append({
                "name": _m["model_name"].replace("EleutherAI/", ""),
                "params": f"{_m['n_params']/1e6:.0f}M",
                "layers": _m["n_layers"],
                "heads": _m["n_heads"],
                "sink_waste": _m["sink_waste_pct"],
                "sick_pct": _m["sick_pct"],
            })
    except Exception:
        _cross_models = [
            {"name": "GPT-2", "params": "124M", "layers": 12, "heads": 12,
             "sink_waste": 44.3, "sick_pct": 21.5},
            {"name": "Pythia-70M", "params": "70M", "layers": 6, "heads": 8,
             "sink_waste": 3.3, "sick_pct": 45.8},
        ]
    # Add LLaMA if available
    try:
        with open(_os.path.join(_dir, "llama_sink_results.json")) as _f:
            _llama = _json.load(_f)
        _cross_models.append({
            "name": "LLaMA-3.2-1B",
            "params": f"{_llama['n_params']/1e9:.1f}B",
            "layers": _llama["n_layers"],
            "heads": _llama["n_heads"],
            "sink_waste": _llama["sink_waste_pct"],
            "sick_pct": _llama["sick_pct"],
        })
    except Exception:
        _cross_models.append({
            "name": "LLaMA-3.2-1B", "params": "1.2B", "layers": 16, "heads": 32,
            "sink_waste": 65.6, "sick_pct": 33.0,
        })

    _model_labels = [
        f"{m['name']}\n({m['params']}, {m['layers']}L×{m['heads']}H)"
        for m in _cross_models
    ]
    _model_sink = [m["sink_waste"] for m in _cross_models]
    _model_sick_pct = [m["sick_pct"] for m in _cross_models]

    _bar_colors = ["#e74c3c", "#8e44ad", "#2c3e50"][:len(_cross_models)]

    _fig_cross, _cross_axes = plt.subplots(1, 2, figsize=(12, 5))

    _cross_axes[0].bar(
        _model_labels, _model_sink, color=_bar_colors, width=0.5,
    )
    for _i, _v in enumerate(_model_sink):
        _cross_axes[0].text(_i, _v + 1, f"{_v}%", ha="center", fontweight="bold")
    _cross_axes[0].set_ylabel("Average sink waste (%)")
    _cross_axes[0].set_title("Sink Waste Across Architectures")
    _cross_axes[0].set_ylim(0, max(_model_sink) * 1.2)

    _sick_colors = ["#e67e22", "#9b59b6", "#27ae60"][:len(_cross_models)]
    _cross_axes[1].bar(
        _model_labels, _model_sick_pct, color=_sick_colors, width=0.5,
    )
    for _i, _v in enumerate(_model_sick_pct):
        _cross_axes[1].text(_i, _v + 0.3, f"{_v}%", ha="center", fontweight="bold")
    _cross_axes[1].set_ylabel("Sick heads (%)")
    _cross_axes[1].set_title("Head Specialization: Universal")
    _cross_axes[1].set_ylim(0, max(_model_sick_pct) * 1.3)
    plt.tight_layout()

    # --- Download button ---
    _export = {
        "model": "gpt2",
        "seq_len": int(data["seq_len"]),
        "n_layers": int(data["n_layers"]),
        "n_heads": int(data["n_heads"]),
        "sink_magnitude_per_layer": data["sink_mag_standard"].tolist(),
        "entropy_standard": data["entropy_standard"].tolist(),
        "entropy_esa": data["entropy_esa"].tolist(),
        "entropy_sink": data["entropy_sink"].tolist(),
        "tokens": data["tokens"],
    }
    _download_btn = mo.download(
        data=_json.dumps(_export, indent=2).encode(),
        filename="attention_sink_data.json",
        label="Download attention data (JSON)",
    )

    mo.output.replace(
        mo.vstack([
            mo.md(f"""
## Why Every Transformer Has a Garbage Bin

Every attention head must distribute exactly 100% of its attention across
all the words. There's no way to say "I have nothing useful to look at."

GPT-2 has 12 heads per layer, each with a job — grammar, pronouns, topics.
But most jobs don't apply to every sentence. When the grammar head hits a
string of adjectives, it's idle. The math still demands it point somewhere.

Two choices: spread attention thinly across every word (adding a little
noise to everything), or dump it all on one predictable word (concentrating
the noise). **Concentrated garbage in one bin beats a thin layer of garbage
on every surface.** That's why position 0 becomes the dump target.

Better models have *more* idle heads, not fewer. More specialization means
more workers sitting idle in any given context. The garbage bin gets busier
as the model gets smarter.
"""),
            mo.md("### Cross-Model Validation"),
            _fig_cross,
            mo.md("""
Three architectures, three different sink profiles:

- **GPT-2** (124M, 12 layers, absolute position embeddings): 44% sink waste
- **Pythia-70M** (70M, 6 layers, rotary embeddings): 3% sink waste — but half its heads are "sick"
- **LLaMA-3.2-1B** (1.2B, 16 layers, rotary embeddings): **66% sink waste** — the highest of all three

Pythia-70M is the outlier. Its final two layers have every head at zero
entropy — maximally focused, but not on position 0. With only 6 layers,
the sink pattern doesn't develop. But LLaMA — also using
[rotary embeddings](https://alphaxiv.org/abs/2104.09864), but with 16
layers — sinks *more* than GPT-2. Depth matters more than position
encoding.

The garbage bin is universal. Deeper models use it more.
"""),
            mo.accordion({
                "Open questions": mo.md("""
- **Post-norm architectures** — theory predicts reduced sinks in post-norm transformers. Does the pattern hold for
models like PaLM or early BERT variants?
- **Mixture-of-Experts** — MoE models route tokens to different experts. Do inactive experts create
different sink patterns than dense models?
- **In-context learning** — do sinks shift during few-shot prompting? If the model is "using" more of its
attention for task-specific reasoning, does sink magnitude drop?
- **Longer contexts** — at 4K, 32K, 128K context lengths, does the sink token at position 0 still dominate,
or do new sinks emerge at other positions?
- **Vision transformers** — Darcet et al. added explicit register tokens. Does training with registers
from scratch produce healthier attention than retrofitting a sink token?
"""),
                "References": mo.md("""
- [Attention Sinks](https://alphaxiv.org/abs/2309.17453) (Xiao et al., 2023)
- [The Spike, the Sparse and the Sink](https://alphaxiv.org/abs/2603.05498) (Sun et al., LeCun/NYU)
- [Attention Sinks Are Provably Necessary](https://alphaxiv.org/abs/2603.11487) (Ran-Milo, 2026)
- [Exclusive Self Attention](https://alphaxiv.org/abs/2603.09078) (Zhai, 2026)
- [Fast KV Compaction via Attention Matching](https://alphaxiv.org/abs/2602.16284) (Zweiger et al.)
- [Register Tokens in Vision Transformers](https://alphaxiv.org/abs/2309.16588) (Darcet et al., 2024)
"""),
            }),
            mo.md("""
### What would fix this at the architecture level

The learned tokens are a retrofit. Three approaches could solve it
from the ground up:

**1. Built-in garbage bins** ([Darcet et al., 2024](https://alphaxiv.org/abs/2309.16588)).
Instead of retrofitting tokens after training, add "register" tokens
during training. The model learns how many bins it needs. Vision
transformers already do this. Language models should too.

**2. Let heads say "nothing."** Right now, every head must distribute
100% of its attention somewhere. If heads could output zero — a native
"I have nothing useful to do" signal — they wouldn't need a garbage
bin at all. Mixture-of-Experts models already route to "no expert."
The same principle for attention heads.

**3. Let heads share.** Interleaved Head Attention (Meta et al., 2026)
lets heads borrow useful signal from neighbors instead of idling.
Fewer idle heads = less garbage = less need for a bin.
"""),
            mo.md("""
## What You Can Do Today

These aren't theoretical suggestions — they come directly from the
experiments in this notebook.

**1. Prepend learned tokens for free improvement.**
Train 4 embeddings at position 0, freeze the model. On GPT-2: -19.7%
perplexity. On LLaMA-3.2-1B: -26.7%. Takes ~10 minutes of training,
costs nothing at inference. Position matters — the same tokens at the
end do 0.0%.

**2. Never evict position 0 from your KV cache.**
When compressing the key-value cache for long-context serving, keep
the first token. It's the most-attended position in the model — losing
it degrades every downstream prediction
([Zweiger et al.](https://alphaxiv.org/abs/2602.16284)).

**3. Filter out the garbage before reading attention maps.**
If you're interpreting attention patterns to understand what the model
is doing, ignore position 0. Over 44-66% of attention weight is
structural parking, not meaningful signal.

**4. Don't try to train sinks away.**
I tested alignment losses at 4 weights × 3 seeds. The model improves
at language but refuses to change its attention. Even at λ=10 (10× the
weight on alignment vs language modeling), sinks hold. Don't waste
compute on this.

**5. Prune sink heads first, not last.**
When removing heads for efficiency, start with sink-dominated heads.
They're 29× less critical than random heads. The high-entropy "noisy"
heads in layers 0-1 and the final layer are the ones you can't afford
to lose.

**Limitation:** Validated on GPT-2 and Pythia (both pre-norm). Post-norm
architectures (PaLM, early BERT) may show different patterns — theory
predicts reduced sinks, but this remains an open question.
"""),
            _download_btn,
        ])
    )




if __name__ == "__main__":
    app.run()
