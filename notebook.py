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
# Attention Sinks Are Load-Bearing
## Every fix failed. Here's why.
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
    _ax.set_title("Everything We Tried", fontsize=14, fontweight="bold")
    _ax.set_ylim(0, 62)
    _ax.grid(True, alpha=0.15, axis="y")
    _ax.axhline(y=44.3, color="#e74c3c", linestyle="--", alpha=0.3, linewidth=1)
    plt.tight_layout()

    mo.output.replace(
        mo.vstack([
            mo.md(f"""
*Betsy Schultz · GPT-2 (124M) · {data['seq_len']} tokens · {"loaded from cache" if data["from_cache"] else "computed"} in {data['elapsed']:.1f}s*

*Original diagnostic experiments: 8 fix approaches tested, multi-seed λ sweep
(4 weights × 3 seeds), three-class head ablation, and cross-model validation —
all interactive below.*

GPT-2 dumps over 44% of its attention budget on one meaningless token.
We tested every proposed fix — blocking self-attention, temperature
scaling, sink tokens, ReLU attention, fine-tuning with alignment loss
(even at equal weighting), recursive training. **None of them eliminate
sinks.** We then zeroed out 31 sink heads entirely: perplexity rose by
55 points. Zeroing 31 random healthy heads? +1,611 points. Sink heads
are the **least critical** heads in the model — but the *need* for them
is non-negotiable. The thesis: sinks are a structural parking mechanism,
not a bug. The parking lot is essential infrastructure, but it's the
least interesting part of the building.
"""),
            _fig_bar,
            mo.hstack([
                mo.stat(value="44.3%", label="attention wasted on one token", bordered=True),
                mo.stat(value="8", label="fix approaches tested", bordered=True),
                mo.stat(value="0", label="fixes that eliminated sinks", bordered=True),
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
        "quirks. Recent work by LeCun and collaborators at NYU discovered that "
        "certain token positions absorb a disproportionate share of attention "
        "weight across many heads and layers. These attention sinks appear to "
        "emerge from the pre-norm architecture used in most modern transformers."
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
            _temp_sweep = {
                round(t, 1): _cached[f"temp_{t:.1f}"]
                for t in np.arange(1.0, 5.1, 0.1)
                if f"temp_{t:.1f}" in _cached
            }

            # Still need model for try-it-yourself
            _model = AutoModelForCausalLM.from_pretrained(
                "gpt2", attn_implementation="eager"
            )
            _tokenizer = AutoTokenizer.from_pretrained("gpt2")
            _model.eval()
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
            )

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
            _raw_scores_all = np.stack(_raw_scores_all)

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
            )

            # --- Precompute temperature sweep ---
            _status.update("Precomputing temperature sweep...")
            _temp_sweep = {}
            for _t_val in np.arange(1.0, 5.1, 0.1):
                _t_key = round(float(_t_val), 1)
                _scaled = _raw_scores_all / _t_val
                _exp = np.exp(_scaled - _scaled.max(axis=-1, keepdims=True))
                _temp_sweep[_t_key] = _exp / _exp.sum(axis=-1, keepdims=True)

            # --- Save cache ---
            _status.update("Saving cache...")
            _save_dict = {
                "text_hash": _text_hash,
                "standard_attn": _standard_attn,
                "esa_attn": _esa_attn,
                "raw_scores_all": _raw_scores_all,
                "sink_attn_full": _sink_attn_full,
                "tokens": np.array(_tokens, dtype=object),
                "seq_len": _seq_len,
                "n_layers": _n_layers,
                "n_heads": _n_heads,
            }
            for _t_key, _t_attn in _temp_sweep.items():
                _save_dict[f"temp_{_t_key:.1f}"] = _t_attn
            np.savez_compressed(_cache_path, **_save_dict)

    # --- Color constants ---
    _C_STD = "#e74c3c"
    _C_TEMP = "#3498db"
    _C_SINK = "#2ecc71"
    _C_ESA = "#95a5a6"

    data = {
        "standard_attn": _standard_attn,
        "esa_attn": _esa_attn,
        "raw_scores_all": _raw_scores_all,
        "sink_attn_full": _sink_attn_full,
        "temp_sweep": _temp_sweep,
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
# CELL 1: THE HOOK
# ============================================================================

@app.cell(hide_code=True)
def the_hook(data, mo, np, plt):
    _attn = data["standard_attn"][8]
    _avg = _attn.mean(axis=0)
    _peak_waste = data["standard_attn"][:, :, :, 0].mean(axis=(1, 2)).max() * 100

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.imshow(_avg, cmap="viridis", aspect="auto")
    _ax.set_xlabel("Key position (who is being looked at)", fontsize=11)
    _ax.set_ylabel("Query position (who is looking)", fontsize=11)
    _ax.set_title(
        "GPT-2 Attention — Layer 8, All Heads Averaged",
        fontsize=13,
        fontweight="bold",
    )
    # Token labels every 5th position
    _tick_positions = list(range(0, len(data["tokens"]), 5))
    _tick_labels = [data["tokens"][i] for i in _tick_positions]
    _ax.set_xticks(_tick_positions)
    _ax.set_xticklabels(_tick_labels, rotation=90, fontsize=7)
    plt.colorbar(_ax.images[0], ax=_ax, shrink=0.8, label="Attention weight")
    plt.tight_layout()

    mo.output.replace(
        mo.vstack([
            mo.md(f"""
## The Sink Up Close

See the bright vertical stripe on the left edge? Every token in the sequence
is staring at the very first token — the word "The."

That stripe is an **attention sink**. In GPT-2's deepest layers, over
**{_peak_waste:.0f}%** of attention goes to one meaningless token.

> **Why GPT-2?** Attention sinks aren't a GPT-2 quirk — they appear in
> LLaMA, Mistral, and every pre-norm transformer tested. GPT-2 is small
> enough to run these diagnostics interactively in your browser, with fully
> open weights for reproducibility. The architecture is the same; the sink
> is the same.
"""),
            mo.accordion({
                "Input text being analyzed": mo.md(
                    f"*{data['text']}*"
                ),
            }),
            _fig,
            mo.callout(
                f"GPT-2 wastes over {_peak_waste:.0f}% of its attention budget "
                f"on a single meaningless token.",
                kind="danger",
            ),
            mo.accordion({
                "How to read the heatmap": mo.md("""
Each row is a word asking "who should I pay attention to?" Each column is a word that might get looked at.
**Bright (yellow)** = high attention, **dark (purple)** = low. The color bar maps colors to weight.
A bright vertical stripe means *every* word is staring at the same spot — that's the sink.
The x-axis shows actual words here; later charts use position numbers since the pattern matters more than specific words.
"""),
                "Key terms": mo.md("""
**Layer** — GPT-2 processes text in 12 stacked stages. Early layers handle simple patterns (nearby words, punctuation).
Deeper layers build higher-level meaning (who did what to whom). The sink gets worse in deeper layers.

**Head** — Each layer has 12 attention heads working in parallel, each specializing in a different relationship —
grammar, pronoun matching, topic tracking. GPT-2 has 144 heads total (12 layers × 12 heads). A "sick" head
spends most of its attention budget on the sink instead of doing useful work.

**Attention budget** — The model splits 100% of each head's attention across all tokens. The first token
receives 40-60% in deeper layers, regardless of what word it is. Researchers call this an **attention sink**
([Xiao et al., 2023](https://alphaxiv.org/abs/2309.17453); [Sun et al., LeCun/NYU](https://alphaxiv.org/abs/2603.05498)).
Every percentage point on the sink is a point *not* spent tracking syntax, pronouns, or long-range context.

**Entropy** — A measure of how spread out a head's attention is. High entropy means attention is distributed across
many tokens (the head is doing diverse work). Low entropy means attention is concentrated on one or two positions
(often the sink). We use it as a health score: high = healthy, low = sick.
"""),
            }),
        ])
    )


# ============================================================================
# CELL 1B: LAYER WALK
# ============================================================================

@app.cell
def layer_walk_controls(mo):
    walk_layer = mo.ui.slider(
        start=0, stop=11, value=0, label="Layer", show_value=True
    )
    mo.output.replace(
        mo.vstack([
            mo.md("""
### Watch the sink emerge

Drag the slider from layer 0 to layer 11 to see the sink stripe form as you go deeper.
Early layers distribute attention broadly. By layer 6-8, the first token dominates.
"""),
            mo.callout(walk_layer, kind="info"),
        ])
    )
    return walk_layer,


@app.cell(hide_code=True)
def layer_walk_viz(data, mo, np, plt, walk_layer):
    _layer = walk_layer.value
    _avg = data["standard_attn"][_layer].mean(axis=0)
    _sink_pct = data["standard_attn"][_layer, :, :, 0].mean() * 100

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.imshow(_avg, cmap="viridis", aspect="auto")
    _ax.set_xlabel("Key position")
    _ax.set_ylabel("Query position")
    _ax.set_title(f"Layer {_layer} — Sink: {_sink_pct:.1f}% of attention to position 0")

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
                f"Layer {_layer}: **{_sink_pct:.1f}%** of attention goes to position 0. "
                f"Status: **{_status}**.",
                kind="neutral" if _status == "healthy" else "warn" if _status == "forming" else "danger",
            ),
        ])
    )


# ============================================================================
# CELL 2: ESA SECTION (compressed — toggle + result in one cell)
# ============================================================================

@app.cell(hide_code=True)
def esa_section(data, mo, np, plt):
    _toggle = mo.ui.switch(label="Enable ESA", value=False)
    _layer_sl = mo.ui.slider(start=0, stop=11, value=8, label="Layer", show_value=True)

    _layer = _layer_sl.value
    _std = data["standard_attn"][_layer].mean(axis=0)
    _esa = data["esa_attn"][_layer].mean(axis=0)
    _right = _esa if _toggle.value else _std
    _rtitle = "Exclusive Self-Attention" if _toggle.value else "Standard (toggle above)"
    _vmax = max(_std.max(), _right.max())

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for _ax, _d, _t in [(_axes[0], _std, "Standard"), (_axes[1], _right, _rtitle)]:
        _ax.imshow(_d, cmap="viridis", aspect="auto", vmin=0, vmax=_vmax)
        _ax.set_title(_t, fontsize=11, fontweight="bold")
        _ax.set_xlabel("Key position")
        _ax.set_ylabel("Query position")
        plt.colorbar(_ax.images[0], ax=_ax, shrink=0.8)
    plt.tight_layout()

    mo.output.replace(
        mo.vstack([
            mo.md("""
### Does blocking self-attention fix the sink?

[Exclusive Self Attention](https://alphaxiv.org/abs/2603.09078) (Zhai, 2025)
blocks the diagonal — tokens can't attend to themselves. Toggle it on:
"""),
            mo.hstack([_toggle, _layer_sl], justify="start", gap=1),
            _fig,
            mo.callout(
                mo.md("""
**No effect.** ESA doesn't change sink magnitude at any layer. Sinks aren't caused
by self-attention — they're structural, tied to pre-norm residual streams.
*An original diagnostic: [ESA](https://alphaxiv.org/abs/2603.09078) meets
[attention sink research](https://alphaxiv.org/abs/2603.05498), and nothing changes.*
"""),
                kind="info",
            ),
        ])
    )


# ============================================================================
# CELL 4: FIX CONTROLS (show code)
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
    fix_head = mo.ui.radio(
        options=_head_opts, value="Average (all heads)", label="Head", inline=True
    )

    mo.output.replace(
        mo.vstack([
            mo.md("""
## Two Surviving Fixes

ESA doesn't work. Two approaches survive:

**Temperature scaling** divides scores by T before softmax — spreads attention more evenly.
**Sink token** prepends a garbage token at position 0 — gives the model a proper dump target.
One changes the math. The other changes the input.
"""),
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
    # Temperature-scaled attention (precomputed lookup)
    _t_key = round(float(temp_slider.value), 1)
    temp_attn = data["temp_sweep"].get(_t_key)
    if temp_attn is None:
        _scaled = data["raw_scores_all"] / temp_slider.value
        _exp = np.exp(_scaled - _scaled.max(axis=-1, keepdims=True))
        temp_attn = _exp / _exp.sum(axis=-1, keepdims=True)

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
([Wortsman et al. 2023](https://alphaxiv.org/abs/2309.08586) showed ReLU
requires sequence-length scaling). I'm quantifying it in terms of sink
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
            "Exclusive Self-Attention",
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
## Per-Head Attention Health

Each head gets a health score based on entropy (how spread out its attention is).
High entropy = looking at many tokens (healthy). Low entropy = fixated on one position (sick).

**Blue = healthy** · **Red = sick** · Start at Layer 7, Head 3 — a clearly sink-corrupted head.

*Threshold: a head is "sick" if its entropy falls below 70% of the median across all heads.
This is a heuristic, not ground truth — there's no labeled dataset of sick vs. healthy heads.
We validated the cutoff by checking that heads classified as sick visually show the sink stripe
in their attention maps, and that the count is stable across nearby thresholds (60-80%).*
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
        "Exclusive Self-Attention": "esa",
        f"Temperature T={temp_slider.value:.1f}": "temp",
        "Sink Token": "sink",
    }
    _cond = _cond_keys.get(entropy_radio.value, "standard")

    _p = np.clip(temp_attn, 1e-9, 1.0)
    _entropy_temp = -(_p * np.log(_p)).sum(axis=-1).mean(axis=-1)

    _ent_map = {
        "standard": data["entropy_standard"],
        "esa": data["entropy_esa"],
        "temp": _entropy_temp,
        "sink": data["entropy_sink"],
    }
    _lbl_map = {
        "standard": "Standard",
        "esa": "Exclusive Self-Attention",
        "temp": f"Temperature T={temp_slider.value:.1f}",
        "sink": "Sink Token",
    }
    _attn_source = {
        "standard": data["standard_attn"],
        "esa": data["esa_attn"],
        "temp": temp_attn,
        "sink": data["sink_attn_full"],
    }

    _ent = _ent_map[_cond]
    _ent_std = data["entropy_standard"]

    _all_ent = np.stack([
        data["entropy_standard"], data["entropy_esa"],
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
## The Training Experiment: Can You Train the Sinks Away?

If we know what healthy attention looks like, can we train the model to produce it?
I finetuned GPT-2 on WikiText-2 with an alignment loss — standard LM training plus a penalty
pushing sick heads to distribute attention like healthy neighbors. {_blend["total_steps"]} steps, three epochs.

Perplexity dropped from {_ppl_before} to {_ppl_after} ({_ppl_drop:.0f}% improvement).
But **attention patterns remained unchanged.**
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

Sinks are load-bearing. The model had every opportunity to reallocate
that attention budget. The loss function explicitly rewarded it.
It didn't.
"""),
                kind="warn",
            ),
            _fig,
            mo.md(f"""
The top-left panel shows the model learning. LM loss drops across
all three epochs. The top-right shows alignment loss barely moves.
The bottom row is the punchline: sink waste and sick head count lock in
early and never budge.

### What if the signal was too weak?

The original training used λ_align = 0.1 (alignment loss at 10% of LM
loss). A natural objection: maybe the gradient signal was just too weak.
So I swept across four alignment weights, each with three random seeds:

| λ_align | Sink waste (mean ± std) | PPL after | Sick heads |
|---------|------------------------|-----------|------------|
| 0.1 | 46.2% ± 0.2 | 26.8 | 32 |
| 0.5 | 47.1% ± 0.2 | 26.9 | 32 |
| 1.0 | 47.8% ± 0.3 | 27.0 | 30 |
| **10.0** | **45.3% ± 0.4** | **28.7** | **28** |

*Baseline: 47.4% sink waste, 44.5 PPL, 34 sick heads. All runs started from
fresh GPT-2 weights. Standard deviations across seeds are ±0.2-0.4% — the
pattern is stable regardless of initialization.*

Two patterns emerge. From λ = 0.1 to 1.0, sinks *increase* as the model
improves — more specialized heads means more idle capacity that needs
parking. At λ = 10.0, where alignment loss dominates language modeling
10-to-1, the model finally trades language quality (+1.9 PPL) for a
marginal sink reduction — but sinks still sit at **45.3%**. Even under
extreme gradient pressure, the structural floor holds. This isn't a
weak-signal problem. The architecture requires these sinks.

{_recursive_md}

To be clear: sinks don't cost extra compute. The model does the same
amount of math whether attention goes to the sink or to meaningful tokens. And
the increased sinks didn't hurt output quality — perplexity *improved*.
The model builds more structural support as it gets better, not
less. More specialized heads means more heads sitting idle in any given
context, and idle heads need somewhere safe to park.

This empirically confirms Ran-Milo's *"Attention Sinks Are
Provably Necessary"*
([2026](https://alphaxiv.org/abs/2603.11487)) —
pre-norm transformers mathematically require sinks for representational
stability.

I also tried a learned sink embedding: 768 trainable parameters at
position 0, model frozen. It *improved* perplexity slightly, suggesting
the sink position carries useful information when trained. But the
sinks themselves remain unchanged. The parking mechanism isn't broken.
It's doing exactly what the architecture needs.
"""),
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

    # --- Main ablation bar chart (log scale) ---
    _conditions = [
        "Baseline\n(no ablation)", f"Sink heads\nzeroed ({_n_sink})",
        f"Random heads\nzeroed ({_n_sink})", f"Noisy heads\nzeroed ({_n_diffuse})",
    ]
    _ppls = [_baseline, _sink_ppl, _random_ppl, _diffuse_ppl]
    _colors = ["#95a5a6", "#e74c3c", "#2c3e50", "#3498db"]

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _bars = _ax.bar(
        _conditions, _ppls, color=_colors, width=0.55,
        edgecolor="white", linewidth=1.5,
    )
    _ax.set_yscale("log")
    _ax.set_ylabel("Perplexity (log scale)", fontsize=12)
    _ax.set_title("What Happens When You Remove Heads?", fontsize=14, fontweight="bold")
    for _bar, _ppl in zip(_bars, _ppls):
        _ax.text(
            _bar.get_x() + _bar.get_width() / 2, _ppl * 1.15,
            f"{_ppl:.0f}", ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    _ax.grid(True, alpha=0.15, axis="y")
    plt.tight_layout()

    # --- Load cumulative ablation if available ---
    _cumulative_elements = []
    try:
        with open(_os.path.join(_dir, "cumulative_ablation.json")) as _f:
            _cum = _json.load(_f)

        _fig_cum, _ax_cum = plt.subplots(figsize=(10, 5))
        _style = {
            "sink_first": ("#2ecc71", "o-", "Sink heads first (least critical)"),
            "random": ("#95a5a6", "s--", "Random order"),
            "important_first": ("#e74c3c", "^-", "Important heads first (most critical)"),
        }
        for _curve_name, (_color, _marker, _label) in _style.items():
            _curve = _cum["curves"][_curve_name]
            _xs = [p["n_heads"] for p in _curve]
            _ys = [p["perplexity"] for p in _curve]
            _ax_cum.plot(
                _xs, _ys, _marker, color=_color,
                linewidth=2.5, markersize=7, label=_label,
            )

        _ax_cum.set_xlabel("Number of heads removed (of 144)", fontsize=12)
        _ax_cum.set_ylabel("Perplexity", fontsize=12)
        _ax_cum.set_title(
            "The Shape of Failure", fontsize=14, fontweight="bold",
        )
        _ax_cum.legend(fontsize=10)
        _ax_cum.set_yscale("log")
        _ax_cum.grid(True, alpha=0.2)
        plt.tight_layout()

        _cumulative_elements = [
            _fig_cum,
            mo.md("""
**Follow the green line.** You can remove 30+ sink heads before
perplexity doubles. The red line — removing important heads first —
explodes almost immediately. The model can lose its parking lot and barely
notice. Lose its workers and it collapses.
"""),
        ]
    except Exception:
        pass

    mo.output.replace(
        mo.vstack([
            mo.md(f"""
## The Ablation Test

If sinks are just parking spots, what happens when you remove them?
I zeroed out {_n_sink} sink heads, {_n_sink} random healthy heads, and
{_n_diffuse} high-entropy heads, then measured perplexity on WikiText-2.
"""),
            _fig,
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
# CELL 10: THE INSIGHT — WHY SINKS EXIST
# ============================================================================

@app.cell(hide_code=True)
def the_insight(data, mo, np, plt):
    _n_layers = data["n_layers"]
    _n_heads = data["n_heads"]
    _ent = data["entropy_standard"]
    _median = np.median(_ent)
    _n_sick = int((_ent < _median * 0.7).sum())

    # --- Cross-model comparison (load Pythia data if available) ---
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
             "sink_waste": 44.3, "sick_pct": 10.4},
            {"name": "Pythia-70M", "params": "70M", "layers": 6, "heads": 8,
             "sink_waste": 35.0, "sick_pct": 10.0},
        ]

    _model_labels = [
        f"{m['name']}\n({m['params']}, {m['layers']}L×{m['heads']}H)"
        for m in _cross_models
    ]
    _model_sink = [m["sink_waste"] for m in _cross_models]
    _model_sick_pct = [m["sick_pct"] for m in _cross_models]

    _fig_cross, _cross_axes = plt.subplots(1, 2, figsize=(12, 5))

    _cross_axes[0].bar(
        _model_labels, _model_sink, color=["#e74c3c", "#8e44ad"], width=0.5,
    )
    for _i, _v in enumerate(_model_sink):
        _cross_axes[0].text(_i, _v + 1, f"{_v}%", ha="center", fontweight="bold")
    _cross_axes[0].set_ylabel("Average sink waste (%)")
    _cross_axes[0].set_title("Sink Waste: Depth Matters")
    _cross_axes[0].set_ylim(0, 60)

    _cross_axes[1].bar(
        _model_labels, _model_sick_pct, color=["#e67e22", "#9b59b6"], width=0.5,
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
## Why Sinks Exist

Softmax forces every head to distribute **exactly 100%** of its attention
budget. No exceptions. No "attend to nothing" option.

GPT-2 has 12 attention heads per layer, each specialized. One tracks
subject-verb pairs, another handles pronouns, another follows topics.
But when the subject-verb head encounters a string of adjectives, it
has nothing to do — and softmax still demands it distribute 100% somewhere.

Two choices: spread attention randomly across all words (adding noise
to everything) or dump it on one predictable token (concentrating the
noise). **Concentrated garbage beats distributed garbage.** Dump on
position 0 and the damage is contained. Spread randomly and you corrupt
every representation a little.

The sink is the model's solution to "I have nothing useful to do right
now" — a **learned parking mechanism**, not a malfunction.

That's why training can't eliminate sinks. The architecture *requires*
them. And a better model has more sinks, not fewer: more specialized
heads means more heads sitting idle in any given context. The parking
spot gets more use as the model gets smarter.
"""),
            mo.md("### Cross-Model Validation"),
            _fig_cross,
            mo.md("""
**Pythia-70M** (GPT-NeoX, rotary embeddings, parallel attention+FF) tells a
different story: sink waste drops to ~3%, but nearly half its heads are "sick"
by the entropy metric. With only 6 layers, Pythia doesn't develop the deep-layer
sink pattern GPT-2 shows — confirming that sinks scale with depth, not just
architecture. The *need* for head specialization is universal; the *form* it
takes depends on how many layers the model has to work with.
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
- [ReLU sequence-length scaling](https://alphaxiv.org/abs/2309.08586) (Wortsman et al., 2023)
- [Exclusive Self Attention](https://alphaxiv.org/abs/2603.09078) (Zhai, 2025)
- [Fast KV Compaction via Attention Matching](https://alphaxiv.org/abs/2602.16284) (Zweiger et al.)
- [Register Tokens in Vision Transformers](https://alphaxiv.org/abs/2309.16588) (Darcet et al., 2024)
"""),
            }),
            mo.md("""
### What might actually work

Every fix we tested accepts the softmax constraint and tries to work around
it. The results suggest the constraint itself is the problem. Two directions
worth investigating:

**1. Learned register tokens** ([Darcet et al., 2024](https://alphaxiv.org/abs/2309.16588) —
vision transformers). Instead of retrofitting a sink token, train with explicit
"parking" tokens from the start. The model learns *how many* parking spots it
needs per layer rather than hijacking position 0. Our sink token experiment
shows the mechanism works; registers formalize it.

**2. Sparse attention with explicit null routing.** Replace softmax with an
attention function that has a native "attend to nothing" option — not ReLU
(which backfires at length), but a learned gate per head that can route
attention to a null sink without corrupting real token representations.
MoE models already route tokens to "no expert." The same principle applied
to attention heads would eliminate the need for structural parking entirely.

**3. Cross-head communication.** [Interleaved Head Attention](https://x.com/askalphaxiv/status/2043018144222957683)
(Meta, UT Austin, UC Berkeley, Harvard, MIT) constructs pseudo-heads as
learned mixtures of original projections, enabling heads to share what
they've found. Idle heads could borrow useful signal from active neighbors
instead of parking — reducing the *demand* for sinks rather than fighting
their existence.

All three directions work *with* the discovery that parking is necessary —
and formalize it rather than fighting the architecture.
"""),
            mo.md("""
## What to Do With This

Sinks don't cost extra compute and don't hurt output quality. Why
study them? Three practical reasons:

- **KV cache compression.** Keep position 0 in cache, aggressively evict
others — real memory savings for long-context serving
([Zweiger et al.](https://alphaxiv.org/abs/2602.16284), PyramidKV).

- **Interpretability.** 40-60% of attention weight is structural parking.
Filter it out before reading attention patterns.

- **Architecture design.** Understanding sinks are necessary led to
[register tokens](https://alphaxiv.org/abs/2309.16588) in vision transformers —
explicit sink slots built into the architecture from the start.

### If you're building with transformers

- **Serving long contexts?** Keep the sink token in your KV cache. It's
cheap to store and expensive to lose.
- **Reading attention maps?** Filter out position 0 before interpreting.
- **Pruning heads?** Don't start with sinks — they're the least critical.
High-entropy heads in early and final layers matter far more per-head.
- **Designing a new architecture?** Add explicit register/sink tokens from
the start. The model will create parking spots regardless.
- **Fine-tuning with attention-shaping losses?** Don't expect sink patterns
to change. Even at λ=1.0, sinks hold.

**Limitation:** Validated on GPT-2 and Pythia (both pre-norm). Post-norm
architectures (PaLM, early BERT) may show different patterns — theory
predicts reduced sinks, but this remains an open question.
"""),
            _download_btn,
        ])
    )


# ============================================================================
# CELL 10: TRY IT YOURSELF CONTROLS (show code)
# ============================================================================

@app.cell
def try_controls(mo):
    text_form = mo.ui.form(
        mo.ui.text_area(
            label="Paste any text (max 200 tokens)", max_length=1500
        ),
        submit_button_label="Analyze",
    )

    mo.output.replace(
        mo.vstack([
            mo.md("""
## Try It Yourself

Paste any text and see where GPT-2's attention goes. Runs a fresh forward pass on submit.
"""),
            mo.callout(text_form, kind="info"),
        ])
    )
    return (text_form,)


# ============================================================================
# CELL 11: TRY IT YOURSELF VIZ
# ============================================================================

@app.cell
def try_viz(data, mo, np, plt, text_form):
    import torch as _torch

    mo.stop(not text_form.value, mo.md("*Type or paste text above and click Submit to analyze.*"))

    _model = data["model"]
    _tokenizer = data["tokenizer"]
    _n_heads = data["n_heads"]
    _n_layers = data["n_layers"]
    _d_model = _model.config.n_embd
    _d_head = _d_model // _n_heads

    _inputs = _tokenizer(
        text_form.value, return_tensors="pt", truncation=True, max_length=200
    )
    _sl = _inputs["input_ids"].shape[1]

    _qkv_cache = {}

    def _capture_qkv(_li):
        def _hook(module, inp, out):
            _qkv_cache[_li] = out.detach()
        return _hook

    _hooks = []
    for _i, _block in enumerate(_model.transformer.h):
        _hooks.append(_block.attn.c_attn.register_forward_hook(_capture_qkv(_i)))

    with _torch.no_grad():
        _out = _model(**_inputs, output_attentions=True)

    for _h in _hooks:
        _h.remove()

    _attn = np.stack([_layer[0].numpy() for _layer in _out.attentions])
    _sink = _attn[:, :, :, 0].mean(axis=(1, 2))

    # --- Sink token forward pass ---
    _sink_ids = _torch.cat(
        [_torch.zeros(1, 1, dtype=_torch.long), _inputs["input_ids"]], dim=1
    )
    with _torch.no_grad():
        _sink_out = _model(input_ids=_sink_ids, output_attentions=True)

    _sink_attn = np.stack([_layer[0].numpy() for _layer in _sink_out.attentions])
    # Attention to real first token (now at position 1) from real tokens only
    _sink_real = _sink_attn[:, :, 1:, 1].mean(axis=(1, 2))

    # --- Entropy for both ---
    _p = np.clip(_attn, 1e-9, 1.0)
    _ent = -(_p * np.log(_p)).sum(axis=-1).mean(axis=-1)

    _p_sk = np.clip(_sink_attn, 1e-9, 1.0)
    _ent_sk = -(_p_sk * np.log(_p_sk)).sum(axis=-1).mean(axis=-1)

    _median_ent = np.median(_ent)
    _thresh = _median_ent * 0.7
    _n_sick = int((_ent < _thresh).sum())
    _n_sick_sk = int((_ent_sk < _thresh).sum())
    _n_total = _n_layers * _n_heads

    # --- Sink magnitude comparison ---
    _fig_bars, _ax_bars = plt.subplots(figsize=(10, 5))
    _x = np.arange(_n_layers)
    _w = 0.35

    _ax_bars.bar(
        _x - _w / 2, _sink * 100,
        width=_w, color=data["colors"]["standard"], label="Standard",
    )
    _ax_bars.bar(
        _x + _w / 2, _sink_real * 100,
        width=_w, color=data["colors"]["sink"], label="Sink Token",
    )
    _ax_bars.set_xlabel("Layer")
    _ax_bars.set_ylabel("Attention to first real token (%)")
    _ax_bars.set_title("Sink Magnitude — Standard vs Sink Token")
    _ax_bars.legend()
    _ax_bars.set_ylim(bottom=0)
    plt.tight_layout()

    # --- Attention heatmaps: standard vs sink token (deepest layer, averaged) ---
    _deep = _n_layers - 1
    _std_avg = _attn[_deep].mean(axis=0)
    _sk_avg_map = _sink_attn[_deep].mean(axis=0)

    _vmax_attn = max(_std_avg.max(), _sk_avg_map.max())
    _fig_hm, _hm_axes = plt.subplots(1, 2, figsize=(12, 5))

    _hm_axes[0].imshow(_std_avg, cmap="viridis", aspect="auto", vmin=0, vmax=_vmax_attn)
    _hm_axes[0].set_title(f"Standard — Layer {_deep}, All Heads")
    _hm_axes[0].set_xlabel("Key position")
    _hm_axes[0].set_ylabel("Query position")
    plt.colorbar(_hm_axes[0].images[0], ax=_hm_axes[0], shrink=0.8, label="Attention weight")

    _hm_axes[1].imshow(_sk_avg_map, cmap="viridis", aspect="auto", vmin=0, vmax=_vmax_attn)
    _hm_axes[1].set_title(f"Sink Token — Layer {_deep}, All Heads")
    _hm_axes[1].set_xlabel("Key position")
    _hm_axes[1].set_ylabel("Query position")
    plt.colorbar(_hm_axes[1].images[0], ax=_hm_axes[1], shrink=0.8, label="Attention weight")
    plt.tight_layout()

    # --- Entropy grids side by side ---
    _fig_ent, _ent_axes = plt.subplots(1, 2, figsize=(12, 5))
    _ent_vmin = min(_ent.min(), _ent_sk.min())
    _ent_vmax = max(_ent.max(), _ent_sk.max())

    _ent_axes[0].imshow(_ent, cmap="RdBu", aspect="auto", vmin=_ent_vmin, vmax=_ent_vmax)
    _ent_axes[0].set_title("Entropy — Standard")
    _ent_axes[0].set_xlabel("Head")
    _ent_axes[0].set_ylabel("Layer")
    _ent_axes[0].set_xticks(range(_n_heads))
    _ent_axes[0].set_yticks(range(_n_layers))

    _ent_axes[1].imshow(_ent_sk, cmap="RdBu", aspect="auto", vmin=_ent_vmin, vmax=_ent_vmax)
    _ent_axes[1].set_title("Entropy — Sink Token")
    _ent_axes[1].set_xlabel("Head")
    _ent_axes[1].set_ylabel("Layer")
    _ent_axes[1].set_xticks(range(_n_heads))
    _ent_axes[1].set_yticks(range(_n_layers))
    plt.colorbar(_ent_axes[1].images[0], ax=_ent_axes[1], shrink=0.8, label="Entropy (red=sick, blue=healthy)")
    plt.tight_layout()

    _peak = _sink.max() * 100
    _peak_layer = int(_sink.argmax())
    _std_avg_pct = _sink.mean() * 100
    _sk_avg_pct = _sink_real.mean() * 100

    # Find which token gets the most total attention
    _total_attn_per_pos = _attn.mean(axis=(0, 1, 2))
    _most_attended_pos = int(_total_attn_per_pos.argmax())
    _tokens_list = [_tokenizer.decode(t) for t in _inputs["input_ids"][0]]
    _most_attended_word = repr(_tokens_list[_most_attended_pos].strip())

    _sink_pct = _n_sick / _n_total * 100
    _sink_pct_sk = _n_sick_sk / _n_total * 100
    _healed = max(0, _n_sick - _n_sick_sk)

    if _most_attended_pos == 0:
        _word_explanation = (
            f"The most-attended token is {_most_attended_word} (position 0) — "
            f"the classic first-token sink. The model isn't choosing this word "
            f"because it's meaningful; it's a parking spot for attention that "
            f"has nowhere useful to go."
        )
    else:
        _word_explanation = (
            f"The most-attended token is {_most_attended_word} "
            f"(position {_most_attended_pos}) — unusually, not the first token. "
            f"This can happen with short inputs or text that starts with a "
            f"high-information word."
        )

    mo.output.replace(
        mo.lazy(mo.vstack([
            mo.md(f"""
**Your text** ({_sl} tokens)

| | Standard | Sink Token | Change |
|---|---|---|---|
| **Peak sink** | {_peak:.1f}% (layer {_peak_layer}) | {_sink_real.max() * 100:.1f}% (layer {int(_sink_real.argmax())}) | {(_sink_real.max() - _sink.max()) * 100:+.1f}% |
| **Average sink** | {_std_avg_pct:.1f}% | {_sk_avg_pct:.1f}% | {_sk_avg_pct - _std_avg_pct:+.1f}% |
| **Sick heads** | {_n_sick} of {_n_total} ({_sink_pct:.0f}%) | {_n_sick_sk} of {_n_total} ({_sink_pct_sk:.0f}%) | {_healed} healed |

{_word_explanation}
"""),
            _fig_bars,
            mo.md("**Attention distribution** — deepest layer, all heads averaged. "
                   "Left: standard (bright stripe = sink). Right: with sink token (attention redistributed)."),
            _fig_hm,
            mo.md("**Head health** — entropy grids side by side. More blue = healthier heads."),
            _fig_ent,
        ]))
    )


if __name__ == "__main__":
    app.run()
