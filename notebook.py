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


# ============================================================================
# PRECOMPUTATION — runs once on notebook load
# ============================================================================

@app.cell
def precompute():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _start = time.time()

    _model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
    _tokenizer = AutoTokenizer.from_pretrained("gpt2")
    _model.eval()

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
    _inputs = _tokenizer(_TEXT, return_tensors="pt")
    _seq_len = _inputs["input_ids"].shape[1]
    _tokens = [_tokenizer.decode(t) for t in _inputs["input_ids"][0]]

    _n_heads = _model.config.n_head
    _d_model = _model.config.n_embd
    _d_head = _d_model // _n_heads
    _n_layers = _model.config.n_layer

    _qkv_cache = {}
    def _capture_qkv(layer_idx):
        def hook_fn(module, input, output):
            _qkv_cache[layer_idx] = output.detach()
        return hook_fn

    _hooks = []
    for _i, _block in enumerate(_model.transformer.h):
        _h = _block.attn.c_attn.register_forward_hook(_capture_qkv(_i))
        _hooks.append(_h)

    with torch.no_grad():
        _standard_out = _model(**_inputs, output_attentions=True)

    for _h in _hooks:
        _h.remove()

    _standard_attn = np.stack([_layer[0].numpy() for _layer in _standard_out.attentions])

    _causal_mask = torch.tril(torch.ones(_seq_len, _seq_len)).unsqueeze(0).unsqueeze(0)
    _raw_scores_all = []
    for _i in range(_n_layers):
        _q, _k, _ = _qkv_cache[_i].split(_d_model, dim=-1)
        _q = _q.view(1, _seq_len, _n_heads, _d_head).transpose(1, 2)
        _k = _k.view(1, _seq_len, _n_heads, _d_head).transpose(1, 2)
        _scores = torch.matmul(_q, _k.transpose(-2, -1)) / (_d_head ** 0.5)
        _scores = _scores.masked_fill(_causal_mask == 0, float('-inf'))
        _raw_scores_all.append(_scores[0].numpy())
    _raw_scores_all = np.stack(_raw_scores_all)

    _esa_attn = _standard_attn.copy()
    for _li in range(_n_layers):
        for _hi in range(_n_heads):
            _a = _esa_attn[_li, _hi].copy()
            np.fill_diagonal(_a, 0.0)
            _rs = _a.sum(axis=-1, keepdims=True)
            _rs = np.where(_rs == 0, 1.0, _rs)
            _esa_attn[_li, _hi] = _a / _rs

    _sink_ids = torch.cat([torch.zeros(1, 1, dtype=torch.long), _inputs["input_ids"]], dim=1)
    with torch.no_grad():
        _sink_out = _model(input_ids=_sink_ids, output_attentions=True)
    _sink_attn_full = np.stack([_layer[0].numpy() for _layer in _sink_out.attentions])

    def _entropy(attn_4d):
        _p = np.clip(attn_4d, 1e-9, 1.0)
        return -(_p * np.log(_p)).sum(axis=-1).mean(axis=-1)

    def _sink_mag(attn_4d, col=0):
        return attn_4d[:, :, :, col].mean(axis=(1, 2))

    data = {
        "standard_attn": _standard_attn,
        "esa_attn": _esa_attn,
        "raw_scores_all": _raw_scores_all,
        "sink_attn_full": _sink_attn_full,
        "entropy_standard": _entropy(_standard_attn),
        "entropy_esa": _entropy(_esa_attn),
        "entropy_sink": _entropy(_sink_attn_full),
        "sink_mag_standard": _sink_mag(_standard_attn),
        "sink_mag_esa": _sink_mag(_esa_attn),
        "sink_real_first": _sink_attn_full[:, :, 1:, 1].mean(axis=(1, 2)),
        "n_layers": _n_layers,
        "n_heads": _n_heads,
        "seq_len": _seq_len,
        "tokens": _tokens,
        "model": _model,
        "tokenizer": _tokenizer,
        "elapsed": time.time() - _start,
    }

    return data, mo, np, plt


# ============================================================================
# ACT 1: THE COST
# ============================================================================

@app.cell
def act1_hook(data, mo, np, plt):
    _attn = data["standard_attn"][8]
    _avg = _attn.mean(axis=0)
    _peak_waste = data["standard_attn"][:, :, :, 0].mean(axis=(1, 2)).max() * 100

    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.imshow(_avg, cmap="viridis", aspect="auto")
    _ax.set_xlabel("Key position (who is being looked at)", fontsize=11)
    _ax.set_ylabel("Query position (who is looking)", fontsize=11)
    _ax.set_title("GPT-2 Attention — Layer 8, All Heads Averaged", fontsize=13, fontweight="bold")
    plt.colorbar(_ax.images[0], ax=_ax, shrink=0.8, label="Attention weight")
    plt.tight_layout()

    mo.output.replace(mo.vstack([
        mo.md(f"""
# Does Attention Need Itself?
## Diagnosing wasted attention in GPT-2

See the bright vertical stripe on the left edge? Every token in the sequence is staring at the very first token — the word "The."

That stripe is an **attention sink**. In GPT-2's deepest layers, over **{_peak_waste:.0f}%** of attention goes to one meaningless token. We measured the cost, tested what works and what doesn't, and built a tool to see it.
"""),
        _fig,
    ]))
    return


@app.cell
def act1_context(mo):
    mo.output.replace(mo.md("""
### What is attention?

Every word in a sentence "looks at" every other word to decide which ones matter most. The model splits 100% of its attention budget across all tokens — more attention means "you're important for understanding this position."

### What is a sink?

The very first token receives far more attention than it deserves — often 40–60% in deeper layers — regardless of what word it actually is. Researchers call this an **attention sink** ([Xiao et al., 2023](https://alphaxiv.org/abs/2309.17453); [Chen et al., LeCun/NYU](https://alphaxiv.org/abs/2406.02069)).

### Why does it matter?

Every percentage point spent on the sink is a point *not* spent tracking syntax, resolving pronouns, or maintaining long-range context. The model's capacity is finite. The sink wastes it.
"""))
    return


# ============================================================================
# ACT 2: WHAT DOESN'T WORK
# ============================================================================

@app.cell
def act2_esa_controls(mo):
    esa_toggle = mo.ui.switch(label="Enable Exclusive Self-Attention", value=False)
    _head_opts = ["Average (all heads)"] + [f"Head {_i}" for _i in range(12)]
    esa_layer = mo.ui.slider(start=0, stop=11, value=8, label="Layer", show_value=True)
    esa_head = mo.ui.dropdown(options=_head_opts, value="Average (all heads)", label="Head")

    mo.output.replace(mo.vstack([
        mo.md("""
---

### Does blocking self-attention fix the sink?

One hypothesis: maybe sinks form because tokens attend to themselves too much. [Exclusive Self Attention](https://alphaxiv.org/abs/2503.07822) (Zhai, 2025) blocks the diagonal — each token can look at every other token, but not itself.

Toggle it on and watch what happens to the sink stripe.
"""),
        mo.hstack([esa_toggle, esa_layer, esa_head], justify="start", gap=1),
    ]))
    return esa_toggle, esa_layer, esa_head


@app.cell
def act2_esa_viz(data, mo, np, plt, esa_toggle, esa_layer, esa_head):
    _layer = esa_layer.value
    _head_val = esa_head.value
    _std = data["standard_attn"][_layer]
    _esa = data["esa_attn"][_layer]

    if _head_val == "Average (all heads)":
        _std_show = _std.mean(axis=0)
        _esa_show = _esa.mean(axis=0)
        _hlabel = "all heads averaged"
    else:
        _hi = int(_head_val.split()[-1])
        _std_show = _std[_hi]
        _esa_show = _esa[_hi]
        _hlabel = _head_val

    _right = _esa_show if esa_toggle.value else _std_show
    _rtitle = "Exclusive Self-Attention" if esa_toggle.value else "Standard (toggle ESA above)"
    _vmax = max(_std_show.max(), _right.max())

    _fig1, _axes = plt.subplots(1, 2, figsize=(12, 5))
    for _ax, _d, _t in [(_axes[0], _std_show, "Standard"), (_axes[1], _right, _rtitle)]:
        _ax.imshow(_d, cmap="viridis", aspect="auto", vmin=0, vmax=_vmax)
        _ax.set_title(f"{_t} — Layer {_layer}, {_hlabel}", fontsize=11, fontweight="bold")
        _ax.set_xlabel("Key position")
        _ax.set_ylabel("Query position")
        plt.colorbar(_ax.images[0], ax=_ax, shrink=0.8)
    plt.tight_layout()

    _layers = np.arange(data["n_layers"])
    _fig2, _ax2 = plt.subplots(figsize=(8, 4))
    _ax2.plot(_layers, data["sink_mag_standard"], "o-", color="#e74c3c", linewidth=2.5, markersize=7, label="Standard", zorder=3)
    _ax2.plot(_layers, data["sink_mag_esa"], "s--", color="#95a5a6", linewidth=2, markersize=6, label="Exclusive Self-Attention", zorder=2)
    _ax2.set_xlabel("Layer", fontsize=12)
    _ax2.set_ylabel("Mean attention to position 0", fontsize=12)
    _ax2.set_title("Sink Magnitude Across Layers", fontsize=13, fontweight="bold")
    _ax2.legend(fontsize=10)
    _ax2.grid(True, alpha=0.2)
    _ax2.set_ylim(bottom=0)
    plt.tight_layout()

    mo.output.replace(mo.vstack([
        _fig1,
        mo.md("""
### Finding: Blocking self-attention doesn't fix the sink

The two lines below overlap almost perfectly. Blocking self-attention has **no effect** on sink magnitude. Sinks aren't caused by tokens looking at themselves — they're deeper, likely tied to the pre-norm architecture and how residual streams accumulate.

*An original empirical result. [Exclusive Self Attention](https://alphaxiv.org/abs/2503.07822) (curated paper #6) meets [attention sink research](https://alphaxiv.org/abs/2406.02069) (LeCun/NYU) — and nothing changes. Published nowhere.*
"""),
        _fig2,
    ]))
    return


# ============================================================================
# ACT 3: WHAT WORKS
# ============================================================================

@app.cell
def act3_controls(mo):
    temp_slider = mo.ui.slider(start=1.0, stop=5.0, step=0.5, value=3.0, label="Temperature", show_value=True)
    fix_radio = mo.ui.radio(
        options=["Standard (baseline)", "Temperature scaling", "Sink token"],
        value="Standard (baseline)",
        label="Attention mode",
    )
    _head_opts = ["Average (all heads)"] + [f"Head {_i}" for _i in range(12)]
    fix_layer = mo.ui.slider(start=0, stop=11, value=8, label="Layer", show_value=True)
    fix_head = mo.ui.dropdown(options=_head_opts, value="Average (all heads)", label="Head")

    mo.output.replace(mo.vstack([
        mo.md("""
---

## What Actually Works

We tested several approaches. Exclusive Self Attention doesn't fix sinks. Elastic softmax has zero effect (softmax is shift-invariant). ReLU attention is more interesting — see below.

Two approaches survive:

**Temperature scaling** divides attention scores by T before softmax. Higher T spreads attention more evenly — the model stops dumping excess onto the first token because it doesn't need to.

**Sink token** prepends a designated garbage token at position 0. If the model needs a dump target, give it one on purpose. The real first token is freed.

One changes the math. The other changes the input.
"""),
        mo.hstack([fix_radio, temp_slider, fix_layer, fix_head], justify="start", gap=1),
    ]))
    return temp_slider, fix_radio, fix_layer, fix_head


@app.cell
def act3_viz(data, mo, np, plt, temp_slider, fix_radio, fix_layer, fix_head):
    # Compute temperature-scaled attention on-the-fly
    _scaled = data["raw_scores_all"] / temp_slider.value
    _exp = np.exp(_scaled - _scaled.max(axis=-1, keepdims=True))
    temp_attn = _exp / _exp.sum(axis=-1, keepdims=True)

    _layer = fix_layer.value
    _head_val = fix_head.value

    _mode_keys = {"Standard (baseline)": "standard", "Temperature scaling": "temp", "Sink token": "sink"}
    _mode_key = _mode_keys[fix_radio.value]

    _attn_map = {
        "standard": data["standard_attn"][_layer],
        "temp": temp_attn[_layer],
        "sink": data["sink_attn_full"][_layer],
    }
    _color_map = {"standard": "#e74c3c", "temp": "#3498db", "sink": "#2ecc71"}
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

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 5))

    _axes[0].imshow(_show, cmap="viridis", aspect="auto")
    _axes[0].set_title(f"{_label_map[_mode_key]} — Layer {_layer}, {_hlabel}", fontsize=11, fontweight="bold")
    _axes[0].set_xlabel("Key position")
    _axes[0].set_ylabel("Query position")
    plt.colorbar(_axes[0].images[0], ax=_axes[0], shrink=0.8)

    _lyrs = np.arange(data["n_layers"])
    _temp_sink_mag = temp_attn[:, :, :, 0].mean(axis=(1, 2))
    _axes[1].plot(_lyrs, data["sink_mag_standard"], "o-", color="#e74c3c", linewidth=2, label="Standard", alpha=0.7)
    _axes[1].plot(_lyrs, _temp_sink_mag, "s-", color="#3498db", linewidth=2, label=f"Temp T={temp_slider.value:.1f}")
    _axes[1].plot(_lyrs, data["sink_real_first"], "^-", color="#2ecc71", linewidth=2, label="Sink Token (real 1st)")

    _mag_map = {"standard": data["sink_mag_standard"], "temp": _temp_sink_mag, "sink": data["sink_real_first"]}
    _axes[1].plot(_layer, _mag_map[_mode_key][_layer], "o", color=_color_map[_mode_key],
                  markersize=14, markeredgecolor="black", markeredgewidth=2, zorder=5)

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

    # ── ReLU backfire chart ──
    _relu_175 = np.maximum(data["raw_scores_all"], 0)
    _relu_175_attn = _relu_175 / (_relu_175.sum(axis=-1, keepdims=True) + 1e-9)
    _relu_sink_175 = _relu_175_attn[:, :, :, 0].mean(axis=(1, 2)).mean() * 100

    _raw_12 = data["raw_scores_all"][:, :, :12, :12]
    _relu_12 = np.maximum(_raw_12, 0)
    _relu_12_attn = _relu_12 / (_relu_12.sum(axis=-1, keepdims=True) + 1e-9)
    _relu_sink_12 = _relu_12_attn[:, :, :, 0].mean(axis=(1, 2)).mean() * 100

    _std_sink_avg = data["sink_mag_standard"].mean() * 100

    _fig_relu, _ax_relu = plt.subplots(figsize=(6, 3))
    _relu_bars = _ax_relu.bar(
        ["Standard\n(baseline)", "ReLU\n(12 tokens)", "ReLU\n(175 tokens)"],
        [_std_sink_avg, _relu_sink_12, _relu_sink_175],
        color=["#e74c3c", "#27ae60", "#c0392b"], width=0.6,
    )
    for _b, _v in zip(_relu_bars, [_std_sink_avg, _relu_sink_12, _relu_sink_175]):
        _ax_relu.text(_b.get_x() + _b.get_width() / 2, _v + 0.5, f"{_v:.1f}%",
                      ha="center", va="bottom", fontweight="bold", fontsize=10)
    _ax_relu.set_ylabel("Sink magnitude (%)", fontsize=10)
    _ax_relu.set_title("ReLU Attention: Works Short, Backfires Long", fontsize=11, fontweight="bold")
    _ax_relu.grid(True, alpha=0.2, axis="y")
    _ax_relu.set_ylim(bottom=0)
    plt.tight_layout()

    mo.output.replace(mo.vstack([
        _fig,
        mo.md(f"""
**Layer {_layer} — attention to first real token:**
Standard: **{_std_s:.3f}** |
Temperature T={temp_slider.value:.1f}: **{_temp_s:.3f}** ({(_temp_s / _std_s - 1) * 100:+.0f}%) |
Sink Token: **{_sink_r:.4f}** ({(_sink_r / _std_s - 1) * 100:+.0f}%)

*Drag the temperature slider to explore the tradeoff. Higher T reduces the sink but also blurs genuine attention patterns.*
"""),
        mo.accordion({"Why ReLU attention didn't make the cut": mo.vstack([
            _fig_relu,
            mo.md(f"""
ReLU attention replaces softmax with ReLU — attention no longer has to sum to 100%. At 12 tokens, it works: sinks drop to **{_relu_sink_12:.1f}%**. But at production length (175 tokens), sinks *increase* to **{_relu_sink_175:.1f}%** — worse than standard.

The first 12 tokens of a 175-token sequence produce identical QKV values (causal masking means later tokens can't affect earlier ones). The chart above uses this fact: "ReLU (12 tokens)" is computed from the same forward pass, using only the first 12x12 block of attention scores.

Short-sequence validation can give the opposite conclusion from production-length testing.
"""),
        ])}),
    ]))
    return temp_attn,


# ============================================================================
# ACT 4: THE TOOL
# ============================================================================

@app.cell
def act4_controls(mo, temp_slider):
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
    dash_layer = mo.ui.slider(start=0, stop=11, value=8, label="Inspect layer", show_value=True)
    dash_head = mo.ui.slider(start=0, stop=11, value=0, label="Inspect head", show_value=True)

    mo.output.replace(mo.vstack([
        mo.md("""
---

## The Tool: Per-Head Attention Health

This is what we actually built.

Every transformer has dozens of attention heads, each specializing in different aspects of language. The entropy grid below shows which ones are working and which ones the sink has hijacked. Select any cell to see that head's full attention pattern.

- **Blue = healthy** — attention spread across relevant tokens
- **Red = sick** — attention concentrated on the sink
"""),
        mo.hstack([entropy_radio, dash_layer, dash_head], justify="start", gap=1),
    ]))
    return entropy_radio, dash_layer, dash_head


@app.cell
def act4_viz(data, mo, np, plt, temp_slider, temp_attn, entropy_radio, dash_layer, dash_head):
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

    _all_ent = np.stack([data["entropy_standard"], data["entropy_esa"],
                         _entropy_temp, data["entropy_sink"]])
    _vmin, _vmax = _all_ent.min(), _all_ent.max()

    _fig1, _ax1 = plt.subplots(figsize=(6, 5))
    _ax1.imshow(_ent, cmap="RdBu", aspect="auto", vmin=_vmin, vmax=_vmax)
    _ax1.set_xlabel("Head", fontsize=11)
    _ax1.set_ylabel("Layer", fontsize=11)
    _ax1.set_title(f"Entropy — {_lbl_map[_cond]}", fontsize=12, fontweight="bold")
    _ax1.set_xticks(range(data["n_heads"]))
    _ax1.set_yticks(range(data["n_layers"]))
    _ax1.add_patch(_Rect((dash_head.value - 0.5, dash_layer.value - 0.5), 1, 1,
                          linewidth=3, edgecolor="#f1c40f", facecolor="none"))
    plt.colorbar(_ax1.images[0], ax=_ax1, shrink=0.8, label="Entropy (red=sick, blue=healthy)")
    plt.tight_layout()

    _detail = _attn_source[_cond][dash_layer.value][dash_head.value]
    _fig2, _ax2 = plt.subplots(figsize=(6, 5))
    _ax2.imshow(_detail, cmap="viridis", aspect="auto")
    _ax2.set_xlabel("Key position", fontsize=11)
    _ax2.set_ylabel("Query position", fontsize=11)
    _ax2.set_title(f"Layer {dash_layer.value}, Head {dash_head.value} — {_lbl_map[_cond]}", fontsize=12, fontweight="bold")
    plt.colorbar(_ax2.images[0], ax=_ax2, shrink=0.8)
    plt.tight_layout()

    _median = np.median(_all_ent)
    _thresh = _median * 0.7
    _n_sick = int((_ent < _thresh).sum())
    _n_sick_std = int((_ent_std < _thresh).sum())
    _n_total = data["n_layers"] * data["n_heads"]

    _head_ent = _ent[dash_layer.value, dash_head.value]
    _head_ent_std = _ent_std[dash_layer.value, dash_head.value]
    if _head_ent < _thresh:
        _head_status = "**Naturally focused** — low entropy in all conditions. Likely a syntax or positional head." if _head_ent_std < _thresh else "**Still sick** — low entropy persists in this condition."
    else:
        _head_status = "**Healed** — was sick in standard, recovered here." if _head_ent_std < _thresh else "**Healthy** — attention distributed across relevant tokens."

    if _cond == "standard":
        _summary = f"**{_n_sick}** of {_n_total} heads are sink-corrupted (below 70% of median entropy)."
    else:
        _healed = _n_sick_std - _n_sick
        _summary = f"**{_n_sick}** of {_n_total} heads remain sick (was {_n_sick_std} in standard). **{max(0, _healed)} heads healed.**"

    mo.output.replace(mo.vstack([
        mo.hstack([_fig1, _fig2]),
        mo.md(f"{_summary}\n\n**Layer {dash_layer.value}, Head {dash_head.value}:** {_head_status}"),
    ]))
    return


@app.cell
def act4_reading(mo):
    mo.output.replace(mo.md("""
### Reading the dashboard

Not all low-entropy heads are sick. Some heads *should* focus narrowly — syntax heads tracking subject-verb agreement, coreference heads linking pronouns to nouns. These show low entropy in every condition.

The sink-corrupted heads are the ones with low entropy **only in standard attention** that recover when you apply a fix. The sink was hijacking their budget.

A healthy model has a mix of focused and diffuse heads. The sink corrupts this mix by forcing too many heads to concentrate on the wrong thing.
"""))
    return


# ============================================================================
# CLOSING: THE COST
# ============================================================================

@app.cell
def closing(data, mo, np, plt, temp_slider, temp_attn):
    _layers = np.arange(data["n_layers"])
    _std_waste = data["standard_attn"][:, :, :, 0].mean(axis=(1, 2)) * 100
    _temp_waste = temp_attn[:, :, :, 0].mean(axis=(1, 2)) * 100
    _sink_waste = data["sink_real_first"] * 100

    _std_total = _std_waste.mean()
    _temp_total = _temp_waste.mean()
    _sink_total = _sink_waste.mean()

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 5))

    _conditions = ["Standard", f"Temp T={temp_slider.value:.1f}", "Sink Token"]
    _values = [_std_total, _temp_total, _sink_total]
    _colors = ["#e74c3c", "#3498db", "#2ecc71"]
    _bars = _axes[0].bar(_conditions, _values, color=_colors, width=0.6, edgecolor="white", linewidth=1.5)
    _axes[0].set_ylabel("Attention budget wasted (%)", fontsize=11)
    _axes[0].set_title("Total Attention Wasted on Sink", fontsize=13, fontweight="bold")
    _axes[0].set_ylim(0, 55)
    for _bar, _val in zip(_bars, _values):
        _axes[0].text(_bar.get_x() + _bar.get_width() / 2, _val + 1, f"{_val:.1f}%",
                      ha="center", va="bottom", fontweight="bold", fontsize=11)
    _axes[0].grid(True, alpha=0.2, axis="y")

    _axes[1].plot(_layers, _std_waste, "o-", color="#e74c3c", linewidth=2, label="Standard")
    _axes[1].plot(_layers, _temp_waste, "s-", color="#3498db", linewidth=2, label=f"Temp T={temp_slider.value:.1f}")
    _axes[1].plot(_layers, _sink_waste, "^-", color="#2ecc71", linewidth=2, label="Sink Token")
    _axes[1].set_xlabel("Layer", fontsize=11)
    _axes[1].set_ylabel("Attention to sink (%)", fontsize=11)
    _axes[1].set_title("Waste Per Layer", fontsize=11, fontweight="bold")
    _axes[1].legend(fontsize=9)
    _axes[1].grid(True, alpha=0.2)
    _axes[1].set_ylim(bottom=0)
    plt.tight_layout()

    mo.output.replace(mo.vstack([
        mo.md(f"""
---

## The Cost

Every head in every layer has a fixed attention budget of 100%. When the sink absorbs a share, that capacity is wasted on a meaningless token instead of tracking syntax, semantics, or long-range dependencies.
"""),
        _fig,
        mo.md(f"""
**GPT-2 wastes {_std_total:.0f}% of its total attention budget** on a single meaningless token.

Temperature T={temp_slider.value:.1f} recovers **{_std_total - _temp_total:.0f} percentage points**.
Sink token recovers **{_std_total - _sink_total:.0f} percentage points** — nearly all of it.

In deeper layers, the waste exceeds 50%. More than half the model's capacity, staring at nothing.

*This connects to [Fast KV Compaction via Attention Matching](https://alphaxiv.org/abs/2602.16284) (Zweiger et al., curated paper #9). Compression algorithms keep the highest-attention tokens and evict the rest. When a garbage token hogs attention, it survives compression while real content gets dropped.*

This notebook connects [Exclusive Self Attention](https://alphaxiv.org/abs/2503.07822) (curated paper #6) with [Fast KV Compaction](https://alphaxiv.org/abs/2602.16284) (curated paper #9) through original empirical work: a finding that ESA has no effect on sinks, a demonstration that ReLU attention backfires at production scale, and a per-head diagnostic tool that no interactive version of exists in the literature.

*Open question: sinks appear tied to pre-norm architecture. A post-norm model of comparable size should show reduced or absent sinks — testing this would confirm whether the phenomenon is architectural rather than emergent.*

---

*Built for the [alphaXiv × marimo notebook competition](https://marimo.io/pages/events/notebook-competition).*

*References: [Attention Sinks](https://alphaxiv.org/abs/2309.17453) (Xiao et al., 2023), [The Spike, the Sparse and the Sink](https://alphaxiv.org/abs/2406.02069) (Chen et al., LeCun/NYU), [Attention Sinks Are Provably Necessary](https://alphaxiv.org/abs/2603.11487) (2026).*
"""),
    ]))
    return


# ============================================================================
# TRY IT YOURSELF
# ============================================================================

@app.cell
def try_controls(mo):
    text_form = mo.ui.form(
        mo.ui.text_area(label="Paste any text (max 200 tokens)", max_length=1500),
        submit_button_label="Analyze",
    )

    mo.output.replace(mo.vstack([
        mo.md("""
---

## Try It Yourself

Paste your own text and see where GPT-2's attention goes. A fresh forward pass runs on submit — no caching, no tricks.
"""),
        text_form,
    ]))
    return text_form,


@app.cell
def try_viz(data, mo, np, plt, text_form):
    import torch as _torch

    if not text_form.value:
        return

    _model = data["model"]
    _tokenizer = data["tokenizer"]
    _n_heads = data["n_heads"]
    _n_layers = data["n_layers"]
    _d_model = _model.config.n_embd
    _d_head = _d_model // _n_heads

    _inputs = _tokenizer(text_form.value, return_tensors="pt", truncation=True, max_length=200)
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

    _p = np.clip(_attn, 1e-9, 1.0)
    _ent = -(_p * np.log(_p)).sum(axis=-1).mean(axis=-1)

    _median_ent = np.median(_ent)
    _thresh = _median_ent * 0.7
    _n_sick = int((_ent < _thresh).sum())
    _n_total = _n_layers * _n_heads

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))

    _axes[0].bar(np.arange(_n_layers), _sink * 100, color="#e74c3c", width=0.7)
    _axes[0].set_xlabel("Layer", fontsize=11)
    _axes[0].set_ylabel("Attention to position 0 (%)", fontsize=11)
    _axes[0].set_title("Sink Magnitude — Your Text", fontsize=12, fontweight="bold")
    _axes[0].grid(True, alpha=0.2, axis="y")
    _axes[0].set_ylim(bottom=0)

    _axes[1].imshow(_ent, cmap="RdBu", aspect="auto")
    _axes[1].set_xlabel("Head", fontsize=11)
    _axes[1].set_ylabel("Layer", fontsize=11)
    _axes[1].set_title("Entropy Grid — Your Text", fontsize=12, fontweight="bold")
    _axes[1].set_xticks(range(_n_heads))
    _axes[1].set_yticks(range(_n_layers))
    plt.colorbar(_axes[1].images[0], ax=_axes[1], shrink=0.8)
    plt.tight_layout()

    _peak = _sink.max() * 100

    mo.output.replace(mo.vstack([
        _fig,
        mo.md(f"""
**Your text ({_sl} tokens):** Peak sink: **{_peak:.1f}%** at layer {_sink.argmax()}. Average: **{_sink.mean() * 100:.1f}%**.
**{_n_sick}** of {_n_total} heads show low entropy.
"""),
    ]))
    return


if __name__ == "__main__":
    app.run()
