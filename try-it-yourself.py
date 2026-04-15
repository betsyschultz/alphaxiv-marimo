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


@app.cell(hide_code=True)
def title():
    import marimo as mo
    mo.output.replace(
        mo.md("""
# Try It Yourself
## Paste any text and see where GPT-2's attention goes.

The sink pattern holds regardless of content — try code vs. prose vs. a
grocery list. The first token always absorbs disproportionate attention
in deeper layers, even when the content is completely different.
""")
    )


@app.cell
def setup():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        "font.family": "sans-serif",
    })

    with mo.status.spinner("Loading GPT-2..."):
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model.eval()

    mo.output.replace(
        mo.callout(mo.md("**GPT-2 loaded.** Paste text below or pick an example."), kind="success")
    )
    return model, mo, np, plt, tokenizer, torch


@app.cell
def picker(mo):
    example_picker = mo.ui.dropdown(
        options={
            "": "",
            "Python code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nfor i in range(20):\n    print(f'fib({i}) = {fibonacci(i)}')",
            "Poetry (Dickinson)": "Because I could not stop for Death,\nHe kindly stopped for me;\nThe carriage held but just ourselves\nAnd Immortality.\n\nWe slowly drove, he knew no haste,\nAnd I had put away\nMy labor, and my leisure too,\nFor his civility.",
            "Grocery list": "eggs, milk, bread, butter, apples, chicken thighs, olive oil, garlic, onions, pasta, canned tomatoes, mozzarella, basil, salt, pepper, rice, black beans, avocados, limes, cilantro",
            "Legal text": "Notwithstanding any other provision of this Agreement, the indemnifying party shall not be liable for any indirect, incidental, consequential, special, or exemplary damages arising out of or related to this Agreement, including but not limited to loss of revenue, loss of profits, loss of business, or loss of data, even if such party has been advised of the possibility of such damages.",
        },
        value="",
        label="Pick an example",
    )
    mo.output.replace(example_picker)
    return (example_picker,)


@app.cell
def text_input(example_picker, mo):
    custom_text = mo.ui.text_area(
        value=example_picker.value or "",
        placeholder="Paste any text here to see its attention sink pattern...",
        label="Your text (truncated to 512 tokens)",
        max_length=2000,
        full_width=True,
    )
    mo.output.replace(custom_text)
    return (custom_text,)


@app.cell(hide_code=True)
def viz(custom_text, mo, model, np, plt, tokenizer, torch):
    _text = custom_text.value.strip()
    mo.stop(
        not _text,
        mo.callout(
            mo.md("*Type or paste text above to analyze its attention pattern.*"),
            kind="neutral",
        ),
    )

    with mo.status.spinner("Computing attention..."):
        _inputs = tokenizer(
            _text, return_tensors="pt", truncation=True, max_length=512,
        )
        _seq = _inputs["input_ids"].shape[1]
        _toks = [tokenizer.decode(t) for t in _inputs["input_ids"][0]]

        with torch.no_grad():
            _out = model(**_inputs, output_attentions=True)

        _attn = np.stack([l[0].numpy() for l in _out.attentions])
        _n_layers = _attn.shape[0]
        _sink_per_layer = _attn[:, :, :, 0].mean(axis=(1, 2)) * 100
        _overall_sink = _sink_per_layer.mean()

    _deep = _attn[-1].mean(axis=0)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))

    _axes[0].imshow(_deep, cmap="viridis", aspect="auto")
    _axes[0].set_title(
        f"Layer {_n_layers - 1} — {_sink_per_layer[-1]:.1f}% to position 0",
        fontweight="bold",
    )
    _axes[0].set_xlabel("Key position")
    _axes[0].set_ylabel("Query position")
    plt.colorbar(_axes[0].images[0], ax=_axes[0], shrink=0.8, label="Attention weight")

    _axes[1].plot(
        range(_n_layers), _sink_per_layer, "o-",
        color="#e74c3c", linewidth=2,
    )
    _axes[1].set_xlabel("Layer")
    _axes[1].set_ylabel("Attention to position 0 (%)")
    _axes[1].set_title("Sink Magnitude by Layer", fontweight="bold")
    _axes[1].set_ylim(bottom=0)
    _axes[1].grid(True, alpha=0.2)
    plt.tight_layout()

    _first_tok = _toks[0].strip() or "(space)"
    _status = "dominant" if _overall_sink > 30 else "moderate" if _overall_sink > 15 else "mild"
    mo.output.replace(
        mo.vstack([
            _fig,
            mo.callout(
                mo.md(
                    f"**Your text:** {_seq} tokens · **{_overall_sink:.1f}%** average "
                    f"attention to position 0 · First token: *{_first_tok}* · "
                    f"Status: **{_status}**"
                ),
                kind="danger" if _status == "dominant" else "warn" if _status == "moderate" else "success",
            ),
        ])
    )


if __name__ == "__main__":
    app.run()
