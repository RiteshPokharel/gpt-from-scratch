"""
See what the trained model pays attention to.

    python visualize_attention.py

Feeds a line of text through the model and draws, for every head in every
layer, which earlier characters each character looks at.
Each heatmap: row = the character doing the looking, column = the character
being looked at, brightness = how much attention. Everything above the
diagonal is black because of the causal mask (no looking at the future).

Saves:
  out/attention_grid.png   all 6 layers x 6 heads
  out/attention_heads.png  a few heads up close, with the characters labelled
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import gpt
from gpt import GPT, device, encode

text = "ROMEO:\nIs the day so young?\n\nBENVOLIO:\nBut new struck nine.\n"
if len(sys.argv) > 1:
    text = sys.argv[1].replace("\\n", "\n")

ckpt = torch.load("out/shakespeare.pt", map_location=device)
model = GPT().to(device)
model.load_state_dict(ckpt["model"])
model.eval()

for block in model.blocks:
    block.attn.save_weights = True

idx = torch.tensor([encode(text)], dtype=torch.long, device=device)
with torch.no_grad():
    model(idx)

# attn[layer][head] is a (T, T) matrix
attn = torch.stack([block.attn.last_weights[0] for block in model.blocks])  # (layers, heads, T, T)
n_layers, n_heads, T, _ = attn.shape
labels = ["\\n" if c == "\n" else ("␣" if c == " " else c) for c in text]


def describe(w):
    """Summarise a head in numbers: where does it look, on average?"""
    T = w.shape[0]
    rows = torch.arange(T)
    to_self = w[rows, rows].mean().item()
    to_prev = w[rows[1:], rows[1:] - 1].mean().item()
    nl_cols = [i for i, c in enumerate(text) if c == "\n"]
    to_newline = w[:, nl_cols].sum(dim=1).mean().item() if nl_cols else 0.0
    dist = (w * (rows[:, None] - rows[None, :]).clamp(min=0).float()).sum(dim=1).mean().item()
    return to_self, to_prev, to_newline, dist


# ---------------------------------------------------------------- overview grid
fig, axes = plt.subplots(n_layers, n_heads, figsize=(n_heads * 2.2, n_layers * 2.2), dpi=130)
for L in range(n_layers):
    for H in range(n_heads):
        ax = axes[L, H]
        ax.imshow(attn[L, H] ** 0.5, cmap="magma", vmin=0, vmax=1)  # sqrt makes faint attention visible
        ax.set_xticks([])
        ax.set_yticks([])
        if L == 0:
            ax.set_title(f"head {H + 1}", fontsize=10)
        if H == 0:
            ax.set_ylabel(f"layer {L + 1}", fontsize=10)
fig.suptitle("Attention in every layer and head (brighter = more attention)", fontsize=13)
fig.tight_layout()
fig.savefig("out/attention_grid.png")
plt.close(fig)

# ---------------------------------------------------------------- pick distinct heads
stats = {(L, H): describe(attn[L, H]) for L in range(n_layers) for H in range(n_heads)}
print("layer head | self  prev  newline | avg distance back")
for (L, H), (s, p, nl, d) in stats.items():
    print(f"  {L + 1}    {H + 1}  | {s:.2f}  {p:.2f}  {nl:.2f}    | {d:5.1f}")

# These three heads were picked by looking at the grid and the numbers above
# for the trained checkpoint. After retraining, the same roles may move to
# different heads, so check the grid and update this list.
chosen = [
    ("the previous characters", (0, 1)),          # layer 1, head 2
    ("spaces and line breaks", (1, 5)),           # layer 2, head 6
    ("inside a speaker's name", (5, 5)),          # layer 6, head 6
]

fig, axes = plt.subplots(1, len(chosen), figsize=(7.5 * len(chosen), 7.5), dpi=130)
axes = [axes] if len(chosen) == 1 else axes
for ax, (name, (L, H)) in zip(axes, chosen):
    ax.imshow(attn[L, H] ** 0.5, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(labels, fontsize=6, family="monospace")
    ax.set_yticklabels(labels, fontsize=6, family="monospace")
    ax.set_xlabel("character being looked at")
    ax.set_ylabel("character doing the looking")
    ax.set_title(f"Layer {L + 1}, head {H + 1}: {name}", fontsize=12)
fig.tight_layout()
fig.savefig("out/attention_heads.png")
print("\nsaved out/attention_grid.png and out/attention_heads.png")
print("close-ups:", ", ".join(f"{n} = layer {L + 1} head {H + 1}" for n, (L, H) in chosen))
