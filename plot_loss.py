"""Plot the training curves.   python plot_loss.py

out/loss.csv             -> out/loss.png              (Shakespeare)
out/tinystories_loss.csv -> out/tinystories_loss.png  (TinyStories, if present)
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    rows = list(csv.DictReader(open(path)))
    return ([int(r["step"]) for r in rows], [float(r["train_loss"]) for r in rows], [float(r["val_loss"]) for r in rows])


steps, train, val = read("out/loss.csv")
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
ax.plot(steps, train, label="train", linewidth=2)
ax.plot(steps, val, label="validation (unseen text)", linewidth=2)
ax.axhline(2.49, color="gray", linestyle="--", linewidth=1, label="bigram baseline")
ax.set_xlabel("training step")
ax.set_ylabel("loss (lower is better)")
ax.set_title("Character-level GPT on Shakespeare (10.8M parameters)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("out/loss.png")
print("saved out/loss.png")

if os.path.exists("out/tinystories_loss.csv"):
    steps, train, val = read("out/tinystories_loss.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    # step 0 (random weights, loss ~8.4) is left out so the rest of the curve is readable
    ax.plot(steps[1:], train[1:], label="train", linewidth=2)
    ax.plot(steps[1:], val[1:], label="validation (unseen stories)", linewidth=2)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss per token (lower is better)")
    ax.set_title("BPE GPT on TinyStories (27.4M parameters)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("out/tinystories_loss.png")
    print("saved out/tinystories_loss.png")
