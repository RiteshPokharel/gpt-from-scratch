"""Plot out/loss.csv to out/loss.png.   python plot_loss.py"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("out/loss.csv")))
steps = [int(r["step"]) for r in rows]
train = [float(r["train_loss"]) for r in rows]
val = [float(r["val_loss"]) for r in rows]

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
