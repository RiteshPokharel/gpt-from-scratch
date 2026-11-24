"""
Export everything the comparison page shows into web/data/*.json.

    python web/export_data.py

Reads the two checkpoints, the loss logs and the TinyStories sample log
(on the CPU, so it can run while the GPU is training). The live examples are
recorded separately by record_demo.mjs.
"""

import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import gpt as char_gpt  # noqa: E402  (loads data/shakespeare.txt on import)
from bpe import Tokenizer  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

OUT = os.path.join(ROOT, "web", "data")
device = "cpu"
os.makedirs(OUT, exist_ok=True)


def dump(name, obj):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote web/data/{name} ({os.path.getsize(os.path.join(OUT, name)) / 1024:.0f} KB)")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- Shakespeare model
char_ckpt = torch.load("out/shakespeare.pt", map_location=device)
char_model = char_gpt.GPT().to(device)
char_model.load_state_dict(char_ckpt["model"])
char_model.eval()
chars = char_ckpt["chars"]
char_params = sum(p.numel() for p in char_model.parameters())



# ---------------------------------------------------------------- TinyStories model
# The training run overwrites out/tinystories.pt every eval, so copy it once
# it has not changed for a few seconds, then load the copy.
def stable_copy(path):
    for _ in range(60):
        m1, s1 = os.path.getmtime(path), os.path.getsize(path)
        time.sleep(3)
        if (m1, s1) == (os.path.getmtime(path), os.path.getsize(path)):
            dst = os.path.join(tempfile.gettempdir(), "tinystories_export.pt")
            shutil.copyfile(path, dst)
            if os.path.getmtime(path) == m1:
                return dst
    raise RuntimeError("checkpoint kept changing")


ts_path = stable_copy("out/tinystories.pt")
ts_ckpt = torch.load(ts_path, map_location=device)
os.remove(ts_path)
cfg = GPTConfig(**ts_ckpt["config"])
ts_model = GPT(cfg).to(device)
ts_model.load_state_dict(ts_ckpt["model"])
ts_model.eval()
tok = Tokenizer.load("data/tinystories/tokenizer.json")
eot_id = cfg.vocab_size - 1




# ---------------------------------------------------------------- characters per token
# Needed to turn TinyStories loss (per token) into bits per character.
val = np.memmap("data/tinystories/val.bin", dtype=np.uint16, mode="r")[:2_000_000]
ids = val.tolist()
n_chars = 0
start = 0
for i, t in enumerate(ids + [eot_id]):
    if t == eot_id:
        n_chars += len(tok.decode(ids[start:i])) + 1  # the end-of-story token counts as one character
        start = i + 1
chars_per_token = n_chars / len(ids)

# ---------------------------------------------------------------- loss curves + specs
char_loss = [{"step": int(r["step"]), "train": float(r["train_loss"]), "val": float(r["val_loss"])}
             for r in read_csv("out/loss.csv")]
ts_rows = read_csv("out/tinystories_loss.csv")
ts_loss = [{"step": int(r["step"]), "train": float(r["train_loss"]), "val": float(r["val_loss"])} for r in ts_rows]
best_char = min(char_loss, key=lambda r: r["val"])
best_ts = min(ts_loss, key=lambda r: r["val"])
last = ts_rows[-1]
ts_steps_total = 30000
elapsed_h = float(last["elapsed_h"])
h_per_step = elapsed_h / max(1, int(last["step"]))
tokens_per_step = 64 * cfg.block_size
char_train_chars = int(0.9 * len(char_gpt.text))

ln2 = math.log(2)

# a real excerpt from each training file, and how many stories TinyStories has
char_excerpt = "\n\n".join(char_gpt.text.split("\n\n")[:4])
train_tokens = np.memmap("data/tinystories/train.bin", dtype=np.uint16, mode="r")
n_stories = int(np.count_nonzero(train_tokens == eot_id))
with open("data/tinystories/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8") as f:
    story = f.read(20_000).split("<|endoftext|>")[1].strip()
cut = story[:420]
story_excerpt = cut[: cut.rfind(".") + 1]
# about 1,200 characters of real stories, to show how much text each model can see at once
with open("data/tinystories/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8") as f:
    window_text = " ".join(x.strip().replace("\n", " ") for x in f.read(20_000).split("<|endoftext|>")[1:4])[:1250]
models = {
    "updated": time.strftime("%Y-%m-%d %H:%M"),
    "char": {
        "name": "Shakespeare",
        "params": char_params,
        "layers": char_gpt.n_layer, "heads": char_gpt.n_head, "embd": char_gpt.n_embd,
        "context_tokens": char_gpt.block_size, "context_chars": char_gpt.block_size,
        "vocab": len(chars), "chars": "".join(chars),
        "tokenizer": "characters",
        "dataset": "TinyShakespeare", "dataset_chars": len(char_gpt.text),
        "dataset_lines": char_gpt.text.count("\n"), "excerpt": char_excerpt, "file": "shakespeare.txt",
        "train_tokens": char_train_chars,
        "dropout": char_gpt.dropout, "precision": "fp32", "weight_tying": False,
        "batch": char_gpt.batch_size, "lr": "1e-3 → 1e-4 cosine",
        "best_step": best_char["step"], "best_val": best_char["val"],
        "steps_run": char_loss[-1]["step"],
        "tokens_seen": best_char["step"] * char_gpt.batch_size * char_gpt.block_size,
        "epochs": best_char["step"] * char_gpt.batch_size * char_gpt.block_size / char_train_chars,
        "train_minutes": 27.4,
        "bpc": best_char["val"] / ln2,
        "loss": char_loss,
    },
    "bpe": {
        "name": "TinyStories",
        "params": ts_model.num_params(),
        "layers": cfg.n_layer, "heads": cfg.n_head, "embd": cfg.n_embd,
        "context_tokens": cfg.block_size, "context_chars": round(cfg.block_size * chars_per_token),
        "vocab": cfg.vocab_size,
        "tokenizer": "byte-level BPE",
        "dataset": "TinyStories V2", "dataset_chars": os.path.getsize("data/tinystories/TinyStoriesV2-GPT4-train.txt"),
        "stories": n_stories, "excerpt": story_excerpt, "file": "TinyStoriesV2-GPT4-train.txt",
        "window_text": window_text,
        "train_tokens": os.path.getsize("data/tinystories/train.bin") // 2,
        "dropout": cfg.dropout, "precision": "bf16", "weight_tying": True,
        "batch": 64, "lr": "6e-4 → 6e-5 cosine",
        "best_step": best_ts["step"], "best_val": best_ts["val"],
        "steps_run": int(last["step"]), "steps_total": ts_steps_total,
        "tokens_seen": int(last["step"]) * tokens_per_step,
        "epochs": int(last["step"]) * tokens_per_step / (os.path.getsize("data/tinystories/train.bin") // 2),
        "elapsed_h": elapsed_h,
        "eta_h": h_per_step * (ts_steps_total - int(last["step"])),
        "done": int(last["step"]) >= ts_steps_total,
        "chars_per_token": chars_per_token,
        "bpc": best_ts["val"] / ln2 / chars_per_token,
        "loss": ts_loss,
    },
}
dump("models.json", models)

# ---------------------------------------------------------------- samples through training
text = open("out/tinystories_samples.txt", encoding="utf-8").read()
blocks = re.findall(r"=== step (\d+), val loss ([\d.]+) ===\n(.*?)(?=\n\n=== step|\Z)", text, re.S)
replay = [{"step": int(s), "val": float(v), "text": t.replace("�", "").strip()} for s, v, t in blocks]
dump("replay.json", replay)

print(f"chars/token {chars_per_token:.3f} | char bpc {models['char']['bpc']:.3f} | "
      f"bpe bpc {models['bpe']['bpc']:.3f} (step {best_ts['step']})")
