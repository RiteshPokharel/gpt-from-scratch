"""
Generate a story from the trained TinyStories model.

    python sample_tinystories.py
    python sample_tinystories.py --prompt "The little girl" --temp 0.8

Needs out/tinystories.pt and data/tinystories/tokenizer.json, either from your
own training run or downloaded from the GitHub release.
"""

import argparse

import torch

from bpe import Tokenizer
from model import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="Once upon a time")
parser.add_argument("--n", type=int, default=300, help="tokens to generate at most")
parser.add_argument("--temp", type=float, default=0.8, help="temperature")
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--ckpt", type=str, default="out/tinystories.pt")
parser.add_argument("--tokenizer", type=str, default="data/tinystories/tokenizer.json")
args = parser.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(args.ckpt, map_location=device)
cfg = GPTConfig(**ckpt["config"])
model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
tok = Tokenizer.load(args.tokenizer)
eot = cfg.vocab_size - 1  # the end-of-story token
print(f"[checkpoint from step {ckpt['step']}, val loss {ckpt['val_loss']:.3f}]\n")

idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
out = model.generate(idx, args.n, temperature=args.temp, top_k=args.top_k, stop_id=eot)[0].tolist()
print(tok.decode([i for i in out if i != eot]))
