"""
Generate text from the trained Shakespeare checkpoint.

    python sample.py
    python sample.py --n 1000 --temp 0.8
    python sample.py --prompt "ROMEO:"
"""

import argparse

import torch

from gpt import GPT, decode, device, encode

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=500, help="characters to generate")
parser.add_argument("--temp", type=float, default=1.0, help="temperature")
parser.add_argument("--prompt", type=str, default="\n", help="text to start from")
parser.add_argument("--ckpt", type=str, default="out/shakespeare.pt")
args = parser.parse_args()

ckpt = torch.load(args.ckpt, map_location=device)
model = GPT().to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"[checkpoint from step {ckpt['step']}, val loss {ckpt['val_loss']:.3f}]\n")

start = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
out = model.generate(start, args.n, temperature=args.temp)
print(decode(out[0].tolist()))
