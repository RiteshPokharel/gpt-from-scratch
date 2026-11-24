"""
Train the 27M parameter GPT on TinyStories.

    python prepare_tinystories.py          # once: tokenizer + token files
    python train_tinystories.py --bench    # measure step time
    python train_tinystories.py            # train (Ctrl+C is safe; state is checkpointed)
    python train_tinystories.py --resume   # continue from the last checkpoint

Every eval it saves out/tinystories.pt (model, optimizer, step), appends to
out/tinystories_loss.csv and writes a sample story to out/tinystories_samples.txt.
"""

import argparse
import csv
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from bpe import Tokenizer
from model import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--bench", action="store_true", help="time a few steps and exit")
parser.add_argument("--resume", action="store_true", help="continue from out/tinystories.pt")
parser.add_argument("--max_steps", type=int, default=30000)  # ~6 h on an M5 in bf16, ~490M tokens (about 0.9 of the dataset)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=6e-4)
parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
parser.add_argument("--eval_every", type=int, default=500)
args = parser.parse_args()

data_dir = "data/tinystories"
out_dir = "out"
ckpt_path = os.path.join(out_dir, "tinystories.pt")
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(1337)

warmup_steps = 500
min_lr = args.lr / 10
eval_iters = 50

# ---------------------------------------------------------------- data
# np.memmap reads straight from disk, so a 1 GB token file never has to fit in memory.
if args.bench and not os.path.exists(os.path.join(data_dir, "train.bin")):
    vocab_size = 4097
    train_data = val_data = np.random.randint(0, vocab_size, size=10_000_000).astype(np.uint16)
    tok = None
else:
    tok = Tokenizer.load(os.path.join(data_dir, "tokenizer.json"))
    vocab_size = len(tok.vocab) + 1  # + end-of-story token
    train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

cfg = GPTConfig(vocab_size=vocab_size)


def get_batch(split):
    source = train_data if split == "train" else val_data
    starts = np.random.randint(0, len(source) - cfg.block_size - 1, size=args.batch_size)
    x = torch.stack([torch.from_numpy(source[i : i + cfg.block_size].astype(np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(source[i + 1 : i + 1 + cfg.block_size].astype(np.int64)) for i in starts])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------- model + optimizer
model = GPT(cfg).to(device)
print(f"device: {device} | parameters: {model.num_params() / 1e6:.1f}M | precision: {args.precision}", flush=True)

# Weight decay on matrices only, not on biases or LayerNorm parameters.
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95),
)

# bf16 autocast for matmuls; master weights stay fp32.
autocast = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=args.precision == "bf16")


def lr_at(step):
    if step < warmup_steps:
        return args.lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, args.max_steps - warmup_steps))
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * progress))


def sync():
    if device == "mps":
        torch.mps.synchronize()


def train_step():
    x, y = get_batch("train")
    with autocast:
        _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss


# ---------------------------------------------------------------- bench
if args.bench:
    for i in range(25):
        if i == 5:
            sync()
            t0 = time.time()
        train_step()
    sync()
    per_step = (time.time() - t0) / 20
    tokens_per_step = args.batch_size * cfg.block_size
    print(f"{per_step * 1000:.0f} ms/step | {tokens_per_step / per_step:,.0f} tokens/s")
    for hours in (6, 8, 10):
        steps = int(hours * 3600 / per_step * 0.95)  # ~5% for evals
        print(f"  {hours} h ≈ {steps:,} steps ≈ {steps * tokens_per_step / 1e6:,.0f}M tokens")
    raise SystemExit


# ---------------------------------------------------------------- train
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with autocast:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def sample_story(prompt="Once upon a time", n=200):
    model.eval()
    eot_id = vocab_size - 1
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    with autocast:
        out = model.generate(idx, n, temperature=0.8, top_k=50, stop_id=eot_id)
    model.train()
    ids = [i for i in out[0].tolist() if i != eot_id]
    return tok.decode(ids)


os.makedirs(out_dir, exist_ok=True)
start_step, best_val, elapsed_before = 0, float("inf"), 0.0
if args.resume:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step, best_val, elapsed_before = ckpt["step"] + 1, ckpt["best_val"], ckpt.get("elapsed_h", 0.0)
    print(f"resumed from step {ckpt['step']} (best val {best_val:.3f})", flush=True)

loss_file = os.path.join(out_dir, "tinystories_loss.csv")
new_log = not (args.resume and os.path.exists(loss_file))
log = open(loss_file, "w" if new_log else "a", newline="")
writer = csv.writer(log)
if new_log:
    writer.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_h"])
samples = open(os.path.join(out_dir, "tinystories_samples.txt"), "w" if new_log else "a")

t_start = time.time()
for step in range(start_step, args.max_steps + 1):
    lr = lr_at(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    if step % args.eval_every == 0 or step == args.max_steps:
        losses = estimate_loss()
        elapsed = elapsed_before + (time.time() - t_start) / 3600
        print(f"step {step:6d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
              f"| lr {lr:.1e} | {elapsed:.2f} h", flush=True)
        writer.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr:.2e}", f"{elapsed:.3f}"])
        log.flush()
        story = sample_story()
        samples.write(f"=== step {step}, val loss {losses['val']:.3f} ===\n{story}\n\n")
        samples.flush()
        best_val = min(best_val, losses["val"])
        # Always save the latest state so a crash or Ctrl+C loses at most one eval interval.
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                    "best_val": best_val, "val_loss": losses["val"], "config": asdict(cfg),
                    "elapsed_h": elapsed}, ckpt_path)

    if step == args.max_steps:
        break
    train_step()

print(f"\ndone. best val loss {best_val:.3f}")
print(sample_story())
