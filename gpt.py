"""
Character-level GPT trained on Shakespeare.

Decoder-only transformer: 6 blocks of multi-head causal self-attention + MLP,
pre-LayerNorm, learned position embeddings, 256-character context. Same data
pipeline and training loop as the bigram baseline.

Train:    python gpt.py            (saves out/shakespeare.pt and out/loss.csv)
Sample:   python sample.py
Quick speed test:  python gpt.py --bench
"""

import csv
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# ---------------------------------------------------------------- settings
# Model size. These give ~10.8M parameters.
n_layer = 6        # how many transformer blocks are stacked
n_head = 6         # attention heads per block
n_embd = 384       # size of the vector that represents each character
block_size = 256   # context: how many previous characters the model can see
dropout = 0.2

# Training.
batch_size = 64
max_steps = 5000
eval_every = 250
eval_iters = 100
max_lr = 1e-3      # peak learning rate, after linear warmup
min_lr = 1e-4      # final learning rate, after cosine decay
warmup_steps = 100

device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = "out"

# ---------------------------------------------------------------- data (same as bigram.py)
text = open("data/shakespeare.txt", encoding="utf-8").read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    source = train_data if split == "train" else val_data
    starts = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in starts])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------- model
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    One fused linear layer produces q, k, v for all heads; the output is split
    into heads, attended over with a causal mask, and projected back.
    """

    def __init__(self):
        super().__init__()
        self.head_size = n_embd // n_head                      # 64 numbers per head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)   # query, key, value for every head
        self.proj = nn.Linear(n_embd, n_embd)                  # mixes the heads' outputs back together
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # Causal mask: position t may only attend to positions <= t.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.save_weights = False  # keep attention weights for visualize_attention.py
        self.last_weights = None

    def forward(self, x):
        B, T, C = x.shape  # batch, time (characters), channels (n_embd)

        q, k, v = self.qkv(x).split(n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size)
        q = q.view(B, T, n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, n_head, self.head_size).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)  # (B, nh, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        if self.save_weights:
            self.last_weights = weights.detach().cpu()
        weights = self.attn_dropout(weights)

        out = weights @ v                                    # (B, nh, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # concatenate heads
        return self.resid_dropout(self.proj(out))


class MLP(nn.Module):
    """Position-wise feed-forward network (4x expansion, GELU)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-LayerNorm transformer block: attention then MLP, each with a residual connection."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)    # token embedding
        self.pos_emb = nn.Embedding(block_size, n_embd)      # learned position embedding
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)            # logits over the 65 characters
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # GPT-2 style init: N(0, 0.02), zero biases.
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)  # (B, T, n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # crop to the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ---------------------------------------------------------------- training helpers
def lr_at(step):
    """Linear warmup, then cosine decay from max_lr to min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def sample_text(model, n=300):
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    return decode(model.generate(start, n)[0].tolist())


# ---------------------------------------------------------------- train
if __name__ == "__main__":
    model = GPT().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device} | parameters: {n_params / 1e6:.2f}M", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.99), weight_decay=0.1)

    if "--bench" in sys.argv:
        for i in range(25):
            if i == 5:  # ignore the first steps (kernel compilation)
                torch.mps.synchronize() if device == "mps" else None
                t0 = time.time()
            x, y = get_batch("train")
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.mps.synchronize() if device == "mps" else None
        per_step = (time.time() - t0) / 20
        print(f"{per_step * 1000:.0f} ms/step -> {max_steps} steps ≈ {per_step * max_steps / 60:.1f} min (+ evals)")
        sys.exit()

    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "loss.csv"), "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_min"])

    best_val = float("inf")
    t_start = time.time()
    for step in range(max_steps + 1):
        lr = lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % eval_every == 0 or step == max_steps:
            losses = estimate_loss(model)
            elapsed = (time.time() - t_start) / 60
            print(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
                  f"| lr {lr:.1e} | {elapsed:.1f} min", flush=True)
            writer.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr:.2e}", f"{elapsed:.2f}"])
            log.flush()
            if losses["val"] < best_val:  # checkpoint on best validation loss
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "step": step, "val_loss": best_val,
                            "chars": chars}, os.path.join(out_dir, "shakespeare.pt"))

        if step == max_steps:
            break

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min | best val loss {best_val:.3f}")
    print("\n=== sample ===")
    print(sample_text(model, 500))
