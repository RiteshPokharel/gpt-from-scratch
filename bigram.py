"""
Bigram baseline: predicts the next character from the current character only.

It has no context beyond one character, so it sets the floor that the GPT has
to beat. The data pipeline, loss and training loop are shared with gpt.py;
only the model differs.

    python bigram.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# ---------------------------------------------------------------- settings
batch_size = 32
block_size = 8       # context length in characters
max_steps = 3000
eval_every = 300
learning_rate = 1e-2
device = "mps" if torch.backends.mps.is_available() else "cpu"

# ---------------------------------------------------------------- 1. data
text = open("data/shakespeare.txt", encoding="utf-8").read()

# Character-level vocabulary: every distinct character in the file.
chars = sorted(set(text))
vocab_size = len(chars)  # 65

stoi = {ch: i for i, ch in enumerate(chars)}  # string -> integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer -> string
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

# Last 10% is held out for validation.
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    """Random batch of snippets. y is x shifted left by one (next-character targets)."""
    source = train_data if split == "train" else val_data
    starts = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in starts])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------- 2. model
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 65 x 65 table: row i holds the next-character logits given character i.
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)  # (B, T, vocab)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Autoregressive sampling, one character at a time."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, iters=100):
    """Mean loss over `iters` batches for each split."""
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def sample(model, n=300):
    start = torch.zeros((1, 1), dtype=torch.long, device=device)  # '\n'
    return decode(model.generate(start, n)[0].tolist())


# ---------------------------------------------------------------- 3. train
if __name__ == "__main__":
    model = BigramModel(vocab_size).to(device)
    print(f"device: {device}, parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n=== before training ===")
    print(sample(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("\n=== training ===")
    for step in range(max_steps + 1):
        if step % eval_every == 0:
            losses = estimate_loss(model)
            print(f"step {step:5d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}")

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\n=== after training ===")
    print(sample(model))
