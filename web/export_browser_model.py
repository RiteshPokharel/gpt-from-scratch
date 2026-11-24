"""
Export both models for the in-browser demo: web/models/<name>.json + <name>.bin.

    python web/export_browser_model.py

Weight matrices are stored as int8 with one float32 scale per row (a quarter
of the float32 size); LayerNorm weights and biases stay float32. Both models
are written with the same tensor names, so one JavaScript engine runs either.
Also writes web/models/reference.json: PyTorch's top-10 next-token guesses for
a few prompts, used to check the JavaScript engine (node web/js/test_engine.mjs).
"""

import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))
sys.path.insert(0, ROOT)
from bpe import Tokenizer  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from nets import CharGPT  # noqa: E402

OUT = os.path.join(ROOT, "web", "models")
CHUNK = 12_000_000
os.makedirs(OUT, exist_ok=True)


def stable_load(path):
    """The training run rewrites its checkpoint every eval; load a copy once it is quiet."""
    for _ in range(60):
        m = os.path.getmtime(path)
        time.sleep(3)
        if m == os.path.getmtime(path):
            tmp = os.path.join(tempfile.gettempdir(), "browser_export.pt")
            shutil.copyfile(path, tmp)
            if m == os.path.getmtime(path):
                ck = torch.load(tmp, map_location="cpu")
                os.remove(tmp)
                return ck
    raise RuntimeError(f"{path} kept changing")


def write(name, tensors, meta):
    """tensors: list of (name, array, quantize)."""
    manifest = {**meta, "tensors": {}}
    blob = bytearray()

    def append(arr):
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(arr.tobytes())
        return off

    for tname, t, quant in tensors:
        a = t.detach().float().numpy()
        entry = {"shape": list(a.shape)}
        if quant:
            scale = np.abs(a).max(axis=1) / 127.0
            scale[scale == 0] = 1.0
            q = np.clip(np.round(a / scale[:, None]), -127, 127).astype(np.int8)
            entry.update(dtype="int8", offset=append(q), scale=append(scale.astype(np.float32)))
        else:
            entry.update(dtype="f32", offset=append(a.astype(np.float32)))
        manifest["tensors"][tname] = entry
    # split into 12 MB parts: static hosts often cap single files at 15 MB or so
    for old in os.listdir(OUT):
        if old.startswith(f"{name}.bin"):
            os.remove(os.path.join(OUT, old))
    parts = [blob[i:i + CHUNK] for i in range(0, len(blob), CHUNK)]
    for k, part in enumerate(parts):
        with open(os.path.join(OUT, f"{name}.bin.{k}"), "wb") as f:
            f.write(part)
    manifest["parts"] = len(parts)
    manifest["bytes"] = len(blob)
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False)
    print(f"web/models/{name}: {len(blob) / 1e6:.1f} MB in {len(parts)} parts")


def tensor_list(sd, n_layer, mlp, tied, head_bias):
    t = [("tok_emb", sd["token_emb.weight"], True), ("pos_emb", sd["pos_emb.weight"], True)]
    for l in range(n_layer):
        p = f"blocks.{l}."
        t += [
            (f"{l}.ln1.w", sd[p + "ln1.weight"], False), (f"{l}.ln1.b", sd[p + "ln1.bias"], False),
            (f"{l}.qkv.w", sd[p + "attn.qkv.weight"], True),
            (f"{l}.proj.w", sd[p + "attn.proj.weight"], True), (f"{l}.proj.b", sd[p + "attn.proj.bias"], False),
            (f"{l}.ln2.w", sd[p + "ln2.weight"], False), (f"{l}.ln2.b", sd[p + "ln2.bias"], False),
            (f"{l}.fc.w", sd[p + mlp + "0.weight"], True), (f"{l}.fc.b", sd[p + mlp + "0.bias"], False),
            (f"{l}.out.w", sd[p + mlp + "2.weight"], True), (f"{l}.out.b", sd[p + mlp + "2.bias"], False),
        ]
    t += [("ln_f.w", sd["ln_f.weight"], False), ("ln_f.b", sd["ln_f.bias"], False)]
    if not tied:
        t.append(("head.w", sd["head.weight"], True))
    if head_bias:
        t.append(("head.b", sd["head.bias"], False))
    return t


reference = {}

# ---------------------------------------------------------------- Shakespeare
ck = torch.load(os.path.join(ROOT, "out", "shakespeare.pt"), map_location="cpu")
chars = ck["chars"]
char_net = CharGPT(len(chars))
char_net.load_state_dict(ck["model"])
char_net.eval()
write("shakespeare", tensor_list(ck["model"], 6, "mlp.net.", tied=False, head_bias=True), {
    "name": "Shakespeare", "tokenizer": "char", "chars": "".join(chars),
    "vocab": len(chars), "block_size": 256, "n_layer": 6, "n_head": 6, "n_embd": 384,
    "tied": False, "eot": None, "step": ck["step"], "val_loss": ck["val_loss"],
})
stoi = {c: i for i, c in enumerate(chars)}
reference["shakespeare"] = []
with torch.no_grad():
    for prompt in ["ROMEO:\n", "First Citizen:\nWe are", "To be, or not to be"]:
        ids = [stoi[c] for c in prompt]
        p = torch.softmax(char_net(torch.tensor([ids]))[0, -1], -1)
        v, i = torch.topk(p, 10)
        reference["shakespeare"].append({"ids": ids, "top": i.tolist(), "p": v.tolist()})

# ---------------------------------------------------------------- TinyStories
ck = stable_load(os.path.join(ROOT, "out", "tinystories.pt"))
cfg = GPTConfig(**ck["config"])
ts_net = GPT(cfg)
ts_net.load_state_dict(ck["model"])
ts_net.eval()
write("tinystories", tensor_list(ck["model"], cfg.n_layer, "mlp.", tied=True, head_bias=False), {
    "name": "TinyStories", "tokenizer": "bpe",
    "vocab": cfg.vocab_size, "block_size": cfg.block_size, "n_layer": cfg.n_layer, "n_head": cfg.n_head,
    "n_embd": cfg.n_embd, "tied": True, "eot": cfg.vocab_size - 1, "step": ck["step"], "val_loss": ck["val_loss"],
})
tok = Tokenizer.load(os.path.join(ROOT, "data", "tinystories", "tokenizer.json"))
reference["tinystories"] = []
with torch.no_grad():
    for prompt in ["Once upon a time, there was a little", "The dog ran to the", "Lily wanted to"]:
        ids = tok.encode(prompt)
        p = torch.softmax(ts_net(torch.tensor([ids]))[0][0, -1], -1)
        v, i = torch.topk(p, 10)
        reference["tinystories"].append({"ids": ids, "top": i.tolist(), "p": v.tolist()})

with open(os.path.join(OUT, "reference.json"), "w") as f:
    json.dump(reference, f)
print(f"tinystories checkpoint: step {ck['step']}, val loss {ck['val_loss']:.3f}")
