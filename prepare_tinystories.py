"""
Tokenize TinyStories into binary token files for training.

    python prepare_tinystories.py

  1. train the BPE tokenizer on the first 20 MB of the training text
  2. encode every story, appending an end-of-story token
  3. write token ids as uint16 to data/tinystories/train.bin and val.bin

Encoded chunks are cached: the corpus reuses a small set of words, so each
distinct chunk is encoded only once.
"""

import argparse
import os
import time
from array import array
from collections import Counter

import numpy as np

from bpe import Tokenizer, split_text

parser = argparse.ArgumentParser()
parser.add_argument("--vocab_size", type=int, default=4096)
parser.add_argument("--tok_train_mb", type=int, default=20, help="MB of text to train the tokenizer on")
parser.add_argument("--max_train_mb", type=int, default=0, help="only encode this many MB of training text (0 = all)")
parser.add_argument("--data_dir", type=str, default="data/tinystories")
args = parser.parse_args()

EOT = "<|endoftext|>"  # how TinyStories marks the end of a story
train_path = os.path.join(args.data_dir, "TinyStoriesV2-GPT4-train.txt")
val_path = os.path.join(args.data_dir, "TinyStoriesV2-GPT4-valid.txt")
tok_path = os.path.join(args.data_dir, "tokenizer.json")

# ---------------------------------------------------------------- 1. tokenizer
with open(train_path, encoding="utf-8") as f:
    sample = f.read(args.tok_train_mb * 1_000_000)
sample = sample.replace(EOT, "\n")

print(f"training tokenizer: {args.vocab_size} tokens on {args.tok_train_mb} MB...", flush=True)
t0 = time.time()
tok = Tokenizer()
tok.train(sample, vocab_size=args.vocab_size, verbose=True)
tok.save(tok_path)
eot_id = len(tok.vocab)  # one extra id, after all the BPE tokens, means "story over"
print(f"done in {time.time() - t0:.0f} s. saved {tok_path}. end-of-story id = {eot_id}", flush=True)
assert eot_id < 2**16, "ids must fit in uint16"

# ---------------------------------------------------------------- 2. encode
cache = {}


def encode_chunk(chunk):
    ids = cache.get(chunk)
    if ids is None:
        ids = tok.encode(chunk)
        cache[chunk] = ids
    return ids


def encode_file(path, out_path, max_mb=0):
    t0 = time.time()
    tokens = array("H")  # uint16 storage; a Python list would need ~20 GB
    n_stories = 0
    read_bytes = 0
    with open(path, encoding="utf-8") as f:
        story_lines = []
        for line in f:
            read_bytes += len(line)
            if EOT in line:
                before = line.split(EOT)[0]
                story_lines.append(before)
                story = "".join(story_lines).strip()
                if story:
                    for chunk in split_text(story):
                        tokens.extend(encode_chunk(chunk))
                    tokens.append(eot_id)
                    n_stories += 1
                story_lines = []
                if n_stories % 100_000 == 0 and n_stories:
                    print(f"  {n_stories:,} stories, {len(tokens) / 1e6:.0f}M tokens, "
                          f"{time.time() - t0:.0f} s", flush=True)
                if max_mb and read_bytes > max_mb * 1_000_000:
                    break
            else:
                story_lines.append(line)
    arr = np.frombuffer(tokens, dtype=np.uint16)
    arr.tofile(out_path)
    print(f"{out_path}: {n_stories:,} stories, {len(arr):,} tokens ({time.time() - t0:.0f} s)", flush=True)
    return arr


val = encode_file(val_path, os.path.join(args.data_dir, "val.bin"))
train = encode_file(train_path, os.path.join(args.data_dir, "train.bin"), args.max_train_mb)

# ---------------------------------------------------------------- 3. sanity check
counts = Counter(val.tolist())
print(f"\nvocab used in val: {len(counts)} of {eot_id + 1} ids")
print("most common tokens:", [tok.decode([i]) if i != eot_id else "<EOT>" for i, _ in counts.most_common(15)])
print("first story, decoded back:")
first = val[: np.argmax(val == eot_id)].tolist()
print(tok.decode(first)[:500])
