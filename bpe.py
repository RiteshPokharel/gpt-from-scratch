"""
Byte-level byte-pair encoding (BPE) tokenizer.

Starts from the 256 byte values and repeatedly merges the most frequent
adjacent pair into a new token until the vocabulary reaches the target size.
Text is pre-split into word-like chunks (GPT-2 style) so merges never cross
word boundaries.

    aaabdaaabac -> (aa=Z) ZabdZabac -> (Za=Y) YbdYbac -> (Yb=X) XdXac

Usage:
    tok = Tokenizer()
    tok.train(text, vocab_size=4096)
    ids = tok.encode("Once upon a time")
    tok.decode(ids)                      # -> "Once upon a time"
    tok.save("out/tokenizer.json")

Tests:  python test_bpe.py
"""

import json
import re

# Pre-tokenization: a leading space stays attached to the following word,
# so " the" and "the" are distinct chunks.
SPLIT_PATTERN = re.compile(r" ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


def split_text(text):
    """'Once upon a time.' -> ['Once', ' upon', ' a', ' time', '.']"""
    return SPLIT_PATTERN.findall(text)


def get_pair_counts(ids, counts=None, weight=1):
    """Count how often each neighbouring pair appears in `ids`.

    get_pair_counts([1, 2, 3, 1, 2]) -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}

    `counts`: an existing dict to add into (to count across many chunks).
    `weight`: add this much per occurrence (a chunk that appears 500 times
              in the text counts 500 times).
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + weight
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`, left to right.

    merge([1, 2, 3, 1, 2], (1, 2), 256) -> [256, 3, 256]
    merge([1, 1, 1], (1, 1), 256)       -> [256, 1]
    """
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2  # skip both halves of the pair
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self):
        # merges: (id_a, id_b) -> new_id, in the order they were learned.
        self.merges = {}
        # vocab: id -> the bytes it stands for. Starts with the 256 single bytes.
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size, verbose=False):
        """Learn `vocab_size - 256` merges from `text`."""
        # Work on distinct chunks weighted by frequency instead of the raw
        # text: far fewer sequences to rescan after each merge.
        chunk_counts = {}
        for chunk in split_text(text):
            chunk_counts[chunk] = chunk_counts.get(chunk, 0) + 1
        chunks = [(list(c.encode("utf-8")), n) for c, n in chunk_counts.items()]

        for new_id in range(256, vocab_size):
            counts = {}
            for ids, n in chunks:
                get_pair_counts(ids, counts, weight=n)
            if not counts:  # nothing left to merge
                break
            pair = max(counts, key=counts.get)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            chunks = [(merge(ids, pair, new_id), n) for ids, n in chunks]
            if verbose and (new_id - 256) % 250 == 0:
                print(f"merge {new_id}: {self.vocab[pair[0]]!r} + {self.vocab[pair[1]]!r} "
                      f"-> {self.vocab[new_id]!r} ({counts[pair]:,} times)", flush=True)

    def encode(self, text):
        """Text -> list of token ids."""
        out = []
        for chunk in split_text(text):
            ids = list(chunk.encode("utf-8"))
            while len(ids) > 1:
                # Apply the earliest-learned merge first; later merges build on earlier ones.
                pairs = get_pair_counts(ids)
                pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                ids = merge(ids, pair, self.merges[pair])
            out.extend(ids)
        return out

    def decode(self, ids):
        """List of token ids -> text. Invalid UTF-8 (e.g. a split character) is replaced."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        data = {"merges": [[a, b, new] for (a, b), new in self.merges.items()]}
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path) as f:
            data = json.load(f)
        for a, b, new in data["merges"]:
            tok.merges[(a, b)] = new
            tok.vocab[new] = tok.vocab[a] + tok.vocab[b]
        return tok
