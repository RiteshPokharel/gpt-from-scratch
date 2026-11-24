"""Tests for bpe.py.   python test_bpe.py  (stops at the first failure)"""

import os
import sys
import time

from bpe import Tokenizer, get_pair_counts, merge, split_text

passed = 0


def check(name, got, expected):
    global passed
    if got != expected:
        print(f"FAIL  {name}\n      expected: {expected!r}\n      got:      {got!r}")
        sys.exit(1)
    passed += 1
    print(f"ok    {name}")


# ---------------------------------------------------------------- split_text
def t_split():
    check("split_text", split_text("Once upon a time."), ["Once", " upon", " a", " time", "."])


# ---------------------------------------------------------------- get_pair_counts
def t_pairs():
    check("pair counts", get_pair_counts([1, 2, 3, 1, 2]), {(1, 2): 2, (2, 3): 1, (3, 1): 1})
    check("pair counts, empty and single", (get_pair_counts([]), get_pair_counts([5])), ({}, {}))
    c = get_pair_counts([1, 2], weight=10)
    check("pair counts with weight", c, {(1, 2): 10})
    check("pair counts adds into existing dict", get_pair_counts([1, 2, 9], counts=c), {(1, 2): 11, (2, 9): 1})


# ---------------------------------------------------------------- merge
def t_merge():
    check("merge", merge([1, 2, 3, 1, 2], (1, 2), 256), [256, 3, 256])
    check("merge with overlap", merge([1, 1, 1], (1, 1), 256), [256, 1])
    check("merge, pair not present", merge([1, 2, 3], (3, 1), 256), [1, 2, 3])
    original = [1, 2, 1, 2]
    merge(original, (1, 2), 256)
    check("merge doesn't change its input", original, [1, 2, 1, 2])


# ---------------------------------------------------------------- train
def t_train():
    tok = Tokenizer()
    tok.train("aaabdaaabac", vocab_size=259)
    check("train makes vocab_size - 256 merges", len(tok.merges), 3)
    check("first merge is the most common pair, 'aa'", list(tok.merges.items())[0], ((97, 97), 256))
    check("merge ids count up from 256", sorted(tok.merges.values()), [256, 257, 258])
    check("vocab has vocab_size entries", len(tok.vocab), 259)
    check("vocab[256] is the bytes of 'aa'", tok.vocab[256], b"aa")

    tok = Tokenizer()
    tok.train("the cat the cat the hat", vocab_size=262)
    check("merges don't cross word boundaries", all(b" " not in v[1:] for v in tok.vocab.values()), True)


# ---------------------------------------------------------------- encode / decode
def t_encode_decode():
    tok = Tokenizer()
    check("decode raw bytes", tok.decode([104, 105]), "hi")
    check("encode with no merges = bytes", tok.encode("hi"), [104, 105])

    tok.train("aaabdaaabac", vocab_size=259)
    check("encode compresses", len(tok.encode("aaabdaaabac")), 5)

    sample = "Once upon a time, a little dog named Max found a big red ball.\n\n\"Yay!\" he said."
    tok = Tokenizer()
    tok.train(sample * 20, vocab_size=320)
    ids = tok.encode(sample)
    check("round trip: decode(encode(x)) == x", tok.decode(ids), sample)
    check("trained tokenizer uses fewer tokens than bytes", len(ids) < len(sample.encode()), True)
    weird = "héllo wörld 👋  tabs\tand\nnewlines\n"
    check("round trip on unicode and odd spacing", tok.decode(tok.encode(weird)), weird)
    check("decode doesn't crash on half a character", isinstance(tok.decode([240, 159]), str), True)

    tok = Tokenizer()
    tok.merges = {(97, 98): 256, (256, 99): 257, (98, 99): 258}
    tok.vocab.update({256: b"ab", 257: b"abc", 258: b"bc"})
    check("encode applies earliest-learned merge first", tok.encode("abc"), [257])


# ---------------------------------------------------------------- save / load + real data
def t_real():
    tok = Tokenizer()
    tok.train("the cat sat on the mat " * 50, vocab_size=280)
    os.makedirs("out", exist_ok=True)
    tok.save("out/_test_tok.json")
    tok2 = Tokenizer.load("out/_test_tok.json")
    os.remove("out/_test_tok.json")
    check("save then load gives the same encoding", tok2.encode("the cat sat"), tok.encode("the cat sat"))

    path = "data/tinystories/TinyStoriesV2-GPT4-valid.txt"
    if not os.path.exists(path):
        print("skip  TinyStories check (dataset not downloaded)")
        return
    text = open(path, encoding="utf-8").read()[:1_000_000]
    t0 = time.time()
    tok = Tokenizer()
    tok.train(text, vocab_size=512)
    secs = time.time() - t0
    ids = tok.encode(text[:20_000])
    check("TinyStories round trip", tok.decode(ids), text[:20_000])
    ratio = len(text[:20_000].encode()) / len(ids)
    print(f"info  1 MB of TinyStories, 256 merges: {secs:.1f} s, {ratio:.2f} bytes per token")


for test in (t_split, t_pairs, t_merge, t_train, t_encode_decode, t_real):
    test()

print(f"\nall {passed} checks passed")
