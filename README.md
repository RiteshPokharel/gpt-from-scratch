# GPT from scratch

Two GPT-style language models written and trained from the ground up in PyTorch, on a laptop (Apple M5, MPS GPU backend), plus an interactive page where they run in the browser.

| Model | Reads text as | Parameters | Trained on | Training time | Validation loss |
|---|---|---|---|---|---|
| Bigram (baseline) | characters | 4,225 | TinyShakespeare | 6 s | 2.49 per char |
| **Shakespeare GPT** | characters (65) | 10.8M | TinyShakespeare, 1.1M chars | 27 min | **1.48** per char |
| **TinyStories GPT** | BPE tokens (4,097) | 27.4M | TinyStories, 2.7M stories | training | training |

Both models follow the decoder-only transformer from [Attention Is All You Need](https://arxiv.org/abs/1706.03762): token + position embeddings, a stack of pre-LayerNorm blocks (multi-head causal self-attention, then an MLP, each with a residual connection), and a final linear layer and softmax over the vocabulary. Everything is written by hand: the attention, the blocks, the training loop and the byte-pair encoding tokenizer. No pretrained weights, no tokenizer libraries.

## The interactive page

`web/` is a static page (plain HTML, CSS and JavaScript, no dependencies) with:

- **A demo.** Pick a model and a starting line and watch it write. Recorded examples play instantly; a custom prompt loads the model (int8 weights, 11 MB / 28 MB) into the browser and runs it there.
- **What happens when you press Generate.** Tokens, embeddings, attention, the next-token probabilities and sampling, on a fixed example per model, with the attention and softmax equations worked through on real numbers.
- **The two models side by side**, and how they learned: loss curves and the stories the TinyStories model wrote every 500 steps of training.

The browser engine (`web/js/gpt.js`) is a from-scratch JavaScript forward pass with a key/value cache. It is checked against PyTorch with `node web/js/test_engine.mjs`: the same top prediction on every test prompt, probabilities within 0.7 percentage points.

Run it locally:

```bash
python3 -m http.server -d web 8000     # then open http://localhost:8000
```

## Shakespeare model

![Training and validation loss](out/loss.png)

Validation loss stopped improving around step 2,000 while training loss kept falling: the model had read its 1M characters about 33 times and was starting to memorise them. The checkpoint with the best validation loss is the one kept.

Generated from the prompt `ROMEO:` at temperature 0.8:

```
ROMEO:
And not a brother than the king shall stay the world:
No foot that thou hast devised me.
And shalt thou call me forth from her duty?

Nurse:
Nay, son, since he cannot live her coat.

ROMEO:
'Tis very rich in mine arms cannot hear her.
Now Capulet! how now, by this speed, the earth doth more flatter
```

### What the attention heads learned

Feeding the line below through the trained model and recording every head's attention weights (`visualize_attention.py`):

```
ROMEO:
Is the day so young?

BENVOLIO:
But new struck nine.
```

In each heatmap, a row is the character doing the looking, a column is the character being looked at, and brighter means more attention. The upper triangle is empty because of the causal mask.

![Three attention heads up close](out/attention_heads.png)

- **Layer 1, head 2: the previous characters.** The bright line just under the diagonal means each character looks at the one or two before it. All six heads in layer 1 do some version of this, which is how the model handles spelling.
- **Layer 2, head 6: spaces and line breaks.** The vertical stripes sit on the spaces and newlines, so this head tracks where the current word started.
- **Layer 6, head 6: inside a speaker's name.** While the model is writing `BENVOLIO:`, every character in the name puts 97–100% of its attention on the blank line just before it. That tells the model it is in the middle of a speaker's name, so it keeps writing capitals until the colon and then starts a new line.

None of these were programmed in. They come out of training to predict the next character.

<details>
<summary>All 36 heads (6 layers × 6 heads)</summary>

![Attention in every layer and head](out/attention_grid.png)

Early layers look locally, middle layers latch onto word and line boundaries, and later layers spread their attention over longer distances.
</details>

## TinyStories model

**Tokenizer.** A byte-level BPE tokenizer (`bpe.py`), trained on the first 20 MB of the stories: it starts from the 256 byte values and repeatedly merges the most frequent adjacent pair, 3,840 times, with GPT-2-style pre-splitting so merges never cross word boundaries. That gives 4,096 tokens plus an end-of-story token, at about 3.9 characters per token. `python test_bpe.py` runs its 25 tests.

**Model.** 8 layers, 8 heads, 512-dimensional embeddings, 256-token context (about 1,000 characters), weight tying between the token embedding and the output layer, PyTorch's fused attention.

**Training.** 30,000 steps of 64 × 256 tokens (about 490M tokens, under one pass over the 557M-token training set), AdamW with 500 warmup steps and cosine decay from 6e-4 to 6e-5, gradient clipping at 1.0, bf16 mixed precision. A checkpoint and a sample story are written every 500 steps.

Results are added when the training run finishes.

## Files

| File | What it is |
|---|---|
| `bigram.py` | Baseline model: predicts the next character from the current one only |
| `gpt.py` | Character-level GPT and its training loop (Shakespeare) |
| `sample.py` | Generate text from the Shakespeare checkpoint |
| `plot_loss.py` | Plot the Shakespeare training curve |
| `visualize_attention.py` | Draw what each attention head looks at |
| `bpe.py`, `test_bpe.py` | Byte-level BPE tokenizer and its tests |
| `model.py` | Configurable GPT used for TinyStories |
| `prepare_tinystories.py` | Train the tokenizer and encode the dataset to token files |
| `train_tinystories.py` | Training loop with warmup, cosine decay, bf16, checkpoints and `--resume` |
| `web/` | The interactive page, the JavaScript engine and the scripts that export data for it |

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Shakespeare
python bigram.py                         # baseline, a few seconds
python gpt.py                            # train the GPT (~30 min on an M5)
python sample.py --prompt "JULIET:" --temp 0.8

# TinyStories (download TinyStoriesV2-GPT4-train.txt and -valid.txt into data/tinystories/ first)
python prepare_tinystories.py            # tokenizer + token files, ~2 min
python train_tinystories.py              # ~7 h on an M5; Ctrl+C is safe, then --resume
```

## Data

- [TinyShakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare), about 1.1 million characters, in `data/shakespeare.txt`.
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan and Li, 2023), version 2: 2.7 million short stories in simple English, about 2.2 GB of text.
