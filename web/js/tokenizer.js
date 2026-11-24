// Both tokenizers, in the browser.
// The BPE encoder is a direct port of bpe.py: same pre-split pattern, same
// rule of applying the earliest-learned merge first.

const SPLIT = / ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+/g;
const utf8 = new TextEncoder();
const strict = new TextDecoder("utf-8", { fatal: true });

export function makeBPE(merges) {
  // merges[i] = [a, b] means tokens a and b were merged into token 256 + i
  const rank = new Map();
  const vocab = [];
  for (let i = 0; i < 256; i++) vocab.push(Uint8Array.of(i));
  merges.forEach(([a, b], i) => {
    const id = 256 + i;
    rank.set(a * 65536 + b, id);
    const bytes = new Uint8Array(vocab[a].length + vocab[b].length);
    bytes.set(vocab[a]);
    bytes.set(vocab[b], vocab[a].length);
    vocab.push(bytes);
  });

  function encodeChunk(chunk) {
    let ids = Array.from(utf8.encode(chunk));
    while (ids.length > 1) {
      let best = Infinity, at = -1;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = rank.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && r < best) { best = r; at = i; }
      }
      if (at < 0) break;
      const pairA = ids[at], pairB = ids[at + 1];
      const merged = [];
      for (let i = 0; i < ids.length; i++) {
        if (i < ids.length - 1 && ids[i] === pairA && ids[i + 1] === pairB) { merged.push(best); i++; }
        else merged.push(ids[i]);
      }
      ids = merged;
    }
    return ids;
  }

  return {
    size: vocab.length,
    encode(text) {
      const out = [];
      for (const m of text.matchAll(SPLIT)) out.push(...encodeChunk(m[0]));
      return out;
    },
    // a token's text, or null when its bytes are only part of a character
    piece(id) {
      try { return strict.decode(vocab[id]); } catch { return null; }
    },
    bytes(id) { return vocab[id]; },
  };
}

export function makeCharTokenizer(chars) {
  const ids = new Map(Array.from(chars).map((c, i) => [c, i]));
  return {
    size: ids.size,
    // one entry per character: its id, or -1 if the vocabulary doesn't have it
    encode(text) { return Array.from(text).map((c) => ({ c, id: ids.has(c) ? ids.get(c) : -1 })); },
  };
}

// How a token is drawn: whitespace made visible but quiet.
export function showWhitespace(s) {
  return s.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "→");
}
