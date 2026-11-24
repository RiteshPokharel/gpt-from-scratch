// Text in, text out: tokenizing, one analysis pass for the explainer, and
// sampling. Used by the browser worker (live mode) and by record_demo.mjs
// (the recorded examples), so both run exactly the same code.

import { GPT, softmax, topK, sample } from "./gpt.js";
import { makeBPE } from "./tokenizer.js";

const EMB_DIMS = 24; // how many embedding numbers the explainer draws per token

export function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeRunner(manifest, buffer, merges) {
  const model = new GPT(manifest, buffer);
  const isChar = manifest.tokenizer === "char";
  const chars = isChar ? Array.from(manifest.chars) : null;
  const stoi = isChar ? new Map(chars.map((c, i) => [c, i])) : null;
  const bpe = isChar ? null : makeBPE(merges);
  const eot = manifest.eot;
  const settings = isChar ? { temperature: 0.8, topK: manifest.vocab } : { temperature: 0.8, topK: 50 };

  const encode = (text) => (isChar ? Array.from(text).filter((c) => stoi.has(c)).map((c) => stoi.get(c)) : bpe.encode(text));
  const label = (id) => {
    if (id === eot) return "<end of story>";
    if (isChar) return chars[id];
    return bpe.piece(id) ?? `‹${[...bpe.bytes(id)].map((b) => b.toString(16).padStart(2, "0")).join(" ")}›`;
  };

  function embedding(id, pos) {
    const out = [];
    const { tok, pos: P } = model;
    for (let c = 0; c < EMB_DIMS; c++) {
      out.push(tok.q[id * tok.cols + c] * tok.s[id] + P.q[pos * P.cols + c] * P.s[pos]);
    }
    return out;
  }

  // Feed the prompt and describe what the model did with it.
  function analyse(text) {
    let ids = encode(text);
    if (!ids.length) ids = encode(isChar ? "\n" : " ");
    ids = ids.slice(-Math.floor(model.B / 2));
    model.reset();
    const maps = Array.from({ length: model.L }, () => Array.from({ length: model.H }, () => []));
    let logits;
    ids.forEach((id) => { logits = model.step(id, maps); });
    const T = ids.length;
    // attention from the last prompt token, averaged over the heads of each layer
    const look = maps.map((heads) => {
      const row = new Array(T).fill(0);
      for (const h of heads) h[T - 1].forEach((w, j) => { row[j] += w / heads.length; });
      return row.map((v) => +v.toFixed(4));
    });
    // p: probability; z: the raw score (logit) it came from, for the softmax example on the page
    const top = topK(softmax(logits), 10).map((t) => ({ id: t.id, text: label(t.id), p: +t.p.toFixed(4), z: +logits[t.id].toFixed(3) }));
    return {
      ids,
      tokens: ids.map(label),
      dropped: Array.from(text).length && isChar ? Array.from(text).filter((c) => !stoi.has(c)).length : 0,
      emb: ids.map((id, i) => embedding(id, i).map((v) => +v.toFixed(3))),
      embDims: model.C,
      heads: model.H,
      vocab: model.V,
      look,
      top,
      logits: Float32Array.from(logits),
    };
  }

  // Continue after analyse(); calls onPiece(text | null for end-of-story, id) for each token.
  function generate(analysis, n, onPiece, rand = Math.random) {
    const history = analysis.ids.slice();
    let logits = analysis.logits;
    let pending = [];
    for (let i = 0; i < n; i++) {
      const id = sample(logits, settings.temperature, settings.topK, rand);
      history.push(id);
      if (id === eot) { onPiece(null, id); pending = []; }
      else if (isChar) onPiece(chars[id], id);
      else {
        pending.push(...bpe.bytes(id));
        try {
          const s = new TextDecoder("utf-8", { fatal: true }).decode(Uint8Array.from(pending));
          pending = [];
          onPiece(s, id);
        } catch { /* part of a multi-byte character: wait for the rest */ }
      }
      if (model.T >= model.B) {
        // context full: keep the most recent half and rebuild the cache from it
        const keep = history.slice(-Math.floor(model.B / 2));
        model.reset();
        keep.forEach((t, k) => { if (k < keep.length - 1) model.step(t); });
        logits = model.step(keep[keep.length - 1]);
      } else {
        logits = model.step(id);
      }
    }
  }

  return { manifest, encode, label, analyse, generate, isChar };
}
