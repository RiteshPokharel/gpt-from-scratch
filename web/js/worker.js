// Runs a model in the background so the page stays responsive.
// Messages in:  {type: "load", name}  |  {type: "run", text, n}
// Messages out: progress, ready, analysis, piece, done, error

import { makeRunner } from "./runner.js";

let runner = null;

async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const reader = res.body.getReader();
  const chunks = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    onBytes(value.length);
  }
  return new Blob(chunks).arrayBuffer();
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      const base = new URL("../models/", import.meta.url);
      const manifest = await (await fetch(new URL(`${data.name}.json`, base))).json();
      let loaded = 0;
      // Parts are raw bytes (name.bin.0, ...). Hosts that only serve text get
      // base64 copies instead, listed in manifest.partNames.
      const b64 = manifest.encoding === "base64";
      const names = manifest.partNames ?? Array.from({ length: manifest.parts }, (_, k) => `${data.name}.bin.${k}`);
      const parts = await Promise.all(names.map(async (file) => {
        const buf = await fetchWithProgress(new URL(file, base), (n) => {
          loaded += b64 ? (n * 3) / 4 : n;
          self.postMessage({ type: "progress", loaded: Math.min(loaded, manifest.bytes), total: manifest.bytes });
        });
        if (!b64) return buf;
        const bin = atob(new TextDecoder().decode(buf));
        const out = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
        return out.buffer;
      }));
      const buffer = new Uint8Array(manifest.bytes);
      let off = 0;
      for (const p of parts) { buffer.set(new Uint8Array(p), off); off += p.byteLength; }
      const merges = manifest.tokenizer === "bpe"
        ? await (await fetch(new URL("../data/tokenizer.json", import.meta.url))).json()
        : null;
      runner = makeRunner(manifest, buffer.buffer, merges);
      self.postMessage({ type: "ready", name: data.name });
    } else if (data.type === "run") {
      const analysis = runner.analyse(data.text);
      const { logits, ...plain } = analysis;
      const id = data.id;
      self.postMessage({ type: "analysis", analysis: plain, id });
      let batch = [];
      let last = performance.now();
      runner.generate(analysis, data.n, (piece, tokenId, alts) => {
        batch.push({ p: piece, a: alts });
        const now = performance.now();
        if (now - last > 30) { self.postMessage({ type: "pieces", pieces: batch, id }); batch = []; last = now; }
      });
      if (batch.length) self.postMessage({ type: "pieces", pieces: batch, id });
      self.postMessage({ type: "done", id });
    }
  } catch (err) {
    self.postMessage({ type: "error", message: err.message, id: data.id });
  }
};
