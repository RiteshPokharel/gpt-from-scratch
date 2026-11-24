// Check the JavaScript engine against PyTorch: node web/js/test_engine.mjs
// Compares the top-10 next-token guesses for the prompts in web/models/reference.json.

import { readFileSync } from "fs";
import { GPT, softmax, topK } from "./gpt.js";

const dir = new URL("../models/", import.meta.url);
const ref = JSON.parse(readFileSync(new URL("reference.json", dir)));
let worst = 0;

for (const name of ["shakespeare", "tinystories"]) {
  const manifest = JSON.parse(readFileSync(new URL(`${name}.json`, dir)));
  const bin = Buffer.concat(Array.from({ length: manifest.parts }, (_, k) => readFileSync(new URL(`${name}.bin.${k}`, dir))));
  const model = new GPT(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
  for (const r of ref[name]) {
    model.reset();
    let logits;
    const t0 = performance.now();
    for (const id of r.ids) logits = model.step(id);
    const ms = (performance.now() - t0) / r.ids.length;
    const p = softmax(logits);
    const ours = topK(p, 10);
    const same = ours.filter((o, i) => o.id === r.top[i]).length;
    const diff = Math.max(...r.top.map((id, i) => Math.abs(p[id] - r.p[i])));
    worst = Math.max(worst, diff);
    console.log(`${name.padEnd(12)} top-1 ${ours[0].id === r.top[0] ? "same" : "DIFF"} | top-10 order ${same}/10 | max prob diff ${(diff * 100).toFixed(2)} pts | ${ms.toFixed(1)} ms/token`);
  }
}
console.log(`worst probability difference: ${(worst * 100).toFixed(2)} percentage points`);
