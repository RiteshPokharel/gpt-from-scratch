// Record the built-in examples: run both models on each starting line with the
// same JavaScript engine the page uses live, and save the results.
//
//     node web/record_demo.mjs        (after python web/export_browser_model.py)
//
// Writes web/data/recordings.json.

import { readFileSync, writeFileSync } from "fs";
import { makeRunner, mulberry32 } from "./js/runner.js";

export const PROMPTS = [
  "Once upon a time",
  "ROMEO:",
  "The little girl",
  "To be, or not to be",
  "One day, a big bear",
  "KING RICHARD III:",
  "Lily wanted to",
  "O Romeo, Romeo",
];
const LENGTH = { shakespeare: 320, tinystories: 160 };
const VARIANTS = 3; // different stories per example, so "Write another" gives a new one

const dir = new URL("./", import.meta.url);
const read = (p) => readFileSync(new URL(p, dir));
const merges = JSON.parse(read("data/tokenizer.json"));
const out = { prompts: PROMPTS, models: {} };
const EXPLAINER_PROMPTS = {
  tinystories: "Once upon a time, there was a little",
  shakespeare: "JULIET:\nO Romeo, Romeo! wherefore art thou R",
};

function loadRunner(name) {
  const manifest = JSON.parse(read(`models/${name}.json`));
  const bin = Buffer.concat(Array.from({ length: manifest.parts }, (_, k) => read(`models/${name}.bin.${k}`)));
  return makeRunner(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), merges);
}

function explainer() {
  const out = {};
  for (const [name, prompt] of Object.entries(EXPLAINER_PROMPTS)) {
    const { logits, ...analysis } = loadRunner(name).analyse(prompt);
    out[name] = { prompt, analysis };
  }
  return out;
}

// node web/record_demo.mjs --explainer   only refreshes the fixed explainer example
if (process.argv.includes("--explainer")) {
  const existing = JSON.parse(read("data/recordings.json"));
  existing.explainer = explainer();
  writeFileSync(new URL("data/recordings.json", dir), JSON.stringify(existing));
  console.log("explainer examples:", Object.values(EXPLAINER_PROMPTS));
  process.exit(0);
}

for (const name of ["shakespeare", "tinystories"]) {
  const manifest = JSON.parse(read(`models/${name}.json`));
  const runner = loadRunner(name);
  const runs = {};
  PROMPTS.forEach((prompt, i) => {
    const variants = [];
    let plain;
    for (let v = 0; v < VARIANTS; v++) {
      const analysis = runner.analyse(prompt); // resets the model, so every story starts fresh
      const pieces = [], alts = [];
      runner.generate(analysis, LENGTH[name], (p, id, a) => { pieces.push(p); alts.push(a); }, mulberry32(1000 + i * 10 + v));
      const { logits, ...rest } = analysis;
      plain = rest;
      variants.push({ pieces, alts });
    }
    runs[prompt] = { analysis: plain, variants };
  });
  out.models[name] = { step: manifest.step, val_loss: manifest.val_loss, length: LENGTH[name], runs };
  console.log(`${name}: ${PROMPTS.length} examples (checkpoint step ${manifest.step})`);
}
// one fixed example for the "What happens when you press Generate" section
out.explainer = explainer();
writeFileSync(new URL("data/recordings.json", dir), JSON.stringify(out));
console.log(`wrote web/data/recordings.json (${(JSON.stringify(out).length / 1024).toFixed(0)} KB)`);
