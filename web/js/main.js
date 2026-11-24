import { lineChart, legend } from "./charts.js";
import { startRain } from "./rain.js";

startRain(document.getElementById("rain"));
import { makeBPE, showWhitespace } from "./tokenizer.js";

const $ = (s, r = document) => r.querySelector(s);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const css = getComputedStyle(document.documentElement);
const COLOR = { shakespeare: css.getPropertyValue("--red").trim(), tinystories: css.getPropertyValue("--phos").trim() };
const NAME = { shakespeare: "Shakespeare", tinystories: "TinyStories" };
const fmt = (n, d = 0) => n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const mb = (bytes) => `${fmt(bytes / 1e6, 1)} MB`;
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const vis = (s) => (s === null ? "end of story" : showWhitespace(s));

const getJSON = (path) => fetch(path).then((r) => {
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
});

// Always open at the top: browsers otherwise jump back to the last scroll position or to a #section in the address.
if ("scrollRestoration" in history) history.scrollRestoration = "manual";
if (location.hash) history.replaceState(null, "", location.pathname + location.search);
scrollTo(0, 0);
addEventListener("pageshow", () => scrollTo(0, 0));
// the top-bar links scroll to their section without putting #section in the address
document.querySelectorAll('.tmux a[href^="#"]').forEach((a) => a.addEventListener("click", (e) => {
  e.preventDefault();
  const behavior = matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
  const href = a.getAttribute("href");
  if (href === "#top") return scrollTo({ top: 0, behavior }); // the name on the left goes back to the start
  document.querySelector(href)?.scrollIntoView({ behavior });
}));

Promise.all([
  getJSON("data/models.json"), getJSON("data/recordings.json"), getJSON("data/replay.json"),
  getJSON("models/shakespeare.json"), getJSON("models/tinystories.json"), getJSON("data/tokenizer.json"),
]).then(([models, recordings, replay, mShak, mTiny, merges]) => {
  demo(recordings, { shakespeare: mShak, tinystories: mTiny });
  // the explainer: one fixed example per model, picked with its own switch
  const howSeg = [...document.querySelectorAll("#how-seg [role=radio]")];
  const showExample = (model) => {
    howSeg.forEach((b) => b.setAttribute("aria-checked", b.dataset.model === model));
    $("#how").dataset.active = model;
    moveThumb($("#how-seg"));
    const steps = $(".steps");
    steps.classList.remove("switching"); void steps.offsetWidth; steps.classList.add("switching");
    explain(recordings.explainer[model].analysis, recordings.explainer[model].prompt, model);
  };
  howSeg.forEach((b) => b.addEventListener("click", () => showExample(b.dataset.model)));
  showExample("tinystories");
  charts(models);
  replayer(replay);
  compare(models, merges);
  navHighlight();
  scrollTo(0, 0);
}).catch((err) => {
  $("#mode").textContent = `Could not load the page data (${err.message}). If you opened the file directly, serve the folder instead: python3 -m http.server -d web`;
});

/* ================================================================ the demo */
function demo(recordings, manifests) {
  // no model is selected until the visitor picks one
  const state = { model: null, workers: {}, ready: {}, timer: null, last: null, variant: {}, runId: 0, loading: null };
  const input = $("#prompt"), go = $("#go"), out = $("#out");

  // choosing a model: the label switches from "Choose your model:" to "Your chosen model:"
  const seg = [...document.querySelectorAll("#try .seg [role=radio]")];
  function selectModel(name) {
    if (name === state.model) return;
    state.model = name;
    seg.forEach((x) => x.setAttribute("aria-checked", x.dataset.model === name));
    showExamples(name);
    $(".console").dataset.active = name;
    document.body.dataset.active = name;
    moveThumb($("#try .seg"));
    if ($(".console").classList.contains("unpicked")) {
      const label = $("#pick-label");
      label.textContent = "Your chosen model:"; // same length as "Choose your model:", so nothing shifts
      label.classList.add("swap");
      $(".console").classList.remove("unpicked");
      $(".console").classList.add("revealed");
      $("#console-body").inert = false;
    }
    // nothing runs (or downloads) until Generate: switching just stops the current text and clears it
    state.runId++; clearInterval(state.timer); if (!state.loading) meter.stop();
    out.textContent = ""; $("#count").textContent = "";
    input.focus();
  }
  seg.forEach((b) => b.addEventListener("click", () => selectModel(b.dataset.model)));

  // three examples that suit the chosen model; tapping one fills the box, Generate runs it
  const EXAMPLES = {
    tinystories: ["Once upon a time", "The little girl", "One day, a big bear"],
    shakespeare: ["ROMEO:", "To be, or not to be", "KING RICHARD III:"],
  };
  function showExamples(name) {
    $("#examples").replaceChildren(...EXAMPLES[name].map((p) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "chip"; b.textContent = p;
      b.addEventListener("click", () => { input.value = p; labels(); input.focus(); });
      return b;
    }));
  }

  $("#ask").addEventListener("submit", (e) => { e.preventDefault(); run(input.value); });
  input.addEventListener("input", labels);

  const recorded = (prompt) => recordings.models[state.model].runs[prompt];

  function labels() {
    if (!state.model) return;
    go.disabled = !!state.loading; // own text: Generate loads the model (with the progress bar)
    [...$("#examples").children].forEach((c) => c.setAttribute("aria-pressed", c.textContent === input.value));
  }

  // ---------------------------------------------------------------- the progress bar under the text box
  // Two ways to move:
  //  - "curve" (nothing to download): races toward ~97% (fast, then slower), 100% when finish() is called
  //  - "download": follows the real download, scaled so a complete download is 90%;
  //    then finish() counts 91, 92 ... 100 over 3 seconds
  const meter = (() => {
    const el = $("#loading"), line = $("#load-bar");
    let n = 20, shown = 0, real = 0, finished = false, frame = 0, started = 0, tau = 1, onDone = null, gen = 0;
    let mode = "curve", tailStart = 0, words = null, wordAt = -1;
    const TAIL_MS = 3000;
    const cells = () => {
      const probe = document.createElement("span");
      probe.textContent = "█".repeat(20);
      line.replaceChildren(probe);
      const w = probe.getBoundingClientRect().width / 20 || 8;
      return Math.max(10, Math.floor((el.clientWidth * 0.5) / w) - 2); // about half the box wide
    };
    const draw = (f) => {
      const pct = Math.round(f * 100), on = Math.round(f * n);
      line.innerHTML = `[${"█".repeat(on)}<span class="off">${"░".repeat(n - on)}</span>]`;
      $("#load-pct").textContent = `${pct}%`;
      el.setAttribute("aria-valuenow", pct);
    };
    const tick = () => {
      const now = performance.now();
      if (words && !finished) { // fixed-length runs share the 3 seconds; downloads keep cycling until done
        const k = mode === "download"
          ? Math.floor((now - started) / 1500) % words.length // 1.5 s per word while downloading
          : Math.min(words.length - 1, Math.floor((now - started) / (WAIT_MS / words.length)));
        if (k !== wordAt) { wordAt = k; $("#load-text").textContent = words[k]; }
      }
      if (mode === "download") {
        if (finished) shown = Math.max(shown, 0.9 + 0.1 * Math.min(1, (now - tailStart) / TAIL_MS)); // 91 ... 100
        else shown = Math.max(shown, shown + (0.9 * real - shown) * 0.2); // glides after the real download
      } else {
        const curve = 0.97 * (1 - Math.exp(-(now - started) / 1000 / tau));
        shown = finished ? Math.min(1, shown + 0.06) : Math.max(shown, Math.min(0.97, Math.max(curve, real)));
      }
      draw(shown);
      if (finished && shown >= 1) { const f = onDone; onDone = null; f?.(); return; }
      frame = requestAnimationFrame(tick);
    };
    return {
      start(label, opts = {}) {
        gen++; cancelAnimationFrame(frame);
        tau = opts.tau || 1.1; mode = opts.mode || "curve"; words = opts.words || null; wordAt = -1;
        real = 0; shown = 0; finished = false; onDone = null; started = performance.now();
        $("#load-text").textContent = label;
        el.hidden = false; n = cells(); draw(0);
        if (!reduceMotion) frame = requestAnimationFrame(tick);
      },
      set(f) { real = f; if (reduceMotion) draw(mode === "download" ? 0.9 * f : f); },
      // complete no sooner than minMs after start, run up to 100%, hold a moment, then hide
      finish(minMs = 0) {
        const mine = gen;
        return new Promise((resolve) => {
          setTimeout(() => {
            if (mine !== gen) return; // a newer start took over the bar
            finished = true; real = 1; tailStart = performance.now();
            if (mode === "download") shown = Math.max(shown, 0.9);
            const end = () => setTimeout(() => { if (mine === gen) el.hidden = true; resolve(); }, 300);
            if (reduceMotion) { draw(1); end(); } else onDone = end;
          }, mode === "download" ? 0 : Math.max(0, minMs - (performance.now() - started)));
        });
      },
      stop() { gen++; cancelAnimationFrame(frame); el.hidden = true; },
    };
  })();
  const WAIT_MS = 3000; // every Generate takes about this long before text appears

  // what the bar says while it runs: things a model really does (plus a little fun), never the same twice in a row
  const WORDS = {
    all: ["Tokenizing…", "Embedding…", "Attending…", "Softmaxing…", "Sampling…", "Consulting the weights…",
      "Multiplying matrices…", "Normalizing layers…", "Warming up the attention heads…", "Brewing logits…",
      "Untangling vectors…", "Pondering probabilities…", "Masking the future…", "Adding residuals…",
      "Querying keys…", "Valuing values…", "Projecting heads…", "Stacking layers…", "Dot-producting…",
      "Scaling by √64…", "Exponentiating…", "Dividing by the total…", "Rolling weighted dice…",
      "Looking back 256 tokens…", "Encoding positions…", "Activating GELUs…", "Crunching floats…",
      "Reading the context…", "Weighing every word…", "Feeding forward…", "Merging heads…",
      "Not peeking ahead…", "Predicting the next token…", "Warming up your CPU…", "Shuffling bytes…",
      "Discombobulating…", "Cogitating…", "Ruminating…", "Percolating…", "Noodling…", "Mulling it over…",
      "Contemplating…", "Deliberating…", "Synthesizing…", "Calibrating…", "Recombobulating…", "Musing…",
      "Waking the transformer…", "Unpacking the weights…", "Dequantizing int8…", "Allocating Float32Arrays…",
      "Spinning up a web worker…", "Warming the KV cache…", "Decoding bytes…", "Merging byte pairs…",
      "Building the vocabulary…", "Unrolling the context…", "Counting attention heads…", "Stretching the embeddings…",
      "Lining up the layers…", "Polishing probabilities…", "Herding tensors…", "Tuning the temperature…",
      "Consulting the loss curve…", "Remembering 2.7 million stories…", "Folding matrices…", "Aligning dimensions…",
      "Summoning the softmax…", "Greasing the gradients…", "Charging the residuals…", "Defragmenting thoughts…"],
    tinystories: ["Once-upon-a-time-ing…", "Sharpening crayons…", "Finding a moral…", "Naming the dog…",
      "Rummaging through 27 million numbers…", "Petting the cat…", "Building a sandcastle…",
      "Finding the lost ball…", "Saying sorry nicely…", "Sharing the toys…", "Going to the park…",
      "Asking Mom…", "Tying shoelaces…", "Counting to ten…", "Planting a little seed…",
      "Finding a happy ending…", "Feeding the ducks…", "Drawing with chalk…", "Looking for the cat…",
      "Baking cookies with Grandma…", "Climbing the big tree…", "Learning to be kind…", "Hugging the teddy bear…"],
    shakespeare: ["Thou-ing and thee-ing…", "Sharpening the quill…", "Consulting the Bard…", "Iambic-pentametering…",
      "Rummaging through 10.8 million numbers…", "Wherefore-ing…", "Exiting, pursued by a bear…",
      "Rhyming couplets…", "Doth-ing and hath-ing…", "Summoning a soliloquy…", "Plotting a tragedy…",
      "Calling the Nurse…", "Forsooth-ing…", "Drawing daggers…", "Raising the curtain…",
      "Brooding on the balcony…", "Composing a sonnet…", "Waiting in the wings…", "Bowing to the King…",
      "Speaking in verse…", "Writing stage directions…", "Swearing by the moon…", "Rehearsing Act III…"],
  };
  const PER_RUN = 3; // one a second across the 3-second start
  const recent = new Set(); // words shown lately, skipped next time
  const shuffle = (a) => a.map((x) => [Math.random(), x]).sort((p, q) => p[0] - q[0]).map((x) => x[1]);
  function statusWords(model, count = PER_RUN) {
    const pick = (pool, k) => {
      const fresh = shuffle(pool.filter((w) => !recent.has(w)));
      return (fresh.length >= k ? fresh : shuffle(pool)).slice(0, k);
    };
    const nOwn = Math.max(1, Math.round(count * (0.2 + Math.random() * 0.2))); // a few that only fit this model
    const chosen = shuffle([...pick(WORDS[model], nOwn), ...pick(WORDS.all, count - nOwn)]);
    chosen.forEach((w) => recent.add(w));
    while (recent.size > 24) recent.delete(recent.values().next().value); // remember the last ~5 runs
    return chosen;
  }

  function loadLive(name) {
    if (state.ready[name]) return Promise.resolve();
    if (state.workers[name]) return state.workers[name].loading;
    const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    const total = manifests[name].bytes;
    state.loading = name; labels();
    // a download can take a while: go through every word once before any repeats
    const words = shuffle([...WORDS.all, ...WORDS[name]]);
    meter.start(words[0], { mode: "download", words });
    const loading = new Promise((resolve, reject) => {
      let last = performance.now(), done = false;
      const fail = (message) => {
        if (done) return;
        done = true; clearInterval(watchdog); worker.terminate(); meter.stop();
        state.loading = null; delete state.workers[name];
        $("#mode").textContent = `The model could not load (${message}). Try again, or use another browser.`;
        go.disabled = false;
        reject(new Error(message));
      };
      // never sit at 0% forever: give up if nothing arrives for 20 seconds
      const watchdog = setInterval(() => { if (performance.now() - last > 20000) fail("no data for 20 seconds"); }, 2000);
      worker.onerror = (e) => { e.preventDefault?.(); fail(e.message || "the loader could not start in this browser"); };
      worker.onmessageerror = () => fail("the model data could not be read");
      worker.onmessage = ({ data }) => {
        last = performance.now();
        if (data.type === "progress") {
          meter.set(data.loaded / total);
        } else if (data.type === "ready") {
          done = true; clearInterval(watchdog);
          meter.finish(WAIT_MS).then(() => {
            state.ready[name] = true; state.loading = null; labels(); resolve();
          });
        } else if (data.type === "error") {
          fail(data.message);
        }
      };
    });
    worker.postMessage({ type: "load", name });
    state.workers[name] = { worker, loading };
    return loading;
  }

  // ---------------------------------------------------------------- output box
  function startOutput(prompt) {
    clearInterval(state.timer);
    out.textContent = "";
    const p = document.createElement("span");
    p.className = "p"; p.textContent = prompt;
    const cursor = document.createElement("span");
    cursor.className = "cursor"; cursor.setAttribute("aria-hidden", "true");
    out.append(p, cursor);
    return { cursor, tokens: 0, chars: 0 };
  }
  function addPiece(view, piece) {
    const s = document.createElement("span");
    if (piece === null) { s.className = "t eot"; s.textContent = "end of story"; }
    else { s.className = "t"; s.textContent = piece; view.chars += Array.from(piece).length; }
    view.cursor.before(s);
    view.tokens++;
    $("#count").textContent = `${view.tokens} tokens · ${view.chars} characters`;
  }
  function finish(view) {
    view.cursor.remove();
    labels();
  }

  // ---------------------------------------------------------------- run
  // Every run gets an id; starting a new run (or switching model) makes the old one stop.
  async function run(prompt) {
    if (!state.model || state.loading) return;
    if (!prompt.trim()) { input.focus(); return; }
    const id = ++state.runId;
    clearInterval(state.timer);
    $("#mode").textContent = ""; // only ever shows a problem; clear it for the new run
    out.textContent = ""; $("#count").textContent = ""; // the old text goes at once, even while the model loads
    const rec = !state.ready[state.model] && recorded(prompt);
    if (!state.ready[state.model] && !rec) {
      try { await loadLive(state.model); } catch { return; } // the download itself shows the bar
    } else {
      const words = statusWords(state.model);
      meter.start(words[0], { tau: 1.1, words });
      await meter.finish(WAIT_MS);
    }
    if (id !== state.runId) return;
    state.last = `${state.model}|${prompt}`;
    labels();
    const view = startOutput(prompt);
    if (rec) {
      const key = state.last;
      state.variant[key] = key in state.variant ? (state.variant[key] + 1) % rec.variants.length : 0;
      const pieces = rec.variants[state.variant[key]];
      if (reduceMotion) { pieces.forEach((p) => addPiece(view, p)); finish(view); return; }
      let i = 0;
      const per = state.model === "shakespeare" ? 2 : 1; // roughly the live speed of each model
      state.timer = setInterval(() => {
        if (id !== state.runId) { clearInterval(state.timer); return; }
        for (let k = 0; k < per && i < pieces.length; k++) addPiece(view, pieces[i++]);
        if (i >= pieces.length) { clearInterval(state.timer); finish(view); }
      }, 28);
      return;
    }
    const { worker } = state.workers[state.model];
    const n = recordings.models[state.model].length;
    worker.onmessage = ({ data }) => {
      if (data.id !== id || id !== state.runId) return; // a run that was replaced
      if (data.type === "pieces") data.pieces.forEach((p) => addPiece(view, p));
      else if (data.type === "done") finish(view);
      else if (data.type === "error") { $("#mode").textContent = `Something went wrong: ${data.message}`; finish(view); }
    };
    worker.postMessage({ type: "run", text: prompt, n, id });
  }

  labels();
}

/* ================================================================ the five steps */
// One fixed example per model (recorded by record_demo.mjs), the same for every visitor.
function explain(a, prompt, model) {
  const isChar = model === "shakespeare";
  const firstPiece = a.top[0].text;
  const shown = prompt.replace(/\n/g, " ↵ ");
  const last = a.tokens[a.tokens.length - 1];
  $("#how-intro").textContent = isChar
    ? `Here is what the Shakespeare model does with the line "${shown}", step by step, to choose the next letter.`
    : `Here is what the TinyStories model does with the starting line "${shown}", step by step, to choose the next word.`;

  // 1. tokens
  $("#s1-text").textContent = isChar
    ? `This model cuts text into single characters, and knows only the 65 that appear in Shakespeare. Spaces and line breaks count too, so this line is ${a.tokens.length} tokens.`
    : `This model cuts text into common chunks it learned from the stories, 4,097 in all. This line is ${a.tokens.length} tokens.`;
  $("#s1").innerHTML = a.tokens.map((t, i) =>
    `<span class="tcard"><span class="tt">${esc(vis(t))}</span><span class="tid">${a.ids[i]}</span></span>`).join("");

  // 2. embeddings
  const rows = a.emb.slice(-8), toks = a.tokens.slice(-8);
  const max = Math.max(...rows.flat().map(Math.abs)) || 1;
  $("#s2-text").textContent = `Each token's ID is swapped for a list of ${a.embDims} numbers the model learned in training, plus a few that encode its position. Below are the first 24 numbers for ${rows.length < a.tokens.length ? "the last 8 tokens" : "each token"}. ${isChar ? "Red" : "Green"} is positive, blue is negative.`;
  $("#s2").innerHTML = `<div class="emb">${rows.map((r, i) => `
    <span class="emb-label">${esc(vis(toks[i]))}</span>
    <span class="emb-row">${r.map((v) => `<i style="background:${v >= 0 ? `rgba(var(--accent-rgb),${(Math.abs(v) / max).toFixed(2)})` : `rgba(110,170,255,${(Math.abs(v) / max).toFixed(2)})`}"></i>`).join("")}<span class="more">…</span></span>`).join("")}</div>`;

  // 3. attention from the last token, one row per layer
  const T = a.tokens.length, from = Math.max(0, T - 14);
  $("#s3-text").textContent = `To choose what comes next, the last token "${vis(last)}" looks back at the tokens before it. This is called attention. Each row is one of the model's ${a.look.length} layers. Brighter means it looked there more.`;
  $("#s3").innerHTML = `<div class="att" style="--cols:${T - from}">
    <span></span>${a.tokens.slice(from).map((t) => `<span class="att-col">${esc(vis(t))}</span>`).join("")}
    ${a.look.map((row, l) => `<span class="att-row">layer ${l + 1}</span>${row.slice(from).map((w) =>
      `<i title="${(w * 100).toFixed(0)}%" style="opacity:${Math.max(0.04, Math.sqrt(w)).toFixed(2)}"></i>`).join("")}`).join("")}
  </div>`;

  // 4. top ten
  const picked = firstPiece === null ? "<end of story>" : firstPiece;
  const pmax = a.top[0].p;
  $("#s4").innerHTML = `<div class="bars">${a.top.map((t) => `
    <div class="bar${t.text === picked ? " picked" : ""}">
      <span class="bar-tok">${esc(vis(t.text))}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${((t.p / pmax) * 100).toFixed(1)}%"></span></span>
      <span class="bar-pct">${t.p >= 0.001 ? (t.p * 100).toFixed(1) : "<0.1"}%</span>
    </div>`).join("")}</div>`;
  $("#s4-text").textContent = `The model gives every token in its vocabulary a chance of coming next. These are its ten favourites for what comes after "${shown}". The highlighted one is what it picked.`;

  // the math under steps 3 and 4, filled with this run's real numbers
  const heads = a.heads ?? (isChar ? 6 : 8);
  const dk = a.embDims / heads;
  $("#m3-dk").innerHTML = `Each head's vectors have ${dk} numbers, so scores are divided by √${dk} = ${Math.sqrt(dk)} to keep them from growing too big.`;
  $("#m3-heads").innerHTML = `The ${NAME[model]} model runs this ${heads} times side by side (its "heads") in each of its ${a.look.length} layers: ${heads * a.look.length} times for every token it writes. The grid above shows each layer's average over its heads.`;

  const vocab = a.vocab ?? (isChar ? 65 : 4097);
  $("#m4-sigma").textContent = `The sum over all ${fmt(vocab)} tokens it knows. Dividing by it makes the chances add up to 100%.`;
  const sci = (lnx) => { // e^(lnx) written as m × 10^k
    const log10 = lnx / Math.LN10, k = Math.floor(log10), mant = 10 ** (log10 - k);
    return k === 0 ? mant.toFixed(2) : `${mant.toFixed(2)} × 10<sup>${k}</sup>`;
  };
  const rowsSm = a.top.slice(0, 5);
  if (rowsSm[0].z !== undefined) {
    const lnTotal = rowsSm[0].z - Math.log(rowsSm[0].p); // Σ e^z, recovered from the top token's score and chance
    $("#m4-tab").innerHTML = `<thead><tr><th>token</th><th>raw score z</th><th></th><th>e<sup>z</sup></th><th></th><th>chance</th></tr></thead><tbody>${rowsSm.map((t) => `
      <tr><td class="tok">${esc(vis(t.text))}</td><td>${t.z.toFixed(2)}</td><td class="op">→</td><td>${sci(t.z)}</td><td class="op">÷ ${sci(lnTotal)} =</td><td class="res">${(t.p * 100).toFixed(1)}%</td></tr>`).join("")}</tbody>`;
    $("#m4-note").innerHTML = `${sci(lnTotal)} is the sum of e<sup>z</sup> over all ${fmt(vocab)} tokens. The other ${fmt(vocab - 5)} tokens share what is left: ${(100 - rowsSm.reduce((x, t) => x + t.p * 100, 0)).toFixed(1)}%. Real implementations subtract the biggest score first so these numbers never overflow; the result is the same.`;
  }

  // 5. the loop: only the end of the line (what matters for the next token), line breaks shown as ↵
  const flat = prompt.replace(/\n/g, " ↵ ");
  const tail = flat.length > 24 ? `…${flat.slice(-22).trimStart()}` : flat;
  const second = a.top[1];
  $("#s5").innerHTML = `
    <div class="loop-row"><span class="lbox">${esc(tail)}</span><span class="arrow">→</span><span class="lbox model">model</span><span class="arrow">→</span><span class="lbox pick">${esc(vis(firstPiece))}</span></div>
    <div class="loop-row"><span class="lbox">${esc(tail)}<b>${esc(firstPiece)}</b></span><span class="arrow">↺</span><span class="dim">back to step 1, ${isChar ? "one letter at a time" : "until the story ends"}</span></div>`;
  const pct = (p) => (p * 100 >= 1 ? `${Math.round(p * 100)}%` : `${(p * 100).toFixed(1)}%`);
  $("#s5-text").textContent = `It picks a token at random, weighted by those chances. Here it picked "${firstPiece.trim() || vis(firstPiece)}", but "${second.text.trim() || vis(second.text)}" would still win ${pct(second.p)} of the time, which is why every run writes something different. Then it adds the token to the text and starts again from step 1, once for every token it writes.`;
}

/* ================================================================ training */
function charts(m) {
  const pts = (rows, key) => rows.map((r) => ({ x: r.step, y: r[key] }));
  const c = m.char, b = m.bpe;
  const charBox = $("#chart-char .plot");
  lineChart(charBox, [
    { name: "training", color: COLOR.shakespeare, dash: true, points: pts(c.loss, "train") },
    { name: "unseen", color: COLOR.shakespeare, points: pts(c.loss, "val") },
  ], {
    xMax: c.steps_run, yMin: 0.8, yMax: 4.4, label: "Shakespeare model loss by training step",
    notes: [{ x: c.best_step, text: `best: ${fmt(c.best_val, 2)}`, left: true }],
  });
  charBox.after(legend([{ name: "training text", color: COLOR.shakespeare, dash: true }, { name: "unseen text", color: COLOR.shakespeare }]));

  const rows = b.loss.filter((r) => r.step > 0);
  const bpeBox = $("#chart-bpe .plot");
  lineChart(bpeBox, [
    { name: "training", color: COLOR.tinystories, dash: true, points: pts(rows, "train") },
    { name: "unseen", color: COLOR.tinystories, points: pts(rows, "val") },
  ], {
    xMax: b.steps_total, yMin: 1, yMax: 3, label: "TinyStories model loss by training step",
    offscale: `at step 0 the loss was ${fmt(b.loss[0].val, 1)}, off the top`,
    notes: b.done ? [] : [{ x: b.steps_run, text: "still training" }],
  });
  bpeBox.after(legend([{ name: "training text", color: COLOR.tinystories, dash: true }, { name: "unseen text", color: COLOR.tinystories }]));
  $("#bpe-chart-note").textContent = b.done
    ? `Trained for ${fmt(b.steps_total)} steps. The two lines stay together, so it never started memorising: it saw most stories only once.`
    : `Still training: step ${fmt(b.steps_run)} of ${fmt(b.steps_total)}. The two lines stay together, so it isn't memorising.`;
}

function replayer(replay) {
  const slider = $("#replay-slider");
  slider.max = replay.length - 1;
  slider.value = 0; // starts at step 0: random noise
  const show = () => {
    const r = replay[+slider.value];
    $("#replay-step").textContent = fmt(r.step);
    $("#replay-val").textContent = fmt(r.val, 2);
    slider.setAttribute("aria-valuetext", `step ${r.step}, loss ${r.val.toFixed(2)}`);
    $("#replay-out").textContent = r.text;
  };
  slider.addEventListener("input", show);
  show();
}

function compare(m, merges) {
  const c = m.char, b = m.bpe;
  const row = (title, text, vis) => `
    <div class="cmp-row">
      <div class="cmp-text"><h3>${title}</h3><p>${text}</p></div>
      <div class="cmp-vis">${vis}</div>
    </div>`;
  const file = (model, label, text) => `<figure class="file" data-model="${model}"><figcaption><span class="swatch"></span>${esc(label)}</figcaption><pre>${esc(text)}</pre></figure>`;

  // 1. data: excerpts + scale (one square = all of Shakespeare)
  const ratio = Math.round(b.dataset_chars / c.dataset_chars);
  const squares = `<div class="scale" role="img" aria-label="One square for Shakespeare's text, ${fmt(ratio)} squares for the stories">
      <p class="scale-key">How much text each one read: <i class="key-dot"></i> 1 square = all of Shakespeare (${fmt(c.dataset_chars / 1e6, 1)} million characters)</p>
      <div class="scale-row" data-model="shakespeare"><span class="scale-label">Shakespeare</span><span class="dots"><i></i></span><span class="scale-n">1 square</span></div>
      <div class="scale-row" data-model="tinystories"><span class="scale-label">TinyStories</span><span class="dots">${"<i></i>".repeat(ratio)}</span><span class="scale-n">${fmt(ratio)} squares</span></div>
    </div>`;
  const r1 = row("What they learned from",
    `The Shakespeare model read Shakespeare's plays (${fmt(c.dataset_lines)} lines). The TinyStories model read ${fmt(b.stories / 1e6, 1)} million short children's stories, written with words a 3 or 4 year old knows. The two boxes show a real piece of each. Below, you can see how much more text the stories are.`,
    `<div class="files">${file("shakespeare", "A real piece of what the Shakespeare model read", c.excerpt)}${file("tinystories", "A real piece of what the TinyStories model read", b.excerpt)}</div>${squares}`);

  // 2. the same sentence, cut both ways
  const sentence = "Once upon a time, there was a little dog named Max.";
  const bpe = makeBPE(merges);
  const charToks = Array.from(sentence);
  const bpeToks = bpe.encode(sentence).map((id) => bpe.piece(id) ?? "?");
  const boxes = (toks) => toks.map((t) => `<span class="tb">${esc(showWhitespace(t))}</span>`).join("");
  const r2 = row("How they read",
    `A model sees text as a row of tokens. The Shakespeare model uses single letters, so this sentence is ${charToks.length} tokens. The TinyStories model uses word pieces from its own tokenizer, so the same sentence is only ${bpeToks.length}.`,
    `<div class="split" data-model="shakespeare"><span class="split-n">${charToks.length}</span><div class="tbs">${boxes(charToks)}</div></div>
     <div class="split" data-model="tinystories"><span class="split-n">${bpeToks.length}</span><div class="tbs">${boxes(bpeToks)}</div></div>`);

  // 3. size: the stack of layers, drawn to scale
  const stack = (model, layers, width, params, maxW) => `
    <div class="stack" data-model="${model}">
      <div class="layers">${Array.from({ length: layers }, (_, i) => `<span style="width:${(width / maxW) * 100}%">layer ${i + 1}</span>`).join("")}</div>
      <p><b>${fmt(params / 1e6, 1)}M</b> parameters · ${layers} layers · ${width} numbers wide</p>
    </div>`;
  const r3 = row("How big they are",
    `Each model is a stack of identical layers. The TinyStories model has more layers, and each one is wider, so it holds ${fmt(b.params / c.params, 1)}× as many learned numbers (parameters).`,
    `<div class="stacks">${stack("shakespeare", c.layers, c.embd, c.params, b.embd)}${stack("tinystories", b.layers, b.embd, b.params, b.embd)}</div>`);

  // 4. how much text each can see at once, on a real story
  const t = b.window_text;
  const r4 = row("How much they can see at once",
    `Both models see the last 256 tokens. For the Shakespeare model that is 256 letters; for the TinyStories model it is about ${fmt(b.context_chars)} letters, since each token is a whole word piece. Highlighted: how much of this story each one can take in at once.`,
    `<div class="bars2">
       <div class="b2" data-model="shakespeare"><span class="b2-fill" style="width:${(c.context_chars / t.length) * 100}%"></span><span class="b2-val">${c.context_chars} characters</span></div>
       <div class="b2" data-model="tinystories"><span class="b2-fill" style="width:${(b.context_chars / t.length) * 100}%"></span><span class="b2-val">≈ ${fmt(b.context_chars)} characters</span></div>
     </div>
     <p class="window"><span class="w-both">${esc(t.slice(0, c.context_chars))}</span><span class="w-bpe">${esc(t.slice(c.context_chars, b.context_chars))}</span><span class="w-none">${esc(t.slice(b.context_chars))}</span></p>`);

  // 5. training time, to scale
  const charH = c.train_minutes / 60, bpeH = b.done ? b.elapsed_h : b.elapsed_h + b.eta_h;
  const r5 = row("How long they trained",
    `Both trained on the same laptop GPU. The Shakespeare model read its small text about ${fmt(c.epochs, 0)} times over and started memorising it after 27 minutes. The TinyStories model ${b.done ? "trained" : "is training"} for about ${fmt(bpeH, 1)} hours and sees most stories only once.`,
    `<div class="bars2">
       <div class="b2" data-model="shakespeare"><span class="b2-fill" style="width:${(charH / bpeH) * 100}%"></span><span class="b2-val">${fmt(c.train_minutes, 0)} minutes</span></div>
       <div class="b2" data-model="tinystories"><span class="b2-fill${b.done ? "" : " partial"}" style="width:100%"><span class="done" style="width:${(b.elapsed_h / bpeH) * 100}%"></span></span><span class="b2-val">≈ ${fmt(bpeH, 1)} hours${b.done ? "" : `, ${fmt(b.elapsed_h, 1)} so far`}</span></div>
     </div>`);

  // 6. shrinking the models so they can run on a visitor's computer
  const full = { shakespeare: (c.params * 4) / 1e6, tinystories: (b.params * 4) / 1e6 };
  const small = { shakespeare: m.sizes?.shakespeare ?? 10.9, tinystories: m.sizes?.tinystories ?? 27.7 };
  const shrink = (model, name) => `
    <div class="shrink" data-model="${model}">
      <span class="scale-label">${name}</span>
      <div class="shrink-bars">
        <div class="b2"><span class="b2-fill partial" style="width:100%"></span><span class="b2-val">${fmt(full[model], 0)} MB after training</span></div>
        <div class="b2"><span class="b2-fill" style="width:${(small[model] / full[model]) * 100}%"></span><span class="b2-val">${fmt(small[model], 1)} MB in your browser</span></div>
      </div>
    </div>`;
  const r6 = row("How they run in your browser",
    `After training, the TinyStories model is ${fmt(full.tinystories, 0)} MB. That's too big to ask anyone to download. So every number in it is stored with less detail, 8 bits instead of 32. That shrinks it to ${fmt(small.tinystories, 1)} MB (Shakespeare: ${fmt(small.shakespeare, 1)} MB), and the writing comes out the same. A small JavaScript program then runs it right here in your browser. Tested against the original, it picked the same next word in every test.`,
    shrink("tinystories", "TinyStories") + shrink("shakespeare", "Shakespeare"));

  $("#cmp").innerHTML = r1 + r2 + r3 + r4 + r5 + r6;
}


// The coloured thumb behind a model switch: slides to the chosen button and takes its colour.
function moveThumb(seg, flash = true) {
  let thumb = seg.querySelector(".seg-thumb");
  if (!thumb) {
    thumb = document.createElement("span");
    thumb.className = "seg-thumb";
    seg.prepend(thumb);
  }
  const on = seg.querySelector('[aria-checked="true"]');
  if (!on) return;
  const first = !thumb.classList.contains("on");
  if (first) thumb.style.transition = "none"; // appear in place, then slide from there on
  thumb.dataset.model = on.dataset.model;
  // follows the button in both directions: side by side on wide screens, stacked on phones
  thumb.style.width = `${on.offsetWidth}px`;
  thumb.style.height = `${on.offsetHeight}px`;
  thumb.style.transform = `translate(${on.offsetLeft}px, ${on.offsetTop}px)`;
  thumb.classList.add("on");
  if (first) { void thumb.offsetWidth; thumb.style.transition = ""; }
  if (flash && !first) { thumb.classList.remove("flash"); void thumb.offsetWidth; thumb.classList.add("flash"); } // lights up as it lands
}
addEventListener("resize", () => document.querySelectorAll(".seg").forEach((g) => moveThumb(g, false)));
document.fonts?.ready.then(() => document.querySelectorAll(".seg").forEach((g) => moveThumb(g, false))); // widths change once the fonts load

// Arrow keys move between the buttons of a model switch, like a radio group.
document.addEventListener("keydown", (e) => {
  const btn = e.target.closest?.(".seg [role=radio]");
  if (!btn || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
  const all = [...btn.parentElement.querySelectorAll("[role=radio]")];
  const next = all[(all.indexOf(btn) + (e.key === "ArrowLeft" || e.key === "ArrowUp" ? all.length - 1 : 1)) % all.length];
  next.focus(); next.click(); e.preventDefault();
});

function navHighlight() {
  const links = [...document.querySelectorAll(".tmux-windows a")];
  const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      links.forEach((a) => a.removeAttribute("aria-current"));
      byId.get(e.target.id)?.setAttribute("aria-current", "true");
    }
  }, { rootMargin: "-40% 0px -55% 0px" });
  byId.forEach((_, id) => { const s = document.getElementById(id); if (s) io.observe(s); });
}
