import { lineChart, legend } from "./charts.js";
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
    explain(recordings.explainer[model].analysis, recordings.explainer[model].prompt, model);
  };
  howSeg.forEach((b) => b.addEventListener("click", () => showExample(b.dataset.model)));
  showExample("tinystories");
  charts(models);
  replayer(replay);
  compare(models, merges);
  statusBar(models);
  navHighlight();
}).catch((err) => {
  $("#mode").textContent = `Could not load the page data (${err.message}). If you opened the file directly, serve the folder instead: python3 -m http.server -d web`;
});

/* ================================================================ the demo */
function demo(recordings, manifests) {
  const state = { model: "tinystories", workers: {}, ready: {}, timer: null, last: null, variant: {}, speed: {}, runId: 0, loading: null };
  document.body.dataset.active = state.model;
  const input = $("#prompt"), go = $("#go"), out = $("#out"), notice = $("#notice");

  // model switch
  const seg = [...document.querySelectorAll(".seg [role=radio]")];
  seg.forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.model === state.model) return;
    state.model = b.dataset.model;
    seg.forEach((x) => x.setAttribute("aria-checked", x === b));
    $(".console").dataset.active = state.model;
    document.body.dataset.active = state.model;
    run(input.value);
  }));
  $(".console").dataset.active = state.model;

  // examples
  recordings.prompts.forEach((p) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.textContent = p.replace(/\n/g, "↵");
    b.addEventListener("click", () => { input.value = p; run(p); });
    $("#examples").append(b);
  });

  $("#ask").addEventListener("submit", (e) => { e.preventDefault(); run(input.value); });
  input.addEventListener("input", labels);

  const recorded = (prompt) => recordings.models[state.model].runs[prompt];

  function labels() {
    const live = state.ready[state.model];
    const size = mb(manifests[state.model].bytes);
    const again = state.last === `${state.model}|${input.value}`;
    const own = !live && !recorded(input.value) && input.value.trim() !== "";
    go.textContent = own ? "Load model and generate" : again ? "Write another" : "Generate";
    // the download is only mentioned once someone types their own line
    notice.hidden = !own || !!state.loading;
    notice.innerHTML = `<span class="tag">[note]</span> To write from your own text, the model runs here in your browser. It loads once (${size}) and stays in this tab only. Nothing is installed.`;
    go.disabled = !!state.loading;
    const speed = state.speed[state.model];
    $("#mode").innerHTML = live
      ? `<span class="tag">[live]</span> ${state.model} · ${size} in this tab${speed ? ` · ${fmt(speed)} tokens/s` : ""}`
      : `<span class="tag">[recorded]</span> saved run`;
    [...$("#examples").children].forEach((c, i) => c.setAttribute("aria-pressed", recordings.prompts[i] === input.value));
  }

  function loadLive(name) {
    if (state.ready[name]) return Promise.resolve();
    if (state.workers[name]) return state.workers[name].loading;
    const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    const bar = $("#loadbar"), fill = $("#loadfill");
    state.loading = name; labels();
    bar.hidden = false; fill.style.width = "0%";
    $("#mode").innerHTML = `<span class="tag">[loading]</span> ${name} · ${mb(0)} of ${mb(manifests[name].bytes)}`;
    const loading = new Promise((resolve, reject) => {
      worker.onmessage = ({ data }) => {
        if (data.type === "progress") {
          fill.style.width = `${(data.loaded / data.total) * 100}%`;
          $("#mode").innerHTML = `<span class="tag">[loading]</span> ${name} · ${mb(data.loaded)} of ${mb(data.total)}`;
        } else if (data.type === "ready") {
          state.ready[name] = true; state.loading = null; bar.hidden = true; labels(); resolve();
        } else if (data.type === "error") {
          state.loading = null; bar.hidden = true; delete state.workers[name];
          $("#mode").textContent = `The model could not load (${data.message}). Check your connection and try again.`;
          go.disabled = false;
          reject(new Error(data.message));
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
    if (state.loading) return;
    if (!prompt.trim()) { input.focus(); return; }
    const id = ++state.runId;
    clearInterval(state.timer);
    const rec = !state.ready[state.model] && recorded(prompt);
    if (!state.ready[state.model] && !rec) {
      try { await loadLive(state.model); } catch { return; }
      if (id !== state.runId) return;
    }
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
    let analysis = null, first = true, t0 = 0;
    const n = recordings.models[state.model].length, model = state.model;
    worker.onmessage = ({ data }) => {
      if (data.id !== id || id !== state.runId) return; // a run that was replaced
      if (data.type === "analysis") analysis = data.analysis;
      else if (data.type === "pieces") {
        if (first) { first = false; t0 = performance.now(); }
        data.pieces.forEach((p) => addPiece(view, p));
      } else if (data.type === "done") {
        const secs = (performance.now() - t0) / 1000;
        if (secs > 0.2) state.speed[model] = Math.round(view.tokens / secs); // measured on this visitor's machine
        finish(view);
      }
      else if (data.type === "error") { $("#mode").textContent = `Something went wrong: ${data.message}`; finish(view); }
    };
    worker.postMessage({ type: "run", text: prompt, n, id });
  }

  labels();
  run(input.value);
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

  // 5. the loop
  const second = a.top[1];
  $("#s5").innerHTML = `
    <div class="loop-row"><span class="lbox">${esc(prompt)}</span><span class="arrow">→</span><span class="lbox model">model</span><span class="arrow">→</span><span class="lbox pick">${esc(vis(firstPiece))}</span></div>
    <div class="loop-row"><span class="lbox">${esc(prompt)}<b>${esc(firstPiece)}</b></span><span class="arrow">↺</span><span class="dim">back to step 1, until the story ends</span></div>`;
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
  slider.value = 0;
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
  const file = (model, name, text) => `<figure class="file" data-model="${model}"><figcaption><span class="swatch"></span>${esc(name)}</figcaption><pre>${esc(text)}</pre></figure>`;

  // 1. data: excerpts + scale (one square = all of Shakespeare)
  const ratio = Math.round(b.dataset_chars / c.dataset_chars);
  const squares = `<div class="scale" role="img" aria-label="One square for Shakespeare's text, ${fmt(ratio)} squares for the stories">
      <div class="scale-row" data-model="shakespeare"><span class="scale-label">Shakespeare</span><span class="dots"><i></i></span></div>
      <div class="scale-row" data-model="tinystories"><span class="scale-label">TinyStories</span><span class="dots">${"<i></i>".repeat(ratio)}</span></div>
    </div>`;
  const r1 = row("What they learned from",
    `Shakespeare's plays (${fmt(c.dataset_lines)} lines) and ${fmt(b.stories / 1e6, 1)} million children's stories written with words a 3 or 4 year old knows. Each square below is the whole Shakespeare file, ${fmt(c.dataset_chars / 1e6, 1)} million characters. The stories fill ${fmt(ratio)} of them.`,
    `<div class="files">${file("shakespeare", c.file, c.excerpt)}${file("tinystories", b.file, b.excerpt)}</div>${squares}`);

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

  $("#cmp").innerHTML = r1 + r2 + r3 + r4 + r5;
}

function statusBar(m) {
  $("#updated").textContent = `Numbers last updated ${m.updated}.`;
}

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
