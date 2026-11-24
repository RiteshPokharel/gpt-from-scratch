// Hand-drawn SVG line charts with a crosshair tooltip. No chart library.

const NS = "http://www.w3.org/2000/svg";
const W = 520, H = 280;
const M = { top: 18, right: 52, bottom: 30, left: 40 };

function el(name, attrs = {}, parent) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function niceTicks(min, max, count) {
  const span = max - min;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? mag * 10;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(6));
  return ticks;
}

const fmtStep = (v) => (v >= 1000 ? `${+(v / 1000).toFixed(1)}k` : `${v}`);

/**
 * series: [{ name, color, dash, points: [{x, y}] }]
 * opts: { xMax, yMin, yMax, unit, notes: [{x, text}], offscale: text }
 */
export function lineChart(container, series, opts) {
  container.textContent = "";
  const { xMax, yMin, yMax } = opts;
  const iw = W - M.left - M.right, ih = H - M.top - M.bottom;
  const sx = (x) => M.left + (x / xMax) * iw;
  const sy = (y) => M.top + (1 - (Math.min(y, yMax) - yMin) / (yMax - yMin)) * ih;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.label });

  // grid + axes
  const grid = el("g", { class: "grid" }, svg);
  const axis = el("g", { class: "axis" }, svg);
  for (const t of niceTicks(yMin, yMax, 5)) {
    el("line", { x1: M.left, x2: W - M.right, y1: sy(t), y2: sy(t) }, grid);
    const label = el("text", { x: M.left - 8, y: sy(t) + 4, "text-anchor": "end" }, axis);
    label.textContent = t.toFixed(1);
  }
  for (const t of niceTicks(0, xMax, 5)) {
    const label = el("text", { x: sx(t), y: H - 8, "text-anchor": "middle" }, axis);
    label.textContent = fmtStep(t);
  }

  // annotations
  const ann = el("g", { class: "ann" }, svg);
  for (const n of opts.notes ?? []) {
    el("line", { x1: sx(n.x), x2: sx(n.x), y1: M.top, y2: M.top + ih }, ann);
    const t = el("text", { x: sx(n.x) + (n.left ? -6 : 6), y: M.top + 10, "text-anchor": n.left ? "end" : "start" }, ann);
    t.textContent = n.text;
  }
  if (opts.offscale) {
    const t = el("text", { x: M.left + 6, y: M.top + 10 }, ann);
    t.textContent = opts.offscale;
  }

  // lines + direct labels at the end of each line
  const labels = [];
  for (const s of series) {
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
    el("path", { d, class: "series", stroke: s.color, "stroke-dasharray": s.dash ? "5 4" : "none", opacity: s.dash ? 0.7 : 1 }, svg);
    const last = s.points[s.points.length - 1];
    labels.push({ s, y: sy(last.y), x: sx(last.x) });
  }
  // keep end labels from colliding
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 13) labels[i].y = labels[i - 1].y + 13;
  for (const l of labels) {
    const t = el("text", { x: l.x + 6, y: l.y + 4, class: "lbl", fill: l.s.color }, svg);
    t.textContent = l.s.name;
  }

  // hover layer
  const hair = el("line", { class: "hair", y1: M.top, y2: M.top + ih, visibility: "hidden" }, svg);
  const dots = series.map((s) => el("circle", { r: 4, class: "dot", fill: s.color, visibility: "hidden" }, svg));
  const hit = el("rect", { class: "hit", x: M.left, y: M.top, width: iw, height: ih }, svg);
  container.appendChild(svg);

  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  container.appendChild(tip);

  const xs = series[0].points.map((p) => p.x);
  function show(clientX) {
    const box = svg.getBoundingClientRect();
    const scale = box.width / W;
    const x = ((clientX - box.left) / scale - M.left) / iw * xMax;
    let i = 0;
    for (let k = 1; k < xs.length; k++) if (Math.abs(xs[k] - x) < Math.abs(xs[i] - x)) i = k;
    const px = sx(xs[i]);
    hair.setAttribute("x1", px); hair.setAttribute("x2", px); hair.setAttribute("visibility", "visible");
    series.forEach((s, k) => {
      const p = s.points[i];
      dots[k].setAttribute("cx", sx(p.x)); dots[k].setAttribute("cy", sy(p.y)); dots[k].setAttribute("visibility", "visible");
    });
    tip.innerHTML = `step <b>${xs[i].toLocaleString()}</b><br>` +
      series.map((s) => `${s.name} <b>${s.points[i].y.toFixed(3)}</b>`).join("<br>");
    tip.hidden = false;
    const left = px * scale;
    const tipW = tip.offsetWidth;
    tip.style.left = `${left + 12 + tipW > box.width ? left - tipW - 12 : left + 12}px`;
    tip.style.top = `${M.top * scale}px`;
  }
  function hide() {
    tip.hidden = true;
    hair.setAttribute("visibility", "hidden");
    dots.forEach((d) => d.setAttribute("visibility", "hidden"));
  }
  hit.addEventListener("pointermove", (e) => show(e.clientX));
  hit.addEventListener("pointerdown", (e) => show(e.clientX));
  hit.addEventListener("pointerleave", hide);
}

export function legend(items) {
  const div = document.createElement("div");
  div.className = "legend";
  div.innerHTML = items.map((it) => `<span><i class="${it.dash ? "dash" : ""}" style="color:${it.color}"></i>${it.name}</span>`).join("");
  return div;
}
