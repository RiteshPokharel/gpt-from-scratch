// The GPT forward pass in plain JavaScript, one token at a time with a
// key/value cache. Same maths as model.py / gpt.py:
//   x = token_emb + pos_emb
//   per block: x += proj(attention(LN1(x)));  x += out(GELU(fc(LN2(x))))
//   logits = head(LN_f(x))
// Weight matrices are int8 with one scale per row (see export_browser_model.py).

export class GPT {
  constructor(manifest, buffer) {
    this.m = manifest;
    const { n_embd: C, n_head: H, n_layer: L, block_size: B } = manifest;
    this.C = C; this.H = H; this.L = L; this.B = B; this.hs = C / H;
    this.V = manifest.vocab;
    const t = (name) => {
      const e = manifest.tensors[name];
      if (!e) return null;
      const n = e.shape.reduce((a, b) => a * b, 1);
      return e.dtype === "int8"
        ? { q: new Int8Array(buffer, e.offset, n), s: new Float32Array(buffer, e.scale, e.shape[0]), rows: e.shape[0], cols: e.shape[1] }
        : new Float32Array(buffer, e.offset, n);
    };
    this.tok = t("tok_emb"); this.pos = t("pos_emb");
    this.layers = [];
    for (let l = 0; l < L; l++) {
      this.layers.push({
        ln1w: t(`${l}.ln1.w`), ln1b: t(`${l}.ln1.b`), qkv: t(`${l}.qkv.w`),
        proj: t(`${l}.proj.w`), projb: t(`${l}.proj.b`),
        ln2w: t(`${l}.ln2.w`), ln2b: t(`${l}.ln2.b`),
        fc: t(`${l}.fc.w`), fcb: t(`${l}.fc.b`), out: t(`${l}.out.w`), outb: t(`${l}.out.b`),
      });
    }
    this.lnfw = t("ln_f.w"); this.lnfb = t("ln_f.b");
    this.head = manifest.tied ? this.tok : t("head.w");
    this.headb = t("head.b");
    // scratch buffers, allocated once
    this.x = new Float32Array(C); this.h = new Float32Array(C); this.a = new Float32Array(C);
    this.qkvBuf = new Float32Array(3 * C); this.mlp = new Float32Array(4 * C);
    this.att = new Float32Array(B); this.logits = new Float32Array(this.V);
    this.K = Array.from({ length: L }, () => new Float32Array(B * C));
    this.Vc = Array.from({ length: L }, () => new Float32Array(B * C));
    this.T = 0;
  }

  reset() { this.T = 0; }

  // Feed one token; returns the logits for the next one.
  // If `maps` is given, the attention weights of this position are pushed to maps[l][h].
  step(id, maps) {
    const { C, H, hs, L, x, h, a } = this;
    const t = this.T;
    if (t >= this.B) throw new Error("context full");
    embedRow(x, this.tok, id);
    addRow(x, this.pos, t);
    const scale = 1 / Math.sqrt(hs);
    for (let l = 0; l < L; l++) {
      const ly = this.layers[l];
      layerNorm(h, x, ly.ln1w, ly.ln1b);
      matvec(this.qkvBuf, ly.qkv, h, null);
      const q = this.qkvBuf, K = this.K[l], Vc = this.Vc[l];
      K.set(q.subarray(C, 2 * C), t * C);
      Vc.set(q.subarray(2 * C, 3 * C), t * C);
      a.fill(0);
      for (let hd = 0; hd < H; hd++) {
        const o = hd * hs;
        let max = -Infinity;
        for (let j = 0; j <= t; j++) {
          let s = 0;
          const kb = j * C + o;
          for (let d = 0; d < hs; d++) s += q[o + d] * K[kb + d];
          s *= scale;
          this.att[j] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let j = 0; j <= t; j++) { const e = Math.exp(this.att[j] - max); this.att[j] = e; sum += e; }
        for (let j = 0; j <= t; j++) {
          const w = this.att[j] / sum;
          this.att[j] = w;
          const vb = j * C + o;
          for (let d = 0; d < hs; d++) a[o + d] += w * Vc[vb + d];
        }
        if (maps) maps[l][hd].push(Float32Array.from(this.att.subarray(0, t + 1)));
      }
      matvec(h, ly.proj, a, ly.projb);
      for (let i = 0; i < C; i++) x[i] += h[i];
      layerNorm(h, x, ly.ln2w, ly.ln2b);
      matvec(this.mlp, ly.fc, h, ly.fcb);
      for (let i = 0; i < this.mlp.length; i++) this.mlp[i] = gelu(this.mlp[i]);
      matvec(h, ly.out, this.mlp, ly.outb);
      for (let i = 0; i < C; i++) x[i] += h[i];
    }
    layerNorm(h, x, this.lnfw, this.lnfb);
    matvec(this.logits, this.head, h, this.headb);
    this.T = t + 1;
    return this.logits;
  }
}

function matvec(out, W, x, bias) {
  const { q, s, rows, cols } = W;
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    const base = r * cols;
    for (let c = 0; c < cols; c++) acc += q[base + c] * x[c];
    out[r] = acc * s[r] + (bias ? bias[r] : 0);
  }
}

function embedRow(out, W, r) {
  const { q, s, cols } = W;
  const sc = s[r], base = r * cols;
  for (let c = 0; c < cols; c++) out[c] = q[base + c] * sc;
}

function addRow(out, W, r) {
  const { q, s, cols } = W;
  const sc = s[r], base = r * cols;
  for (let c = 0; c < cols; c++) out[c] += q[base + c] * sc;
}

function layerNorm(out, x, w, b) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) { const d = x[i] - mean; v += d * d; }
  const inv = 1 / Math.sqrt(v / n + 1e-5);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
}

// exact GELU, x * Phi(x), with erf from Abramowitz & Stegun 7.1.26 (error < 1.5e-7)
function gelu(x) {
  const z = x / Math.SQRT2;
  const sign = z < 0 ? -1 : 1;
  const az = Math.abs(z);
  const t = 1 / (1 + 0.3275911 * az);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-az * az);
  return 0.5 * x * (1 + sign * y);
}

export function softmax(logits, temperature = 1) {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const p = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) { p[i] = Math.exp((logits[i] - max) / temperature); sum += p[i]; }
  for (let i = 0; i < p.length; i++) p[i] /= sum;
  return p;
}

export function topK(p, k) {
  const idx = Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, k);
  return idx.map((i) => ({ id: i, p: p[i] }));
}

// Sample with temperature and top-k, as in training.
export function sample(logits, temperature, k, rand = Math.random) {
  const top = topK(softmax(logits, temperature), k);
  const total = top.reduce((s, t) => s + t.p, 0);
  let r = rand() * total;
  for (const t of top) { r -= t.p; if (r <= 0) return t.id; }
  return top[top.length - 1].id;
}
