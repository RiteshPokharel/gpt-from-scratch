// Background rain, Matrix-style: each column spells a word downward, one character per step,
// leaving a fading trail: 1s, 0s and symbols, words the two models read and write, and glowing $.
// Kept faint, about 13 steps per second, and still for visitors who prefer reduced motion.
// At the top of the page it fills the screen; scrolling down fades it out over the content
// column, so further down it only falls in the empty space at the sides.

const WORDS = [
  "once", "upon", "a", "time", "little", "girl", "boy", "dog", "ball", "happy", "said", "mom", "park", "the", "and",
  "ROMEO", "JULIET", "thou", "thee", "doth", "hath", "lord", "love", "king", "O",
  "attn", "softmax", "QKᵀ", "√d", "Σ", "logits", "GPT", "256", "0", "1", "01", "10", "eˣ",
];
const MONEY = ["$", "$$", "$$$"];
const GLYPHS = ["1011", "0110", "1001", "11010", "0101", "1", "0", "10", "01", "111", "000",
  "#", "##", "@", "%", "&", "*", "<>", "{}", "|", "+=", "::", "#1", "0x"];
const CELL = 22;         // px between columns and between characters
const STEP_MS = 75;      // one character per column per step
const GREEN = "57, 255, 122", GOLD = "255, 209, 102", LIGHT = "214, 255, 228";
const DIM = 0.56;        // normal glyphs draw at this share of full strength; the cursor lights them to full
const RADIUS = 150;      // px around the cursor that lights up

export function startRain(canvas) {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const ctx = canvas.getContext("2d");
  let cols = [], w = 0, h = 0, last = 0, left = 0, right = 0;
  const pointer = { x: -1e4, y: -1e4, until: 0 }; // where the light is, and (for taps) until when
  addEventListener("pointermove", (e) => { if (e.pointerType === "mouse") Object.assign(pointer, { x: e.clientX, y: e.clientY, until: Infinity }); }, { passive: true });
  addEventListener("pointerdown", (e) => { if (e.pointerType !== "mouse") Object.assign(pointer, { x: e.clientX, y: e.clientY, until: performance.now() + 900 }); }, { passive: true });
  document.addEventListener("pointerleave", () => { pointer.until = 0; });
  const lightAt = (x, y) => {
    if (performance.now() > pointer.until) return 0;
    const d = Math.hypot(x - pointer.x, y - pointer.y);
    return d < RADIUS ? (1 - d / RADIUS) ** 1.5 : 0;
  };
  const content = document.querySelector(".screen");

  const isSide = (x) => { const cx = x * CELL + CELL / 2; return cx < left - CELL / 2 || cx > right + CELL / 2; };
  const any = (list) => list[Math.floor(Math.random() * list.length)];
  const newDrop = (startHigh, side = false) => {
    const roll = Math.random();
    const money = roll < 0.25;                         // about 1 in 4: glowing $
    const word = money ? any(MONEY) : roll < 0.65 ? any(GLYPHS) : any(WORDS); // ~40% 1s, 0s and symbols
    return {
      word, money, i: 0,
      row: startHigh ? -Math.floor(Math.random() * 40) : -Math.floor(Math.random() * 8),
      gap: side ? 4 + Math.floor(Math.random() * 9) : 6 + Math.floor(Math.random() * 15), // rows of nothing before the next word: the sides a little busier
    };
  };

  function resize() {
    const dpr = Math.min(2, devicePixelRatio || 1);
    w = innerWidth; h = innerHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.font = `500 17px "IBM Plex Mono", ui-monospace, monospace`;
    ctx.textAlign = "center";
    const n = Math.ceil(w / CELL);
    const r = content?.getBoundingClientRect(); // the content column, where the text is
    left = r ? r.left : 0; right = r ? r.right : w;
    cols = Array.from({ length: n }, (_, k) => cols[k] ?? newDrop(true, isSide(k)));
    ctx.fillStyle = "#050a07";
    ctx.fillRect(0, 0, w, h);
  }

  function step() {
    ctx.fillStyle = "rgba(5, 10, 7, 0.16)"; // fade what is already there: the trail
    ctx.fillRect(0, 0, w, h);
    const away = Math.min(1, scrollY / (h * 0.5)); // 0 at the top, 1 after half a screen of scrolling
    if (performance.now() <= pointer.until) { // a faint glow under the cursor; the fade above turns it into a trail
      const g = ctx.createRadialGradient(pointer.x, pointer.y, 0, pointer.x, pointer.y, RADIUS);
      g.addColorStop(0, `rgba(${GREEN}, 0.05)`); g.addColorStop(1, `rgba(${GREEN}, 0)`);
      ctx.fillStyle = g; ctx.fillRect(pointer.x - RADIUS, pointer.y - RADIUS, RADIUS * 2, RADIUS * 2);
    }
    cols.forEach((d, x) => {
      const cx = x * CELL + CELL / 2;
      const over = cx > left - CELL / 2 && cx < right + CELL / 2; // over the text column?
      const fade = over ? 1 - away : 1;
      if (d.row >= 0 && d.i < d.word.length && fade > 0.02) {
        const y = d.row * CELL;
        const lit = lightAt(cx, y);
        if (d.money) { // money stands out: bold, a little bigger, glowing gold
          ctx.save();
          ctx.font = `700 21px "IBM Plex Mono", ui-monospace, monospace`;
          ctx.shadowColor = `rgba(${GOLD}, ${0.9 * fade})`; ctx.shadowBlur = 10 + 14 * lit;
          ctx.fillStyle = `rgba(${GOLD}, ${(DIM + (1 - DIM) * lit) * fade})`;
          ctx.fillText(d.word[d.i], cx, y);
          ctx.restore();
        } else if (lit > 0) { // under the light: brighter, paler, glowing
          ctx.save();
          ctx.shadowColor = `rgba(${GREEN}, ${lit})`; ctx.shadowBlur = 12 * lit;
          ctx.fillStyle = `rgba(${lit > 0.45 ? LIGHT : GREEN}, ${Math.min(1, DIM * 0.85 + lit) * fade})`;
          ctx.fillText(d.word[d.i], cx, y);
          ctx.restore();
        } else {
          ctx.fillStyle = `rgba(${GREEN}, ${(d.i === d.word.length - 1 ? 0.95 : 0.75) * DIM * fade})`;
          ctx.fillText(d.word[d.i], cx, y);
        }
      }
      d.row++;
      if (d.row > 0) d.i = Math.min(d.word.length, d.i + 1);
      if (d.i >= d.word.length && d.gap-- <= 0) {
        const next = newDrop(false, isSide(x));
        next.row = d.row + 1;           // continue down the same column
        if (next.row * CELL > h + CELL * 2) next.row = -Math.floor(Math.random() * 10);
        cols[x] = next;
      }
      if (d.row * CELL > h + CELL * 4) cols[x] = newDrop(false, isSide(x));
    });
  }

  function loop(t) {
    if (t - last >= STEP_MS) { last = t; step(); }
    requestAnimationFrame(loop);
  }

  addEventListener("resize", resize);
  resize();
  requestAnimationFrame(loop);
}
