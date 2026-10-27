// Drawing helpers shared by all tabs of the lecture page: fish glyphs, a 2-D / 3-D canvas plot with equal axes, colour
// scales and small HTML helpers for matrices.

const $ = id => document.getElementById(id);
const MC = "#ffd54f", AC = "#4fc3f7", BC = "#ff8a65", GOOD = "#7bd88f", BAD = "#ff6b6b", MUTED = "#8a8a8a";
const TAU = Math.PI * 2;
const fmt = (x, d = 1) => (Math.abs(x) < 0.5 * 10 ** -d ? 0 : x).toFixed(d).replace("-", "−");
const lerp = (a, b, t) => a + (b - a) * t;

function viridis(t) {
  const s = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  t = Math.min(1, Math.max(0, t)) * 4; const i = Math.min(3, Math.floor(t)), f = t - i;
  return `rgb(${s[i].map((v, j) => Math.round(v + f * (s[i + 1][j] - v)))})`;
}

// ── fish ────────────────────────────────────────────────────────────────────────────────────────────────────────

// A fish facing right: body = ellipse with length L and thickness T (cm), tail of length F (cm); k = pixels per cm.
function drawFish(g, x, y, L, T, F, k, color, alpha = 1, outline = null) {
  const a = L * k / 2, b = T * k / 2, f = F * k, th = b * 0.45 + Math.min(f * 0.3, k * 3.2);
  g.save(); g.globalAlpha *= alpha; g.fillStyle = color;
  g.beginPath(); g.moveTo(x - a * 0.8, y); g.lineTo(x - a - f, y - th); g.lineTo(x - a - f * 0.65, y); g.lineTo(x - a - f, y + th); g.closePath(); g.fill();
  g.beginPath(); g.ellipse(x, y, a, b, 0, 0, TAU); g.fill();
  if (outline) { g.strokeStyle = outline; g.lineWidth = 1.2; g.stroke(); }
  if (b > 3) {                                                       // eye, only where it is big enough to see
    g.fillStyle = "#fff"; g.beginPath(); g.arc(x + a * 0.55, y - b * 0.2, Math.max(1.2, b * 0.2), 0, TAU); g.fill();
    g.fillStyle = "#111"; g.beginPath(); g.arc(x + a * 0.58, y - b * 0.2, Math.max(0.6, b * 0.1), 0, TAU); g.fill();
  }
  g.restore();
}
// pixels per cm that fit a fish of the largest size (body up to 42 cm + tail up to 28 cm) into a w x h box
const FISH_K = (w, h) => Math.min(w / 60, h / 22);
// a small canvas element with one fish drawn on it
function fishCanvas(L, T, F, color, w = 96, h = 56) {
  const c = document.createElement("canvas"), r = window.devicePixelRatio || 1;
  c.width = w * r; c.height = h * r; c.style.width = w + "px"; c.style.height = h + "px";
  const g = c.getContext("2d"); g.scale(r, r);
  drawFish(g, w / 2 + F * FISH_K(w, h) / 2, h / 2, L, T, F, FISH_K(w, h), color);
  return c;
}

// ── plot with equal axes (2-D, or 3-D that can be rotated with the mouse) ─────────────────────────────────────────

class Plot {
  constructor(canvas, dim = 2) {
    this.c = canvas; this.g = canvas.getContext("2d"); this.dim = dim; this.pad = 30;
    this.lo = [0, 0, 0]; this.hi = [1, 1, 1]; this.zoom = 1; this.px = 0; this.py = 0; this.az = -0.6; this.el = 0.32;
    this.resize();
  }
  resize() {
    const r = window.devicePixelRatio || 1, w = this.c.clientWidth, h = this.c.clientHeight;
    if (w && (this.c.width !== Math.round(w * r) || this.c.height !== Math.round(h * r))) { this.c.width = Math.round(w * r); this.c.height = Math.round(h * r); }
    this.g.setTransform(r, 0, 0, r, 0, 0); this.W = w; this.H = h; if (this.ctr) this.refit();
  }
  // visible data box (equal scale on all axes, so that circles stay circles and orthogonal vectors look orthogonal)
  fit(lo, hi) { this.lo = Array.from(lo); this.hi = Array.from(hi); this.ctr = this.lo.map((l, k) => (l + this.hi[k]) / 2); this.resize(); }
  refit() {
    const sp = this.hi.map((h, k) => Math.max(h - this.lo[k], 1e-12 * (1 + Math.abs(h))));
    if (this.dim === 2) this.s0 = Math.min((this.W - 2 * this.pad) / sp[0], (this.H - 2 * this.pad) / sp[1]);
    else this.s0 = (0.95 + 0.2 * Math.min(1, Math.max(0, this.W / this.H - 1))) * Math.min(this.W, this.H) / 2 / (Math.hypot(...sp.slice(0, 3)) / 2);   // wide canvases can take a bigger box
  }
  get s() { return this.s0 * this.zoom; }
  reset() { this.zoom = 1; this.px = this.py = 0; if (this.dim === 3) { this.az = -0.6; this.el = 0.32; } }
  // data -> screen: [x, y, depth]
  P(v) {
    if (this.dim === 2) return [this.W / 2 + this.px + this.s * (v[0] - this.ctr[0]), this.H / 2 + this.py - this.s * (v[1] - this.ctr[1]), 0];
    const x = v[0] - this.ctr[0], y = v[1] - this.ctr[1], z = (v[2] || 0) - this.ctr[2], ca = Math.cos(this.az), sa = Math.sin(this.az);
    const x1 = x * ca + z * sa, z1 = -x * sa + z * ca, ce = Math.cos(this.el), se = Math.sin(this.el);
    const y2 = y * ce - z1 * se, z2 = y * se + z1 * ce;
    return [this.W / 2 + this.px + this.s * x1, this.H / 2 + this.py - this.s * y2, z2];
  }
  // screen -> data (2-D only)
  inv(sx, sy) { return [this.ctr[0] + (sx - this.W / 2 - this.px) / this.s, this.ctr[1] - (sy - this.H / 2 - this.py) / this.s]; }
  clear() { this.resize(); this.g.clearRect(0, 0, this.W, this.H); }

  line(a, b, style = "#888", w = 1, dash = null, alpha = 1) {
    const g = this.g, p = this.P(a), q = this.P(b);
    g.save(); g.globalAlpha = alpha; g.strokeStyle = style; g.lineWidth = w; if (dash) g.setLineDash(dash);
    g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke(); g.restore();
  }
  poly(pts, fill, stroke = null, alpha = 1, w = 1) {
    const g = this.g; g.save(); g.globalAlpha = alpha; g.beginPath();
    pts.forEach((v, i) => { const p = this.P(v); i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]); }); g.closePath();
    if (fill) { g.fillStyle = fill; g.fill(); } if (stroke) { g.strokeStyle = stroke; g.lineWidth = w; g.stroke(); } g.restore();
  }
  arrow(a, b, style = "#fff", w = 2, head = 9, alpha = 1) {
    const g = this.g, p = this.P(a), q = this.P(b), ang = Math.atan2(q[1] - p[1], q[0] - p[0]);
    if (Math.hypot(q[0] - p[0], q[1] - p[1]) < 2) return;
    g.save(); g.globalAlpha = alpha; g.strokeStyle = g.fillStyle = style; g.lineWidth = w;
    g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0] - Math.cos(ang) * head * 0.6, q[1] - Math.sin(ang) * head * 0.6); g.stroke();
    g.beginPath(); g.moveTo(q[0], q[1]); g.lineTo(q[0] - head * Math.cos(ang - 0.4), q[1] - head * Math.sin(ang - 0.4));
    g.lineTo(q[0] - head * Math.cos(ang + 0.4), q[1] - head * Math.sin(ang + 0.4)); g.closePath(); g.fill(); g.restore();
  }
  dot(v, r, fill, stroke = null, alpha = 1, lw = 2) {
    const g = this.g, p = this.P(v); g.save(); g.globalAlpha = alpha; g.beginPath(); g.arc(p[0], p[1], r, 0, TAU);
    if (fill) { g.fillStyle = fill; g.fill(); } if (stroke) { g.strokeStyle = stroke; g.lineWidth = lw; g.stroke(); } g.restore();
  }
  text(v, s, color = "#ddd", dx = 0, dy = 0, size = 12, align = "left", bold = false) {
    const g = this.g, p = this.P(v); g.save(); g.font = `${bold ? "600 " : ""}${size}px system-ui, sans-serif`; g.fillStyle = color;
    g.textAlign = align; g.textBaseline = "middle"; g.strokeStyle = "#1a1a1ad0"; g.lineWidth = 3; g.strokeText(s, p[0] + dx, p[1] + dy);
    g.fillText(s, p[0] + dx, p[1] + dy); g.restore();
  }
  // a closed curve in a plane given by two direction vectors (for circles / ellipses / spheres in any dimension)
  ellipse(c, u, v, ru, rv, style, w = 1.5, alpha = 1, fill = null) {
    const pts = []; for (let i = 0; i < 72; i++) { const t = TAU * i / 72; pts.push(c.map((x, k) => x + u[k] * ru * Math.cos(t) + v[k] * rv * Math.sin(t))); }
    this.poly(pts, fill, style, alpha, w);
  }

  // ── axes ──
  niceStep(span, target = 6) {
    const raw = span / target, m = 10 ** Math.floor(Math.log10(raw)), f = raw / m;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * m;
  }
  // 2-D: tick labels on the left and bottom edge of the canvas, faint grid, optional axes through the origin
  axes2D(xl, yl, { origin = false, grid = true, sci = false } = {}) {
    const g = this.g, [x0, y1] = this.inv(0, 0), [x1, y0] = this.inv(this.W, this.H);
    const sx = this.niceStep(x1 - x0), sy = this.niceStep(y1 - y0), lab = v => sci ? (v === 0 ? "0" : v.toExponential(0).replace("e-", "e−")) : fmt(v, Math.max(0, -Math.floor(Math.log10(Math.min(sx, sy)))));
    g.save(); g.font = "10px system-ui, sans-serif"; g.fillStyle = "#777"; g.strokeStyle = "#ffffff12"; g.lineWidth = 1;
    for (let v0 = Math.ceil(x0 / sx - 1e-9); v0 * sx <= x1; v0++) { const v = Math.abs(v0 * sx) < sx * 1e-6 ? 0 : v0 * sx, p = this.P([v, 0]);
      if (grid) { g.beginPath(); g.moveTo(p[0], 0); g.lineTo(p[0], this.H); g.stroke(); }
      g.textAlign = "center"; g.textBaseline = "bottom"; g.fillText(lab(v), p[0], this.H - 3); }
    for (let v0 = Math.ceil(y0 / sy - 1e-9); v0 * sy <= y1; v0++) { const v = Math.abs(v0 * sy) < sy * 1e-6 ? 0 : v0 * sy, p = this.P([0, v]);
      if (grid) { g.beginPath(); g.moveTo(0, p[1]); g.lineTo(this.W, p[1]); g.stroke(); }
      g.textAlign = "left"; g.textBaseline = "middle"; g.fillText(lab(v), 4, p[1] - 0); }
    if (origin) { const o = this.P([0, 0]); g.strokeStyle = "#aaa"; g.lineWidth = 1.3;
      g.beginPath(); g.moveTo(0, o[1]); g.lineTo(this.W, o[1]); g.moveTo(o[0], 0); g.lineTo(o[0], this.H); g.stroke(); }
    g.fillStyle = "#aaa"; g.font = "11px system-ui, sans-serif"; g.textAlign = "right"; g.textBaseline = "bottom"; g.fillText(xl, this.W - 6, this.H - 16);
    g.textAlign = "right"; g.textBaseline = "top"; g.fillText(yl, this.W - 6, 6); g.restore();
  }
  // 3-D: wire-frame box around the data range with its labelled axes; with `origin` also the three axes through 0
  axes3D(labels, { origin = false } = {}) {
    const [l, h] = [this.lo, this.hi], c = (i, j, k) => [i ? h[0] : l[0], j ? h[1] : l[1], k ? h[2] : l[2]];
    const edges = []; for (const a of [0, 1]) for (const b of [0, 1]) { edges.push([c(0, a, b), c(1, a, b)], [c(a, 0, b), c(a, 1, b)], [c(a, b, 0), c(a, b, 1)]); }
    for (const [p, q] of edges) this.line(p, q, "#fff", 1, null, 0.14);
    const floor = []; for (let i = 1; i < 4; i++) { const t = i / 4;
      floor.push([[lerp(l[0], h[0], t), l[1], l[2]], [lerp(l[0], h[0], t), l[1], h[2]]], [[l[0], l[1], lerp(l[2], h[2], t)], [h[0], l[1], lerp(l[2], h[2], t)]]); }
    for (const [p, q] of floor) this.line(p, q, "#fff", 1, null, 0.06);
    if (origin) labels.forEach((_, k) => { const a = [0, 0, 0], b = [0, 0, 0]; a[k] = l[k]; b[k] = h[k]; this.line(a, b, "#aaa", 1.3); });
    const mid = k => { const v = [l[0], l[1], l[2]]; v[k] = (l[k] + h[k]) / 2; return v; };
    this.text(mid(0), labels[0], "#aaa", 0, 16, 11, "center"); this.text(mid(1), labels[1], "#aaa", -16, 0, 11, "right"); this.text(mid(2), labels[2], "#aaa", 14, 12, 11, "left");
    this.text(c(0, 0, 0), fmt(l[0], 0), "#777", 0, 10, 10, "center"); this.text(c(1, 0, 0), fmt(h[0], 0), "#777", 0, 10, 10, "center");
    this.text(c(0, 0, 0), fmt(l[1], 0), "#777", -8, 0, 10, "right"); this.text(c(0, 1, 0), fmt(h[1], 0), "#777", -8, 0, 10, "right");
  }

  // mouse: drag = pan (2-D) or rotate (3-D), wheel = zoom, double-click = reset; `onMove(x, y)` / `onClick(x, y)` get canvas px
  bind({ onMove, onClick, onLeave, onChange, rotate = null, pan = true, zoomable = true }) {
    const c = this.c; let drag = null, moved = false;
    c.addEventListener("mousedown", e => { drag = [e.offsetX, e.offsetY]; moved = false; });
    window.addEventListener("mouseup", () => { drag = null; });
    c.addEventListener("mousemove", e => {
      if (drag) {
        const dx = e.offsetX - drag[0], dy = e.offsetY - drag[1];
        if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
        if (rotate ?? this.dim === 3) { this.az += dx * 0.008; this.el = Math.max(-1.5, Math.min(1.5, this.el + dy * 0.008)); }
        else if (pan) { this.px += dx; this.py += dy; }
        drag = [e.offsetX, e.offsetY]; onChange && onChange(); return;
      }
      onMove && onMove(e.offsetX, e.offsetY);
    });
    c.addEventListener("mouseleave", () => onLeave && onLeave());
    c.addEventListener("click", e => { if (!moved) onClick && onClick(e.offsetX, e.offsetY); });
    c.addEventListener("dblclick", () => { this.reset(); onChange && onChange(); });
    if (zoomable) c.addEventListener("wheel", e => {
      e.preventDefault(); const f = Math.exp(-e.deltaY * 0.0015);
      this.px = e.offsetX - this.W / 2 - f * (e.offsetX - this.W / 2 - this.px); this.py = e.offsetY - this.H / 2 - f * (e.offsetY - this.H / 2 - this.py);
      this.zoom *= f; onChange && onChange();
    }, { passive: false });
  }
}

// range [lo, hi] (per axis) of a set of points with a relative margin, optionally forced to include the origin
function rangeOf(X, n, d, { margin = 0.08, zero = false } = {}) {
  const lo = Array(d).fill(Infinity), hi = Array(d).fill(-Infinity);
  for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) { lo[k] = Math.min(lo[k], X[i * d + k]); hi[k] = Math.max(hi[k], X[i * d + k]); }
  if (zero) for (let k = 0; k < d; k++) { lo[k] = Math.min(lo[k], 0); hi[k] = Math.max(hi[k], 0); }
  for (let k = 0; k < d; k++) { const m = (hi[k] - lo[k]) * margin || 1; lo[k] -= m; hi[k] += m; }
  return { lo, hi };
}

// ── HTML helpers ────────────────────────────────────────────────────────────────────────────────────────────────

// a matrix as a bracketed table: rows = array of arrays of strings; opts.head = column headings, opts.rowHead = row labels
function matrixHTML(rows, { head = null, rowHead = null, cls = "", hl = null, cellCls = null, foot = null } = {}) {
  const th = head ? `<tr class="mh">${rowHead ? "<td></td>" : ""}${head.map(h => `<td>${h}</td>`).join("")}</tr>` : "";
  const body = rows.map((r, i) => `<tr data-r="${i}" class="${hl === i ? "hl" : ""}">${rowHead ? `<td class="rh">${rowHead[i]}</td>` : ""}` +
    r.map((v, j) => `<td data-r="${i}" data-c="${j}" class="${cellCls ? cellCls(i, j) || "" : ""}">${v}</td>`).join("") + "</tr>").join("");
  const ft = foot ? foot.map(r => `<tr class="mf">${rowHead ? "<td></td>" : ""}${r.map(v => `<td>${v}</td>`).join("")}</tr>`).join("") : "";
  return `<table class="mat ${cls}">${th}${body}${ft}</table>`;
}
const vecT = v => `(${Array.from(v, x => fmt(x, 1)).join(", ")})`;

// ── small charts (bars, lines) and a heat map, drawn on plain canvases ──────────────────────────────────────────────

function chartCtx(cv) {
  const r = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight; cv.width = Math.round(w * r); cv.height = Math.round(h * r);
  const q = cv.getContext("2d"); q.setTransform(r, 0, 0, r, 0, 0); q.clearRect(0, 0, w, h); return { q, w, h };
}
// grouped vertical bars: bars = [{label, v, color, v2, color2}]
function chartBars(cv, bars, { title = "", legend = null, fmtv = v => fmt(v, 3), top = null } = {}) {
  const { q, w, h } = chartCtx(cv), L = 34, B = 30, T = 22, mx = top ?? Math.max(1e-12, ...bars.map(b => Math.max(b.v, b.v2 ?? 0)));
  const bw = (w - L - 8) / Math.max(1, bars.length);
  q.font = "10px system-ui, sans-serif"; q.fillStyle = "#888"; q.textAlign = "left"; q.fillText(title, 8, 13);
  if (legend) { q.textAlign = "right"; let x = w - 8; for (const [c, t] of [...legend].reverse()) { q.fillStyle = "#aaa"; q.fillText(t, x, 13); x -= q.measureText(t).width + 4; q.fillStyle = c; q.fillRect(x - 8, 6, 8, 8); x -= 16; } }
  q.strokeStyle = "#444"; q.beginPath(); q.moveTo(L, h - B); q.lineTo(w - 4, h - B); q.stroke();
  q.fillStyle = "#777"; q.textAlign = "right"; q.fillText(fmtv(mx), L - 3, T + 4); q.fillText("0", L - 3, h - B);
  bars.forEach((b, k) => {
    const x = L + k * bw, two = b.v2 !== undefined, ww = Math.max(2, bw * (two ? 0.38 : 0.7)), y = v => h - B - (h - B - T) * Math.max(0, v) / mx;
    q.fillStyle = b.color; q.fillRect(x + bw * 0.1, y(b.v), ww, h - B - y(b.v));
    if (two) { q.fillStyle = b.color2; q.fillRect(x + bw * 0.1 + ww, y(b.v2), ww, h - B - y(b.v2)); }
    q.fillStyle = "#999"; q.textAlign = "center"; q.fillText(b.label, x + bw * 0.5, h - B + 11); if (b.label2) q.fillText(b.label2, x + bw * 0.5, h - B + 22);
  });
}
// lines: spec = {series:[{x,y,color,width,dash}], xmin, xmax, ymin, ymax, title, xlabel, vlines:[{x,color,label}], points:[{x,y,color}], shades:[{x0,x1,color,label}], logy}
function chartLines(cv, spec) {
  const { q, w, h } = chartCtx(cv), L = 40, B = 24, T = 20, R = 10, lg = spec.logy;
  const f = v => lg ? Math.log10(Math.max(v, 1e-12)) : v;
  let [x0, x1] = [spec.xmin, spec.xmax], [y0, y1] = [f(spec.ymin), f(spec.ymax)];
  const X = x => L + (w - L - R) * (x - x0) / ((x1 - x0) || 1), Y = y => h - B - (h - B - T) * (f(y) - y0) / ((y1 - y0) || 1);
  q.font = "10px system-ui, sans-serif"; q.fillStyle = "#888"; q.textAlign = "left"; q.fillText(spec.title || "", 8, 13);
  for (const s of spec.shades || []) { q.fillStyle = s.color; q.fillRect(X(s.x0), T, X(s.x1) - X(s.x0), h - B - T); q.fillStyle = "#aaa"; q.textAlign = "left"; q.fillText(s.label || "", X(s.x0) + 4, T + 10); }
  q.strokeStyle = "#444"; q.beginPath(); q.moveTo(L, T); q.lineTo(L, h - B); q.lineTo(w - R, h - B); q.stroke();
  q.fillStyle = "#777"; q.textAlign = "right"; q.fillText(lg ? spec.ymax.toPrecision(2) : fmt(spec.ymax, spec.ydec ?? 2), L - 3, T + 4); q.fillText(lg ? spec.ymin.toPrecision(2) : fmt(spec.ymin, spec.ydec ?? 2), L - 3, h - B);
  q.textAlign = "center"; q.fillText(fmt(x0, spec.xdec ?? 0), L, h - B + 12); q.fillText(fmt(x1, spec.xdec ?? 0), w - R, h - B + 12); if (spec.xlabel) q.fillText(spec.xlabel, (L + w - R) / 2, h - 4);
  for (const v of spec.vlines || []) { q.strokeStyle = v.color; q.setLineDash([3, 3]); q.beginPath(); q.moveTo(X(v.x), T); q.lineTo(X(v.x), h - B); q.stroke(); q.setLineDash([]); if (v.label) { q.fillStyle = v.color; q.textAlign = "left"; q.fillText(v.label, X(v.x) + 3, T + 10); } }
  for (const s of spec.series) {
    q.strokeStyle = s.color; q.lineWidth = s.width || 2; q.setLineDash(s.dash || []); q.beginPath();
    s.x.forEach((x, i) => { const px = X(x), py = Math.max(T - 2, Math.min(h - B + 2, Y(s.y[i]))); i ? q.lineTo(px, py) : q.moveTo(px, py); }); q.stroke(); q.setLineDash([]); q.lineWidth = 1;
  }
  for (const p of spec.points || []) { q.fillStyle = p.color; q.beginPath(); q.arc(X(p.x), Y(p.y), p.r || 3.5, 0, TAU); q.fill(); }
}
// heat map of a symmetric n x n matrix with rows / columns in `order`; a cross marks `focus`
function drawHeat(cv, M, n, order, focus, { gamma = 0.5 } = {}) {
  const { q, w, h } = chartCtx(cv), side = Math.min(w, h) - 40, x0 = (w - side) / 2, y0 = 32, img = document.createElement("canvas"); img.width = img.height = n;
  const ig = img.getContext("2d"), id = ig.createImageData(n, n); let mx = 0;
  for (let i = 0; i < n * n; i++) mx = Math.max(mx, M[i]);
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
    const v = Math.pow(M[order[a] * n + order[b]] / mx, gamma), c = viridis(v).match(/\d+/g).map(Number), k = 4 * (a * n + b);
    id.data[k] = c[0]; id.data[k + 1] = c[1]; id.data[k + 2] = c[2]; id.data[k + 3] = 255;
  }
  ig.putImageData(id, 0, 0); q.imageSmoothingEnabled = false; q.drawImage(img, x0, y0, side, side);
  const f = order.indexOf(focus); q.strokeStyle = MC; q.lineWidth = 1.5; q.strokeRect(x0, y0 + side * f / n, side, side / n); q.strokeRect(x0 + side * f / n, y0, side / n, side);
}
