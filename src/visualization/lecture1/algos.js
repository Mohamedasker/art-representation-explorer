// Numerical building blocks of the lecture-1 page: PCA, k-NN, t-SNE and UMAP, written out step by step so that the
// page can show every intermediate quantity.  No dependencies; checked against numpy / scikit-learn / umap-learn (see the README).
// Data matrices are flat Float64Arrays, row-major: row i = X[i*d .. i*d+d-1].

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function gaussian(rng) {
  let u = 0; while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

// ── linear algebra ──────────────────────────────────────────────────────────────────────────────────────────────

// Eigen-decomposition of a symmetric n x n matrix (cyclic Jacobi rotations): values in descending order and the
// matching unit eigenvectors as the COLUMNS of the flat n x n matrix `vectors`.
function eigSym(A, n) {
  const a = Float64Array.from(A), v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = a[p * n + q];
      if (Math.abs(apq) < 1e-300) continue;
      const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
      const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = a[k * n + p], akq = a[k * n + q];
        a[k * n + p] = c * akp - s * akq; a[k * n + q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = a[p * n + k], aqk = a[q * n + k];
        a[p * n + k] = c * apk - s * aqk; a[q * n + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = v[k * n + p], vkq = v[k * n + q];
        v[k * n + p] = c * vkp - s * vkq; v[k * n + q] = s * vkp + c * vkq;
      }
    }
  }
  const order = [...Array(n).keys()].sort((i, j) => a[j * n + j] - a[i * n + i]);
  const values = Float64Array.from(order, i => a[i * n + i]), vectors = new Float64Array(n * n);
  order.forEach((src, dst) => { for (let k = 0; k < n; k++) vectors[k * n + dst] = v[k * n + src]; });
  return { values, vectors };
}

const matmul = (A, B, n, m, p) => {            // (n x m) · (m x p)
  const C = new Float64Array(n * p);
  for (let i = 0; i < n; i++) for (let k = 0; k < m; k++) { const a = A[i * m + k]; for (let j = 0; j < p; j++) C[i * p + j] += a * B[k * p + j]; }
  return C;
};

// ── PCA, step by step ───────────────────────────────────────────────────────────────────────────────────────────

function columnMean(X, n, d) {
  const mu = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) mu[k] += X[i * d + k] / n;
  return mu;
}
function centered(X, n, d, mu) {
  const Xc = new Float64Array(n * d);
  for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) Xc[i * d + k] = X[i * d + k] - mu[k];
  return Xc;
}
function covariance(Xc, n, d) {                // (1/(n-1)) Xcᵀ Xc
  const C = new Float64Array(d * d);
  for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a * d + b] += Xc[i * d + a] * Xc[i * d + b] / (n - 1);
  return C;
}
// All the quantities of the PCA recipe.  Signs of the eigenvectors are arbitrary, so they are fixed: the largest loading
// of PC1 and PC2 is positive, and the basis is right-handed (a rotation, not a mirror) so that it can be animated.
function pcaSteps(X, n, d) {
  const mu = columnMean(X, n, d), Xc = centered(X, n, d, mu), C = covariance(Xc, n, d), { values, vectors } = eigSym(C, d);
  const W = Float64Array.from(vectors);
  const flip = j => { for (let k = 0; k < d; k++) W[k * d + j] = -W[k * d + j]; };
  for (let j = 0; j < Math.min(2, d); j++) {
    let big = 0; for (let k = 0; k < d; k++) if (Math.abs(W[k * d + j]) > Math.abs(W[big * d + j])) big = k;
    if (W[big * d + j] < 0) flip(j);
  }
  if (d === 2) { W[0 * 2 + 1] = -W[1 * 2 + 0]; W[1 * 2 + 1] = W[0 * 2 + 0]; }          // v2 = v1 rotated by +90 degrees
  if (d === 3) {
    const c = [0, 1, 2].map(k => W[((k + 1) % 3) * 3 + 0] * W[((k + 2) % 3) * 3 + 1] - W[((k + 2) % 3) * 3 + 0] * W[((k + 1) % 3) * 3 + 1]);
    const det = c[0] * W[0 * 3 + 2] + c[1] * W[1 * 3 + 2] + c[2] * W[2 * 3 + 2];       // v3 . (v1 x v2)
    if (det < 0) flip(2);
  }
  const total = values.reduce((s, v) => s + v, 0);
  return { n, d, mu, Xc, C, values, W, ratio: Float64Array.from(values, v => v / total), Z: matmul(Xc, W, n, d, d) };
}

// ── distances and neighbours ────────────────────────────────────────────────────────────────────────────────────

function sqDistMatrix(X, n, d) {
  const D = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    let s = 0; for (let k = 0; k < d; k++) { const t = X[i * d + k] - X[j * d + k]; s += t * t; }
    D[i * n + j] = D[j * n + i] = s;
  }
  return D;
}
// indices of the other points sorted by distance to point i (nearest first)
function neighbourOrder(D2, n, i) {
  const o = []; for (let j = 0; j < n; j++) if (j !== i) o.push(j);
  return o.sort((a, b) => D2[i * n + a] - D2[i * n + b]);
}

// ── t-SNE ───────────────────────────────────────────────────────────────────────────────────────────────────────

// Conditional probabilities p(j|i) = exp(-|xi-xj|²/2σi²) / Σk exp(-|xi-xk|²/2σk²), with σi found by bisection so that
// the perplexity 2^H(P_i) equals the requested value; then symmetrised p_ij = (p(j|i) + p(i|j)) / 2n.
function tsneAffinities(D2, n, perplexity) {
  const target = Math.log(perplexity), cond = new Float64Array(n * n), sigma = new Float64Array(n), entropy = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let dmin = Infinity; for (let j = 0; j < n; j++) if (j !== i) dmin = Math.min(dmin, D2[i * n + j]);
    let lo = 0, hi = Infinity, beta = 1 / Math.max(dmin, 1e-12), H = 0;           // beta = 1 / (2 σ²)
    for (let it = 0; it < 200; it++) {
      let sum = 0, sd = 0;
      for (let j = 0; j < n; j++) if (j !== i) { const e = Math.exp(-(D2[i * n + j] - dmin) * beta); cond[i * n + j] = e; sum += e; sd += e * (D2[i * n + j] - dmin); }
      H = Math.log(sum) + beta * sd / sum;
      if (Math.abs(H - target) < 1e-7) break;
      if (H > target) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; } else { hi = beta; beta = (beta + lo) / 2; }
    }
    let sum = 0; for (let j = 0; j < n; j++) if (j !== i) sum += cond[i * n + j];
    for (let j = 0; j < n; j++) cond[i * n + j] = j === i ? 0 : cond[i * n + j] / sum;
    sigma[i] = Math.sqrt(1 / (2 * beta)); entropy[i] = H;
  }
  const P = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) P[i * n + j] = Math.max((cond[i * n + j] + cond[j * n + i]) / (2 * n), i === j ? 0 : 1e-12);
  return { P, cond, sigma, entropy };
}

// Student-t similarities of a map Y (n x 2): num_ij = 1 / (1 + |yi-yj|²),  q_ij = num_ij / Σ_{k≠l} num_kl.
function tsneQ(Y, n) {
  const num = new Float64Array(n * n); let Z = 0;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1], v = 1 / (1 + dx * dx + dy * dy);
    num[i * n + j] = num[j * n + i] = v; Z += 2 * v;
  }
  return { num, Z };
}
// gradient of KL(P||Q) w.r.t. the map:  dC/dy_i = 4 Σj (exag·p_ij - q_ij) num_ij (y_i - y_j)
function tsneGradient(P, Y, n, exag = 1) {
  const { num, Z } = tsneQ(Y, n), grad = new Float64Array(2 * n);
  let kl = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    if (i === j) continue;
    const p = P[i * n + j], q = Math.max(num[i * n + j] / Z, 1e-300), c = 4 * (exag * p - q) * num[i * n + j];
    grad[2 * i] += c * (Y[2 * i] - Y[2 * j]); grad[2 * i + 1] += c * (Y[2 * i + 1] - Y[2 * j + 1]);
    kl += p * Math.log(p / q);
  }
  return { grad, kl, num, Z };
}
// Gradient descent with momentum; frames[t] is the map after t iterations (frames[0] = the initialisation).
function tsneRun(P, Y0, n, { lr = 50, exag = 12, exagIters = 250, iters = 750, momEarly = 0.5, momLate = 0.8 } = {}) {
  const frames = [Float64Array.from(Y0)], kl = new Float64Array(iters + 1), upd = new Float64Array(2 * n);
  let Y = Float64Array.from(Y0);
  for (let t = 0; t < iters; t++) {
    const early = t < exagIters, g = tsneGradient(P, Y, n, early ? exag : 1), mom = early ? momEarly : momLate;
    kl[t] = g.kl;
    for (let k = 0; k < 2 * n; k++) { upd[k] = mom * upd[k] - lr * g.grad[k]; Y[k] += upd[k]; }
    frames.push(Float64Array.from(Y));
  }
  kl[iters] = tsneGradient(P, Y, n, 1).kl;
  return { frames, kl };
}
// Initial maps: tiny Gaussian noise, or the first two principal components rescaled to the same tiny spread
function tsneInit(kind, n, seed, X, d, scale = 1e-4) {
  const Y = new Float64Array(2 * n);
  if (kind === "pca") {
    const s = pcaSteps(X, n, d); let sd = 0;
    for (let i = 0; i < n; i++) sd += s.Z[i * d] ** 2 / n;
    for (let i = 0; i < n; i++) { Y[2 * i] = s.Z[i * d] / Math.sqrt(sd) * scale; Y[2 * i + 1] = s.Z[i * d + 1] / Math.sqrt(sd) * scale; }
  } else { const r = mulberry32(seed); for (let k = 0; k < 2 * n; k++) Y[k] = gaussian(r) * scale; }
  return Y;
}

// ── UMAP ────────────────────────────────────────────────────────────────────────────────────────────────────────

// Fit 1 / (1 + a x^(2b)) to the target curve (1 below min_dist, exp(-(x-min_dist)/spread) above): Levenberg-Marquardt.
function umapFitAB(minDist, spread = 1) {
  const xs = [], ys = [];
  for (let i = 0; i < 300; i++) { const x = 3 * spread * (i + 0.5) / 300; xs.push(x); ys.push(x < minDist ? 1 : Math.exp(-(x - minDist) / spread)); }
  const f = (a, b, x) => 1 / (1 + a * Math.pow(x, 2 * b)), sse = (a, b) => xs.reduce((s, x, i) => s + (f(a, b, x) - ys[i]) ** 2, 0);
  let a = 1.5, b = 1, lam = 1e-2, cur = sse(a, b);
  for (let it = 0; it < 200; it++) {
    let jaa = 0, jab = 0, jbb = 0, ga = 0, gb = 0;
    xs.forEach((x, i) => {
      const p = Math.pow(x, 2 * b), den = (1 + a * p) ** 2, da = -p / den, db = -a * p * 2 * Math.log(x) / den, r = f(a, b, x) - ys[i];
      jaa += da * da; jab += da * db; jbb += db * db; ga += da * r; gb += db * r;
    });
    const A = jaa * (1 + lam), B = jab, D = jbb * (1 + lam), det = A * D - B * B;
    const na = a - (D * ga - B * gb) / det, nb = b - (-B * ga + A * gb) / det;
    if (na > 0 && nb > 0 && sse(na, nb) < cur) { a = na; b = nb; cur = sse(a, b); lam = Math.max(lam / 3, 1e-9); } else lam *= 4;
    if (lam > 1e9) break;
  }
  return { a, b };
}

// The fuzzy simplicial set of the data (umap-learn's convention: the point itself counts as one of the n_neighbors, so
// k = n_neighbors - 1 other points are used): for every point the distances to its k nearest neighbours, rho_i (distance
// to the nearest one), sigma_i (so that Σj exp(-(d_ij - rho_i)/σ_i) = log2 n_neighbors), the directed weights w(i→j),
// and the symmetric "fuzzy union"  W = A + Aᵀ - A∘Aᵀ.
function umapGraph(D2, n, nNeighbors) {
  const k = nNeighbors - 1, idx = new Int32Array(n * k), dist = new Float64Array(n * k), rho = new Float64Array(n), sigma = new Float64Array(n);
  const A = new Float64Array(n * n), target = Math.log2(nNeighbors);
  for (let i = 0; i < n; i++) {
    const o = neighbourOrder(D2, n, i).slice(0, k);
    o.forEach((j, r) => { idx[i * k + r] = j; dist[i * k + r] = Math.sqrt(D2[i * n + j]); });
    rho[i] = dist[i * k];
    let lo = 0, hi = Infinity, mid = 1;
    for (let it = 0; it < 64; it++) {
      let s = 0; for (let r = 0; r < k; r++) s += Math.exp(-Math.max(0, dist[i * k + r] - rho[i]) / mid);
      if (Math.abs(s - target) < 1e-6) break;
      if (s > target) { hi = mid; mid = (lo + hi) / 2; } else { lo = mid; mid = hi === Infinity ? mid * 2 : (lo + hi) / 2; }
    }
    sigma[i] = mid;
    for (let r = 0; r < k; r++) A[i * n + idx[i * k + r]] = Math.exp(-Math.max(0, dist[i * k + r] - rho[i]) / mid);
  }
  const W = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) W[i * n + j] = A[i * n + j] + A[j * n + i] - A[i * n + j] * A[j * n + i];
  return { k, nNeighbors, idx, dist, rho, sigma, A, W };
}

// Spectral layout: eigenvectors 2 and 3 of the normalised graph Laplacian I - D^-1/2 W D^-1/2 (scaled to [-10, 10]).
function umapSpectralInit(W, n, seed) {
  const deg = new Float64Array(n), L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) deg[i] += W[i * n + j]; }
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) L[i * n + j] = (i === j ? 1 : 0) - W[i * n + j] / Math.sqrt(deg[i] * deg[j] || 1);
  const { vectors } = eigSym(L, n), Y = new Float64Array(2 * n), r = mulberry32(seed);       // ascending order = last columns
  let m = 0;
  for (let i = 0; i < n; i++) { Y[2 * i] = vectors[i * n + (n - 2)]; Y[2 * i + 1] = vectors[i * n + (n - 3)]; m = Math.max(m, Math.abs(Y[2 * i]), Math.abs(Y[2 * i + 1])); }
  for (let k = 0; k < 2 * n; k++) Y[k] = Y[k] / m * 10 + 1e-4 * gaussian(r);
  return Y;
}
function umapRandomInit(n, seed) { const r = mulberry32(seed), Y = new Float64Array(2 * n); for (let k = 0; k < 2 * n; k++) Y[k] = (r() * 2 - 1) * 10; return Y; }

// Share of each point's k nearest neighbours in the input space that are still among its k nearest in the map (0..1).
function knnSets(D2, n, k) {
  const out = new Int32Array(n * k);
  for (let i = 0; i < n; i++) neighbourOrder(D2, n, i).slice(0, k).forEach((j, r) => { out[i * k + r] = j; });
  return out;
}
function knnPreservation(inNN, Y, n, k) {
  const D = sqDistMatrix(Y, n, 2); let hit = 0;
  for (let i = 0; i < n; i++) {
    const mine = new Set(neighbourOrder(D, n, i).slice(0, k));
    for (let r = 0; r < k; r++) if (mine.has(inNN[i * k + r])) hit++;
  }
  return hit / (n * k);
}
const clip4 = v => v > 4 ? 4 : v < -4 ? -4 : v;
// Stochastic optimisation of the layout, as in umap-learn: in epoch e every edge (i, j) is visited with probability
// ~ w_ij; it pulls its end points together and pushes a few random points away from i.  frames[e] = layout after e epochs.
function umapRun(W, Y0, n, a, b, { epochs = 500, lr = 1, negRate = 5, gamma = 1, seed = 42 } = {}) {
  let wmax = 0; for (let k = 0; k < n * n; k++) wmax = Math.max(wmax, W[k]);
  const head = [], tail = [], eps = [];                                   // both directions of every edge
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j && W[i * n + j] >= wmax / epochs) { head.push(i); tail.push(j); eps.push(wmax / W[i * n + j]); }
  const E = head.length, nextS = Float64Array.from(eps), nextN = Float64Array.from(eps, v => v / negRate), epn = Float64Array.from(eps, v => v / negRate);
  const rng = mulberry32(seed), Y = Float64Array.from(Y0), frames = [Float64Array.from(Y)];
  for (let ep = 0; ep < epochs; ep++) {
    const alpha = lr * (1 - ep / epochs);
    for (let e = 0; e < E; e++) {
      if (nextS[e] > ep) continue;
      const i = head[e], j = tail[e];
      let dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1], d2 = dx * dx + dy * dy;
      if (d2 > 0) {
        const c = -2 * a * b * Math.pow(d2, b - 1) / (a * Math.pow(d2, b) + 1), gx = clip4(c * dx), gy = clip4(c * dy);
        Y[2 * i] += gx * alpha; Y[2 * i + 1] += gy * alpha; Y[2 * j] -= gx * alpha; Y[2 * j + 1] -= gy * alpha;
      }
      nextS[e] += eps[e];
      const nNeg = Math.floor((ep - nextN[e]) / epn[e]);
      for (let p = 0; p < nNeg; p++) {
        const k = Math.floor(rng() * n); if (k === i) continue;
        dx = Y[2 * i] - Y[2 * k]; dy = Y[2 * i + 1] - Y[2 * k + 1]; d2 = dx * dx + dy * dy;
        let gx = 4, gy = 4;
        if (d2 > 0) { const c = 2 * gamma * b / ((0.001 + d2) * (a * Math.pow(d2, b) + 1)); gx = clip4(c * dx); gy = clip4(c * dy); }
        Y[2 * i] += gx * alpha; Y[2 * i + 1] += gy * alpha;
      }
      nextN[e] += nNeg * epn[e];
    }
    frames.push(Float64Array.from(Y));
  }
  return { frames };
}

// ── placing a NEW point on an existing map ───────────────────────────────────────────────────────────────────────

// t-SNE has no mapping function: a new point gets its own optimisation.  p_j = conditional probabilities of the new point
// to the existing ones (same perplexity), start at the p-weighted mean of their map positions, then minimise
// KL(p || q) over its position only (gradient 2 Σ (p_j - q_j) num_j (y* - y_j), backtracking line search).
function tsneNewPoint(d2row, Y, n, perplexity) {
  const target = Math.log(perplexity); let dmin = Infinity; for (let j = 0; j < n; j++) dmin = Math.min(dmin, d2row[j]);
  const p = new Float64Array(n); let lo = 0, hi = Infinity, beta = 1 / Math.max(dmin, 1e-12);
  for (let it = 0; it < 200; it++) {
    let sum = 0, sd = 0; for (let j = 0; j < n; j++) { p[j] = Math.exp(-(d2row[j] - dmin) * beta); sum += p[j]; sd += p[j] * (d2row[j] - dmin); }
    const H = Math.log(sum) + beta * sd / sum; if (Math.abs(H - target) < 1e-7) break;
    if (H > target) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; } else { hi = beta; beta = (beta + lo) / 2; }
  }
  let sp = 0; for (let j = 0; j < n; j++) sp += p[j]; for (let j = 0; j < n; j++) p[j] /= sp;
  let y = [0, 0]; for (let j = 0; j < n; j++) { y[0] += p[j] * Y[2 * j]; y[1] += p[j] * Y[2 * j + 1]; }
  const cost = (u, v) => { let Z = 0; const num = new Float64Array(n); for (let j = 0; j < n; j++) { num[j] = 1 / (1 + (u - Y[2 * j]) ** 2 + (v - Y[2 * j + 1]) ** 2); Z += num[j]; }
    let c = 0; for (let j = 0; j < n; j++) if (p[j] > 1e-12) c += p[j] * Math.log(p[j] / (num[j] / Z)); return { c, num, Z }; };
  const path = [y.slice()], costs = [];
  for (let it = 0; it < 60; it++) {
    const { c, num, Z } = cost(y[0], y[1]); costs.push(c);
    let gx = 0, gy = 0; for (let j = 0; j < n; j++) { const f = 2 * (p[j] - num[j] / Z) * num[j]; gx += f * (y[0] - Y[2 * j]); gy += f * (y[1] - Y[2 * j + 1]); }
    let step = 4; const g2 = gx * gx + gy * gy; if (g2 < 1e-20) break;
    while (step > 1e-9 && cost(y[0] - step * gx, y[1] - step * gy).c > c - 1e-4 * step * g2) step /= 2;
    y = [y[0] - step * gx, y[1] - step * gy]; path.push(y.slice());
  }
  return { y, path, p, costs };
}
// UMAP.transform: weights to the k nearest existing points (rho, sigma as for the training points), start at the weighted
// mean of their map positions, then a few epochs of the usual attract / repel updates in which only the new point moves.
function umapNewPoint(d2row, Y, n, nNeighbors, a, b, seed = 7, epochs = 40) {
  const k = nNeighbors - 1, ord = [...Array(n).keys()].sort((i, j) => d2row[i] - d2row[j]).slice(0, k), dist = ord.map(j => Math.sqrt(d2row[j])), rho = dist[0], target = Math.log2(nNeighbors);
  let lo = 0, hi = Infinity, sig = 1;
  for (let it = 0; it < 64; it++) { let s = 0; for (const d of dist) s += Math.exp(-Math.max(0, d - rho) / sig); if (Math.abs(s - target) < 1e-6) break; if (s > target) { hi = sig; sig = (lo + hi) / 2; } else { lo = sig; sig = hi === Infinity ? sig * 2 : (lo + hi) / 2; } }
  const w = dist.map(d => Math.exp(-Math.max(0, d - rho) / sig)), sw = w.reduce((s, x) => s + x, 0);
  let y = [0, 0]; ord.forEach((j, r) => { y[0] += w[r] * Y[2 * j] / sw; y[1] += w[r] * Y[2 * j + 1] / sw; });
  const rng = mulberry32(seed), path = [y.slice()];
  for (let e = 0; e < epochs; e++) {
    const alpha = 1 - e / epochs;
    ord.forEach((j, r) => {
      if (rng() > w[r]) return;
      let dx = y[0] - Y[2 * j], dy = y[1] - Y[2 * j + 1], d2 = dx * dx + dy * dy;
      if (d2 > 0) { const c = -2 * a * b * Math.pow(d2, b - 1) / (a * Math.pow(d2, b) + 1); y[0] += clip4(c * dx) * alpha; y[1] += clip4(c * dy) * alpha; }
      for (let t = 0; t < 5; t++) {
        const m = Math.floor(rng() * n); dx = y[0] - Y[2 * m]; dy = y[1] - Y[2 * m + 1]; d2 = dx * dx + dy * dy;
        let gx = 4, gy = 4; if (d2 > 0) { const c = 2 * b / ((0.001 + d2) * (a * Math.pow(d2, b) + 1)); gx = clip4(c * dx); gy = clip4(c * dy); }
        y[0] += gx * alpha; y[1] += gy * alpha;
      }
    });
    path.push(y.slice());
  }
  return { y, path, neighbours: ord, weights: w, sigma: sig, rho };
}

if (typeof module !== "undefined") module.exports = { mulberry32, gaussian, eigSym, matmul, columnMean, centered, covariance, pcaSteps,
  sqDistMatrix, neighbourOrder, tsneAffinities, tsneQ, tsneGradient, tsneRun, tsneInit, umapFitAB, umapGraph, umapSpectralInit,
  umapRandomInit, knnSets, knnPreservation, umapRun, tsneNewPoint, umapNewPoint };
