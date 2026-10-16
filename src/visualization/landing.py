"""
Landing page (``index.html``) for the explorer: one card per (dataset, representation) with a
live preview scatter, the kind of model, size, and how well k-means clusters agree with the art
styles, plus a comparison table and an overview of the colour-by options.  Self-contained HTML.
"""

from __future__ import annotations

import json
from pathlib import Path

# order shown on the page; (substring of the stem, short label, family, description)
MODEL_INFO = [
    ("cnn_supervised", "ResNet-50 (trained here)", "Supervised, in-domain",
     "Trained from scratch on ArtBench + WikiArt to predict style, genre and artist (multi-task)."),
    ("resnet50_imagenet", "ResNet-50 (ImageNet)", "Supervised, off-the-shelf",
     "Standard ImageNet-classification ResNet-50; never trained on art."),
    ("autoencoder", "Convolutional autoencoder", "Unsupervised",
     "Trained only to reconstruct the images (ArtBench + WikiArt). Latent code, pooled to 1024-D. "
     "Captures colour, layout and texture; no labels involved."),
    ("dinov2", "DINOv2 ViT-B/14", "Self-supervised",
     "Pretrained with self-distillation on large image collections; no labels and no text."),
    ("clip", "CLIP ViT-B/32", "Language-supervised",
     "Pretrained to match images with captions; embeddings carry a lot of semantic content."),
]
DATASETS = {"artbench_train": ("ArtBench-10 · train", "50,000 paintings · 10 styles"),
            "artbench_test": ("ArtBench-10 · test", "10,000 held-out paintings"),
            "artbench_all": ("ArtBench-10 · train + test", "60,000 paintings together, coloured by split"),
            "wikiart_all": ("WikiArt · train + test", "81,444 paintings together, coloured by split"),
            "wikiart_train": ("WikiArt · train", "73,143 paintings · 27 styles · genre and artist labels"),
            "wikiart_test": ("WikiArt · test", "8,301 held-out paintings")}

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Art representation explorer</title>
<style>
:root { --bg:#0e0f12; --panel:#17191e; --panel2:#1d2027; --line:#2a2e37; --fg:#eceef2; --muted:#8c93a1;
        --accent:#ffb454; --accent2:#7aa2ff; }
* { box-sizing:border-box; }
body { margin:0; background:radial-gradient(1200px 500px at 15% -10%, #1c2233 0%, transparent 60%), var(--bg);
       color:var(--fg); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1180px; margin:0 auto; padding:0 20px 60px; }
header.hero { padding:56px 0 28px; }
.hero h1 { font-size:38px; line-height:1.15; margin:0 0 12px; letter-spacing:-.02em; }
.hero h1 span { background:linear-gradient(90deg,var(--accent),#ff7a7a 60%,var(--accent2)); -webkit-background-clip:text;
       background-clip:text; color:transparent; }
.hero p { margin:0; max-width:780px; color:var(--muted); font-size:16px; }
.stats { display:flex; flex-wrap:wrap; gap:10px; margin-top:22px; }
.stat { background:var(--panel); border:1px solid var(--line); border-radius:999px; padding:6px 14px; font-size:13px;
        color:var(--muted); } .stat b { color:var(--fg); font-weight:600; }
.tabs { display:flex; gap:8px; margin:34px 0 18px; }
.tab { cursor:pointer; background:var(--panel); border:1px solid var(--line); color:var(--muted); border-radius:10px;
       padding:9px 16px; font-size:14px; } .tab small { display:block; font-size:11px; opacity:.8; }
.tab.on { color:var(--fg); border-color:var(--accent); box-shadow:0 0 0 1px var(--accent) inset; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:16px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:14px; overflow:hidden;
        transition:transform .15s, border-color .15s, box-shadow .15s; }
.card a.main { display:block; text-decoration:none; color:inherit; }
.card .sets { padding:0 16px 14px; display:flex; flex-wrap:wrap; gap:6px; align-items:center; font-size:12px; color:var(--muted); }
.card .sets a { color:var(--fg); text-decoration:none; background:var(--panel2); border:1px solid var(--line);
                border-radius:999px; padding:2px 9px; } .card .sets a:hover { border-color:var(--accent); color:var(--accent); }
.card:hover { transform:translateY(-3px); border-color:var(--accent); box-shadow:0 10px 30px #0008; }
.prev { display:block; width:100%; height:190px; background:#0b0c0f; }
.body { padding:14px 16px 16px; }
.fam { display:inline-block; font-size:11px; letter-spacing:.06em; text-transform:uppercase; padding:2px 8px;
       border-radius:999px; margin-bottom:8px; border:1px solid currentColor; }
.f-Unsupervised { color:#6ee7b7; } .f-Self-supervised { color:#93c5fd; } .f-Language-supervised { color:#f0abfc; }
.f-Supervised { color:#fcd34d; }
.card h3 { margin:0 0 4px; font-size:18px; }
.card p { margin:0 0 12px; color:var(--muted); font-size:13px; min-height:58px; }
.kv { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
.kv div { background:var(--panel2); border-radius:8px; padding:6px 8px; }
.kv b { display:block; font-size:16px; font-variant-numeric:tabular-nums; } .kv span { font-size:11px; color:var(--muted); }
.open { margin-top:12px; font-size:13px; color:var(--accent); }
section { margin-top:46px; }
section > h2 { font-size:20px; margin:0 0 6px; } section > p.lead { margin:0 0 16px; color:var(--muted); }
table { width:100%; border-collapse:collapse; font-size:13.5px; background:var(--panel); border:1px solid var(--line);
        border-radius:12px; overflow:hidden; }
th, td { padding:9px 12px; text-align:right; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums; }
th:first-child, td:first-child { text-align:left; } th { color:var(--muted); font-weight:500; background:var(--panel2); }
tr:last-child td { border-bottom:0; } td.best { color:var(--accent); font-weight:600; }
.chips { display:flex; flex-wrap:wrap; gap:8px; }
.chip { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 12px; font-size:13px;
        min-width:220px; flex:1 1 220px; } .chip b { display:block; margin-bottom:2px; }
.chip span { color:var(--muted); font-size:12px; }
footer { margin-top:48px; color:var(--muted); font-size:12.5px; }
</style></head>
<body><div class="wrap">
<header class="hero">
  <h1>Exploring <span>image representations</span> of art</h1>
  <p>Embeddings of paintings from ArtBench-10 and WikiArt, compared across unsupervised, self-supervised, language- and
     label-supervised models. Open a card for an interactive map (UMAP, t-SNE, PCA) that you can colour by cluster, art style,
     colour, painting date, metadata and CLIP-derived content, mood and aesthetics.</p>
  <div class="stats" id="stats"></div>
</header>
<div class="tabs" id="tabs"></div>
<div class="grid" id="grid"></div>
<section><h2>How do the embeddings compare?</h2>
  <p class="lead">k-means clusters (k = number of art styles, run on the embedding) versus the true art styles.
     Higher is better; the best model per column is highlighted.</p>
  <div id="table"></div></section>
<section><h2>What can you colour the maps by?</h2>
  <p class="lead">Pick the &ldquo;Colour by&rdquo; menu in any map. Click a legend entry to isolate those points.</p>
  <div class="chips" id="features"></div></section>
<footer>Embeddings cover the whole dataset (train + test); each map shows a random sample of __NS__ images, with other samples and the full set one click away. The autoencoder and the supervised ResNet were trained on the train split only.
  Content, mood and aesthetic scores come from CLIP zero-shot prompts and are subjective, noisy estimates, not ground truth.</footer>
</div>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const PAL = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
  "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5","#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5"];
const col = i => PAL[i % PAL.length];
const fmt = (x, d = 3) => x === null || x === undefined ? "–" : x.toFixed(d);

$("stats").innerHTML = D.stats.map(s => `<span class="stat"><b>${s[0]}</b> ${s[1]}</span>`).join("");
let current = D.datasets[0].key;
function renderTabs() {
  $("tabs").innerHTML = D.datasets.map(d => `<div class="tab ${d.key === current ? "on" : ""}" data-k="${d.key}">${d.name}<small>${d.sub}</small></div>`).join("");
  document.querySelectorAll(".tab").forEach(t => t.onclick = () => { current = t.dataset.k; renderTabs(); renderGrid(); });
}
function renderGrid() {
  const items = D.entries.filter(e => e.dataset === current);
  $("grid").innerHTML = items.map(e => `<div class="card"><a class="main" href="${e.href}">
    <canvas class="prev" data-s="${e.stem}"></canvas>
    <div class="body"><span class="fam f-${e.family.split(",")[0].replace(/\s/g, "-")}">${e.family}</span>
      <h3>${e.title}</h3><p>${e.desc}</p>
      <div class="kv"><div><b>${e.dim}</b><span>dimensions</span></div>
        <div><b>${fmt(e.nmi)}</b><span>cluster/style NMI</span></div>
        <div><b>${fmt(e.sil)}</b><span>silhouette</span></div></div>
      <div class="open">Open interactive map &rarr;</div></div></a>
    ${(e.sets || []).length > 1 ? `<div class="sets">image sets: ${e.sets.map(s => `<a href="${s.href}">${s.label}</a>`).join("")}</div>` : ""}</div>`).join("");
  document.querySelectorAll("canvas.prev").forEach(drawPreview);
}
function drawPreview(cv) {
  const e = D.entries.find(x => x.stem === cv.dataset.s), r = devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight; cv.width = w * r; cv.height = h * r;
  const g = cv.getContext("2d"); g.scale(r, r);
  const pad = 14, P = e.preview;
  g.globalAlpha = .85;
  for (let i = 0; i < P.x.length; i++) {
    g.fillStyle = col(P.c[i]); g.beginPath();
    g.arc(pad + P.x[i] * (w - 2 * pad), pad + (1 - P.y[i]) * (h - 2 * pad), 2.1, 0, 6.3); g.fill();
  }
}
addEventListener("resize", () => document.querySelectorAll("canvas.prev").forEach(drawPreview));

// comparison table
(function () {
  const models = D.models, ds = D.datasets;
  let head = `<tr><th>model</th>` + ds.map(d => `<th>${d.name}<br>NMI</th><th>${d.name}<br>ARI</th><th>${d.name}<br>silhouette</th>`).join("") + `</tr>`;
  const best = {};
  ds.forEach(d => ["nmi", "ari", "sil"].forEach(m => {
    best[d.key + m] = Math.max(...D.entries.filter(e => e.dataset === d.key).map(e => e[m] ?? -9)); }));
  const rows = models.map(m => `<tr><td><b>${m.title}</b><br><span style="color:var(--muted);font-size:12px">${m.family}</span></td>` +
    ds.map(d => { const e = D.entries.find(x => x.dataset === d.key && x.model === m.key);
      return ["nmi", "ari", "sil"].map(k => !e ? "<td>–</td>" :
        `<td class="${e[k] === best[d.key + k] ? "best" : ""}">${fmt(e[k])}</td>`).join(""); }).join("") + `</tr>`);
  $("table").innerHTML = `<table>${head}${rows.join("")}</table>`;
})();
$("features").innerHTML = D.features.map(f => `<div class="chip"><b>${f[0]}</b><span>${f[1]}</span></div>`).join("");
renderTabs(); renderGrid();
</script></body></html>
"""


def _kind(stem: str):
    for key, title, family, desc in MODEL_INFO:
        if key in stem:
            return key, title, family, desc
    return stem, stem, "Representation", ""


def write_index(path: str | Path, built: list[dict], n_samples: int) -> Path:
    """``built`` = the dicts returned by ``scripts/08_build_explorer.build`` (with preview / summary fields)."""
    entries = []
    for b in built:
        ds = "artbench" if b["stem"].startswith("artbench") else "wikiart"
        sp = b["stem"].rsplit("_", 1)[-1]
        ds = f"{ds}_{sp if sp in ('train', 'test') else 'all'}"
        key, title, family, desc = _kind(b["stem"])
        entries.append({"stem": b["stem"], "href": f"{b['stem']}.html", "dataset": ds, "model": key,
                        "title": title, "family": family, "desc": desc, "sets": b.get("sets_links", []),
                        **b["summary"]})
    order = {k: i for i, (k, *_r) in enumerate(MODEL_INFO)}
    entries.sort(key=lambda e: (e["dataset"], order.get(e["model"], 99)))
    models = [{"key": k, "title": t, "family": f} for k, t, f, _ in MODEL_INFO if any(e["model"] == k for e in entries)]
    datasets = [{"key": k, "name": n, "sub": s} for k, (n, s) in DATASETS.items() if any(e["dataset"] == k for e in entries)]

    feats = {}
    for b in built:
        for g, labels in b.get("attr_groups", {}).items():
            feats.setdefault(g, set()).update(labels)
    descr = {"CLIP": "zero-shot scores from CLIP text prompts (subjective)",
             "WikiArt metadata": "genre and artist from the dataset", "ArtBench": "parsed from file names and titles",
             "Image statistics": "computed from the pixels"}
    features = [["Cluster · style · colour theme", "k-means clusters in embedding or PCA space, art style, colour palettes, "
                 "dominant colour, lightness, colourfulness, painting date"]]
    for g, names in sorted(feats.items()):
        d = next((v for k, v in descr.items() if g.startswith(k)), "")
        sample = ", ".join(sorted(names)[:4]) + (" …" if len(names) > 4 else "")
        features.append([f"{g}  ({len(names)})", f"{d}. {sample}"])

    n_ds = len(datasets)
    stats = [[str(len(entries)), "interactive maps"], [str(len(models)), "representations"],
             [str(n_ds), "datasets"], ["UMAP · t-SNE · PCA", "projections"],
             [f"{n_samples:,}", "images per map"]]
    data = {"entries": entries, "models": models, "datasets": datasets, "features": features, "stats": stats}
    html = TEMPLATE.replace("__NS__", f"{n_samples:,}").replace("__DATA__", json.dumps(data, separators=(",", ":")))
    path = Path(path)
    path.write_text(html)
    return path
