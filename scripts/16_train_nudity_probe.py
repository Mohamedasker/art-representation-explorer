#!/usr/bin/env python3
"""
Measure how reliable the nudity signals are, and train a probe that is better than CLIP zero-shot.

WikiArt carries a human-assigned genre; ``nude_painting`` is a curated label for paintings of nudes.  It is
imperfect ground truth (nudes in other genres are labelled negative, so precision is under-estimated), but it
lets every signal be scored on the same labels:

  * CLIP ViT-L/14 zero-shot probability (the "clip" score from scripts/15_flag_nudity.py)
  * NudeNet exposed-part confidence
  * a logistic-regression probe on the CLIP ViT-L/14 image features, trained on the WikiArt genre labels
    (5-fold cross-validated predictions for WikiArt, so no image is scored by a model that saw its label;
     a model fit on all WikiArt images scores ArtBench)

Prints average precision, ROC-AUC and the precision reached at 80 / 90 / 95 % recall for each signal, and writes
``data/nudity/probe_scores.npz`` (read by scripts/15_flag_nudity.py --flags-only).

    python scripts/16_train_nudity_probe.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

OUT = REPO / "data" / "nudity"


def precision_at_recall(y, s, recalls=(0.8, 0.9, 0.95)):
    p, r, thr = precision_recall_curve(y, s)
    out = {}
    for rec in recalls:
        ok = np.flatnonzero(r[:-1] >= rec)
        i = ok[-1] if len(ok) else 0              # highest threshold that still reaches the recall
        out[rec] = (float(p[i]), float(thr[i]))
    return out


def report(name, y, s):
    ap, auc = average_precision_score(y, s), roc_auc_score(y, s)
    par = precision_at_recall(y, s)
    print(f"  {name:<26} AP {ap:.3f}  AUC {auc:.3f}   " +
          "  ".join(f"P@R{int(100 * r)}={p:.2f} (thr {t:.2f})" for r, (p, t) in par.items()))
    return par


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    wa = np.load(OUT / "scores_wikiart.npz", allow_pickle=False)
    ab = np.load(OUT / "scores_artbench.npz", allow_pickle=False)
    f_wa = np.load(OUT / "clipL14_features_wikiart.npy").astype(np.float32)
    f_ab = np.load(OUT / "clipL14_features_artbench.npy").astype(np.float32)
    y = wa["meta"].astype(int)
    print(f"WikiArt: {len(y):,} images, {int(y.sum()):,} labelled nude_painting ({100 * y.mean():.1f}%)")

    print("\nagainst the WikiArt genre label 'nude_painting':")
    report("CLIP L/14 zero-shot", y, wa["clip"])
    report("NudeNet (exposed parts)", y, wa["nn"])

    oof = np.zeros(len(y), dtype=np.float32)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(f_wa, y):
        clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced")
        clf.fit(f_wa[tr] * 10, y[tr])
        oof[te] = clf.predict_proba(f_wa[te] * 10)[:, 1]
    par = report("CLIP L/14 probe (5-fold CV)", y, oof)

    # combined with NudeNet: probe OR NudeNet >= 0.5
    comb = np.maximum(oof, (wa["nn"] >= 0.5).astype(np.float32))
    report("probe OR NudeNet>=0.5", y, comb)

    final = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(f_wa * 10, y)
    p_ab = final.predict_proba(f_ab * 10)[:, 1].astype(np.float32)
    np.savez_compressed(OUT / "probe_scores.npz", keys_wikiart=wa["keys"], probe_wikiart=oof,
                        keys_artbench=ab["keys"], probe_artbench=p_ab,
                        recall90_threshold=par[0.9][1])
    print(f"\nArtBench (no ground truth): {int((p_ab >= par[0.9][1]).sum()):,} of {len(p_ab):,} images above the "
          f"90%-recall probe threshold {par[0.9][1]:.2f}; zero-shot >= 0.5: {int((ab['clip'] >= 0.5).sum()):,}; "
          f"NudeNet >= 0.35: {int((ab['nn'] >= 0.35).sum()):,}")
    print(f"saved -> {OUT / 'probe_scores.npz'}")


if __name__ == "__main__":
    main()
