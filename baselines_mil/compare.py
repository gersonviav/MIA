"""
Comparación estadística en TEST entre el modelo propuesto y los baselines.

  python compare.py --ref Propuesto \
     --run "Propuesto=runs/propuesto/seed*/preds_test.csv" \
     --run "Sultani=runs/sultani/seed*/preds_test.csv" \
     --run "RTFM=runs/rtfm/seed*/preds_test.csv" \
     --run "UR-DMU=runs/urdmu/seed*/preds_test.csv" --out results/comparacion

Salidas:
  * media ± d.e. entre semillas de cada métrica;
  * IC 95 % por bootstrap estratificado por clase (B=2000) del promedio entre semillas;
  * diferencia pareada vs. el modelo de referencia (IC 95 % y p bilateral por bootstrap);
  * prueba de McNemar exacta sobre aciertos (voto suave entre semillas), 3 clases y binaria.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata

def _f1(y, p, c):
    tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
    d = 2 * tp + fp + fn
    return 2 * tp / d if d > 0 else 0.0


def _auc(yb, s):
    """AUC-ROC por Mann-Whitney (equivale a roc_auc_score, con empates promediados)."""
    n1 = int(yb.sum()); n0 = len(yb) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[yb].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _ap(yb, s):
    """Average precision (área bajo la curva precisión-recall), igual que average_precision_score."""
    n1 = int(yb.sum())
    if n1 == 0:
        return np.nan
    o = np.argsort(-s, kind="mergesort")
    ys, ss = yb[o].astype(float), s[o]
    last = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]   # último índice de cada umbral distinto
    tps = np.cumsum(ys)[last]
    prec = tps / (last + 1)
    rec = tps / n1
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


# Versiones en numpy (mismo resultado que sklearn, ~50 veces más rápidas para el bootstrap)
METRICS = {
    "F1-macro": lambda y, p, s: (_f1(y, p, 0) + _f1(y, p, 1) + _f1(y, p, 2)) / 3,
    "F1-Normal": lambda y, p, s: _f1(y, p, 0),
    "F1-Aggression": lambda y, p, s: _f1(y, p, 1),
    "F1-Theft": lambda y, p, s: _f1(y, p, 2),
    "F1-anomalía": lambda y, p, s: _f1((y > 0).astype(int), (p > 0).astype(int), 1),
    "AUC-video": lambda y, p, s: _auc(y > 0, s),
    "AP-video": lambda y, p, s: _ap(y > 0, s),
}


CLS = {"Normal": 0, "Aggression": 1, "Theft": 2}


def normalize(df):
    """Acepta también el formato de tu eval_final.py:
    video_id,label,pred,p_Normal,p_Aggression,p_Theft,max_snippet_score"""
    if "y_true" not in df:
        df["y_true"] = df["label"].map(CLS)
    if "y_pred" not in df:
        df["y_pred"] = df["pred"].map(CLS)
    if "anomaly_score" not in df:
        df["anomaly_score"] = 1.0 - df["p_Normal"]  # mismo criterio que en el capítulo
    assert df[["y_true", "y_pred"]].notna().all().all(), "Etiquetas fuera de Normal/Aggression/Theft"
    df[["y_true", "y_pred"]] = df[["y_true", "y_pred"]].astype(int)
    return df


def load_runs(spec):
    name, pattern = spec.split("=", 1)
    files = sorted(glob.glob(pattern))
    assert files, f"Sin archivos para {spec}"
    dfs = [normalize(pd.read_csv(f)).sort_values("video_id").reset_index(drop=True) for f in files]
    ids = dfs[0]["video_id"].tolist()
    for d in dfs[1:]:
        assert d["video_id"].tolist() == ids, f"{name}: semillas con videos distintos"
    return name, dfs


def seed_avg(dfs, idx, fn):
    return np.nanmean([fn(d.y_true.values[idx], d.y_pred.values[idx], d.anomaly_score.values[idx]) for d in dfs])


def stratified_boot_idx(y, B, rng):
    groups = [np.where(y == c)[0] for c in np.unique(y)]
    for _ in range(B):
        yield np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])


def mcnemar_exact(c_a, c_b):
    b = int((c_a & ~c_b).sum())
    c = int((~c_a & c_b).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, p


def majority(dfs):
    """Voto suave entre semillas: argmax del promedio de probabilidades (evita empates)."""
    P = np.mean([d[["p_Normal", "p_Aggression", "p_Theft"]].values for d in dfs], axis=0)
    return P.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/comparacion")
    args = ap.parse_args()

    runs = dict(load_runs(s) for s in args.run)
    ids = runs[args.ref][0]["video_id"].tolist()
    ref_set = set(ids)
    for name, dfs in list(runs.items()):
        new = []
        for d in dfs:
            faltan = ref_set - set(d["video_id"])
            assert not faltan, (f"{name} no tiene predicciones para {len(faltan)} videos de test del "
                                f"modelo de referencia, p.ej. {sorted(faltan)[:5]}")
            extra = len(d) - len(ref_set)
            if extra > 0:
                print(f"[INFO] {name}: {extra} videos extra fuera del test de referencia; "
                      f"se evalúa solo sobre los {len(ref_set)} videos comunes")
            new.append(d[d["video_id"].isin(ref_set)].sort_values("video_id").reset_index(drop=True))
        runs[name] = new
        assert all(d["video_id"].tolist() == ids for d in new), f"{name}: orden de videos inconsistente"
    y = runs[args.ref][0]["y_true"].values
    n = len(y)
    rng = np.random.default_rng(args.seed)
    boots = list(stratified_boot_idx(y, args.B, rng))
    full = np.arange(n)

    # Bootstrap de cada modelo y métrica calculado una sola vez (mismos remuestreos para todos,
    # así la diferencia pareada ref − otro usa exactamente los mismos índices).
    cache = {}
    for k, (name, dfs) in enumerate(runs.items(), 1):
        print(f"[{k}/{len(runs)}] bootstrap {name} ({len(dfs)} semillas, B={args.B})...", flush=True)
        for mname, fn in METRICS.items():
            cache[name, mname] = np.array([seed_avg(dfs, i, fn) for i in boots])

    rows, diffs = [], []
    for name, dfs in runs.items():
        row = {"Modelo": name, "semillas": len(dfs)}
        for mname, fn in METRICS.items():
            per_seed = [fn(d.y_true.values, d.y_pred.values, d.anomaly_score.values) for d in dfs]
            bs = cache[name, mname]
            row[mname] = np.mean(per_seed)
            row[mname + "_sd"] = np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0.0
            row[mname + "_ci"] = (round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4))
            if name != args.ref:
                ref = runs[args.ref]
                d = cache[args.ref, mname] - bs
                obs = seed_avg(ref, full, fn) - seed_avg(dfs, full, fn)
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                diffs.append({"Comparación": f"{args.ref} − {name}", "Métrica": mname, "Δ": obs,
                              "IC95_inf": np.nanpercentile(d, 2.5), "IC95_sup": np.nanpercentile(d, 97.5),
                              "p_bootstrap": min(p, 1.0)})
        rows.append(row)

    mc = []
    pr = majority(runs[args.ref])
    for name, dfs in runs.items():
        if name == args.ref:
            continue
        pb = majority(dfs)
        for lab, ca, cb in (("3 clases", pr == y, pb == y), ("binaria", (pr > 0) == (y > 0), (pb > 0) == (y > 0))):
            b, c, p = mcnemar_exact(ca, cb)
            mc.append({"Comparación": f"{args.ref} vs {name}", "Nivel": lab,
                       "solo_ref_acierta": b, "solo_otro_acierta": c, "p_McNemar": p})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "tabla_metricas.csv", index=False)
    pd.DataFrame(diffs).to_csv(out / "diferencias_bootstrap.csv", index=False)
    pd.DataFrame(mc).to_csv(out / "mcnemar.csv", index=False)

    # Tabla LaTeX (media ± d.e. entre semillas)
    cols = list(METRICS)
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Modelo & " + " & ".join(cols) + r" \\", r"\midrule"]
    for r in rows:
        cells = [f"{100 * r[c]:.1f} $\\pm$ {100 * r[c + '_sd']:.1f}" for c in cols]
        lines.append(f"{r['Modelo']} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tabla_metricas.tex").write_text("\n".join(lines))
    print(tab[["Modelo", "semillas"] + cols].round(4).to_string(index=False))
    print(pd.DataFrame(diffs).round(4).to_string(index=False))
    print(pd.DataFrame(mc).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
