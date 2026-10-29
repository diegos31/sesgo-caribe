"""Auditoria de desempeno por grupo (MARCADO vs NO_MARCADO).

Convenciones
------------
* Unidad de analisis: la RESENIA (texto). El grupo describe el TEXTO (presencia de
  marcas dialectales segun la guia de anotacion), NO la procedencia del autor.
* Brecha  D(m) = m(NO_MARCADO) - m(MARCADO), con NO_MARCADO re-ponderado para tener
  la misma composicion (clase real x tipo de lugar) que MARCADO. D > 0 => peor
  desempeno en textos marcados.
* IC por bootstrap percentil dentro de celdas (grupo x estrato). Prueba de
  permutacion (etiquetas de grupo barajadas dentro de estrato) como control placebo.
* La paridad demografica y la igualdad de probabilidades (Fairlearn) se reportan
  como indicadores AUXILIARES sobre el subconjunto binario NEG/POS: la primera
  puede diferir por diferencias reales de satisfaccion y no implica discriminacion.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                               equalized_odds_difference, selection_rate)
from sklearn.metrics import accuracy_score, f1_score

from .config import CLASSES, MARCADO, NO_MARCADO
from .db import now
from .design import n_per_group

METRICS = ["accuracy", "macro_f1", "recall_NEG", "recall_NEU", "recall_POS", "sel_POS"]


def audit_frame(conn: sqlite3.Connection, run_id: int, split_version: str) -> pd.DataFrame:
    q = ("SELECT r.id, r.y_class AS y_true, p.y_pred, r.place_type, r.polarity, r.n_chars, g.dialect_gold AS group_label "
         "FROM data_split s JOIN review r ON r.id=s.review_id "
         "JOIN gold_label g ON g.review_id=r.id "
         "JOIN prediction p ON p.review_id=r.id AND p.run_id=? "
         "WHERE s.version=? AND s.role='audit' AND g.dialect_gold IN ('MARCADO','NO_MARCADO')")
    df = pd.read_sql_query(q, conn, params=(run_id, split_version))
    df["place_type"] = df["place_type"].fillna("NA")
    return df


def _wmetrics(y, p, w) -> dict:
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(w, dtype=float)
    out = {"accuracy": float(np.sum(w * (y == p)) / w.sum()),
           "macro_f1": float(f1_score(y, p, labels=list(CLASSES), average="macro",
                                      sample_weight=w, zero_division=0))}
    for c in CLASSES:
        m = y == c
        out[f"recall_{c}"] = float(np.sum(w[m] * (p[m] == c)) / w[m].sum()) if w[m].sum() > 0 else np.nan
    out["sel_POS"] = float(np.sum(w * (p == "POS")) / w.sum())
    return out


def matched_weights(df: pd.DataFrame, strata_cols: list[str]) -> tuple[np.ndarray, dict]:
    """Peso 1 para MARCADO; para NO_MARCADO n_marc(s)/n_nomarc(s) en los estratos
    con ambos grupos; 0 en estratos sin solape (se informa la cobertura)."""
    key = df[strata_cols].astype(str).agg("|".join, axis=1)
    isM = (df["group_label"] == MARCADO).to_numpy()
    nM, nU = key[isM].value_counts(), key[~isM].value_counts()
    w = np.zeros(len(df))
    w[isM] = [1.0 if nU.get(k, 0) > 0 else 0.0 for k in key[isM]]
    ratio = {k: nM.get(k, 0) / nU[k] for k in nU.index}
    w[~isM] = [ratio[k] for k in key[~isM]]
    cov = {"n_marcado_total": int(isM.sum()),
           "n_marcado_con_solape": int(round(w[isM].sum())),
           "n_no_marcado_total": int((~isM).sum()),
           "estratos_sin_solape": sorted(set(nM.index) ^ set(nU.index))}
    return w, cov


def gaps(df: pd.DataFrame, strata_cols: list[str]) -> dict:
    w, _ = matched_weights(df, strata_cols)
    isM = (df["group_label"] == MARCADO).to_numpy()
    mM = _wmetrics(df.y_true[isM], df.y_pred[isM], w[isM]) if w[isM].sum() > 0 else None
    mU = _wmetrics(df.y_true[~isM], df.y_pred[~isM], w[~isM]) if w[~isM].sum() > 0 else None
    if mM is None or mU is None:
        return {m: np.nan for m in METRICS + ["gap_NEG_minus_POS"]}
    g = {m: mU[m] - mM[m] for m in METRICS}
    g["gap_NEG_minus_POS"] = g["recall_NEG"] - g["recall_POS"]
    return g


def _boot_df(df: pd.DataFrame, strata_cols: list[str], rng: np.random.RandomState) -> pd.DataFrame:
    cell = df["group_label"] + "|" + df[strata_cols].astype(str).agg("|".join, axis=1)
    idx = []
    for _, ix in df.groupby(cell).indices.items():
        idx.extend(rng.choice(ix, size=len(ix), replace=True))
    return df.iloc[idx].reset_index(drop=True)


def bootstrap_gaps(df: pd.DataFrame, strata_cols: list[str], B: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame([gaps(_boot_df(df, strata_cols, rng), strata_cols) for _ in range(B)])


def permutation_null(df: pd.DataFrame, strata_cols: list[str], B: int, seed: int) -> pd.DataFrame:
    """Control placebo: baraja la etiqueta de grupo DENTRO de cada estrato."""
    rng = np.random.RandomState(seed)
    key = df[strata_cols].astype(str).agg("|".join, axis=1)
    rows = []
    for _ in range(B):
        d = df.copy()
        lab = d["group_label"].to_numpy().copy()
        for _, ix in d.groupby(key).indices.items():
            lab[ix] = lab[rng.permutation(ix)]
        d["group_label"] = lab
        rows.append(gaps(d, strata_cols))
    return pd.DataFrame(rows)


def decision_rule(est: float, lo: float, hi: float, delta_min: float) -> str:
    """Regla pre-registrada (sobre la brecha D de la metrica primaria)."""
    if np.isnan(est):
        return "no_evaluable"
    if lo > 0 and est >= delta_min:
        return "evidencia_a_favor_de_H1"
    if hi < delta_min:
        return "evidencia_en_contra_de_H1_relevante"
    return "no_concluyente"


def binary_auxiliary(df: pd.DataFrame) -> dict:
    """Indicadores auxiliares de Fairlearn en el subconjunto binario NEG/POS."""
    b = df[df.y_true.isin(["NEG", "POS"])]
    if b.group_label.nunique() < 2 or len(b) == 0:
        return {}
    yt, yp = (b.y_true == "POS").astype(int), (b.y_pred == "POS").astype(int)
    return {"n_binario": int(len(b)),
            "demographic_parity_difference": float(demographic_parity_difference(yt, yp, sensitive_features=b.group_label)),
            "equalized_odds_difference": float(equalized_odds_difference(yt, yp, sensitive_features=b.group_label)),
            "tasa_POS_real_por_grupo": b.groupby("group_label").apply(lambda x: float((x.y_true == "POS").mean()),
                                                                      include_groups=False).to_dict()}


def unmatched_by_group(df: pd.DataFrame, n_boot: int, seed: int, level: float) -> pd.DataFrame:
    """Metricas por grupo SIN emparejar, con IC bootstrap de Fairlearn (MetricFrame)."""
    q = [(1 - level) / 2, 1 - (1 - level) / 2]
    mf = MetricFrame(metrics={"accuracy": accuracy_score,
                              "macro_f1": lambda y, p: f1_score(y, p, labels=list(CLASSES), average="macro", zero_division=0)},
                     y_true=df.y_true, y_pred=df.y_pred, sensitive_features=df.group_label,
                     n_boot=n_boot, ci_quantiles=q, random_state=seed)
    out = mf.by_group.copy()
    for k, name in enumerate(("ic_inf", "ic_sup")):
        ci = mf.by_group_ci[k]
        for m in ci.columns:
            out[f"{m}_{name}"] = ci[m]
    out["n"] = df.groupby("group_label").size()
    return out


def run_audit(conn: sqlite3.Connection, run_id: int, split_version: str, cfg: dict) -> dict:
    df = audit_frame(conn, run_id, split_version)
    strata = cfg["estratos_emparejamiento"]
    bs, perm = cfg["bootstrap"], cfg["permutacion"]
    dmin, prim = cfg["decision"]["delta_min"], cfg["decision"]["metrica_primaria"]
    w, cov = matched_weights(df, strata)
    est = gaps(df, strata)
    boot = bootstrap_gaps(df, strata, bs["B"], bs["semilla"])
    a = (1 - bs["nivel_ic"]) / 2
    ci = {m: (float(boot[m].quantile(a)), float(boot[m].quantile(1 - a))) for m in boot.columns}
    null = permutation_null(df, strata, perm["B"], bs["semilla"] + 1)
    pval = {m: float((1 + (null[m] >= est[m]).sum()) / (1 + len(null))) for m in null.columns}
    tm = cfg["tamano_muestra"]
    n_req = n_per_group(tm["p_no_marcado_supuesto"], tm["delta_deteccion"], tm["alpha"], tm["potencia"])
    n_min = int(min((df.group_label == MARCADO).sum(), (df.group_label == NO_MARCADO).sum()))
    suficiente = n_min >= n_req
    dec1 = decision_rule(est[prim], *ci[prim], dmin)
    dec2 = decision_rule(est["gap_NEG_minus_POS"], *ci["gap_NEG_minus_POS"], 0.0)
    if not suficiente:
        dec1 = dec2 = f"n_insuficiente (minimo por grupo {n_min} < {n_req} requerido)"
    res = {"n_auditoria": int(len(df)), "n_requerido_por_grupo": n_req, "n_suficiente": bool(suficiente), "n_marcado": int((df.group_label == MARCADO).sum()),
           "n_no_marcado": int((df.group_label == NO_MARCADO).sum()), "cobertura": cov,
           "brechas": {m: {"estimado": est[m], "ic": ci[m], "p_permutacion_una_cola": pval[m]} for m in est},
           "decision_H1": dec1, "decision_H2_brecha_NEG_menos_POS": dec2,
           "sin_emparejar": unmatched_by_group(df, min(bs["B"], 500), bs["semilla"], bs["nivel_ic"]).reset_index().to_dict("records"),
           "auxiliares_binarios": binary_auxiliary(df)}
    rows = [(run_id, "pareada", m, est[m], ci[m][0], ci[m][1], res["n_marcado"], res["n_no_marcado"],
             f"p_perm={pval[m]:.4f}", now()) for m in est]
    conn.executemany("INSERT INTO audit_result(run_id, scope, metric, estimate, ci_low, ci_high, n_marked, "
                     "n_unmarked, note, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return res
