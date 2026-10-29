import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score
from sesgo import fairness


def _df(n=600, acc_marked=0.9, acc_unmarked=0.9, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.choice(["NEG", "NEU", "POS"], n, p=[.25, .15, .6])
    g = rng.choice(["MARCADO", "NO_MARCADO"], n)
    pt = rng.choice(["Hotel", "Restaurant", "Attractive"], n)
    acc = np.where(g == "MARCADO", acc_marked, acc_unmarked)
    p = np.where(rng.rand(n) < acc, y, rng.choice(["NEG", "NEU", "POS"], n))
    return pd.DataFrame({"y_true": y, "y_pred": p, "group_label": g, "place_type": pt})


S = ["y_true", "place_type"]


def test_no_gap_when_groups_equal():
    d = fairness.gaps(_df(4000), S)
    assert abs(d["accuracy"]) < 0.03


def test_detects_planted_gap_and_sign():
    d = fairness.gaps(_df(4000, acc_marked=0.6, acc_unmarked=0.9), S)
    assert d["accuracy"] > 0.15  # positivo = peor en MARCADO


def test_bootstrap_ci_covers_estimate_and_is_reproducible():
    df = _df(800, 0.7, 0.9)
    est = fairness.gaps(df, S)["macro_f1"]
    b1 = fairness.bootstrap_gaps(df, S, 200, 5); b2 = fairness.bootstrap_gaps(df, S, 200, 5)
    assert b1.equals(b2)
    lo, hi = b1["macro_f1"].quantile([.025, .975])
    assert lo < est < hi and lo > 0


def test_permutation_null_centers_near_zero_and_placebo_pvalue_small_for_planted_gap():
    df = _df(800, 0.6, 0.9)
    est = fairness.gaps(df, S)["accuracy"]
    null = fairness.permutation_null(df, S, 200, 3)
    assert abs(null["accuracy"].mean()) < 0.03 and (null["accuracy"] >= est).mean() < 0.02


def test_matches_fairlearn_by_group_when_unweighted():
    df = _df(500)
    mf = MetricFrame(metrics=accuracy_score, y_true=df.y_true, y_pred=df.y_pred, sensitive_features=df.group_label)
    for g in ("MARCADO", "NO_MARCADO"):
        s = df[df.group_label == g]
        m = fairness._wmetrics(s.y_true, s.y_pred, np.ones(len(s)))
        assert abs(m["accuracy"] - mf.by_group[g]) < 1e-12


def test_weights_reproduce_marked_composition():
    df = _df(900)
    w, cov = fairness.matched_weights(df, S)
    key = df[S].astype(str).agg("|".join, axis=1)
    u = df.group_label == "NO_MARCADO"
    wu = pd.Series(w[u.to_numpy()]).groupby(key[u].to_numpy()).sum()
    mk = key[~u].value_counts()
    assert all(abs(wu[k] - mk[k]) < 1e-9 for k in wu.index)


def test_decision_rule():
    assert fairness.decision_rule(0.08, 0.02, 0.14, 0.05) == "evidencia_a_favor_de_H1"
    assert fairness.decision_rule(0.01, -0.02, 0.04, 0.05) == "evidencia_en_contra_de_H1_relevante"
    assert fairness.decision_rule(0.04, -0.02, 0.10, 0.05) == "no_concluyente"
