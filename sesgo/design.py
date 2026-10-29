"""Calculos de diseno: tamano minimo por grupo (dos proporciones, bilateral)."""
from __future__ import annotations

import math

import pandas as pd
from scipy.stats import norm


def n_per_group(p_ref: float, delta: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """n por grupo para detectar una diferencia `delta` entre dos proporciones
    (p_ref y p_ref - delta), prueba bilateral, grupos independientes de igual tamano
    (Fleiss, Levin y Paik, 2003)."""
    p2 = p_ref - delta
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    n = z ** 2 * (p_ref * (1 - p_ref) + p2 * (1 - p2)) / delta ** 2
    return math.ceil(n)


def sample_size_table(p_refs=(0.60, 0.70, 0.80), deltas=(0.05, 0.08, 0.10), alpha=0.05, power=0.80) -> pd.DataFrame:
    return pd.DataFrame([{"p_ref": p, "delta": d, "n_por_grupo": n_per_group(p, d, alpha, power)}
                         for p in p_refs for d in deltas])
