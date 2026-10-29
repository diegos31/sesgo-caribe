"""Informe reproducible (Markdown + JSON) de una corrida completa."""
from __future__ import annotations

import json
import platform
import sqlite3
from pathlib import Path

import pandas as pd
import sklearn
import fairlearn

from . import agreement
from .config import config_hash
from .db import now


def _fmt(x, d=3):
    return "n/d" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{d}f}"


def build_report(conn: sqlite3.Connection, cfg: dict, split_version: str, runs: dict, audits: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    src = [r[0] for r in conn.execute("SELECT name FROM corpus_source")]
    synthetic = any(s.upper().startswith("FIXTURE") for s in src)
    L: list[str] = ["# Informe de corrida", ""]
    if synthetic:
        L += ["> **ADVERTENCIA: corpus SINTETICO de prueba (fixture).** Las cifras demuestran que el flujo se ejecuta de",
              "> extremo a extremo; **no tienen interpretacion cientifica** ni evidencian sesgo alguno.", ""]
    L += [f"- Fecha (UTC): {now()}", f"- Hash de la configuracion congelada: `{config_hash(cfg)}`",
          f"- Python {platform.python_version()}, scikit-learn {sklearn.__version__}, Fairlearn {fairlearn.__version__}",
          f"- Fuentes: {', '.join(src)}", ""]
    b = conn.execute("SELECT * FROM ingest_batch ORDER BY id DESC LIMIT 1").fetchone()
    L += ["## 1. Ingesta", f"Leidas {b['n_read']}, insertadas {b['n_inserted']}, duplicadas {b['n_duplicates']}, rechazadas {b['n_rejected']}.",
          f"Resumen: `{b['summary_json']}`", ""]
    n_tot = conn.execute("SELECT COUNT(*) FROM review").fetchone()[0]
    n_asg = conn.execute("SELECT round, status, COUNT(*) c FROM assignment GROUP BY round, status").fetchall()
    L += ["## 2. Anotacion", f"Resenias en corpus: {n_tot}. Tareas: " + "; ".join(f"ronda {r['round']} {r['status']}={r['c']}" for r in n_asg), ""]
    pw = agreement.pairwise_label_agreement(conn, "dialect_label")
    L += ["Concordancia (ronda 1, `dialect_label`):", "", pw.round(3).to_markdown(index=False), ""]
    fl = agreement.fleiss_pilot(conn, "dialect_label", pilot_name="piloto_calibracion")
    L += [f"Fleiss (piloto de calibracion, los 3 anotadores sobre el mismo lote): n={fl['n_items']}, "
          f"kappa={_fmt(fl['kappa_fleiss'])} ({fl.get('interpretacion', 'n/d')})", ""]
    g = pd.read_sql_query("SELECT dialect_gold, COUNT(*) n FROM gold_label GROUP BY dialect_gold", conn)
    L += ["Etiqueta final de grupo:", "", g.to_markdown(index=False), ""]
    sp = pd.read_sql_query("SELECT role, COUNT(*) n FROM data_split WHERE version=? GROUP BY role", conn, params=(split_version,))
    L += [f"## 3. Particion `{split_version}`", "", sp.to_markdown(index=False), ""]
    L += ["## 4. Modelos y auditoria", ""]
    for name, rid in runs.items():
        m = json.loads(conn.execute("SELECT metrics_json FROM model_run WHERE id=?", (rid,)).fetchone()[0])
        a = audits[name]
        L += [f"### {name} (run {rid})", f"Macro-F1 dev={_fmt(m.get('macro_f1_dev'))}, test={_fmt(m.get('macro_f1_test'))}; "
              f"auditoria n={a['n_auditoria']} (MARCADO={a['n_marcado']}, NO_MARCADO={a['n_no_marcado']}).", "",
              f"Cobertura del emparejamiento: `{a['cobertura']}`", "",
              "| Metrica | Brecha D = NO_MARCADO - MARCADO | IC 95 % | p permutacion (1 cola) |", "|---|---|---|---|"]
        for k, v in a["brechas"].items():
            L.append(f"| {k} | {_fmt(v['estimado'])} | [{_fmt(v['ic'][0])}, {_fmt(v['ic'][1])}] | {_fmt(v['p_permutacion_una_cola'], 4)} |")
        L += ["", f"Decision H1 (macro-F1, delta_min={cfg['decision']['delta_min']}): **{a['decision_H1']}**  ",
              f"Decision H2 (brecha NEG menos POS en recall): **{a['decision_H2_brecha_NEG_menos_POS']}**", ""]
        if a["auxiliares_binarios"]:
            L += [f"Auxiliares (subconjunto NEG/POS, Fairlearn): `{a['auxiliares_binarios']}`", ""]
    md = "\n".join(L)
    path = out_dir / "informe_corrida.md"
    path.write_text(md, encoding="utf-8")
    (out_dir / "informe_corrida.json").write_text(json.dumps({"config_hash": config_hash(cfg), "auditorias": audits},
                                                             ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"markdown_path": str(path), "markdown": md}
