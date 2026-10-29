"""Flujo de extremo a extremo sobre el FIXTURE SINTETICO (prueba de integracion).

Los 'anotadores' simulados leen la verdad de referencia del fixture y le agregan
ruido con semilla fija; envian sus anotaciones por la MISMA funcion de servicio
(`submit_annotation`) que usa la interfaz. Los resultados NO tienen interpretacion cientifica.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import agreement, annotation, db, fairness, ingest, models, report, sampling, splits
from .annotation import PREJ_TARGETS
from .config import ROOT, config_hash, load_config
from .lexicon import load_lexicon
from .spans import Span, snap_to_tokens
from .text import tokenize

ALIASES = ["anotador_1", "anotador_2", "anotador_3"]


def load_truth(path: Path) -> dict:
    return {t["id"]: t for t in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


def simulated_annotation(conn, task: dict, truth: dict, rng: np.random.RandomState,
                         p_flip=0.08, p_ambig=0.03, p_flip_prej=0.05) -> dict:
    text = task["text"]
    t = truth.get(task["review_id"], {"dialect": [], "prej": [], "target": None})
    spans: list[Span] = []
    marked = bool(t["dialect"])
    prej = bool(t["prej"])
    r = rng.rand()
    if r < p_ambig:
        dialect = "AMBIGUO"
    elif r < p_ambig + p_flip:
        marked = not marked
        dialect = "MARCADO" if marked else "NO_MARCADO"
    else:
        dialect = "MARCADO" if marked else "NO_MARCADO"
    if dialect == "MARCADO":
        src = t["dialect"] or []
        if src:
            spans += [Span(s["start"], s["end"], "DIALECT") for s in src]
        else:  # falso positivo del anotador: marca un token cualquiera
            toks = [x for x in tokenize(text) if x[0].isalpha()]
            a, b = toks[rng.randint(len(toks))][1:]
            spans.append(Span(a, b, "DIALECT"))
    if dialect == "AMBIGUO" and t["dialect"] and rng.rand() < 0.5:
        s = t["dialect"][0]
        spans.append(Span(s["start"], s["end"], "DIALECT"))
    if rng.rand() < p_flip_prej:
        prej = not prej
    pl, target = ("PRESENTE" if prej else "AUSENTE"), None
    if prej:
        src = t["prej"] or []
        if src:
            spans += [Span(s["start"], s["end"], "PREJ") for s in src]
        else:
            toks = [x for x in tokenize(text) if x[0].isalpha()]
            a, b = toks[rng.randint(len(toks))][1:]
            spans.append(Span(a, b, "PREJ"))
        target = t["target"] or PREJ_TARGETS[-1]
    spans = [snap_to_tokens(text, s) for s in spans]
    return dict(dialect_label=dialect, prejudice_label=pl, spans=spans, prejudice_target=target)


def run_demo(db_path: str | Path, out_dir: Path | None = None, boot_B: int | None = None, verbose=True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    cfg = load_config()
    if boot_B:
        cfg["bootstrap"]["B"] = boot_B
        cfg["permutacion"]["B"] = boot_B
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    conn = db.connect(db_path)
    db.init_db(conn)
    fx = ROOT / "data" / "fixture"
    log("[1/9] Ingesta + sanitizacion")
    ing = ingest.ingest_file(conn, fx / "reviews_fixture.csv", source_name="FIXTURE_SINTETICO",
                            country_filter="Colombia", source_note="Datos sinteticos de prueba")
    log("     ", ing)
    entries = load_lexicon(ROOT / "data" / "lexicon" / "lexicon_fixture.csv")
    log("[2/9] Disponibilidad (viabilidad)")
    feas = sampling.feasibility(conn, entries, country="Colombia")
    log(feas.groupby("y_class")[["n_reviews", "n_lex_candidates"]].sum().to_string())
    log("[3/9] Muestreo estratificado (piloto de calibracion + muestra principal)")
    pil = sampling.draw_sample(conn, "piloto_calibracion", seed=11, n_random=3, n_lexicon=0,
                               entries=entries, country="Colombia")
    smp = sampling.draw_sample(conn, "muestra_principal", seed=cfg["bootstrap"]["semilla"], n_random=14, n_lexicon=12,
                               entries=entries, country="Colombia", exclude_draw_ids=(pil["draw_id"],))
    log("     ", {"piloto": pil["n_items"], "principal": smp["n_items"]})
    log("[4/9] Asignacion independiente (piloto: 3 anotadores; principal: 2 por resenia)")
    annotation.add_annotators(conn, ALIASES)
    gid = annotation.register_guideline(conn, "v1", ROOT / "docs" / "guia_anotacion_v1.md")
    n_pil = annotation.assign_independent(conn, pil["draw_id"], ALIASES, k=3, seed=1)
    n_assign = annotation.assign_independent(conn, smp["draw_id"], ALIASES, k=2, seed=1)
    log("      tareas ronda 1: piloto", n_pil, "| principal", n_assign)
    truth = load_truth(fx / "truth_fixture.jsonl")
    rng = np.random.RandomState(7)

    def drain(alias):
        n = 0
        while (task := annotation.next_task(conn, alias)) is not None:
            ann = simulated_annotation(conn, task, truth, rng)
            annotation.submit_annotation(conn, task["assignment_id"], alias, gid, **ann)
            n += 1
        return n

    log("[5/9] Anotacion (anotadores simulados -> submit_annotation)")
    done = {a: drain(a) for a in ALIASES}
    log("     ", done)
    n3 = annotation.flag_disagreements(conn)
    log("      discrepancias -> tercera anotacion:", n3)
    done2 = {a: drain(a) for a in ALIASES}
    gold = annotation.compute_gold(conn)
    log("      etiqueta final:", gold)
    log("      tramos anotados por capa:", dict(conn.execute("SELECT layer, COUNT(*) FROM span GROUP BY layer").fetchall()))
    log("[6/9] Concordancia entre anotadores")
    pw = agreement.pairwise_label_agreement(conn, "dialect_label")
    log(pw.round(3).to_string(index=False))
    log("      Fleiss (piloto):", agreement.fleiss_pilot(conn, "dialect_label", "piloto_calibracion"))
    log("      Cohen, capa de expresion prejuiciosa (prejudice_label):")
    log(agreement.pairwise_label_agreement(conn, "prejudice_label").round(3).to_string(index=False))
    log("      Kappa por token (BIO, capa DIALECT):")
    log(agreement.token_level_agreement(conn, "DIALECT").round(3).to_string(index=False))
    log("[7/9] Particion (auditoria reservada, anti-fuga)")
    sp = splits.make_splits(conn, "split-v1", cfg["particion"]["semilla"], cfg["particion"]["dev"],
                            cfg["particion"]["test"], audit_draw_ids=(smp["draw_id"],))
    log("     ", sp)
    runs = {}
    log("[8/9] Entrenamiento de lineas base")
    for name, backend in (("mayoritaria", "majority"), ("tfidf_lr", "tfidf_lr")):
        runs[name] = models.run_experiment(conn, name, backend, "split-v1", cfg)
        m = json.loads(conn.execute("SELECT metrics_json FROM model_run WHERE id=?", (runs[name],)).fetchone()[0])
        log(f"      {name}: macro-F1 dev={m['macro_f1_dev']:.3f} | test={m['macro_f1_test']:.3f} (n_train={m['n_train']})")
    log("[9/9] Auditoria (IC bootstrap, permutacion)")
    audits = {name: fairness.run_audit(conn, rid, "split-v1", cfg) for name, rid in runs.items()}
    for name, a in audits.items():
        g = a["brechas"]["macro_f1"]
        log(f"      {name}: n={a['n_auditoria']} (MARCADO={a['n_marcado']}, NO_MARCADO={a['n_no_marcado']}, "
            f"requerido por grupo={a['n_requerido_por_grupo']})")
        log(f"        D(macro-F1)={g['estimado']:+.3f}  IC95=[{g['ic'][0]:+.3f}, {g['ic'][1]:+.3f}]  "
            f"p permutacion (1 cola)={g['p_permutacion_una_cola']:.4f}")
        log(f"        H1: {a['decision_H1']}")
        log(f"        H2: {a['decision_H2_brecha_NEG_menos_POS']}")
    out_dir = Path(out_dir or ROOT / "reports")
    rep = report.build_report(conn, cfg, "split-v1", runs, audits, out_dir)
    log("Informe:", rep["markdown_path"])
    conn.close()
    return {"ingesta": ing, "muestra": smp, "piloto": pil, "anotaciones_r1": done, "anotaciones_r2": done2,
            "gold": gold, "acuerdo": pw, "particion": sp, "runs": runs, "auditorias": audits, "informe": rep}
