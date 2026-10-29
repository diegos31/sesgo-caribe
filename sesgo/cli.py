"""Linea de comandos: python -m sesgo <comando>."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import agreement, annotation, db, design, fairness, ingest, models, report, sampling, splits
from .config import DEFAULT_DB, ROOT, load_config
from .lexicon import load_lexicon


def _conn(a):
    c = db.connect(a.db)
    db.init_db(c)
    return c


def main(argv=None):
    p = argparse.ArgumentParser(prog="sesgo", description="Auditoria de desempeno por marcas dialectales en resenias")
    p.add_argument("--db", default=str(DEFAULT_DB))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db")
    s = sub.add_parser("ingest"); s.add_argument("file"); s.add_argument("--source", required=True)
    s.add_argument("--country"); s.add_argument("--min-chars", type=int, default=20)
    s.add_argument("--map", nargs="*", default=[], help="clave=Columna (title,text,polarity,type,country)")
    s = sub.add_parser("feasibility"); s.add_argument("--lexicon", required=True); s.add_argument("--country")
    s = sub.add_parser("sample"); s.add_argument("--name", required=True); s.add_argument("--seed", type=int, required=True)
    s.add_argument("--n-random", type=int, required=True); s.add_argument("--n-lexicon", type=int, default=0)
    s.add_argument("--lexicon", required=True); s.add_argument("--country")
    s = sub.add_parser("assign"); s.add_argument("--draw", type=int, required=True)
    s.add_argument("--annotators", required=True); s.add_argument("--k", type=int, default=2)
    sub.add_parser("flag-disagreements")
    sub.add_parser("gold")
    sub.add_parser("agreement")
    s = sub.add_parser("split"); s.add_argument("--version", required=True)
    s.add_argument("--audit-draws", nargs="*", type=int)
    s = sub.add_parser("train"); s.add_argument("--name", required=True); s.add_argument("--backend", required=True, choices=list(models.REGISTRY))
    s.add_argument("--split", required=True)
    s = sub.add_parser("audit"); s.add_argument("--run", type=int, required=True); s.add_argument("--split", required=True)
    sub.add_parser("sample-size")
    s = sub.add_parser("demo"); s.add_argument("--boot", type=int, default=500)

    a = p.parse_args(argv)
    cfg = load_config()
    if a.cmd == "demo":
        from .demo import run_demo
        run_demo(ROOT / "work" / "demo.db", boot_B=a.boot)
        return
    if a.cmd == "sample-size":
        tm = cfg["tamano_muestra"]
        print("n minimo por grupo =", design.n_per_group(tm["p_no_marcado_supuesto"], tm["delta_deteccion"], tm["alpha"], tm["potencia"]))
        print(design.sample_size_table().to_string(index=False))
        return
    c = _conn(a)
    if a.cmd == "init-db":
        print("BD lista:", a.db)
    elif a.cmd == "ingest":
        cm = dict(kv.split("=", 1) for kv in a.map)
        print(json.dumps(ingest.ingest_file(c, a.file, a.source, cm, a.country, a.min_chars), ensure_ascii=False, indent=2))
    elif a.cmd == "feasibility":
        print(sampling.feasibility(c, load_lexicon(a.lexicon), a.country).to_string(index=False))
    elif a.cmd == "sample":
        print(sampling.draw_sample(c, a.name, a.seed, a.n_random, a.n_lexicon, load_lexicon(a.lexicon), a.country))
    elif a.cmd == "assign":
        al = a.annotators.split(",")
        annotation.add_annotators(c, al)
        print("tareas creadas:", annotation.assign_independent(c, a.draw, al, a.k))
    elif a.cmd == "flag-disagreements":
        print("tareas de ronda 2:", annotation.flag_disagreements(c))
    elif a.cmd == "gold":
        print(annotation.compute_gold(c))
    elif a.cmd == "agreement":
        print(agreement.pairwise_label_agreement(c, "dialect_label").round(3).to_string(index=False))
        print(agreement.pairwise_label_agreement(c, "prejudice_label").round(3).to_string(index=False))
        print(agreement.token_level_agreement(c, "DIALECT").round(3).to_string(index=False))
    elif a.cmd == "split":
        pt = cfg["particion"]
        print(splits.make_splits(c, a.version, pt["semilla"], pt["dev"], pt["test"], tuple(a.audit_draws) if a.audit_draws else None))
    elif a.cmd == "train":
        print("run_id:", models.run_experiment(c, a.name, a.backend, a.split, cfg))
    elif a.cmd == "audit":
        r = fairness.run_audit(c, a.run, a.split, cfg)
        print(json.dumps({k: r[k] for k in ("n_auditoria", "n_marcado", "n_no_marcado", "decision_H1", "decision_H2_brecha_NEG_menos_POS")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
