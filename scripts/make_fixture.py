"""Genera un corpus SINTETICO de prueba (fixture). NO son datos de investigacion.

Los terminos 'marcadores' y 'prejuiciosos' son palabras INVENTADAS: asi se prueba
todo el flujo sin afirmar nada sobre ninguna variedad lingüistica real ni
reproducir estereotipos. Reproducible: semilla fija.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sesgo.text import normalize_text, text_hash  # noqa: E402

SEED = 20261028
MARKERS = ["zumbarro", "mucarra", "chiriguel", "pocotán", "bambarú", "tolosón"]
PEJ = ["bruzaco", "fulgardo"]
ROLES = ["meseros", "guías", "taxistas", "vendedores"]

OPEN = {
    "Hotel": ["Nos hospedamos tres noches en este hotel", "Estuvimos en el hotel por trabajo", "Reservamos una habitación para el fin de semana"],
    "Restaurant": ["Fuimos a cenar a este restaurante", "Almorzamos aquí un sábado", "Probamos el restaurante por recomendación de un amigo"],
    "Attractive": ["Visitamos este lugar con la familia", "Hicimos el recorrido un domingo", "Fuimos a conocer este sitio turístico"],
}
ASPECT = {
    "Hotel": ["la habitación", "el desayuno", "la piscina", "la recepción", "el aire acondicionado"],
    "Restaurant": ["el plato fuerte", "el mesero", "el postre", "la carta", "el tiempo de espera"],
    "Attractive": ["el guía", "la entrada", "la vista", "el recorrido", "la señalización"],
}
POS = ["{a} fue excelente", "{a} estuvo muy bien", "{a} superó nuestras expectativas", "quedamos encantados con {a}", "{a} es muy bueno"]
NEG = ["{a} fue pésimo", "{a} estuvo muy mal", "{a} nos decepcionó por completo", "no volvería por {a}", "{a} es muy deficiente"]
NEU = ["{a} fue aceptable", "{a} estuvo normal, ni bueno ni malo", "{a} cumple sin más", "{a} tiene cosas por mejorar"]
CLOSE_POS = ["Lo recomiendo.", "Volveríamos sin dudarlo.", "Gran experiencia en general."]
CLOSE_NEG = ["No lo recomiendo.", "Fue una mala experiencia.", "Esperábamos mucho más."]
CLOSE_NEU = ["Una experiencia regular.", "Depende de lo que busque cada quien."]
TITLES = {"pos": ["Muy buena experiencia", "Lo recomiendo", "Excelente lugar"],
          "neg": ["No volvería", "Decepcionante", "Mala experiencia"],
          "neu": ["Regular", "Aceptable", "Ni bueno ni malo"]}
DAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def compose(rng, ptype, pol, marked, pej):
    sent = "pos" if pol >= 4 else ("neg" if pol <= 2 else "neu")
    bank = {"pos": POS, "neg": NEG, "neu": NEU}
    closer = {"pos": CLOSE_POS, "neg": CLOSE_NEG, "neu": CLOSE_NEU}[sent]
    parts = [f"{rng.choice(OPEN[ptype])} el {rng.choice(DAYS)}."]
    for _ in range(rng.randint(1, 3)):
        use = sent if rng.rand() > 0.12 else rng.choice(["pos", "neg", "neu"])
        a = rng.choice(ASPECT[ptype])
        s = rng.choice(bank[use]).format(a=a)
        parts.append(s[0].upper() + s[1:] + ".")
    if marked:
        for m in rng.choice(MARKERS, size=rng.randint(1, 3), replace=False):
            parts.append(rng.choice([f"Todo estaba bien {m}.", f"¡Qué {m} el ambiente!", f"El lugar es {m}, la verdad."]))
    if pej:
        parts.append(f"Los {rng.choice(ROLES)} de por aquí son unos {rng.choice(PEJ)}.")
    parts.append(rng.choice(closer))
    if rng.rand() < 0.02:
        parts.append("Escríbanme a viajero@ejemplo.com")
    return rng.choice(TITLES[sent]), " ".join(parts)


def main(n=1000):
    rng = np.random.RandomState(SEED)
    rows, truth = [], {}
    for i in range(n):
        country = rng.choice(["Colombia", "Mexico", "Cuba"], p=[0.85, 0.08, 0.07])
        ptype = rng.choice(["Hotel", "Restaurant", "Attractive"], p=[0.35, 0.30, 0.35])
        pol = int(rng.choice([1, 2, 3, 4, 5], p=[0.08, 0.09, 0.17, 0.30, 0.36]))
        marked = rng.rand() < 0.33
        pej = rng.rand() < 0.07
        title, body = compose(rng, ptype, pol, marked, pej)
        if rng.rand() < 0.25:  # ruido de calificacion: la estrella no siempre coincide con el texto
            pol = int(np.clip(pol + rng.choice([-1, 1]), 1, 5))
        rows.append({"Title": title, "Review": body, "Polarity": pol, "Type": ptype, "Country": country,
                     "Town": "Ciudad Ejemplo", "UserName": f"usuario_{i}"})
        text, _ = normalize_text(f"{title}. {body}")
        dspans = [{"start": m.start(), "end": m.end(), "surface": m.group(0)}
                  for m in re.finditer("|".join(MARKERS), text)]
        pspans = [{"start": m.start(), "end": m.end(), "surface": m.group(0)}
                  for m in re.finditer("|".join(PEJ), text)]
        truth[text_hash(text)[:16]] = {"id": text_hash(text)[:16], "dialect": dspans, "prej": pspans,
                                      "target": "PROFESION_OFICIO" if pspans else None}
    rows += rows[:10]  # duplicados exactos para probar la deduplicacion
    out = ROOT / "data" / "fixture"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "reviews_fixture.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(out / "truth_fixture.jsonl", "w", encoding="utf-8") as fh:
        for t in truth.values():
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"fixture: {len(rows)} filas (incluye 10 duplicadas) -> {out}")


if __name__ == "__main__":
    main()
