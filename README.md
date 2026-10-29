# sesgo-caribe: auditoría de desempeño por marcas dialectales en reseñas turísticas

Proyecto de Ingeniería I (UTB, 2026-02). Base técnica del Avance 4.

**Qué hace.** Ingesta reseñas (CSV/TSV/JSONL), reduce datos personales, estima disponibilidad, toma una muestra
estratificada, gestiona la anotación independiente por 2–3 anotadores (capa dialectal y capa de expresión
prejuiciosa por separado), mide concordancia, reserva un conjunto de auditoría, entrena líneas base de polaridad y
compara el desempeño entre textos MARCADOS y NO_MARCADOS con IC bootstrap y prueba de permutación.

**Qué NO afirma.** El grupo describe el *texto*, no la procedencia del autor. Ningún resultado del demo es un hallazgo:
el demo usa un corpus sintético (`data/fixture`) con términos inventados.

## Instalación y ejecución
```bash
pip install -r requirements.txt
python scripts/make_fixture.py          # regenera el fixture sintético (opcional)
python -m sesgo demo --boot 1000        # flujo completo -> reports/informe_corrida.md
python -m pytest -q                      # 28 pruebas
streamlit run app/streamlit_app.py       # UI (usa work/demo.db; cambie con SESGO_DB)
```

## Regenerar la evidencia de ejecución (hágalo en su propio equipo)
```bash
python -m sesgo demo --boot 1000 | tee evidencia/demo_log.txt
python -m pytest -v | tee evidencia/pytest_log.txt
cp reports/informe_corrida.md evidencia/informe_corrida_fixture.md
```
El demo imprime en pantalla los 9 pasos con sus resultados (concordancia de ambas capas, kappa por token, macro-F1 por modelo y decisión H1/H2). Los p de permutación son de **una cola** (H: D > 0).

## Con datos reales (Rest-Mex 2023, previo registro en la campaña)
```bash
python -m sesgo init-db
python -m sesgo ingest Rest-Mex_train.csv --source RestMex2023 --country Colombia   # confirme nombres de columna; use --map si difieren
python -m sesgo feasibility --lexicon data/lexicon/lexicon_verificado.csv --country Colombia
python -m sesgo sample --name piloto_calibracion --seed 11 --n-random 30 --lexicon data/lexicon/lexicon_verificado.csv --country Colombia
python -m sesgo assign --draw 1 --annotators ana,beto,carlos --k 3        # piloto: 3 anotadores sobre el mismo lote
python -m sesgo sample --name muestra_principal --seed 20261028 --n-random 400 --n-lexicon 400 --lexicon ...
python -m sesgo assign --draw 2 --annotators ana,beto,carlos --k 2
# ... anotación en la UI ...  luego:
python -m sesgo flag-disagreements && python -m sesgo gold && python -m sesgo agreement
python -m sesgo split --version split-v1 --audit-draws 2
python -m sesgo train --name tfidf --backend tfidf_lr --split split-v1
python -m sesgo audit --run 1 --split split-v1
```
BETO / RoBERTuito: `pip install -r requirements-beto.txt` y `--backend beto` o `robertuito` (requiere Hugging Face Hub; ajuste fino con GPU). **No se ejecutaron en este avance.**

## Estructura
`sesgo/` servicios (db, ingest, text, spans, lexicon, sampling, annotation, agreement, splits, models, fairness, report, design, demo, cli) · `app/streamlit_app.py` · `tests/` · `configs/experimento.json` (parámetros congelados) · `docs/guia_anotacion_v1.md` · `evidencia/` (registros de ejecución).

## Pendientes (ver informe)
Lexicón verificado con fuente, descarga de Rest-Mex, calibración de anotadores, ejecución de BETO/RoBERTuito, revisión jurídica institucional, validación del `delta_min` con el docente.
