# Informe de corrida

> **ADVERTENCIA: corpus SINTETICO de prueba (fixture).** Las cifras demuestran que el flujo se ejecuta de
> extremo a extremo; **no tienen interpretacion cientifica** ni evidencian sesgo alguno.

- Fecha (UTC): 2026-10-29T04:27:51+00:00
- Hash de la configuracion congelada: `734d577b32b3e8f8`
- Python 3.12.3, scikit-learn 1.8.0, Fairlearn 0.15.0
- Fuentes: FIXTURE_SINTETICO

## 1. Ingesta
Leidas 1010, insertadas 847, duplicadas 10, rechazadas 153.
Resumen: `{"columnas_descartadas": ["Town", "UserName"], "rechazos": {"fuera_de_filtro_pais": 153}, "redacciones": {"url": 0, "email": 21, "mention": 0, "phone": 0}}`

## 2. Anotacion
Resenias en corpus: 847. Tareas: ronda 1 done=509; ronda 2 done=60

Concordancia (ronda 1, `dialect_label`):

| anotador_a   | anotador_b   | campo         |   n_comunes |   acuerdo_observado |   kappa_cohen | interpretacion   |
|:-------------|:-------------|:--------------|------------:|--------------------:|--------------:|:-----------------|
| anotador_1   | anotador_2   | dialect_label |          99 |               0.808 |         0.629 | sustancial       |
| anotador_1   | anotador_3   | dialect_label |          98 |               0.786 |         0.586 | moderado         |
| anotador_2   | anotador_3   | dialect_label |          98 |               0.857 |         0.718 | sustancial       |

Fleiss (piloto de calibracion, los 3 anotadores sobre el mismo lote): n=27, kappa=0.673 (sustancial)

Etiqueta final de grupo:

| dialect_gold       |   n |
|:-------------------|----:|
| MARCADO            | 131 |
| NO_MARCADO         | 107 |
| PENDIENTE_CONSENSO |   3 |

## 3. Particion `split-v1`

| role   |   n |
|:-------|----:|
| audit  | 211 |
| dev    |  64 |
| test   |  64 |
| train  | 508 |

## 4. Modelos y auditoria

### mayoritaria (run 1)
Macro-F1 dev=0.289, test=0.289; auditoria n=211 (MARCADO=126, NO_MARCADO=85).

Cobertura del emparejamiento: `{'n_marcado_total': 126, 'n_marcado_con_solape': 126, 'n_no_marcado_total': 85, 'estratos_sin_solape': []}`

| Metrica | Brecha D = NO_MARCADO - MARCADO | IC 95 % | p permutacion (1 cola) |
|---|---|---|---|
| accuracy | -0.000 | [0.000, 0.000] | 0.9970 |
| macro_f1 | 0.000 | [0.000, 0.000] | 0.9720 |
| recall_NEG | 0.000 | [0.000, 0.000] | 1.0000 |
| recall_NEU | 0.000 | [0.000, 0.000] | 1.0000 |
| recall_POS | 0.000 | [0.000, 0.000] | 1.0000 |
| sel_POS | 0.000 | [0.000, 0.000] | 1.0000 |
| gap_NEG_minus_POS | 0.000 | [0.000, 0.000] | 1.0000 |

Decision H1 (macro-F1, delta_min=0.05): **n_insuficiente (minimo por grupo 85 < 547 requerido)**  
Decision H2 (brecha NEG menos POS en recall): **n_insuficiente (minimo por grupo 85 < 547 requerido)**

Auxiliares (subconjunto NEG/POS, Fairlearn): `{'n_binario': 139, 'demographic_parity_difference': 0.0, 'equalized_odds_difference': 0.0, 'tasa_POS_real_por_grupo': {'MARCADO': 0.6, 'NO_MARCADO': 0.4444444444444444}}`

### tfidf_lr (run 2)
Macro-F1 dev=0.783, test=0.843; auditoria n=211 (MARCADO=126, NO_MARCADO=85).

Cobertura del emparejamiento: `{'n_marcado_total': 126, 'n_marcado_con_solape': 126, 'n_no_marcado_total': 85, 'estratos_sin_solape': []}`

| Metrica | Brecha D = NO_MARCADO - MARCADO | IC 95 % | p permutacion (1 cola) |
|---|---|---|---|
| accuracy | -0.028 | [-0.125, 0.069] | 0.7183 |
| macro_f1 | -0.015 | [-0.119, 0.086] | 0.6154 |
| recall_NEG | -0.002 | [-0.219, 0.208] | 0.5275 |
| recall_NEU | -0.017 | [-0.228, 0.209] | 0.5385 |
| recall_POS | -0.053 | [-0.160, 0.000] | 0.9960 |
| sel_POS | 0.016 | [-0.068, 0.093] | 0.3407 |
| gap_NEG_minus_POS | 0.051 | [-0.186, 0.278] | 0.3277 |

Decision H1 (macro-F1, delta_min=0.05): **n_insuficiente (minimo por grupo 85 < 547 requerido)**  
Decision H2 (brecha NEG menos POS en recall): **n_insuficiente (minimo por grupo 85 < 547 requerido)**

Auxiliares (subconjunto NEG/POS, Fairlearn): `{'n_binario': 139, 'demographic_parity_difference': 0.19760348583877996, 'equalized_odds_difference': 0.058823529411764705, 'tasa_POS_real_por_grupo': {'MARCADO': 0.6, 'NO_MARCADO': 0.4444444444444444}}`
