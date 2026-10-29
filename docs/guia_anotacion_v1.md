# Guía de anotación v1 (borrador para calibración)

**Unidad de análisis:** la reseña (texto). Los grupos describen el *texto*, no al autor. La etiqueta de país de Rest-Mex corresponde al *lugar reseñado*; no se usa para inferir procedencia ni dialecto del autor.

## 0. Tres capas independientes (no se mezclan)

| Capa | Pregunta | Quién la define | Salida |
|---|---|---|---|
| A. Rasgo dialectal | ¿El texto contiene marcas léxicas u ortográficas de la variedad caribeña colombiana? | Anotador, contra lista de referencia verificada | `dialect_label` + tramos `DIALECT` |
| B. Expresión prejuiciosa | ¿El autor denigra o estereotipa a una persona o grupo? | Anotador | `prejudice_label` + tramos `PREJ` + objetivo |
| C. Desempeño del modelo | ¿El clasificador acierta igual en ambos grupos? | Sistema (no se anota) | métricas con IC |

La polaridad (1–5) **no se anota**: viene de la calificación original. El tema se toma de `place_type`.

## 1. Capa A: rasgo dialectal

- **MARCADO**: al menos un tramo con marca caribeña colombiana *verificable* (entrada de la lista de referencia con fuente, o patrón ortográfico documentado en la bibliografía del equipo).
- **NO_MARCADO**: ninguna marca.
- **AMBIGUO**: hay candidato, pero es un coloquialismo general del español, un regionalismo de otra zona, o no se puede verificar. Va a tercera anotación.
- Un término ofensivo **no** es dialectal por ser ofensivo, ni prejuicioso por ser dialectal.

## 2. Capa B: expresión prejuiciosa

- **PRESENTE**: el autor usa una expresión que denigra o estereotipa a personas o grupos (procedencia, etnia, género, oficio, nivel socioeconómico). Se marca el tramo y el **objetivo** (a quién se dirige), nunca un atributo inferido del autor.
- **AUSENTE**: críticas a negocios, precios o servicios sin generalizar sobre personas.
- **AMBIGUO**: ironía, cita de terceros, uso intragrupal, estereotipo "positivo".

## 3. Procedimiento

1. Leer la reseña completa. 2. Capa A. 3. Capa B. 4. Guardar. 5. No consultar a otros anotadores hasta la reunión de consenso.
Los tramos se seleccionan por **tokens completos**; el sistema guarda posiciones de carácter y deriva BIO.

## 4. Casos ambiguos y decisión

| Caso | Decisión |
|---|---|
| Regionalismo compartido con otras zonas de Colombia | AMBIGUO salvo que la fuente lo marque como caribeño |
| Elisión o aspiración de -s escrita ("lo' niño'") | Candidato; MARCADO solo si el patrón aparece ≥ 2 veces y el equipo lo documentó como regla |
| Palabra ofensiva sin objetivo humano ("este hotel es una basura") | Capa B: AUSENTE |
| Generalización sobre un grupo ("los X son todos unos Y") | Capa B: PRESENTE |
| Cita textual de lo que dijo un tercero | Capa B: AMBIGUO |
| Mezcla de idiomas, emojis, nombres propios | No cuentan como marca dialectal |
| Error de tecleo vs. grafía fonética | Si no hay patrón repetido: no es marca |

## 5. Calibración (≈ 1 mes)

Ronda piloto de 100 reseñas por anotador; reunión de consenso; ajuste de la guía; repetir hasta kappa de Cohen ≥ 0,61 (sustancial, Landis y Koch, 1977) en `dialect_label` y `prejudice_label`. Cada cambio sube la versión de la guía (el hash queda registrado).
