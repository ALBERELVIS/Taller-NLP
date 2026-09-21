# Manifiesto del corpus reconstruido

Generado por `agente.corpus.construir_corpus()`.

## Qué es esto

El corpus original se reparte en `corpus_miax_2026.zip`. Ese ZIP no estaba
disponible en este proyecto, así que `secciones.jsonl`, `chunks.jsonl` y
`xbrl_facts.parquet` se han reconstruido a partir de dos fuentes:

| Fichero | Reconstruido desde |
| --- | --- |
| `chunks.jsonl` | `dataset/indice/chunks_meta.parquet`, volcado directo |
| `secciones.jsonl` | los mismos fragmentos, recompuestos por `inicio_car`/`fin_car` |
| `xbrl_facts.parquet` | `data.sec.gov/api/xbrl/companyfacts`, 6 peticiones cacheadas |

## Contenido

| Campo | Valor |
| --- | --- |
| Secciones | 48 |
| Fragmentos | 1749 |
| Hechos XBRL | 137 |
| Conceptos XBRL distintos | 14 |
| Tokens totales | 649,119 |
| SHA-256 de `chunks.jsonl` | `89c116f73e6b175f844140d9d6ac1ff0e797aad7168d6cefdd974015dc040cfd` |

## Por qué el hash del manifiesto original no sirve aquí

`dataset/indice/MANIFEST.md` declara el SHA-256 del `chunks.jsonl` original y
avisa de que, si no coincide, índice y metadatos no se corresponden. Esa
comprobación no es aplicable a un fichero regenerado: dos serializaciones a
JSON del mismo contenido difieren en el orden de las claves, en el escapado de
los caracteres no ASCII y en los espacios, así que el hash sería distinto
aunque el contenido fuese idéntico.

Se sustituye por dos comprobaciones que sí prueban lo que hay que probar, y
las dos son `assert` que detienen la construcción si fallan:

1. **Alineación índice-metadatos-fragmentos.** El índice FAISS tiene tantos
   vectores como filas tiene `chunks_meta.parquet` y como líneas tiene
   `chunks.jsonl`, y los `chunk_id` son el mismo conjunto en el mismo orden.
2. **Anclas del golden set oficial.** Las 13 frases literales que el golden
   set oficial declara como verdad, con sus desplazamientos exactos, se
   recortan del texto reconstruido y coinciden carácter a carácter. Cubren 5
   compañías, 2 ejercicios y 3 secciones distintas.

Y para la tabla XBRL, otras dos:

3. **Cifras del golden set.** Todos los `concept_xbrl` / `cifra_esperada` de
   las preguntas numéricas y comparativas coinciden al céntimo.
4. **Huecos declarados.** Amazon sigue sin `GrossProfit`, `Liabilities` ni
   `ResearchAndDevelopmentExpense`; Meta y Alphabet siguen sin `GrossProfit`.
   Rellenarlos habría destruido la parte de la evaluación que mide si el
   agente sabe decir que un dato no está.

## Lo único que se pierde

Los 27 huecos de 4 caracteres —todos en Microsoft, todos entre el final de un
párrafo y un encabezado— que el troceador consumió como separadores y que no
quedan cubiertos por ningún fragmento. Se rellenan con saltos de línea. Ningún
ancla del golden set los atraviesa, y tanto el evaluador de citas como
`recall@k` normalizan los espacios en blanco antes de comparar, así que no
pueden afectar a ninguna métrica.
