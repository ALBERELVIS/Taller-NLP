# Manifiesto del índice

Generado el 2026-09-02 14:04 UTC por `source/data_source/indice.py`.

| Campo | Valor |
| --- | --- |
| Modelo | `BAAI/bge-small-en-v1.5` |
| Dimensión | 384 |
| Vectores | 1749 |
| Tipo de índice | `IndexFlatIP` sobre vectores normalizados |
| Dispositivo | cuda |
| Tiempo de codificación | 9.8 s |
| SHA-256 de `chunks.jsonl` | `388ff3671742c2248e8f5cb1c75afbc786edfa0dc6f72a62d1fadf2310ec82b2` |

## Cómo consultarlo

La puntuación es **similitud coseno directa**: los vectores están
normalizados y el índice usa producto interno.

La consulta lleva prefijo; los fragmentos indexados, no:

```python
PREFIJO = "Represent this sentence for searching relevant passages: "
v = modelo.encode([PREFIJO + consulta], normalize_embeddings=True)
puntuaciones, posiciones = indice.search(v.astype('float32'), k)
filas = meta.iloc[posiciones[0]]
```

Omitir el prefijo no da ningún error: solo recupera peor.

## Alineación

La fila *i* de `chunks_meta.parquet` describe el vector *i*. Si el
hash de `chunks.jsonl` de arriba no coincide con el del corpus que
se esté usando, **el índice y los metadatos no se corresponden** y
el retrieval devolverá texto equivocado sin avisar. Regenerar
siempre las dos piezas a la vez.
