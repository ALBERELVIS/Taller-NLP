# Agente investigador sobre informes 10-K de la SEC

Práctica de LLMs aplicados a Finanzas · MIAX

Un agente que responde preguntas en lenguaje natural sobre doce informes 10-K
—seis emisores tecnológicos, ejercicios 2024 y 2025— decidiendo por sí mismo
cuál de cuatro herramientas usar en cada momento, y devolviendo siempre una
respuesta estructurada con la cifra, su unidad, el origen del dato y la cita
literal que lo respalda.

---

## Puesta en marcha

### 1. La clave de OpenAI

Abre `api_key.txt` y pega tu clave, sin comillas y sin nada más:

```
sk-proj-...
```

El fichero está en `.gitignore` y **nunca se sube al repositorio**. Si
prefieres no tocar ficheros, el proyecto también lee la variable de entorno
`OPENAI_API_KEY`, que tiene prioridad sobre el fichero.

No hay ninguna clave escrita en ningún notebook ni en ningún módulo. Todo pasa
por `agente/config.py`.

### 2. Las dependencias

```bash
pip install -r requirements.txt
```

**No hace falta ningún entorno virtual.** Las versiones van fijadas con `==` y
no con rangos: LangChain publica cada pocos días y una API que se mueve por
debajo rompe un notebook sin que se haya tocado una línea de código.

La primera ejecución descarga el modelo de *embeddings* `BAAI/bge-small-en-v1.5`
de HuggingFace (unos 130 MB). A partir de ahí queda en caché.

### 3. Comprobar que todo está en su sitio

```bash
python -c "from agente import config; print(config.resumen_configuracion())"
```

Si imprime un modelo, el sistema puede llamar a OpenAI. Si dice que no, el
mensaje explica por qué: falta la clave, la clave no vale o la cuenta no tiene
crédito. Son tres problemas distintos con tres soluciones distintas, y por eso
se distinguen.

---

## Uso

### Desde Python

```python
from agente import responder, evaluar

estado = responder("¿Cuánto creció el revenue de NVIDIA entre FY2024 y FY2025?")
print(estado["structured_response"].respuesta)
print(estado["structured_response"].cita)

tabla, resumen = evaluar("golden_set.jsonl")
```

### Desde la terminal

```bash
python -m agente.interfaz responder "¿Cuál fue el revenue de NVDA en FY2025?"
python -m agente.interfaz evaluar holdout.jsonl
```

`evaluar()` acepta tanto una ruta a un JSONL como una lista de diccionarios ya
cargada. El enunciado la especifica con una ruta y así se comporta; la
ampliación es compatible y sirve para evaluar subconjuntos sin escribir
ficheros temporales.

---

## Orden de ejecución de los notebooks

| Notebook | Qué hace | Necesita el modelo | Tarda |
| --- | --- | --- | --- |
| `00_Preparacion_del_corpus.ipynb` | Reconstruye `secciones.jsonl`, `chunks.jsonl` y `xbrl_facts.parquet`, y los valida contra el golden set oficial | no | ~3 min |
| `S1_Herramientas_y_Bucle_Alumno.ipynb` | Material de la sesión 1, resuelto: las cuatro herramientas y el bucle ReAct a mano | sí | ~5 min |
| `S2_Robustez_y_Evaluacion_Alumno.ipynb` | Material de la sesión 2, resuelto: retrieval, guardrails y evaluadores | sí | ~10 min |
| `03_Golden_set_propio.ipynb` | Construye y valida `golden_set_propio.jsonl` y `golden_set_ausencias.jsonl` | no | ~1 min |
| `04_Retrieval_experimentos.ipynb` | Mide el `recall@5` de ocho configuraciones de *retrieval* | sí | ~8 min |
| `05_Evaluacion_baseline_vs_final.ipynb` | Evalúa los tres perfiles sobre los tres conjuntos y guarda los resultados crudos | sí | ~70 min |
| `06_Informe.ipynb` | Genera el PDF a partir de `resultados/` | no | ~10 s |
| `07_Ensayo_holdout.ipynb` | Ensayo del día de la evaluación: diez preguntas nuevas, cronometrado | sí | ~8 min |

El notebook 00 hay que ejecutarlo antes que ninguno: es el que deja el corpus
montado. Los demás dependen solo de él y de los que escriben ficheros que ellos
leen, tal como indica la tabla.

**Coste.** Ejecutar todo de punta a punta con `gpt-5-mini` cuesta del orden de
**1 a 2 dólares**, casi todo en el notebook 05, que hace 141 invocaciones del
agente. Los notebooks 06 y 07 leen de disco y no repiten nada.

---

## Estructura

```
agente/                     El paquete. Todo lo que los notebooks importan.
  config.py                 Clave, modelo, precios, rutas y tolerancias
  corpus.py                 Reconstrucción y carga del corpus
  esquema.py                RespuestaFinanciera y el validador del golden set
  herramientas.py           Las cuatro herramientas del agente
  retrieval.py              Denso, filtros, BM25+RRF, reescritura y reranking
  middleware.py             Guardrail XBRL, topes de llamadas, human-in-the-loop
  agente.py                 Los tres perfiles: baseline, filtros, final
  evaluadores.py            Los tres evaluadores del §4, más cita_fundamentada
  golden.py                 Construcción de golden sets: anclas y cifras
  interfaz.py               responder() y evaluar(), con línea de órdenes
  informe.py                Generación del PDF

corpus/                     Reconstruido por el notebook 00
dataset/indice/             El índice FAISS que se entrega con la práctica
resultados/                 CSV y JSONL de todas las evaluaciones
informe/                    El PDF y sus figuras

golden_set.jsonl            El oficial, 20 preguntas
golden_set_propio.jsonl     Las 20 nuestras
golden_set_ausencias.jsonl  7 preguntas cuya respuesta es «ese dato no está»
holdout_simulado.jsonl      10 preguntas nuevas para el ensayo del notebook 07

api_key.txt                 Aquí va la clave. En .gitignore
requirements.txt            Dependencias con versiones fijadas
```

---

## Antes de entregar

Dos cosas que hay que tocar a mano:

1. **Los nombres de la portada del informe.** Están en la constante
   `INTEGRANTES`, al principio de `agente/informe.py`. Mientras conserven el
   marcador, la portada los imprime en rojo con una advertencia y el generador
   avisa por consola.
2. **El campo `autor` de los golden sets**, en `config.AUTOR_GOLDEN`, si el
   enunciado pide identificar al grupo de otra forma.

La última celda del notebook 06 comprueba por código las dos cosas y otras
nueve, incluida que no haya ninguna clave escrita en el paquete y que
`api_key.txt` siga en `.gitignore`.

---

## Notas

**El material recibido estaba incompleto.** Llegó el índice FAISS con sus
metadatos pero no los tres ficheros de corpus que las herramientas leen. El
notebook 00 los reconstruye: las secciones se reensamblan a partir de los
desplazamientos de `chunks_meta.parquet` y los hechos XBRL se descargan de la
API pública de la SEC. La validación no se hace comparando hashes —el hash de
un fichero que acabamos de generar coincide consigo mismo por construcción—
sino contra el golden set oficial, que es una verdad externa que no
controlamos.

**`demo_traza.json` se puede regenerar.** El fichero que se repartió en clase
es provisional y le falta una clave que la propia `miax_s1.demo_apertura()`
lee, por lo que la función termina en `KeyError`. El script
`generar_traza_demo.py` lo regenera a partir de una ejecución real:

```bash
python generar_traza_demo.py --perfil final
```
