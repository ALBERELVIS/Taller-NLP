# Agente investigador sobre informes 10-K de la SEC

Práctica de *LLMs aplicados a Finanzas* · MIAX

Albert Martin Garcia · Jesús Buissón Poyatos

Un agente que responde preguntas en lenguaje natural sobre doce informes
anuales —seis tecnológicas, ejercicios fiscales 2024 y 2025— eligiendo en cada
paso cuál de cuatro herramientas usar. Cada respuesta sale estructurada: la
prosa, la cifra, su unidad, la compañía, el ejercicio, de dónde sale el dato
y la frase literal del informe que lo respalda.

El día de la presentación basta con clonar el repositorio, poner una clave y
llamar a dos funciones. No hay que abrir ningún notebook.

```python
from agente import responder, evaluar

responder("¿Cuál fue el revenue de NVIDIA en FY2025?")
evaluar("holdout.jsonl")
```

---

## La idea

Un RAG clásico recupera *k* fragmentos y luego genera. Eso funciona cuando la
respuesta cabe en un párrafo parecido a la pregunta. En un 10-K deja de
funcionar en cuanto ocurre una de estas tres cosas:

- la respuesta está en dos sitios (comparar FY2024 con FY2025);
- el dato exacto no está en la prosa, sino en un hecho XBRL ya estructurado;
- lo que se pregunta no existe en el corpus, y devolver «los *k* fragmentos
  más parecidos» fabrica una respuesta.

Por eso el retrieval no es la arquitectura: es una herramienta más, al lado
de una consulta numérica exacta y de la lectura completa de una sección. El
modelo decide cuál usar, cuántas veces y con qué argumentos.

De ahí sale el criterio de corrección. Acertar el revenue de NVIDIA leyéndolo
de una tabla partida cuenta como fallo, aunque el número coincida. Ese camino
no se sostiene el día en que el troceador corte la tabla por la mitad, y en
este corpus el 41 % de los fragmentos lleva una tabla partida dentro. Hay un
evaluador escrito para detectar exactamente ese caso.

---

## El corpus

Seis emisores (NVDA, MSFT, AAPL, GOOGL, META, AMZN), dos ejercicios (FY2024 y
FY2025) y cuatro apartados del 10-K:

| Item | Qué es |
| --- | --- |
| 1A | Factores de riesgo |
| 7 | MD&A: la dirección explicando sus propios resultados |
| 7A | Riesgo de mercado |
| 8 | Estados financieros |

Eso son **48 secciones**, **1.749 fragmentos** de unos 500 tokens con 80 de
solape, **649.119 tokens** de prosa y **137 hechos XBRL** de 14 conceptos
us-gaap. El índice vectorial que vino con la práctica es un FAISS
`IndexFlatIP` sobre vectores normalizados de `BAAI/bge-small-en-v1.5`
(384 dimensiones). La consulta lleva el prefijo que pide BGE; los fragmentos
indexados no. Omitirlo no da error: solo recupera peor.

Tres detalles del corpus que cambian las respuestas:

- **`fiscal_year` es el año de cierre, no el de presentación.** Los seis
  emisores cierran en cuatro meses distintos. El FY2025 de NVIDIA cerró en
  enero de 2025; el de Alphabet, en diciembre de 2025.
- **El concepto de ingreso no es el mismo en todas.** NVIDIA usa `Revenues`.
  Apple, Microsoft, Meta y Amazon usan
  `RevenueFromContractWithCustomerExcludingAssessedTax`. Alphabet etiqueta los
  dos en FY2024 y solo `Revenues` en FY2025. Hay que mirar la tabla, no
  copiar el concepto de otra compañía.
- **Hay huecos reales.** Amazon no reporta `GrossProfit`, `Liabilities` ni
  `ResearchAndDevelopmentExpense`. Meta y Alphabet tampoco reportan
  `GrossProfit`. Preguntar por el margen bruto de Amazon tiene como respuesta
  correcta que ese dato no está.

El material de clase llegó con el índice (`dataset/indice/`) y sin los textos
ni la tabla XBRL. El notebook `00` los reconstruye y comprueba la
reconstrucción contra el golden set oficial, que es una verdad externa. El
detalle está en `corpus/MANIFIESTO.md`.

---

## Cómo responde

### Las cuatro herramientas

Los nombres y los parámetros son contrato: el evaluador de trayectoria y las
preguntas ciegas los buscan tal cual. El modelo no ve el cuerpo de la
función. Ve el nombre, los argumentos y el docstring, y con eso decide si
llama. Escribir ese texto es parte del sistema.

| Herramienta | Para qué | Coste |
| --- | --- | --- |
| `list_available()` | Qué compañías, ejercicios y secciones hay | Gratis |
| `get_xbrl_fact(ticker, fiscal_year, concept)` | La cifra exacta, tal como la reportó la compañía | Una consulta local |
| `search_filings(query, ticker, fiscal_year, item, k=5)` | Fragmentos de texto, cada uno con su `chunk_id` | Embeddings + índice |
| `read_section(ticker, fiscal_year, item)` | El texto íntegro de una sección | Hasta decenas de miles de tokens |

La política de enrutado, la misma en los tres perfiles, cabe en cuatro
reglas. Cualquier cifra sale de `get_xbrl_fact`, aunque el número ya haya
aparecido en un fragmento. Las explicaciones, los riesgos y el comentario de
la dirección salen de `search_filings`, con la consulta en inglés y con el
vocabulario del propio informe. Una comparativa son dos consultas XBRL —una
por ejercicio— y después una búsqueda en el Item 7 para la frase en la que la
dirección explica la variación. `read_section` es el último recurso. Si la
compañía no reportó el concepto, la respuesta correcta es decirlo:
`fuente="ninguna"` y `cifra` vacía.

### Lo que devuelve

`RespuestaFinanciera` (`agente/esquema.py`). Los ocho campos del enunciado
van literales: `respuesta`, `cifra`, `unidad`, `ticker`, `ejercicio`,
`fuente`, `cita`, `chunk_id`. `fuente` es `xbrl`, `texto`, `ambas` o
`ninguna`. La prosa va en el idioma de la pregunta; la cita va en inglés,
copiada palabra por palabra del fragmento. Hay dos campos añadidos,
`cifra_anterior` y el ejercicio anterior, con valor por defecto: una
comparativa puede dejar los dos números sin romper el contrato original.

### Los tres perfiles

El enunciado pide comparar el baseline con el sistema final. Dos filas dicen
«mejoró» y no dicen por qué. Hay una fila intermedia para poder atribuir el
cambio.

| Perfil | Retrieval de `search_filings` | Guardrail de cifras |
| --- | --- | --- |
| `baseline` | Denso plano, *k* = 5, sin filtros | No |
| `filtros` | Denso + filtro por ticker, ejercicio y sección | No |
| `final` | Reescritura de la consulta + híbrido BM25/RRF + reranking | Sí |

El modelo, la temperatura, el prompt, el esquema, los docstrings, *k* y los
límites de llamadas son los mismos en los tres. Si el baseline llevara además
un prompt peor, la tabla mediría dos cosas a la vez.

Los límites están en los tres: 8 llamadas a herramienta y 10 al modelo por
pregunta. Son una red de seguridad. Sin ellos, una pregunta cuyo dato no
existe puede dejar al baseline dando vueltas y la evaluación no termina. El
perfil `final` añade, encima, un middleware que extrae la cifra afirmada y la
contrasta con XBRL al 1 % relativo. Si no cuadra, devuelve el desajuste al
modelo **una sola vez**. Un verificador que se queja siempre es un bucle con
otro nombre.

La fusión híbrida suma posiciones, no puntuaciones: un coseno y un BM25 no
viven en la misma escala. Reciprocal Rank Fusion usa `1 / (k + posición)`.

El modelo se elige al ejecutar. Se prueba `gpt-5-mini`, `gpt-4.1-mini`,
`gpt-4o-mini`, `gpt-4.1` y `gpt-5`, en ese orden, y se queda el primero al
que la cuenta tiene acceso. El resultado se cachea en `.cache/`. Se puede
forzar con la variable de entorno `MIAX_MODELO`. La familia `gpt-5` solo
admite temperatura 1; en el resto se fija 0, porque una evaluación con
temperatura alta no se puede repetir.

---

## Cómo se sabe si ha acertado

Tres evaluadores, y cada uno puede devolver «no aplica» (`None`). Una
extractiva no tiene cifra esperada: contarla como fallo de cifra hundiría la
columna por la composición del conjunto, no por el sistema.

| Evaluador | Qué comprueba |
| --- | --- |
| `cita_correcta` | El `chunk_id` existe, es de la compañía y el ejercicio correctos, y el fragmento contiene la cita |
| `cifra_coincide_xbrl` | La cifra cuadra con XBRL dentro del 1 % relativo. El mismo umbral que el guardrail |
| `uso_la_tool_correcta` | La trayectoria pasó por la herramienta que correspondía |

`cita_fundamentada` aprieta la primera: no basta con que la cita sea real,
tiene que ser la frase que responde. Una cita verdadera del año equivocado
suspende, y en estos informes eso pasa de verdad: los factores de riesgo se
copian casi palabra por palabra de un ejercicio a otro.

Los conjuntos viven en `golden/`:

| Fichero | Preguntas | Para qué está |
| --- | --- | --- |
| `golden_set.jsonl` | 20 (6 extractivas, 7 numéricas, 7 comparativas) | El oficial. No lo hemos escrito nosotros |
| `golden_set_propio.jsonl` | 20 (7, 7 y 6) | El nuestro. Mínimo de 6 comparativas que pide el enunciado |
| `golden_set_ausencias.jsonl` | 7, todas numéricas | El dato no está. Mide si el agente lo dice o se lo inventa |
| `holdout_simulado.jsonl` | 10 (4, 3 y 3) | Ensayo del día 24. Se evalúa una vez y no se usa para iterar |
| `golden_set_ejemplo.jsonl` | 3 | Respaldo de la sesión 1, una pregunta de cada familia |

En las extractivas la verdad es una frase literal con sus desplazamientos,
no un `chunk_id`. El notebook 04 cambia el troceado, y en cuanto se toca la
ventana todos los identificadores son otros. Una métrica anclada al
identificador daría cero justo en el experimento hecho para mejorarla.
Ninguna cifra del golden set propio se teclea: se lee de
`xbrl_facts.parquet` al construirlo.

`recall@5` pregunta una sola cosa: ¿hay, entre los cinco primeros, un
fragmento del documento correcto que contenga entera la frase ancla? El
notebook 04 se la pasa al retriever **con los filtros ya resueltos** desde el
golden set. Ese número es un techo. El notebook 05 mide el mismo recall sobre
las búsquedas que el agente decidió hacer él. Las dos cifras no tienen por
qué coincidir, y la diferencia es el hueco entre «el retriever podría» y «el
agente supo pedírselo».

---

## Qué salió de medirlo

Las cifras vivas están en `resultados/` y en
`informe/Informe_MIAX_Agente_10K.pdf`, que se genera leyéndolas. Lo que sigue
es la lectura de esa ejecución, con `gpt-5-mini`.

**El retriever, con filtros de oráculo** (26 preguntas con ancla, los dos
golden sets juntos):

| Configuración | recall@5 |
| --- | ---: |
| Denso plano | 27 % (7/26) |
| + filtro de metadatos | 54 % (14/26) |
| + híbrido BM25, sin reescritura | 46 % (12/26) |
| + reescritura de la consulta | 77 % (20/26) |
| Reescritura + híbrido + reranking (el sistema final) | 77 % (20/26) |

El filtro de metadatos duplica el recall. La reescritura —traducir la
pregunta al inglés del informe antes de buscar— es el salto grande. El
híbrido BM25, aplicado a la consulta sin reescribir, **baja** el recall: un
arreglo correcto sobre el papel que en este corpus no aporta. Re-trocear
(256, 400, 512 o 768 tokens) se queda entre el 50 % y el 58 %. Cambiar el
modelo de embeddings (`all-MiniLM-L6-v2`, `bge-base-en-v1.5`,
`text-embedding-3-small`) no supera a `bge-small` con filtros. El sistema
final se queda con la reescritura, el híbrido y el reranker porque es lo que
mejor coloca la frase ancla, no porque cada pieza mejore la métrica por
separado. El reranker empata con la reescritura sola.

**El agente entero**, misma pregunta, tres perfiles. Acierto global:

| Conjunto | baseline | filtros | final | recall@5 del final |
| --- | ---: | ---: | ---: | ---: |
| Oficial (20) | 80 % | 75 % | 80 % | 54 % |
| Propio (20) | 67 % | 75 % | 70 % | 62 % |
| Ausencias (7): dice que no está | 86 % | 86 % | 100 % | — |

El recall de punta a punta sí sube (en el oficial, del 31 % del baseline al
54 % del final; en el propio, del 23 % al 62 %). El acierto global casi no.
En el conjunto propio la fila intermedia, que solo añade filtros, acierta más
que el sistema final. El coste se mantiene alrededor de **0,7 centavos por
pregunta**; la latencia del final ronda los **23 segundos**, frente a unos
17 del baseline, por la llamada extra de reescritura y el cross-encoder.

Donde el final sí se separa es en las ausencias: las siete veces dice que el
dato no está, y ninguna vez inventa una cifra. El baseline y el de filtros
se inventan una de las siete.

Esa es la lectura que pide la defensa. Mejorar el retrieval mejora el
retrieval. No compra, por sí solo, más respuestas correctas de punta a punta,
y la fila intermedia existe precisamente para que esa frase se pueda decir
con los dos números delante.

---

## Puesta en marcha

### 1. La clave

Copia `api_key.example.txt` a `api_key.txt` y sustituye el contenido por tu
clave, en una sola línea y sin comillas:

```
sk-proj-...
```

`api_key.txt` está en `.gitignore`. Si prefieres no tocar ficheros, la
variable de entorno `OPENAI_API_KEY` tiene prioridad. No hay ninguna clave
escrita en un notebook ni en un módulo: todo pasa por `agente/config.py`.

### 2. Las dependencias

```bash
pip install -r requirements.txt
```

No hace falta un entorno virtual. Las versiones van fijadas con `==`.
LangChain publica cada pocos días, y un rango abierto rompe un notebook sin
que se haya tocado una línea.

La primera búsqueda descarga `BAAI/bge-small-en-v1.5` (unos 130 MB). El
notebook 04, si se ejecuta entero, descarga además el cross-encoder
`BAAI/bge-reranker-base` (unos 280 MB). A partir de ahí quedan en la caché de
HuggingFace.

### 3. Comprobar que se puede llamar al modelo

```bash
python -c "from agente import config; print(config.resumen_configuracion())"
```

Si imprime un modelo, la cuenta responde. Si no, el mensaje distingue tres
problemas distintos: no hay clave, la clave no vale, o la cuenta no tiene
crédito ni acceso a ninguno de los candidatos.

`import agente` es instantáneo a propósito. Cargar FAISS y el codificador
tarda varios segundos, y el notebook 00 importa el paquete antes de que el
corpus exista. `responder` y `evaluar` se cargan la primera vez que se usan.

---

## Uso

Desde Python, perfil `final` si no se dice otra cosa:

```python
from agente import responder, evaluar

estado = responder("¿Cuánto creció el revenue de NVIDIA entre FY2024 y FY2025?")
print(estado["structured_response"].respuesta)
print(estado["structured_response"].cita)

tabla, resumen = evaluar("golden/golden_set.jsonl")
tabla_base, _ = evaluar("golden/golden_set.jsonl", perfil="baseline")
```

Desde la terminal:

```bash
python -m agente.interfaz responder "¿Cuál fue el revenue de NVDA en FY2025?"
python -m agente.interfaz evaluar golden/holdout_simulado.jsonl --perfil final
```

`evaluar()` acepta una ruta a un JSONL o una lista de diccionarios. El
enunciado la especifica con una ruta, y así se comporta; la lista sirve para
evaluar un subconjunto sin escribir un fichero temporal. Cada pregunta estrena
`thread_id`: si compartieran hilo, la número quince vería lo que se consultó
en las catorce anteriores y podría acertar por contagio. Un fallo en una
pregunta se anota en la fila y la evaluación sigue.

---

## Orden de los notebooks

El `00` va primero: deja `corpus/` montado. El resto lee de ahí, y cada uno
escribe los ficheros que el siguiente espera.

| Notebook | Qué hace | Modelo | Tarda |
| --- | --- | --- | --- |
| `00_Preparacion_del_corpus.ipynb` | Reconstruye secciones, fragmentos y hechos XBRL, y los valida contra el golden set oficial | No | ~3 min la primera vez (seis peticiones a la SEC, cacheadas) |
| `01_Herramientas_y_Bucle_Alumno.ipynb` | Sesión 1, resuelta: las cuatro herramientas, el bucle ReAct a mano y el mismo agente con el framework | Sí | ~5 min |
| `02_Robustez_y_Evaluacion_Alumno.ipynb` | Sesión 2, resuelta: abrir el retrieval, los guardrails y los tres evaluadores | Sí | ~10 min |
| `03_Golden_set_propio.ipynb` | Construye y valida el golden set propio y el de ausencias | No | ~1 min |
| `04_Retrieval_experimentos.ipynb` | Mide el `recall@5` de las configuraciones de retrieval, troceado y embeddings | Sí | ~8 min |
| `05_Evaluacion_baseline_vs_final.ipynb` | Evalúa los tres perfiles sobre los tres conjuntos y guarda los resultados crudos | Sí | ~70 min |
| `06_Informe.ipynb` | Compone el PDF a partir de `resultados/`. No recalcula nada | No | ~10 s |
| `07_Ensayo_holdout.ipynb` | Diez preguntas nuevas, cronometradas, como el día de las preguntas ciegas | Sí | ~8 min |

Ejecutar el recorrido entero con `gpt-5-mini` cuesta del orden de **uno o
dos dólares**. Casi todo está en el notebook 05, que hace 141 invocaciones
del agente (3 perfiles × 47 preguntas). El 06 y el 07 leen de disco lo que
ya está medido; el 07 solo añade las diez del ensayo.

---

## Mapa del repositorio

```
agente/                         El paquete. Lo que se importa en un clon limpio
  config.py                     Rutas, clave, modelo, precios, tolerancias
  corpus.py                     Carga y reconstrucción del corpus
  esquema.py                    RespuestaFinanciera y el validador del golden set
  herramientas.py               Las cuatro herramientas
  retrieval.py                  Denso, filtros, BM25+RRF, reescritura, reranking
  middleware.py                 Guardrail XBRL y topes de llamadas
  agente.py                     Los tres perfiles
  evaluadores.py                Cita, cifra, trayectoria y cita fundamentada
  golden.py                     Construcción de golden sets: anclas y cifras
  interfaz.py                   responder() y evaluar()
  informe.py                    El PDF

corpus/                         Lo que deja el notebook 00
  secciones.jsonl               Texto íntegro de las 48 secciones
  chunks.jsonl                  1.749 fragmentos
  xbrl_facts.parquet            137 hechos
  indice/                       FAISS + metadatos, alineados fila a fila
  MANIFIESTO.md                 Cómo se reconstruyó y cómo se comprobó

dataset/indice/                 El índice tal como llegó. No se modifica
golden/                         Los cinco conjuntos de preguntas
resultados/                     CSV y JSONL de retrieval, evaluación y tablas
informe/                        El PDF y las tres figuras
enunciado/                      El enunciado y los contextos de las dos sesiones
modulos/                        miax_s1.py, miax_s2.py y la traza de la demo

api_key.example.txt             Plantilla de la clave
api_key.txt                     La clave de verdad. En .gitignore
requirements.txt                Dependencias con la versión fijada
```

Los CSV de la raíz, `resultados_baseline.csv` y `resultados_final.csv`, son
un volcado parcial de una ejecución intermedia. La tabla del informe se lee
de `resultados/tabla_comparativa.csv` y `resultados/tabla_ausencias.csv`.

---

## Antes de entregar

La última celda del notebook 06 comprueba por código que el PDF existe, que
el golden set propio tiene 20 preguntas y al menos 6 comparativas, que hay
tablas de retrieval, de comparación y de ausencias, que `api_key.txt` sigue
en `.gitignore`, que ninguna clave está escrita en `agente/`, que
`requirements.txt` fija versiones y que `responder` y `evaluar` se importan.

Dos constantes, por si hay que tocarlas:

- **`INTEGRANTES`**, al principio de `agente/informe.py`. Son los nombres de
  la portada. Mientras alguno contenga el marcador `NOMBRE Y APELLIDOS`, la
  portada lo imprime en rojo y el generador avisa por consola.
- **`AUTOR_GOLDEN`**, en `agente/config.py`. Es el campo `autor` de los
  golden sets propios. Ahora vale `grupo-nlp`.

---

## Notas

**El corpus se reconstruye, y la prueba no es un hash.** El ZIP de textos no
llegó. Las secciones se reensamblan con los desplazamientos de
`chunks_meta.parquet` y los hechos XBRL se descargan de la API pública de la
SEC (seis peticiones, cacheadas en `.cache/sec/`). Comparar el hash de un
fichero recién generado consigo mismo no prueba nada: dos serializaciones del
mismo contenido difieren en el orden de las claves. Lo que sí se comprueba es
que las 13 anclas literales del golden set oficial coinciden carácter a
carácter, que las cifras oficiales cuadran al céntimo y que los huecos
declarados —Amazon sin beneficio bruto, Meta y Alphabet sin beneficio bruto—
siguen vacíos. Rellenarlos habría destruido la parte de la evaluación que
mide si el agente sabe decir que un dato no está.

**`modulos/demo_traza.json` se puede regenerar.** La traza que se repartió en
clase es provisional y le falta una clave que `miax_s1.demo_apertura()` lee.
El script la reconstruye a partir de una ejecución real:

```bash
python modulos/generar_traza_demo.py --perfil final
```
