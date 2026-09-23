# Sesión 2 — Retrieval, robustez, guardrails y evaluación del agente 10-K

> Documento destilado para usar como contexto de implementación en Codex. Resume la sesión del 17/09 y la contrasta con el enunciado oficial. El foco es convertir el baseline de la sesión 1 en un sistema medido, trazable y robusto.

## 0. Objetivo de esta fase

La segunda sesión cambia la pregunta de “¿funciona el agente?” a:

- ¿por qué falla?
- ¿qué parte falla: retrieval, routing, modelo o datos?
- ¿cómo sabemos objetivamente que una mejora mejora algo?
- ¿cómo evitamos cifras o citas inventadas?
- ¿cómo comprobamos que el agente siguió la trayectoria correcta?

El resultado final debe ser un experimento reproducible, no una demo que “parece ir bien”.

## 1. Prioridad del contrato oficial

El enunciado oficial manda sobre cualquier comentario informal de clase.

Hay una diferencia importante que no debe perderse:

- En un momento de la sesión se comentó que, “por ahora”, llegar a la respuesta correcta usando una herramienta ineficiente podía considerarse acierto y discutirse después como optimización.
- El **enunciado oficial es más estricto**: acertar por el camino equivocado cuenta como fallo y existe un evaluador de trayectoria para detectarlo.

Por tanto, para la entrega final Codex debe implementar y evaluar el **routing correcto**, no solo la exactitud de la respuesta.

## 2. Empezar ejecutando y diagnosticando el baseline

La sesión 2 presupone que existe:

- un baseline ejecutable;
- un golden set propio;
- resultados guardados.

Antes de modificar nada, ejecutar el baseline y clasificar fallos. En la demostración aparecieron patrones como:

1. **No recuperar evidencia relevante** aunque existe.
2. **Recuperar de empresa/año/sección incorrectos**.
3. **Confundir información semánticamente parecida** en documentos financieros.
4. **No saber comparar dos ejercicios** con una sola recuperación.
5. **Inventar una cifra**.
6. **Usar una herramienta inadecuada o una trayectoria innecesariamente cara**.

Esta taxonomía es más útil que limitarse a un porcentaje agregado: indica qué parte de la arquitectura hay que modificar.

## 3. El retrieval se evalúa por recuperación, no por “sensación”

El baseline usa búsqueda densa sobre embeddings. La práctica exige mejorarla y **medir `recall@k` después de cada cambio**.

Conceptualmente, para una pregunta extractiva con un ancla textual conocida:

```text
recall@k = 1 si alguno de los k chunks recuperados contiene/respalda el ancla
           0 en caso contrario
```

Agregado sobre el conjunto:

```text
recall@k = preguntas con ancla recuperada en top-k / preguntas evaluables
```

El ancla de las preguntas extractivas es una frase literal del informe, no un `chunk_id`, porque al cambiar el chunking cambian los IDs.

## 4. El baseline denso no es suficiente

El índice inicial está construido con `BAAI/bge-small-en-v1.5` sobre 1.749 chunks. El manifiesto del índice especifica el prefijo que debe aplicarse a la **consulta** y no a los fragmentos; el baseline debe respetar esa configuración para que la comparación sea válida.

La clase remarca tres variables que pueden cambiar muchísimo el retrieval:

- **preprocesamiento/limpieza del texto**;
- **chunking**: tamaño, solapamiento, cortes y manejo de tablas;
- **modelo de embeddings**.

No asumir que un modelo de embeddings “más nuevo” debe producir scores numéricamente mayores. Lo que importa es la calidad del ranking/recuperación. Dos modelos pueden tener escalas de similitud distintas.

Por eso, cualquier umbral de similitud debe calibrarse empíricamente para el modelo concreto y no copiarse de otro modelo.

## 5. Limpieza y chunking: la calidad empieza antes del vector search

La sesión mostró por qué los documentos financieros son especialmente delicados:

- tablas rotas entre páginas;
- espacios/tabulaciones redundantes;
- números entre paréntesis que pueden significar negativos o referencias a notas según el contexto;
- estructuras muy repetitivas entre compañías y secciones.

Un mal parseo puede hacer imposible que un buen retriever encuentre la respuesta.

### Principios prácticos

- No cortar una tabla de forma arbitraria si puede evitarse.
- Preservar suficiente contexto semántico alrededor del dato.
- Eliminar ruido que no aporta significado y consume tokens.
- No hacer chunks enormes solo porque el modelo de embeddings los admita: más texto puede diluir la señal.
- Evaluar tamaños/solapamientos con el golden set en vez de elegirlos por intuición.

El enunciado entrega chunks iniciales de aproximadamente 500 tokens con solapamiento 80, pero permite mejorar el troceado.

## 6. Mejora obligatoria 1: filtro por metadatos

El primer arreglo de retrieval demostrado en clase fue filtrar por metadatos como:

- `ticker`;
- `fiscal_year`;
- `item`.

Si la pregunta es sobre NVIDIA FY2025 Item 1A, no tiene sentido dejar que chunks de otras compañías/años/secciones compitan en igualdad de condiciones.

### Comportamiento correcto de filtros opcionales

Los argumentos de `search_filings` son opcionales. Un filtro solo debe aplicarse si viene informado:

```python
if ticker is not None and row["ticker"] != ticker:
    continue

if fiscal_year is not None and int(row["fiscal_year"]) != int(fiscal_year):
    continue

if item is not None and row["item"] != item:
    continue
```

Después de aplicar filtros, acumular resultados hasta reunir `k` válidos; no devolver menos resultados simplemente porque los primeros vecinos globales pertenecían a otro ticker.

### Lección de implementación observada en clase

En pandas, una columna llamada `item` puede chocar con el método `Series.item`. Para columnas con nombres potencialmente conflictivos, usar acceso por corchetes (`row["item"]`) en vez de `row.item`.

### Resultado de demostración

En el notebook de clase, el filtro por metadatos elevó la métrica aproximada de alrededor de 30% a alrededor de 46%. Es un resultado **ilustrativo**, no una cifra objetivo para la entrega.

## 7. Mejora obligatoria 2: combinación BM25 + búsqueda densa

El enunciado exige implementar y medir una búsqueda híbrida que combine:

- una señal léxica tipo BM25;
- la búsqueda semántica densa.

La clase no fija un algoritmo de fusión obligatorio. La forma concreta de combinar ambas señales es una decisión de implementación que debe quedar documentada y, sobre todo, evaluada con `recall@k`.

No asumir que la híbrida siempre mejora. En la demostración concreta, una de las variantes híbridas no movió la métrica. El propio enunciado considera válido reportar un arreglo razonable que no mejora el resultado: **un resultado negativo medido también es un resultado**.

## 8. Mejora obligatoria 3: reescritura de consulta con un LLM

La sesión muestra la reescritura como un mini-agente/tool: antes de hacer retrieval, el sistema transforma la pregunta para que sea más recuperable.

Un uso importante es alinear el idioma de la consulta con el corpus. Los informes están en inglés; una pregunta del usuario en español puede reescribirse/traducirse al inglés antes de generar el embedding.

La reescritura debe conservar la intención y no inventar hechos. Debe mantener intactos los elementos estructurales relevantes:

- ticker;
- ejercicio fiscal;
- sección si está explícita;
- magnitud o concepto buscado.

En la demostración, tras el filtro de metadatos la reescritura llevó la métrica aproximadamente de 46% a 69%. De nuevo, estas cifras son solo referencia del notebook, no valores esperados ni garantizados.

## 9. Preguntas comparativas: el agente debe hacer varias acciones

Una pregunta como “¿cómo cambió X entre FY2024 y FY2025?” no debería resolverse esperando que un único chunk contenga ambos datos.

El agente puede:

1. identificar la magnitud y los dos ejercicios;
2. recuperar/verificar el dato de FY2024;
3. recuperar/verificar el dato de FY2025;
4. calcular la diferencia o variación solicitada;
5. construir la respuesta final con trazabilidad.

Esta capacidad iterativa es una de las razones por las que la práctica usa un agente y no un RAG de una sola tirada.

Para comparativas numéricas, los valores individuales deben venir de XBRL cuando existan como hechos autorizados.

## 10. Guardrail de cifras: XBRL es la fuente autorizada

El enunciado exige un middleware propio que extraiga las cifras de la respuesta y las contraste contra XBRL, devolviendo el desajuste al modelo cuando no cuadren.

La intención es impedir el patrón:

```text
retrieval textual -> LLM ve una tabla -> copia/interpreta un número -> responde
```

cuando existe una ruta exacta:

```text
pregunta numérica -> get_xbrl_fact -> cifra exacta -> respuesta
```

### Comportamiento esperado del guardrail

1. Detectar si la respuesta contiene/declara una cifra relevante.
2. Determinar si la pregunta requiere fuente XBRL.
3. Recuperar/comparar el hecho correcto.
4. Aplicar una tolerancia documentada cuando proceda.
5. Si no coincide, **no aceptar la respuesta silenciosamente**: devolver feedback al agente para que corrija.
6. Si el hecho no existe en el corpus, permitir una respuesta explícita de ausencia en vez de inventar.

La tolerancia debe estar documentada; no ocultar diferencias grandes detrás de un margen arbitrario.

## 11. Guardrail de citas: pedir evidencia literal y verificarla

Para respuestas textuales, la sesión propone pedir al modelo la frase literal que respalda la conclusión y comprobar programáticamente que esa frase existe en el texto original.

Esto tiene dos efectos:

- obliga al modelo a anclarse mejor en la evidencia;
- permite detectar una cita alucinada.

La verificación no debería ser un `substring` frágil sin normalización. El propio profesor advierte que saltos de línea, espacios dobles u otros detalles de formato pueden generar falsos negativos.

Como mínimo, normalizar whitespace antes de comparar. El evaluador final debe comprobar no solo que la cita existe, sino que **realmente respalda lo afirmado**.

## 12. Los tres evaluadores obligatorios

El sistema debe incluir tres evaluadores automáticos:

### 12.1 Evaluador de cita

Comprueba:

- que la cita existe;
- que pertenece al texto/corpus correcto;
- que respalda la afirmación de la respuesta.

### 12.2 Evaluador numérico

Comprueba:

- que la cifra producida coincide con el hecho XBRL correcto;
- que la unidad es coherente;
- que la diferencia está dentro de una tolerancia documentada.

### 12.3 Evaluador de trayectoria

Comprueba:

- qué tools fueron llamadas;
- si la trayectoria incluyó la herramienta esperada;
- si se usó la ruta adecuada para la familia de pregunta.

Este último evaluador es la razón por la que no basta con “acertar de casualidad”.

## 13. Golden set propio

El entregable exige un JSONL de 20 preguntas con respuesta conocida:

- al menos 6 comparativas entre dos ejercicios;
- familia extractiva;
- familia numérica;
- comparativas.

Para extractivas, el ground truth debe anclarse a **una frase literal de una sola frase**, no a un `chunk_id`.

Esquema de ejemplo oficial:

```json
{
  "id": "g3-007",
  "pregunta": "¿Cuál fue el revenue de NVIDIA en el ejercicio 2024?",
  "familia": "numerica",
  "ticker": "NVDA",
  "fiscal_year": 2024,
  "respuesta_esperada": "60.922 millones de dólares",
  "cifra_esperada": 60922000000.0,
  "unidad": "USD",
  "concept_xbrl": "Revenues",
  "item_esperado": null,
  "ancla_texto": null,
  "ancla_inicio": null,
  "ancla_fin": null,
  "chunk_id_esperado": null,
  "herramienta_esperada": ["get_xbrl_fact"],
  "autor": "grupo-3"
}
```

### Buen golden set

Debe cubrir fallos reales, no veinte variaciones triviales de la misma consulta. Conviene incluir:

- distintos tickers;
- ambos ejercicios;
- distintas secciones;
- hechos presentes y ausentes;
- conceptos XBRL con nombres distintos entre emisores;
- preguntas textuales con formulaciones distintas;
- comparativas que requieran dos pasos.

No diseñar el sistema para memorizar estas 20 preguntas; el día de la presentación se ejecutan 10 preguntas ciegas.

## 14. Pipeline de evaluación

El repositorio debe exponer:

```python
def responder(pregunta):
    ...


def evaluar(ruta_jsonl):
    ...
```

`evaluar(...)` debe poder ejecutar un dataset de preguntas y producir resultados suficientes para comparar baseline y final.

### Métricas obligatorias de la tabla baseline vs final

- aciertos por familia;
- `recall@k` de retrieval;
- coste medio por pregunta;
- latencia media;
- llamadas a herramienta por pregunta.

El mejor valor debe quedar resaltado en la tabla final. Coste y latencia son columnas de primer nivel, no notas al pie.

Guardar también los resultados crudos de baseline y final para que la tabla sea regenerable.

## 15. Medición de coste y latencia

La primera sesión ya mostró que el estado/metadatos del proveedor pueden incluir tokens y, según la API, coste. La segunda sesión refuerza que optimizar sin medir no vale.

Por pregunta registrar, cuando sea posible:

```text
latency_total_ms
num_tool_calls
input_tokens
output_tokens
cost
```

Si el proveedor no entrega coste directamente, documentar cómo se calcula. No mezclar costes de modelos distintos sin guardar qué modelo ejecutó cada llamada.

## 16. Middleware: qué es necesario y qué es opcional

En clase se enseñó middleware como mecanismo general para insertar lógica antes/durante/después del modelo. También se enseñaron ejemplos de summarization y human-in-the-loop.

Para **esta práctica**:

- el middleware/guardrail de verificación numérica contra XBRL **sí es requisito oficial**;
- human-in-the-loop **no es necesario**;
- un frontend Streamlit/Gradio es opcional y no sustituye ningún requisito;
- middleware genérico de resumen de contexto puede ser útil, pero no es una prioridad si no mejora las métricas exigidas.

No invertir tiempo en features vistosas antes de cerrar evaluación, routing y reproducibilidad.

## 17. Estrategia experimental recomendada

La forma más defendible de trabajar es cambiar una cosa cada vez y guardar el resultado.

### Etapa A — congelar baseline

```text
baseline_dense
```

Guardar resultados y métricas.

### Etapa B — metadatos

```text
baseline_dense + metadata_filter
```

Medir de nuevo `recall@k`, coste y latencia.

### Etapa C — híbrida

```text
metadata_filter + dense + BM25
```

Medir de nuevo. No asumir mejora.

### Etapa D — query rewrite

```text
metadata_filter + hybrid + query_rewrite
```

Medir de nuevo.

### Etapa E — experimentos adicionales

Solo después de tener las obligatorias:

- modelo de embeddings;
- chunk size/overlap;
- limpieza de texto;
- umbrales;
- prompt/docstrings;
- modelo LLM.

Cada experimento debe quedar asociado a configuración y resultados, para evitar “tocar cosas” sin saber cuál produjo el cambio.

## 18. Separar evaluación de retrieval de evaluación end-to-end

No mezclar dos preguntas diferentes:

### “¿El retriever encontró la evidencia?”

Se responde con `recall@k` respecto al ancla textual.

### “¿El agente dio una respuesta correcta y por la ruta correcta?”

Se responde con los evaluadores de cita, cifra y trayectoria, más la exactitud por familia.

Un retrieval malo puede ser compensado ocasionalmente por `read_section`; eso no significa que el retrieval haya mejorado. Del mismo modo, un retrieval perfecto no garantiza que el LLM interprete bien la evidencia.

## 19. Reproducibilidad del repositorio

Antes de entregar, probar literalmente un clon limpio:

```text
1. clone
2. instalar dependencias
3. proporcionar API key por variable de entorno/secreto
4. ejecutar preparación necesaria
5. importar/ejecutar responder()
6. ejecutar evaluar(golden.jsonl)
7. regenerar resultados/tablas
```

No exigir edición manual de paths ni API keys dentro del código.

Si se generan embeddings/índices propios y no conviene regenerarlos cada vez, decidir explícitamente si se versionan como artefactos o si existe un comando reproducible de construcción. Lo importante es que el evaluador no dependa de pasos tácitos hechos en el notebook del autor.

## 20. Checklist final para Codex

Antes de considerar la práctica terminada, verificar:

- [ ] Las 4 firmas contractuales siguen intactas.
- [ ] `get_xbrl_fact` es la ruta autorizada de cifras.
- [ ] `search_filings` soporta metadatos y devuelve trazabilidad.
- [ ] `read_section` se trata como herramienta cara/fallback.
- [ ] Existe límite de tool calls/rondas.
- [ ] La respuesta usa `RespuestaFinanciera` con todos los campos requeridos.
- [ ] Hay middleware de contraste numérico XBRL.
- [ ] Las citas se pueden verificar contra texto real.
- [ ] Hay trace de tool calls para el evaluador de trayectoria.
- [ ] Golden set: 20 preguntas, al menos 6 comparativas.
- [ ] Extractivas ancladas a frase literal, no a `chunk_id`.
- [ ] Se midió el baseline denso.
- [ ] Se midió filtro por metadatos.
- [ ] Se implementó y midió BM25 + denso.
- [ ] Se implementó y midió query rewriting.
- [ ] Se reporta `recall@k` tras cada mejora.
- [ ] Existen los 3 evaluadores.
- [ ] Existen `responder()` y `evaluar()`.
- [ ] Se guardaron resultados baseline y final.
- [ ] La tabla incluye aciertos por familia, recall, coste, latencia y tool calls.
- [ ] No hay API keys en Git.
- [ ] Un clon limpio funciona sin editar código.

## 21. Qué debería poder explicar el equipo en la presentación

No basta con que Codex genere un sistema que pase tests. El profesor insistió en que el equipo debe poder defender las decisiones.

Hay que poder responder con claridad:

- por qué una pregunta numérica usa XBRL;
- cuándo se usa retrieval y cuándo `read_section`;
- qué fallaba en el baseline;
- qué mejora movió `recall@k` y cuánto;
- qué mejora no funcionó;
- qué coste/latencia añadió cada cambio;
- cómo se verifica una cita;
- cómo se detecta una cifra inconsistente;
- cómo se comprueba la trayectoria;
- por qué el resultado final debería generalizar a las 10 preguntas ciegas.

Ese criterio técnico es parte central de la práctica.

## Fuentes sintetizadas

- Transcripción de la clase del 17/09.
- Enunciado oficial “LLMs aplicados a Finanzas — Un agente investigador sobre informes 10-K de la SEC”.
