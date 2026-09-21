# Sesión 1 — Arquitectura del agente 10-K y contrato de herramientas

> Documento destilado para usar como contexto de implementación en Codex. No es una transcripción. Resume la sesión del 10/09, incorpora las correcciones técnicas compartidas en el chat de clase y contrasta los puntos importantes con el enunciado oficial de la práctica.

## 0. Qué hay que construir

La práctica no consiste en hacer un RAG aislado. El objetivo es construir **un agente investigador sobre informes 10-K** que pueda elegir entre herramientas con distinto coste, precisión y propósito, mantener una trayectoria auditable y devolver una respuesta estructurada.

La idea arquitectónica central es:

1. El LLM recibe una pregunta y conoce las herramientas disponibles.
2. Decide qué herramienta llamar y con qué argumentos.
3. El programa ejecuta la herramienta.
4. El resultado vuelve al contexto operativo del agente.
5. El agente puede llamar a otra herramienta o finalizar.
6. La ejecución debe dejar rastro suficiente para evaluar **respuesta, coste, latencia y trayectoria de herramientas**.

El retrieval es una herramienta del agente, no la arquitectura completa. Para cifras financieras exactas, el sistema debe preferir XBRL; para información textual, retrieval; y la lectura de una sección completa debe ser un recurso caro/de último nivel.

## 1. Fuente de verdad y reglas de prioridad

Cuando haya una diferencia entre una explicación informal de clase y el enunciado oficial, Codex debe tratar **el enunciado oficial como contrato de entrega**.

En particular:

- Las firmas de las cuatro herramientas son contrato y no deben cambiarse.
- La salida estructurada requerida no debe perder ni renombrar campos.
- Los valores numéricos reportados deben contrastarse con XBRL.
- La trayectoria de herramientas importa en la evaluación final.
- El repositorio debe poder ejecutarse sobre un clon limpio sin edición manual.

Las explicaciones de clase sirven para entender la intención y para tomar buenas decisiones de diseño, pero no sustituyen esos requisitos.

## 2. Entorno y preparación de Colab

El corpus ya está preparado. **No hay que descargar los 10-K desde SEC EDGAR.** El profesor remarcó que el material entregado debe ser la base de la práctica.

Se puede trabajar en local o en Google Colab. Para Colab, durante la sesión aparecieron problemas de rutas y se compartieron estas correcciones en el chat.

### Corrección de `sys.path`

Añadir al principio del notebook:

```python
import sys
sys.path.append("/content")
```

### Candidatos para localizar los archivos

En la celda de corpus/índice, la lista compartida en el chat fue:

```python
import pathlib

CANDIDATOS = [
    pathlib.Path("."),
    pathlib.Path("/content"),
    pathlib.Path("/content/drive/MyDrive/MIAX_2026"),
    pathlib.Path("/content/drive/Shareddrives/MIAX_2026"),
    pathlib.Path("/content/dataset"),
]
```

### Corrección dentro de `miax_s1.py`

La variable indicada en el chat debe quedar:

```python
from pathlib import Path

CANDIDATOS_CORPUS = [
    Path("corpus"),
    Path("/content/corpus"),
]
```

Estas correcciones surgieron porque el material se había preparado originalmente en local y algunas rutas adicionales daban problemas en Colab. Mantener las rutas simples y comprobar que los ZIP se han descomprimido donde el código espera encontrarlos.

## 3. Modelo mental del corpus

El corpus oficial contiene:

- Empresas: `NVDA`, `MSFT`, `AAPL`, `GOOGL`, `META`, `AMZN`.
- Ejercicios fiscales: FY2024 y FY2025.
- Secciones por 10-K: `1A`, `7`, `7A` y `8`.
- 48 secciones en total.
- Aproximadamente 650.000 tokens.

Ficheros relevantes:

- `secciones.jsonl`: texto completo por empresa, ejercicio y sección. Fuente de `read_section`.
- `chunks.jsonl`: texto troceado en 1.749 fragmentos con `chunk_id`. Fuente de `search_filings` y trazabilidad textual.
- `xbrl_facts.parquet`: 135 hechos numéricos. **Fuente autorizada para cifras** y ground truth numérico.
- `indice/`: índice FAISS inicial sobre los chunks, construido con `BAAI/bge-small-en-v1.5`.

### Tres trampas de datos que el agente no debe “razonar por analogía”

1. `fiscal_year` no equivale necesariamente al año de presentación del 10-K. Hay que usar el campo del corpus.
2. El concepto XBRL de revenue cambia entre compañías y, en algún caso, entre ejercicios. No asumir un concepto por similitud con otra empresa.
3. Hay hechos que realmente no están reportados en el corpus. La respuesta correcta puede ser “no disponible”; el agente no debe inventar una cifra.

Esto justifica tener una herramienta `list_available()` y tratar la ausencia de datos como un caso legítimo.

## 4. Las cuatro herramientas son la interfaz pública del agente

Las firmas deben mantenerse exactamente. Se puede reimplementar el cuerpo, añadir parámetros opcionales con valor por defecto y añadir herramientas auxiliares, pero no romper este contrato:

```python
@tool
def list_available() -> str:
    """Devuelve las empresas y ejercicios fiscales disponibles en el corpus."""

@tool
def get_xbrl_fact(ticker: str, fiscal_year: int, concept: str) -> str:
    """Devuelve el valor EXACTO de una magnitud financiera tal y como la
    compañía la reportó en XBRL. Es la fuente autorizada para cualquier
    cifra. Úsala SIEMPRE en lugar de leer un número del texto."""

@tool
def search_filings(
    query: str,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
    k: int = 5,
) -> str:
    """Busca fragmentos de texto relevantes en los 10-K del corpus.
    Devuelve k fragmentos, cada uno con su chunk_id para poder citarlo."""

@tool
def read_section(ticker: str, fiscal_year: int, item: str) -> str:
    """Devuelve el TEXTO COMPLETO de una sección. Es CARA: puede devolver
    decenas de miles de tokens."""
```

## 5. Semántica de routing: cuándo debe usar cada herramienta

### `list_available()`

Sirve para saber qué universo de empresas, ejercicios y secciones existe. Su valor es mayor si el corpus puede cambiar o si se quiere evitar hardcodear disponibilidad en el prompt.

Regla útil: antes de concluir que una empresa/ejercicio/dato no existe, el agente debe comprobar el universo disponible cuando haya duda.

### `get_xbrl_fact(...)`

Es la vía correcta para magnitudes financieras exactas. No extraer una cifra de una tabla textual si existe como hecho XBRL.

Razones:

- es exacto;
- evita errores de OCR/Markdown/tablas partidas;
- es barato;
- generaliza mejor que “leer el número” de un fragmento textual.

### `search_filings(...)`

Es la herramienta normal para preguntas cualitativas o extractivas sobre el 10-K. Debe devolver fragmentos suficientemente trazables, incluido `chunk_id`, para poder verificar la cita.

Los filtros `ticker`, `fiscal_year` e `item` son parte importante de la precisión. Si están disponibles en la pregunta o se pueden inferir de forma segura, deben usarse.

### `read_section(...)`

Es el fallback caro. Se utiliza cuando el retrieval no encuentra suficiente evidencia o cuando la pregunta exige contexto amplio que no cabe en unos pocos chunks.

No debe ser la primera opción por defecto: una sección puede tener decenas de miles de tokens, aumentando coste y latencia.

## 6. El docstring de una herramienta es parte de la política del agente

En clase se insistió en que el LLM **no ve el cuerpo de la función** para decidir si la usa. Ve el nombre, el esquema de argumentos y la descripción/docstring que se le expone.

Por tanto, el docstring no es documentación decorativa: es comportamiento.

Un buen docstring debe dejar claro:

- qué problema resuelve la herramienta;
- qué devuelve;
- cuándo debe usarse;
- cuándo no debe usarse;
- qué significa cada argumento si puede haber ambigüedad;
- qué hacer si no hay resultado.

Evitar resolver errores de routing añadiendo indefinidamente frases y excepciones al docstring. Es preferible una descripción corta, precisa y mutuamente diferenciada respecto de las otras herramientas.

## 7. Bucle manual del agente: entenderlo antes de abstraerlo

La sesión construyó conceptualmente el agente como un bucle de razonamiento/acción. El patrón básico es:

```text
messages = [system_prompt, user_question]

for round in range(MAX_ROUNDS):
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if response has no tool calls:
        return final structured answer

    for tool_call in response.tool_calls:
        result = execute_tool(tool_call)
        messages.append(tool_result_message(result))

raise AgentLimitError
```

La diferencia entre “LLM con tools” y “agente” es precisamente ese bucle: el modelo puede observar el resultado de sus acciones y decidir el siguiente paso.

### Límite de iteraciones

Debe existir un límite explícito de rondas/llamadas. En clase se mostró que un agente puede entrar en bucles o buscar demasiado; el enunciado exige además un límite de llamadas a herramientas por invocación.

No confiar únicamente en que el modelo “sepa cuándo parar”.

## 8. Framework: usar abstracción sin perder control

La sesión utilizó LangChain como ejemplo, pero la idea no depende obligatoriamente de ese framework. El profesor lo recomendó porque permite bajar a un nivel de control fino cuando el agente crece.

Para la práctica importa más que la implementación permita:

- registrar tool calls y tool results;
- limitar llamadas;
- imponer salida estructurada;
- inspeccionar el estado/trace;
- insertar middleware/guardrails;
- medir coste y latencia.

Las librerías de agentes cambian rápido. Si Codex genera código con una API actual, debe comprobar la documentación de la versión realmente instalada en el repositorio y no asumir APIs antiguas de LangChain por memoria.

## 9. Salida estructurada obligatoria

El agente no debe devolver únicamente prosa libre. El contrato oficial es:

```python
from typing import Literal
from pydantic import BaseModel, Field


class RespuestaFinanciera(BaseModel):
    """Respuesta trazable a una pregunta sobre informes 10-K."""

    respuesta: str = Field(description="Respuesta en prosa, breve y directa")
    cifra: float | None = Field(
        default=None,
        description="Valor numérico, si la pregunta pide uno",
    )
    unidad: str | None = Field(
        default=None,
        description="USD, shares, porcentaje…",
    )
    ticker: str | None = None
    ejercicio: int | None = None
    fuente: Literal["xbrl", "texto", "ambas", "ninguna"] = Field(
        description="De dónde sale el dato. 'ninguna' si no está en el corpus"
    )
    cita: str | None = Field(
        default=None,
        description="Texto literal del informe que respalda la respuesta",
    )
    chunk_id: str | None = Field(
        default=None,
        description="Identificador del fragmento citado, para verificar",
    )
```

Se pueden añadir campos internos, pero no quitar ni renombrar los anteriores.

La salida estructurada facilita la evaluación automática, evita parsear prosa arbitraria y permite aplicar guardrails de cifra/cita.

## 10. Estado, trazas y observabilidad

Una idea importante de la primera sesión es que el **estado del agente** no es igual a la ventana de contexto del LLM.

El estado puede contener:

- historial de mensajes;
- tool calls;
- resultados de herramientas;
- IDs de conversación;
- métricas de uso;
- datos auxiliares que el LLM no necesita ver directamente.

Para esta práctica, guardar la trayectoria es especialmente útil porque posteriormente hay que evaluar qué herramienta utilizó el agente.

### Qué registrar por pregunta

Como mínimo conviene producir un trace estructurado con:

```text
question_id
start_time / end_time / latency
model
final_response
tool_calls[]:
    tool_name
    arguments
    result summary or reference
    start/end/latency
usage:
    input_tokens
    output_tokens
    cost (si el proveedor lo devuelve)
errors / retries
```

La sesión remarcó que medir tokens/coste permite demostrar que una mejora realmente reduce coste en vez de afirmarlo de forma intuitiva.

## 11. Baseline: congelarlo antes de “mejorar”

La primera sesión deja preparado un agente básico. Para la práctica, el baseline no es un borrador desechable: es **la referencia experimental**.

Antes de cambiar retrieval, prompts, embeddings o tools:

1. ejecutar el baseline;
2. guardar sus resultados;
3. guardar configuración/modelo/versión;
4. medir sus métricas;
5. etiquetar o versionar ese estado en Git.

El enunciado exige comparar baseline frente a sistema final y conservar resultados regenerables. Si se sobreescribe el baseline, se pierde media evaluación.

## 12. Explorar los datos antes de optimizar

El profesor insistió varias veces en “pasearse” por los DataFrames y entender el corpus antes de pedir a un LLM que optimice el sistema.

Antes de implementar mejoras, inspeccionar:

- columnas y tipos de `xbrl_facts`;
- valores reales de `concept` por ticker/año;
- distribución de chunks por sección;
- ejemplos con tablas;
- tamaños/tokens de chunks;
- casos de datos ausentes;
- cómo se representa `item`;
- cómo aparecen las citas en el texto real.

Para este proyecto, el criterio sobre los datos financieros es parte de la solución. No asumir que todos los emisores reportan lo mismo o usan la misma taxonomía.

## 13. Decisiones de implementación recomendadas por la intención de la práctica

Estas son las decisiones que mejor encajan con lo explicado y con el contrato oficial:

- Mantener una capa de datos separada de la lógica del agente.
- Hacer que las tools sean funciones pequeñas, deterministas y fáciles de testear.
- Devolver errores/ausencias como resultados explícitos, no hacer `raise` sin control ante un dato no encontrado.
- Evitar que el agente lea todo el corpus o una sección enorme si puede resolver con una fuente exacta o con pocos chunks.
- No hardcodear conceptos XBRL por analogía entre empresas.
- Mantener el trace del agente fuera de la respuesta final, pero accesible para evaluación.
- Mantener API keys fuera del repositorio.
- Versionar dependencias para que un clon limpio sea reproducible.

## 14. Errores que Codex debe evitar

- Convertir el proyecto en “una pregunta -> un vector search -> una respuesta” sin bucle de herramientas.
- Leer cifras financieras de prosa cuando deben venir de XBRL.
- Usar `read_section` como shortcut general porque “funciona”.
- Cambiar nombres o argumentos de las cuatro tools.
- Ocultar las tool calls, haciendo imposible evaluar trayectoria.
- Devolver una respuesta sin los campos estructurados obligatorios.
- Concluir que un dato no existe sin comprobar correctamente corpus/XBRL.
- Meter cientos de miles de tokens en contexto por comodidad.
- Optimizar antes de guardar un baseline reproducible.
- Depender de una API de framework sin fijar versión o sin comprobar documentación actual.

## 15. Estado esperado al terminar la sesión 1

Al cerrar esta fase, el repositorio debería tener al menos:

```text
- corpus cargando de forma reproducible
- las 4 tools con firmas contractuales
- baseline ejecutable
- agente con bucle y límite de tool calls
- salida RespuestaFinanciera
- trazas de tool calls
- medición básica de latencia/uso
- baseline guardado/versionado
- estructura preparada para el golden set
```

La sesión 2 parte precisamente de aquí: ejecutar el baseline sobre preguntas conocidas, clasificar sus fallos y mejorar el retrieval de forma medida.

## Fuentes sintetizadas

- Transcripción de la clase del 10/09.
- Chat de la clase del 10/09, usado para las correcciones de rutas/Colab.
- Enunciado oficial “LLMs aplicados a Finanzas — Un agente investigador sobre informes 10-K de la SEC”.
