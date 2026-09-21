"""Montaje del agente en sus tres perfiles.

## Por qué tres perfiles y no uno

El §5 del enunciado pide una tabla que compare el sistema baseline con el
final. Una tabla de dos filas responde «mejoró», pero no responde «por qué», y
esa es la pregunta de la defensa. Con una fila intermedia se puede atribuir la
mejora a una causa concreta en lugar de a un paquete de cambios.

| Perfil | Retriever | Guardrail de cifras | Límites | Para qué está |
| --- | --- | --- | --- | --- |
| `baseline` | denso plano, sin filtros | no | sí | el punto de partida del día 10 |
| `filtros` | denso + metadatos | no | sí | aísla cuánto aporta el filtro solo |
| `final` | reescritura + híbrido + reranking | sí | sí | el sistema que se entrega |

Los límites de llamadas están en los tres. No son una mejora, son una red de
seguridad: sin ellos, el baseline puede quedarse dando vueltas en una pregunta
cuyo dato no existe y la evaluación no termina nunca. Dejarlos fuera del
baseline haría que la comparación midiera también «el baseline no termina», que
es un resultado trivial y no informativo.

El resto de las diferencias entre perfiles está deliberadamente reducido al
mínimo: **el mismo modelo, el mismo prompt de sistema, el mismo esquema de
salida y los mismos docstrings en las cuatro herramientas.** Si el baseline
tuviera además un prompt peor, la tabla mediría el prompt y el retriever a la
vez y no se podría atribuir nada.
"""

from __future__ import annotations

from agente import config, esquema, herramientas, middleware

# ---------------------------------------------------------------------------
# El prompt de sistema
# ---------------------------------------------------------------------------
# Es el mismo en los tres perfiles, por lo dicho arriba. Está escrito como una
# política de decisión y no como una descripción de buenas intenciones: cada
# línea dice qué hacer en una situación concreta y reconocible.
#
# Las reglas que están aquí y no en un docstring son las que no pertenecen a
# ninguna herramienta en particular: el reparto entre herramientas en una
# comparativa, y qué hacer cuando el dato no existe. Todo lo que sí pertenece a
# una herramienta concreta vive en su docstring, que es donde el modelo lo lee
# en el momento de decidir si la llama.
SYSTEM = """Eres un analista financiero que responde preguntas sobre informes 10-K de la SEC usando ÚNICAMENTE las herramientas disponibles. Nunca respondes de memoria.

CÓMO ELEGIR LA HERRAMIENTA

- Cualquier CIFRA sale de get_xbrl_fact. Siempre. Aunque veas el número escrito en un fragmento que ya has recuperado, aunque parezca obvio: un número leído de la prosa puede venir de una tabla partida, de un trimestre o de un comparativo de otro año.
- Las EXPLICACIONES, los riesgos, la estrategia y los comentarios de la dirección salen de search_filings.
- Si no estás seguro de que una compañía, un ejercicio o un concepto existan en el corpus, empieza por list_available. Es gratis.
- read_section es el último recurso: devuelve decenas de miles de tokens. Úsala solo si search_filings ya ha fallado sobre esa misma sección y necesitas el contexto completo.

PREGUNTAS COMPARATIVAS

Una pregunta que compara dos ejercicios se resuelve en tres pasos, no en uno:
1. get_xbrl_fact para el ejercicio más reciente.
2. get_xbrl_fact para el anterior. Dos llamadas separadas: no estimes la segunda.
3. search_filings sobre el Item 7 (MD&A) para la frase en la que la dirección explica la variación.
Rellena `cifra` con el ejercicio reciente y `cifra_anterior` con el anterior.

BÚSQUEDAS

El corpus está en INGLÉS. Escribe las consultas en inglés y con el vocabulario del propio informe ("net sales", "gross margin percentage", "provision for income taxes"), no traduciendo la pregunta palabra por palabra. Pasa siempre los filtros que conozcas: ticker, fiscal_year y, si sabes dónde vive la respuesta, item.

CITAS

Cuando te apoyes en un fragmento, copia su chunk_id (el identificador entre corchetes que precede al fragmento) en el campo chunk_id, y copia en el campo cita una frase LITERAL de ese mismo fragmento, palabra por palabra y sin traducir. La cita se verifica automáticamente contra el texto original: una cita traducida o parafraseada cuenta como no verificada.

IDIOMA

El campo `respuesta` va SIEMPRE en el idioma de la pregunta. El campo `cita` va SIEMPRE en el idioma original del informe, que es el inglés, y sin tocar una palabra. Son dos campos con reglas opuestas a propósito: la respuesta la lee una persona y la cita la verifica una comparación literal contra el texto del 10-K.

CUANDO EL DATO NO ESTÁ

El corpus es cerrado. Si una compañía no reportó un concepto, o si la compañía o el ejercicio no están, la respuesta CORRECTA es decirlo: fuente="ninguna", cifra=null y una explicación de qué falta. No lo estimes, no lo deduzcas de otras cifras y no lo sustituyas por el concepto más parecido de otra compañía. Decir "no está en el corpus" cuando no está es un acierto, no un fracaso."""


PERFILES = {
    # perfil: (estrategia de retrieval, ¿guardrail de cifras?)
    "baseline": ("denso_plano", False),
    "filtros": ("filtros", False),
    "final": ("final", True),
}

DESCRIPCION_PERFILES = {
    "baseline": "denso plano, sin filtros, sin guardrail numérico",
    "filtros": "denso + filtros de metadatos, sin guardrail numérico",
    "final": "reescritura + híbrido BM25/RRF + reranking, con guardrail numérico",
}


def construir_agente(
    perfil: str = "final",
    modelo: str | None = None,
    con_hitl: bool = False,
    checkpointer=None,
):
    """El agente compilado, listo para `invoke`.

    Args:
        perfil: `"baseline"`, `"filtros"` o `"final"`.
        modelo: nombre del modelo de OpenAI. Si es `None`, se autodetecta.
        con_hitl: añade el Human-in-the-Loop. **Fuera de la evaluación**: una
            interrupción esperando aprobación humana deja colgada cualquier
            ejecución automática.
        checkpointer: por defecto `InMemorySaver`, que es lo que da memoria
            dentro de un `thread_id`.

    Sobre la memoria: el checkpointer hace que dos preguntas con el mismo
    `thread_id` compartan conversación, y eso permite preguntas de seguimiento
    («¿y el año anterior?»). En la evaluación, **cada pregunta lleva su propio
    `thread_id`**, y eso no es un detalle: si veinte preguntas compartieran
    hilo, la número quince vería el contexto de las catorce anteriores y podría
    acertar por contagio en vez de por haber consultado la herramienta. La
    medición dejaría de ser de la pregunta y pasaría a ser de la conversación.
    """
    if perfil not in PERFILES:
        raise ValueError(
            f"Perfil {perfil!r} desconocido. Disponibles: {list(PERFILES)}"
        )
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver

    estrategia, con_guardrail = PERFILES[perfil]

    # Los topes van SIEMPRE los primeros: se ejecutan en el orden de la lista y
    # un tope colocado detrás del verificador no impide que este se ejecute
    # una vez de más.
    cadena = middleware.limites()
    if con_hitl:
        cadena.append(middleware.human_in_the_loop())
    if con_guardrail:
        cadena.append(middleware.verificar_cifras_contra_xbrl)

    return create_agent(
        model=config.crear_modelo(modelo),
        tools=herramientas.construir_herramientas(estrategia),
        system_prompt=SYSTEM,
        response_format=esquema.RespuestaFinanciera,
        middleware=cadena,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )


# ---------------------------------------------------------------------------
# Observabilidad
# ---------------------------------------------------------------------------
def pretty_trace(resultado, max_chars: int = 220) -> None:
    """Qué herramientas se llamaron, con qué argumentos y qué devolvieron.

    Sin trayectoria no se distingue una respuesta correcta de una respuesta
    correcta por casualidad, que es justo lo que comprueba el evaluador
    `uso_la_tool_correcta`. Es la misma función de clase, con la corrección
    del guardrail marcada aparte para que se vea cuándo intervino.
    """
    mensajes = resultado["messages"] if isinstance(resultado, dict) else resultado
    paso = 0
    for mensaje in mensajes:
        if middleware.MARCA in str(getattr(mensaje, "content", "") or ""):
            print(f"  [!] {middleware.MARCA}: el guardrail devolvió la "
                  f"respuesta al modelo")
        for llamada in getattr(mensaje, "tool_calls", None) or []:
            paso += 1
            argumentos = ", ".join(f"{k}={v!r}" for k, v in llamada["args"].items())
            print(f"  {paso}. {llamada['name']}({argumentos})")
        if type(mensaje).__name__ == "ToolMessage":
            contenido = str(mensaje.content).replace("\n", " ")
            print(
                f"       -> {contenido[:max_chars]}"
                f"{'…' if len(contenido) > max_chars else ''}"
            )
    if isinstance(resultado, dict) and resultado.get("structured_response"):
        r = resultado["structured_response"]
        print(f"\n  respuesta: {r.respuesta}")
        print(f"  fuente: {r.fuente} · cita: {r.chunk_id}")


def herramientas_usadas(resultado) -> list[str]:
    """Los nombres de las herramientas que aparecen en la trayectoria.

    Se devuelven **con repeticiones y en orden**, no como conjunto: una
    comparativa que llama dos veces a `get_xbrl_fact` y otra que la llama una
    sola vez y estima el segundo año no son lo mismo, y el evaluador de
    trayectoria tiene que poder distinguirlas.
    """
    mensajes = resultado["messages"] if isinstance(resultado, dict) else resultado
    return [
        llamada["name"]
        for mensaje in mensajes
        for llamada in (getattr(mensaje, "tool_calls", None) or [])
    ]


def hubo_correccion(resultado) -> bool:
    """¿Intervino el guardrail numérico en esta invocación?

    Es una columna del informe por sí misma. Un guardrail que nunca salta no
    se puede distinguir de uno que no está puesto, y uno que salta en todas las
    preguntas está mal calibrado.
    """
    mensajes = resultado["messages"] if isinstance(resultado, dict) else resultado
    return any(
        middleware.MARCA in str(getattr(m, "content", "") or "") for m in mensajes
    )


def tokens_de(resultado) -> tuple[int, int]:
    """`(entrada, salida)` sumando el uso reportado por todos los mensajes."""
    mensajes = resultado["messages"] if isinstance(resultado, dict) else resultado
    entrada = salida = 0
    for mensaje in mensajes:
        uso = getattr(mensaje, "usage_metadata", None) or {}
        entrada += uso.get("input_tokens", 0) or 0
        salida += uso.get("output_tokens", 0) or 0
    return entrada, salida


def coste_de(resultado, modelo: str | None = None) -> float:
    """Coste en USD de una invocación, según la tabla de precios de `config`.

    Los tokens los reporta la API; el precio lo ponemos nosotros. Por eso
    `config.FECHA_PRECIOS` se imprime junto a cualquier tabla que lleve esta
    columna: un coste calculado con precios de hace seis meses es un número
    con apariencia de dato.
    """
    modelo = modelo or config.modelo_por_defecto()
    precio_entrada, precio_salida = config.precio_de(modelo.split(":", 1)[-1])
    entrada, salida = tokens_de(resultado)
    return (entrada * precio_entrada + salida * precio_salida) / 1e6


def cronometrar(funcion, *args, **kwargs) -> tuple:
    """`(resultado, segundos)`."""
    import time

    comienzo = time.perf_counter()
    resultado = funcion(*args, **kwargs)
    return resultado, time.perf_counter() - comienzo
