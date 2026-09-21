"""Guardrails: el código propio que se mete dentro del bucle del agente.

## Qué es un middleware, en una frase

El bucle del agente es este:

    mientras queden vueltas:
        respuesta = modelo(mensajes)
        si no pide herramientas: terminar
        ejecutar las herramientas y añadir los resultados

Un middleware es un gancho en uno de los huecos de ese bucle. Recibe el estado,
puede leerlo, cambiarlo o cortar la ejecución. No hay magia: `after_model`
significa «justo después de la respuesta del modelo».

## Por qué el verificador numérico es el guardrail que importa aquí

El fallo caro de este dominio no es que el agente no encuentre el dato. Es que
afirme un número con aplomo y el número esté mal. En un sistema financiero eso
no es un error de calidad, es una respuesta inservible, y además es
indistinguible de una correcta para quien la lee.

La idea cabe en una frase: **si la respuesta afirma un número, contrástalo
contra XBRL antes de dejarla salir, y si no cuadra, devuélvele el desajuste al
modelo.** Es lo que convierte el entregable en un sistema de dominio financiero
y no en un chatbot con documentos encima.

## Los dos riesgos de escribirlo mal, y cómo se evitan aquí

**Uno: que no haga nada.** Añadir un mensaje al estado NO hace que el agente
vuelva a pensar. Cuando `after_model` termina, el grafo mira el último
`AIMessage` para decidir a dónde va; si era una respuesta final, el agente
acaba, y los mensajes que se hayan añadido detrás se quedan sin que nadie los
lea. Hacen falta las dos mitades: `can_jump_to=["model"]` al decorar, que
construye la arista del grafo, y `"jump_to": "model"` al devolver, que la usa.

**Dos: que no pare.** Un verificador que se queja siempre es un bucle infinito
con otro nombre, y `ToolCallLimitMiddleware` **no** protege de este: un modelo
que responde sin llamar a ninguna herramienta no gasta llamadas a herramienta.
Por eso aquí hay tres frenos independientes: el verificador corrige **una sola
vez** por invocación, `ModelCallLimitMiddleware` acota las vueltas al modelo, y
`ToolCallLimitMiddleware` acota el gasto en herramientas.
"""

from __future__ import annotations

from langchain.agents.middleware import (
    AgentState,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    after_model,
)
from langgraph.runtime import Runtime

from agente import config, corpus

# Marca que identifica un mensaje puesto por el verificador. Es lo que permite
# saber, mirando solo el estado, si ya se ha corregido en esta invocación.
MARCA = "VERIFICACIÓN AUTOMÁTICA"

# Unidades en las que la cifra NO es un hecho XBRL y por tanto no se puede
# contrastar contra la tabla. Un crecimiento del 62 % es una magnitud derivada
# correcta que no aparece en ningún `xbrl_facts.parquet`, y compararla contra
# los hechos reportados produciría una falsa alarma: el verificador mandaría
# corregir una respuesta que está bien y gastaría una vuelta del modelo.
UNIDADES_NO_VERIFICABLES = {
    "%",
    "porcentaje",
    "percent",
    "percentage",
    "pp",
    "puntos porcentuales",
    "ratio",
    "veces",
    "x",
}


def cuadra(afirmada: float, real: float, tolerancia: float | None = None) -> bool:
    """¿La cifra afirmada coincide con la reportada, con tolerancia relativa?

    La tolerancia existe porque redondear 281.724 millones a «281.700 millones»
    no es inventarse un número. Inventárselo es decir 250.000.

    Es la misma función y la misma constante que usa el evaluador de cifras, y
    tiene que serlo: si el guardrail y el evaluador midieran con criterios
    distintos, una respuesta podría pasar el guardrail y suspender la
    evaluación, y la tabla del informe dejaría de significar nada.
    """
    tolerancia = config.TOLERANCIA_CIFRA if tolerancia is None else tolerancia
    if real == 0:
        return afirmada == 0
    return abs(afirmada - real) / abs(real) <= tolerancia


def _ya_se_corrigio(mensajes) -> bool:
    return any(MARCA in str(getattr(m, "content", "") or "") for m in mensajes)


def _formatear_disponibles(ticker: str, ejercicio: int) -> str:
    filas = corpus.cargar_xbrl()
    filas = filas[(filas.ticker == ticker) & (filas.fiscal_year == int(ejercicio))]
    if filas.empty:
        return "    (no hay ningún hecho XBRL para esa compañía y ejercicio)"
    lineas = []
    for fila in filas.itertuples():
        formato = ",.2f" if fila.unit == "USD/shares" else ",.0f"
        lineas.append(f"    {fila.concept} = {fila.value:{formato}} {fila.unit}")
    return "\n".join(lineas)


@after_model(can_jump_to=["model"])
def verificar_cifras_contra_xbrl(
    state: AgentState, runtime: Runtime
) -> dict | None:
    """Contrasta la cifra de la respuesta con el XBRL y devuelve el desajuste.

    Devolver `None` significa «sigue, no toco nada», y es lo que hace en la
    inmensa mayoría de las invocaciones. Se abstiene en seis casos, y cada uno
    tiene su motivo:

    1. **No hay respuesta estructurada todavía**, o no afirma ninguna cifra.
       No hay nada que verificar.
    2. **Falta el ticker o el ejercicio.** Sin saber de qué compañía y de qué
       año es la cifra, no se puede buscar contra qué compararla. Reclamar aquí
       sería castigar al modelo por un campo que no rellenó, no por el número.
    3. **La respuesta dice `fuente="ninguna"`.** El agente está afirmando que
       el dato no está en el corpus, que es una respuesta legítima y de hecho
       la correcta en varias preguntas. Verificarla contra XBRL no tiene
       sentido.
    4. **La unidad no es un hecho XBRL.** Un porcentaje de crecimiento o un
       múltiplo son magnitudes derivadas correctas que no aparecen en la tabla.
    5. **Ya se corrigió una vez en esta invocación.** El freno del bucle
       infinito, y no es opcional.
    6. **La cifra cuadra.** Que es el caso bueno.

    Cuando sí interviene, la comprobación va en dos niveles de exigencia:

    - Si la respuesta declara el `concepto_xbrl` que consultó, se compara
      **contra ese concepto concreto**. Es la comprobación fuerte: detecta que
      el modelo haya devuelto el ingreso cuando se le preguntaba por el
      beneficio.
    - Si no lo declara, se compara contra el conjunto de hechos reportados por
      esa compañía en ese ejercicio. Es más laxa —una cifra correcta de otro
      concepto pasaría— pero es todo lo que se puede exigir sin saber qué se
      consultó, y sigue atrapando el fallo que importa: el número inventado,
      que no coincide con ninguno.

    El mensaje correctivo dice tres cosas, y las tres hacen falta: qué afirmó,
    qué hay reportado de verdad, y qué se espera que haga. Un mensaje vago no
    corrige nada; el modelo insiste y se gasta la única corrección disponible.
    """
    respuesta = state.get("structured_response")
    if respuesta is None:
        return None

    afirmada = getattr(respuesta, "cifra", None)
    if afirmada is None:
        return None

    ticker = getattr(respuesta, "ticker", None)
    ejercicio = getattr(respuesta, "ejercicio", None)
    if not ticker or not ejercicio:
        return None

    if getattr(respuesta, "fuente", None) == "ninguna":
        return None

    unidad = (getattr(respuesta, "unidad", None) or "").strip().lower()
    if unidad in UNIDADES_NO_VERIFICABLES:
        return None

    if _ya_se_corrigio(state["messages"]):
        return None

    ticker = str(ticker).strip().upper()
    try:
        ejercicio = int(ejercicio)
    except (TypeError, ValueError):
        return None

    xbrl = corpus.cargar_xbrl()
    hechos = xbrl[(xbrl.ticker == ticker) & (xbrl.fiscal_year == ejercicio)]
    if hechos.empty:
        return None  # compañía o ejercicio fuera del corpus: no es cosa de este guardrail

    # --- Nivel 1: contra el concepto declarado -----------------------------
    concepto = getattr(respuesta, "concepto_xbrl", None)
    if concepto:
        fila = hechos[hechos.concept == str(concepto).strip()]
        if not fila.empty:
            real = float(fila.iloc[0].value)
            if cuadra(float(afirmada), real):
                return None
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{MARCA}: la cifra que has dado no cuadra con el "
                            f"XBRL.\n"
                            f"  Has afirmado : {float(afirmada):,.2f}\n"
                            f"  {ticker} FY{ejercicio} reportó "
                            f"'{concepto}' = {real:,.2f} "
                            f"{fila.iloc[0].unit}\n\n"
                            f"Vuelve a llamar a get_xbrl_fact para "
                            f"comprobarlo y corrige tu respuesta con el valor "
                            f"exacto reportado. Si lo que querías responder no "
                            f"corresponde a ese concepto, consulta el concepto "
                            f"correcto. Si el dato no está en el corpus, dilo "
                            f"con fuente=\"ninguna\" y cifra=null en lugar de "
                            f"estimarlo."
                        ),
                    }
                ],
                "jump_to": "model",
            }

    # --- Nivel 2: contra todo lo reportado por esa compañía ese año --------
    if any(cuadra(float(afirmada), float(v)) for v in hechos.value):
        return None

    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"{MARCA}: la cifra que has dado no coincide con NINGÚN "
                    f"hecho que {ticker} reportara en FY{ejercicio}.\n"
                    f"  Has afirmado: {float(afirmada):,.2f} "
                    f"{unidad or '(sin unidad)'}\n"
                    f"  Lo reportado en XBRL es:\n"
                    f"{_formatear_disponibles(ticker, ejercicio)}\n\n"
                    f"Llama a get_xbrl_fact con el concepto que corresponda y "
                    f"corrige tu respuesta con el valor exacto. Si la magnitud "
                    f"que te piden no está en esa lista, la compañía no la "
                    f"reporta: dilo con fuente=\"ninguna\" y cifra=null. No la "
                    f"estimes ni la leas de la prosa del informe."
                ),
            }
        ],
        "jump_to": "model",
    }


# ---------------------------------------------------------------------------
# Los límites
# ---------------------------------------------------------------------------
def limites() -> list:
    """Los dos topes de gasto, en el orden en que tienen que ir.

    **El orden importa y no es cosmético.** Los middleware se ejecutan en el
    orden de la lista, así que un límite colocado después del verificador no
    impide que el verificador se ejecute una vez de más. Los topes van
    primero, siempre.

    **Por qué hacen falta los dos.** `ToolCallLimitMiddleware` corta el caso de
    la sesión 2: la pregunta por el margen bruto de Amazon, que Amazon no
    reporta, con el modelo reintentando variantes del concepto indefinidamente.
    Pero no corta el bucle que puede abrir el verificador, porque un modelo que
    contesta sin llamar a ninguna herramienta no gasta llamadas a herramienta.
    Ese lo corta `ModelCallLimitMiddleware`.

    Los valores, 8 y 10, salen de mirar cuánto necesita la pregunta más
    exigente que se ha planteado: una comparativa entre dos ejercicios gasta
    dos `get_xbrl_fact` más una `search_filings` para la explicación, tres
    llamadas; con un `list_available` de más, una búsqueda fallida que se
    reformula y la corrección del guardrail, se llega a seis. Ocho deja margen
    para lo imprevisto y corta mucho antes de que el coste se dispare.
    """
    return [
        ToolCallLimitMiddleware(run_limit=config.LIMITE_LLAMADAS_HERRAMIENTA),
        ModelCallLimitMiddleware(run_limit=config.LIMITE_LLAMADAS_MODELO),
    ]


def seccion_muy_larga(peticion) -> bool:
    """¿La sección que se va a leer pasa de 20.000 tokens?

    El predicado del Human-in-the-Loop. Un HITL que interrumpe en cada llamada
    no se usa: a la tercera vez la gente aprueba sin leer. El predicado es lo
    que lo hace utilizable, porque interrumpe solo en lo caro.
    """
    argumentos = peticion.tool_call["args"]
    seccion = corpus.seccion(
        str(argumentos.get("ticker", "")).upper(),
        int(argumentos.get("fiscal_year", 0) or 0),
        str(argumentos.get("item", "")),
    )
    return bool(seccion and seccion["n_tokens"] > 20_000)


def human_in_the_loop():
    """Pide aprobación humana solo para las lecturas caras de verdad.

    **No entra en el sistema que se evalúa.** Un middleware que interrumpe
    esperando una respuesta humana es incompatible con `evaluar()`, que ejecuta
    veinte preguntas sin nadie delante: la primera interrupción dejaría la
    invocación a medias y la evaluación colgada. Se deja implementado y
    documentado porque es el guardrail que un sistema en producción sí
    llevaría, y se demuestra en el notebook de la sesión 2 con una invocación
    aparte.
    """
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    return HumanInTheLoopMiddleware(
        interrupt_on={
            "read_section": {
                "allowed_decisions": ["approve", "reject"],
                "when": seccion_muy_larga,
            },
            "search_filings": False,
            "get_xbrl_fact": False,
            "list_available": False,
        },
        description_prefix="Pendiente de aprobación",
    )
