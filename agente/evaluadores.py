"""Los tres evaluadores obligatorios del §4 del enunciado.

Firma común: reciben el ítem del golden set y lo que devolvió el agente, y
devuelven `True`, `False` o `None`.

## Por qué existe el tercer estado

`None` significa «no aplica», y es distinto de `False`. Una pregunta
extractiva no tiene `cifra_esperada`, así que el evaluador de cifras no puede
opinar sobre ella; contarla como fallo hundiría la métrica de todas las
extractivas y haría que la columna «cifra» dijese más sobre la composición del
golden set que sobre el sistema. Los promedios que calcula `resumir()` se
toman solo sobre los casos aplicables.

## Por qué estos tres y no otros

Miden tres cosas independientes que pueden fallar por separado:

- **`cita_correcta`** ataca la alucinación de fuente: que el agente respalde su
  respuesta con un fragmento que no existe o que no dice lo que dice.
- **`cifra_coincide_xbrl`** ataca la alucinación numérica: el fallo caro del
  dominio.
- **`uso_la_tool_correcta`** ataca algo distinto de los dos anteriores, y es el
  que de verdad separa un sistema de uno que tuvo suerte: **acertar por el
  camino equivocado es un fallo.** Una pregunta numérica cuyo número sale bien
  porque el modelo lo leyó de la prosa suspende aquí, y debe suspender: ese
  camino funcionó hoy con esta tabla y dejará de funcionar el día que el
  troceador la parta por la mitad. Y el 41 % de los fragmentos de este corpus
  llevan una tabla partida dentro.
"""

from __future__ import annotations

from agente import config, corpus
from agente.agente import herramientas_usadas
from agente.middleware import cuadra
from agente.retrieval import normalizar

# Cuántos caracteres de la cita se exigen literales. No se exige la cita
# entera por una razón práctica: el modelo recorta, cambia una coma o une dos
# frases del mismo fragmento, y ninguna de esas tres cosas es inventarse una
# fuente. Con los primeros 120 caracteres normalizados ya no hay forma
# razonable de acertar por casualidad —son unas veinte palabras seguidas— y se
# sigue detectando la cita traducida, la parafraseada y la que viene de otro
# documento.
CARACTERES_DE_CITA_EXIGIDOS = 120


def _respuesta_de(resultado):
    if resultado is None:
        return None
    if isinstance(resultado, dict):
        return resultado.get("structured_response")
    return getattr(resultado, "structured_response", None)


# ---------------------------------------------------------------------------
# 1. Cita
# ---------------------------------------------------------------------------
def cita_correcta(item: dict, resultado: dict) -> bool | None:
    """El `chunk_id` citado existe, es del documento correcto y respalda la cita.

    Tres comprobaciones encadenadas, de menos a más exigente, y las tres hacen
    falta porque detectan fallos distintos:

    1. **¿Existe el `chunk_id`?** Si no existe en el corpus, el modelo se lo
       inventó. Es la alucinación de fuente en su forma más burda, y también
       la más fácil de pasar por alto: un identificador con el formato correcto
       parece legítimo.
    2. **¿Es del documento correcto?** Un fragmento real pero de otra compañía
       o de otro ejercicio no respalda la respuesta. Este caso es frecuente de
       verdad, no teórico: los 10-K repiten sus factores de riesgo casi palabra
       por palabra de un año para otro, así que una cita del FY2024 puede
       *parecer* que respalda una afirmación sobre el FY2025.
    3. **¿El texto del fragmento contiene lo que dice la cita?** Es lo que
       distingue citar de inventar. Se compara normalizando espacios y
       mayúsculas, porque un salto de línea de más no es una cita falsa.

    Devuelve `None` —no aplica— para las preguntas numéricas puras que se
    contestan solo con XBRL: ahí no hay fragmento que citar, y exigir uno
    penalizaría al agente por haber usado la herramienta correcta. Para una
    extractiva o una comparativa, en cambio, la ausencia de cita es `False`:
    el enunciado pide respuestas con fuente.
    """
    respuesta = _respuesta_de(resultado)
    if respuesta is None:
        return False

    chunk_id = getattr(respuesta, "chunk_id", None)
    familia = item.get("familia")

    if not chunk_id:
        # Sin cita. En una numérica pura contestada desde XBRL no aplica; en
        # cualquier otra, es un fallo.
        if familia == "numerica" and getattr(respuesta, "fuente", None) == "xbrl":
            return None
        if getattr(respuesta, "fuente", None) == "ninguna":
            return None  # no hay nada que citar si el dato no está
        return False

    fragmento = corpus.chunks_por_id().get(chunk_id)
    if fragmento is None:
        return False  # 1. se lo inventó

    if fragmento["ticker"] != item.get("ticker") or int(
        fragmento["fiscal_year"]
    ) != int(item.get("fiscal_year", -1)):
        return False  # 2. documento equivocado

    cita = getattr(respuesta, "cita", None)
    if not cita:
        return False  # dio identificador pero no copió nada

    esperado = normalizar(cita)[:CARACTERES_DE_CITA_EXIGIDOS]
    return esperado in normalizar(fragmento["texto"])  # 3. respaldo real


# ---------------------------------------------------------------------------
# 2. Cifra
# ---------------------------------------------------------------------------
def cifra_coincide_xbrl(item: dict, resultado: dict) -> bool | None:
    """La cifra afirmada coincide con la del golden set, con tolerancia.

    Se compara con `cuadra()` y no con `==`, y la misma tolerancia del 1 % que
    usa el guardrail. Tiene que ser la misma: si el middleware aceptara un
    redondeo que el evaluador rechaza, una respuesta pasaría el guardrail y
    suspendería la tabla, y la tabla dejaría de medir el sistema.

    `None` cuando el ítem no declara `cifra_esperada`: una extractiva no tiene
    número que comprobar.

    Caso especial, y es el que hace que este evaluador sirva para algo más que
    para contar aciertos: un ítem cuya respuesta correcta es **que el dato no
    está** se marca con `cifra_esperada = None` y `concept_xbrl` puesto. Ahí no
    basta con que el agente no dé cifra; se exige además que lo diga
    explícitamente con `fuente="ninguna"`. Callarse y decir «no está» no son lo
    mismo.
    """
    esperada = item.get("cifra_esperada")
    respuesta = _respuesta_de(resultado)

    if esperada is None:
        if item.get("respuesta_esperada_es_ausencia"):
            if respuesta is None:
                return False
            return (
                getattr(respuesta, "fuente", None) == "ninguna"
                and getattr(respuesta, "cifra", None) is None
            )
        return None

    if respuesta is None:
        return False
    afirmada = getattr(respuesta, "cifra", None)
    if afirmada is None:
        return False
    return cuadra(float(afirmada), float(esperada), config.TOLERANCIA_CIFRA)


# ---------------------------------------------------------------------------
# 3. Trayectoria
# ---------------------------------------------------------------------------
def uso_la_tool_correcta(item: dict, resultado: dict) -> bool | None:
    """La trayectoria pasó por TODAS las herramientas esperadas.

    Es el evaluador que mide el camino y no el resultado, y el que hace que
    este sistema se pueda defender. Acertar el número sin haber llamado a
    `get_xbrl_fact` suspende aquí, y tiene que suspender: significa que el
    número se leyó de la prosa, que funcionó hoy y que no generaliza.

    Para las comparativas se exige además **dos llamadas a `get_xbrl_fact`**
    cuando el ítem declara dos ejercicios. Contar solo herramientas distintas
    dejaría pasar al agente que consulta un año y estima el otro, que es
    precisamente el atajo que las comparativas están puestas para detectar.
    """
    esperadas = item.get("herramienta_esperada") or []
    if not esperadas:
        return None

    usadas = herramientas_usadas(resultado)
    if not set(esperadas).issubset(set(usadas)):
        return False

    if item.get("familia") == "comparativa" and "get_xbrl_fact" in esperadas:
        if usadas.count("get_xbrl_fact") < 2:
            return False
    return True


EVALUADORES = {
    "cita": cita_correcta,
    "cifra": cifra_coincide_xbrl,
    "trayectoria": uso_la_tool_correcta,
}


# ---------------------------------------------------------------------------
# Auxiliar: ¿la cita viene del sitio donde vive la respuesta?
# ---------------------------------------------------------------------------
def cita_fundamentada(item: dict, resultado: dict) -> bool | None:
    """El fragmento citado solapa con el ancla que el golden set declara.

    ## Por qué hace falta, además de `cita_correcta`

    `cita_correcta` comprueba que la cita sea **real**: que el `chunk_id`
    exista, que el fragmento sea del documento correcto y que el texto citado
    esté de verdad ahí. Es lo que pide el §4 del enunciado y ataca la
    alucinación de fuente.

    Lo que `cita_correcta` **no** puede comprobar es si ese fragmento real
    tiene algo que ver con la pregunta. Un 10-K tiene cuarenta fragmentos por
    sección; citar uno cualquiera de la compañía correcta es trivial, y se
    detectó midiendo: con el retrieval más pobre de los tres perfiles, todas
    las preguntas extractivas salían «correctas» porque el agente siempre
    encontraba algo real que citar. Una métrica que no distingue entre el
    mejor y el peor de los sistemas no está midiendo el sistema.

    ## El criterio

    El golden set declara, para cada pregunta extractiva o comparativa, el
    tramo `[ancla_inicio, ancla_fin)` de la sección donde está la frase que
    responde. Cada fragmento del corpus declara su tramo `[inicio_car,
    fin_car)` en esas mismas coordenadas. La pregunta «¿citó desde el sitio
    correcto?» se reduce entonces a si los dos intervalos se solapan, que es
    objetivo y no necesita un juez.

    Se exige **solape**, no contención, a propósito: el troceador parte por
    donde le toca y un ancla puede quedar repartida entre dos fragmentos. En
    ese caso los dos son citas legítimas y exigir que uno contenga el ancla
    entera penalizaría al agente por una decisión del troceador.

    Cuando el ítem no declara ancla —las numéricas puras— devuelve `None`.
    """
    if not item.get("ancla_texto"):
        return None

    respuesta = _respuesta_de(resultado)
    chunk_id = getattr(respuesta, "chunk_id", None) if respuesta else None
    if not chunk_id:
        return False

    fragmento = corpus.chunks_por_id().get(chunk_id)
    if fragmento is None:
        return False

    # Tiene que ser de la misma sección: dos secciones distintas usan el mismo
    # sistema de coordenadas empezando en cero, así que sin esta comprobación
    # un fragmento del Item 7 podría «solapar» con un ancla del Item 1A.
    if (
        fragmento["ticker"] != item.get("ticker")
        or int(fragmento["fiscal_year"]) != int(item.get("fiscal_year"))
        or fragmento["item"] != item.get("item_esperado")
    ):
        return False

    inicio, fin = item.get("ancla_inicio"), item.get("ancla_fin")
    if inicio is None or fin is None:
        # Sin desplazamientos, se cae a la comprobación textual.
        return normalizar(item["ancla_texto"]) in normalizar(fragmento["texto"])

    return fragmento["inicio_car"] < int(fin) and int(inicio) < fragmento["fin_car"]


# ---------------------------------------------------------------------------
# Un cuarto, auxiliar: ¿la respuesta en prosa dice lo que tenía que decir?
# ---------------------------------------------------------------------------
def acierta_familia(item: dict, resultado: dict) -> bool | None:
    """Aproximación al acierto de la respuesta, por familia.

    No es uno de los tres obligatorios y se marca como auxiliar a propósito,
    porque es el único que no se puede comprobar de forma completamente
    objetiva: nadie está leyendo la prosa y decidiendo si convence. Lo que se
    hace es exigir, en cada familia, la condición verificable **sin la cual la
    respuesta no puede ser correcta**.

    - **numérica**: la cifra cuadra con el XBRL. El acierto ES la cifra.
    - **comparativa**: cuadran las cifras de los dos ejercicios —exigir las dos
      es lo que distingue comparar de estimar— **y** la cita procede del tramo
      del informe donde la dirección explica la variación. Una comparativa es
      una cifra más una explicación; sin la segunda está a medias.
    - **extractiva**: no hay número, así que la condición es doble: cita
      verificada (`cita_correcta`) y procedente del sitio correcto
      (`cita_fundamentada`).

    **Qué no mide.** No comprueba que la prosa esté bien redactada ni que
    responda a lo que se preguntaba: comprueba que esté anclada donde tiene
    que estarlo. Un agente que copie la frase correcta y la interprete al
    revés aprobaría. Es el límite del criterio y se dice en el informe en
    lugar de disfrazarlo de métrica objetiva.
    """
    familia = item.get("familia")

    if familia == "numerica":
        return cifra_coincide_xbrl(item, resultado)

    if familia == "comparativa":
        principal = cifra_coincide_xbrl(item, resultado)
        if principal is not True:
            return principal
        anterior = item.get("cifra_anterior_esperada")
        if anterior is not None:
            respuesta = _respuesta_de(resultado)
            afirmada = getattr(respuesta, "cifra_anterior", None)
            if afirmada is None:
                return False
            if not cuadra(float(afirmada), float(anterior), config.TOLERANCIA_CIFRA):
                return False
        fundamentada = cita_fundamentada(item, resultado)
        return True if fundamentada is None else bool(fundamentada)

    if familia == "extractiva":
        if cita_correcta(item, resultado) is not True:
            return False
        fundamentada = cita_fundamentada(item, resultado)
        return True if fundamentada is None else bool(fundamentada)

    return None
