"""Retrieval: lo que había dentro de la caja negra de `search_filings`.

El día 10 esto era una caja cerrada. Dentro había cuatro decisiones —troceado,
modelo de *embeddings*, índice y top-*k*— y cada una se puede hacer mejor o
peor. Este módulo las abre y añade las mejoras que el §4 del enunciado exige
medir.

## Las cinco estrategias, y en qué se diferencian

| Función | Qué añade | Coste |
| --- | --- | --- |
| `denso_plano` | nada: los *k* más parecidos de TODO el corpus | 0 llamadas al LLM |
| `con_filtros` | descarta lo que no es de la compañía, el ejercicio o la sección | 0 llamadas |
| `hibrido` | funde el orden denso con el de BM25 por RRF | 0 llamadas, +1 índice léxico |
| `con_reescritura` | traduce y reformula la consulta con el modelo antes de buscar | 1 llamada por búsqueda |
| `buscar` (sistema final) | reescritura + filtros + híbrido + reranking | 1 llamada + 1 cross-encoder |

## Dos decisiones que conviene justificar antes de leer el código

**El prefijo de BGE va en la consulta y no en los fragmentos.** Lo documenta
`indice/MANIFEST.md` y es el fallo silencioso favorito del curso: omitirlo no
da ningún error, simplemente recupera peor. Como el índice que se entrega se
construyó con esa convención, respetarla no es una opción de diseño sino una
condición para que la comparación con el baseline signifique algo.

**La fusión del híbrido combina posiciones, no puntuaciones.** Una es un coseno
entre -1 y 1 y la otra un número BM25 sin escala fija; sumarlas hace que la
escala arbitraria de una domine a la otra. Reciprocal Rank Fusion suma
`1/(kk + posición)` de cada lista, que solo depende del orden.
"""

from __future__ import annotations

import functools
import json
import re

from agente import config, corpus

# ---------------------------------------------------------------------------
# Codificación
# ---------------------------------------------------------------------------
def codificar(textos: list[str], es_consulta: bool = True):
    """Vectores normalizados, con el prefijo de BGE si son consultas.

    `normalize_embeddings=True` no es decorativo: el índice es un
    `IndexFlatIP`, producto interno, y el producto interno de dos vectores
    normalizados **es** la similitud coseno. Sin normalizar, la puntuación
    dependería de la longitud del texto y dejaría de ser comparable.
    """
    _, _, codificador = corpus.cargar_indice()
    if es_consulta:
        textos = [config.PREFIJO_CONSULTA_BGE + t for t in textos]
    return codificador.encode(
        textos, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")


def fila_a_fragmento(fila, puntuacion: float) -> dict:
    """Una fila de `chunks_meta` en el dict que manejan las herramientas.

    `inicio_car` y `fin_car` viajan dentro a propósito: son lo que permite a
    `recall@k` comprobar por tramo exacto, y no solo por coincidencia de
    texto, si el fragmento recuperado contiene el ancla.
    """
    return {
        "chunk_id": fila["chunk_id"],
        "ticker": fila["ticker"],
        "fiscal_year": int(fila["fiscal_year"]),
        "item": fila["item"],
        "texto": fila["texto"],
        "n_tokens": int(fila["n_tokens"]),
        "contiene_tabla": bool(fila["contiene_tabla"]),
        "inicio_car": int(fila["inicio_car"]),
        "fin_car": int(fila["fin_car"]),
        "puntuacion": round(float(puntuacion), 4),
    }


def formatear_fragmentos(fragmentos: list[dict]) -> str:
    """Los fragmentos, en el texto que ve el modelo.

    Cada uno lleva su `chunk_id` entre corchetes delante. Eso es lo que hace
    posible que el agente cite y que el evaluador de citas compruebe la cita:
    si el identificador no viajara en el texto, el modelo no tendría de dónde
    copiarlo.

    Cuando no hay resultados se devuelve una instrucción y no una cadena
    vacía. Un resultado vacío sin explicación hace que el modelo reintente la
    misma consulta; una instrucción concreta le dice qué cambiar.
    """
    if not fragmentos:
        return (
            "Sin resultados para esa consulta con esos filtros. Prueba a "
            "quitar el filtro de item, a ampliar k, o a reformular la "
            "consulta en inglés con el vocabulario del informe. Si sospechas "
            "que la compañía o el ejercicio no están en el corpus, "
            "compruébalo con list_available."
        )
    return "\n\n---\n\n".join(
        f"[{f['chunk_id']}] {f['ticker']} FY{f['fiscal_year']} "
        f"Item {f['item']} (similitud {f['puntuacion']:.3f})\n{f['texto']}"
        for f in fragmentos
    )


# ---------------------------------------------------------------------------
# 1. Denso plano — el baseline
# ---------------------------------------------------------------------------
def denso_plano(consulta: str, k: int = config.K_POR_DEFECTO, **_) -> list[dict]:
    """Los *k* fragmentos más parecidos de TODO el corpus.

    Es literalmente lo que hacía `search_filings` el día 10: una consulta, un
    `encode`, un `search`, y los *k* primeros vengan de donde vengan. Acepta y
    descarta los argumentos de filtro para poder intercambiarse con las demás
    estrategias en el bucle de medición sin condicionales.
    """
    indice, meta, _ = corpus.cargar_indice()
    puntuaciones, posiciones = indice.search(codificar([consulta]), k)
    return [
        fila_a_fragmento(meta.iloc[int(i)], s)
        for s, i in zip(puntuaciones[0], posiciones[0])
        if int(i) >= 0
    ]


# ---------------------------------------------------------------------------
# 2. Filtro por metadatos — el arreglo más barato
# ---------------------------------------------------------------------------
def con_filtros(
    consulta: str,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
    k: int = config.K_POR_DEFECTO,
) -> list[dict]:
    """Igual que `denso_plano`, pero descartando lo que no cuadra.

    Cada fragmento sabe de qué compañía, ejercicio y sección es. No usar esa
    información es tirar señal que ya está sobre la mesa, y arregla los dos
    fallos más frecuentes del baseline a la vez:

    - la respuesta vive en el Item 7A, que son 37 fragmentos de 1.749, y sin
      filtro compiten contra todo lo demás;
    - los 10-K repiten factores de riesgo casi literales entre ejercicios, así
      que lo más parecido a la pregunta puede ser el año equivocado.

    **Se busca sobre todo el índice y se filtra después**, no al revés. Con
    1.749 vectores eso es instantáneo y da el resultado exacto. Con un corpus
    de verdad habría que filtrar antes, construyendo un índice por partición o
    usando el filtrado nativo de FAISS, y esa es una conversación distinta que
    se comenta en el informe.

    Importante: se acumulan resultados **hasta reunir k válidos**. Devolver
    menos de *k* porque los primeros vecinos globales eran de otra compañía
    sería cambiar el filtro por un recorte.
    """
    indice, meta, _ = corpus.cargar_indice()

    # Máscara de posiciones permitidas. Se calcula una vez, no por candidato.
    permitidas = None
    if ticker or fiscal_year or item:
        mascara = meta.index
        if ticker:
            mascara = mascara.intersection(meta.index[meta["ticker"] == ticker])
        if fiscal_year:
            mascara = mascara.intersection(
                meta.index[meta["fiscal_year"].astype(int) == int(fiscal_year)]
            )
        if item:
            # Con corchetes y no `meta.item`: en pandas, `item` choca con el
            # método `Series.item`.
            mascara = mascara.intersection(meta.index[meta["item"] == item])
        permitidas = set(int(i) for i in mascara)
        if not permitidas:
            return []

    puntuaciones, posiciones = indice.search(codificar([consulta]), indice.ntotal)
    salida: list[dict] = []
    for puntuacion, posicion in zip(puntuaciones[0], posiciones[0]):
        posicion = int(posicion)
        if posicion < 0:
            continue
        if permitidas is not None and posicion not in permitidas:
            continue
        salida.append(fila_a_fragmento(meta.iloc[posicion], puntuacion))
        if len(salida) >= k:
            break
    return salida


# ---------------------------------------------------------------------------
# 3. Híbrido BM25 + denso, fundidos por RRF
# ---------------------------------------------------------------------------
_PALABRAS = re.compile(r"[a-z0-9$%.]+")


def tokenizar(texto: str) -> list[str]:
    """Minúsculas y palabras, conservando `$`, `%` y los puntos decimales.

    No es un detalle. El caso en el que BM25 le gana a un modelo de embeddings
    son los tickers, los años y las cifras: cadenas exactas. Un tokenizador que
    se coma el `$` y los decimales tira justo la señal que se venía a buscar y
    convierte el híbrido en ruido caro.
    """
    return _PALABRAS.findall(texto.lower())


@functools.lru_cache(maxsize=1)
def montar_bm25():
    """`(bm25, chunks)` sobre los 1.749 fragmentos. Se monta una sola vez."""
    from rank_bm25 import BM25Okapi

    chunks = corpus.cargar_chunks()
    return BM25Okapi([tokenizar(c["texto"]) for c in chunks]), chunks


def hibrido(
    consulta: str,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
    k: int = config.K_POR_DEFECTO,
    kk: int = 60,
) -> list[dict]:
    """Funde el orden denso con el orden BM25 mediante Reciprocal Rank Fusion.

        RRF(d) = suma sobre cada lista de  1 / (kk + posicion_de_d)

    **Por qué RRF y no una suma ponderada de puntuaciones.** El denso devuelve
    un coseno acotado en [-1, 1] y BM25 un número sin escala fija que depende
    del corpus y de la longitud de la consulta. Sumarlos requiere normalizar
    dos distribuciones distintas y elegir un peso, y ese peso habría que
    calibrarlo con los mismos datos con los que luego se mide, que es
    exactamente la forma de engañarse a uno mismo. RRF solo mira el orden, no
    tiene peso que ajustar y es lo que recomienda la literatura.

    **Por qué `kk = 60`.** Es el valor del artículo original de Cormack et al.
    (2009) y el que usan por defecto Elasticsearch y Vespa. Amortigua las
    primeras posiciones: sin él, el primero de una lista arrastraría el
    resultado aunque la otra lista lo tuviera en el puesto 300. No se ha
    ajustado a este corpus a propósito, para no calibrar un hiperparámetro
    sobre el mismo golden set con el que después se mide.

    Los filtros se aplican a las dos listas, no solo a la densa: si BM25
    pudiera colar fragmentos del ejercicio equivocado, el filtro dejaría de
    filtrar.
    """
    indice, meta, _ = corpus.cargar_indice()

    # --- lista densa, ya filtrada ------------------------------------------
    densos = con_filtros(consulta, ticker, fiscal_year, item, k=indice.ntotal)
    if not densos:
        return []
    posicion_densa = {f["chunk_id"]: i for i, f in enumerate(densos, 1)}
    por_id = {f["chunk_id"]: f for f in densos}

    # --- lista léxica, restringida a los mismos candidatos -----------------
    bm25, chunks_bm = montar_bm25()
    puntuaciones = bm25.get_scores(tokenizar(consulta))
    ordenados = sorted(
        (
            (puntuaciones[i], c["chunk_id"])
            for i, c in enumerate(chunks_bm)
            if c["chunk_id"] in posicion_densa
        ),
        reverse=True,
    )
    posicion_lexica = {cid: i for i, (_, cid) in enumerate(ordenados, 1)}

    MUY_LEJOS = 10**6
    fusionados = sorted(
        posicion_densa,
        key=lambda cid: -(
            1 / (kk + posicion_densa.get(cid, MUY_LEJOS))
            + 1 / (kk + posicion_lexica.get(cid, MUY_LEJOS))
        ),
    )
    return [por_id[cid] for cid in fusionados[:k]]


# ---------------------------------------------------------------------------
# 4. Reescritura de la consulta con el propio modelo
# ---------------------------------------------------------------------------
INSTRUCCION_REESCRITURA = """Reescribe la pregunta del usuario como una consulta de búsqueda para un índice de informes 10-K escritos en INGLÉS.

Reglas:
- Devuelve SOLO la consulta, sin comillas, sin explicación y sin prefijos.
- Escribe en inglés, usando el vocabulario del propio informe: "net sales",
  "revenue increased", "risk factors", "gross margin percentage",
  "capital expenditures", "provision for income taxes".
- Conserva intactos los elementos estructurales: el ticker, el ejercicio
  fiscal, la sección si está explícita y la magnitud concreta que se busca.
- No inventes hechos, cifras ni nombres que no estén en la pregunta.
- Si la pregunta compara dos ejercicios, escribe la consulta sobre la
  EXPLICACIÓN de la variación, no sobre las cifras: las cifras se consultan en
  XBRL, no se buscan en el texto."""


def _ruta_cache_reescrituras():
    return config.DIR_CACHE / "reescrituras.json"


@functools.lru_cache(maxsize=1)
def _cache_reescrituras() -> dict:
    ruta = _ruta_cache_reescrituras()
    if ruta.is_file():
        try:
            return json.loads(ruta.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _guardar_cache_reescrituras() -> None:
    config.DIR_CACHE.mkdir(exist_ok=True)
    _ruta_cache_reescrituras().write_text(
        json.dumps(_cache_reescrituras(), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


@functools.lru_cache(maxsize=1)
def _reescritor():
    return config.crear_modelo()


def reescribir(pregunta: str, usar_cache: bool = True) -> str:
    """La pregunta del usuario, convertida en consulta para el índice.

    **Por qué hace falta.** El corpus está en inglés y las preguntas están en
    español. `bge-small-en-v1.5` es monolingüe inglés —lo dice en el nombre— y
    BM25 cuenta coincidencias de cadenas, así que ninguna de las dos señales
    cruza bien la frontera del idioma: «¿cuánto creció el revenue de
    Microsoft?» y `Microsoft Cloud revenue increased 23% to $168.9 billion`
    comparten exactamente dos palabras.

    **Qué cuesta.** Una llamada al modelo antes de cada búsqueda. Es el más
    caro de los tres arreglos obligatorios y por eso hay que medir si compensa,
    que es justo lo que hace el notebook 04.

    **Por qué se cachea en disco.** Dos motivos independientes. Uno,
    reproducibilidad: la tabla del informe se regenera sin volver a pagar ni
    depender de que el modelo dé la misma respuesta. Dos, coste: la evaluación
    se ejecuta muchas veces mientras se itera, y reescribir la misma pregunta
    cincuenta veces es tirar dinero sin aprender nada. La caché vive en
    `.cache/`, que está en `.gitignore`, así que un clon limpio la reconstruye.

    Si no hay clave o el modelo falla, devuelve la pregunta original en vez de
    lanzar. Degradar a «sin reescritura» produce peor recall pero permite que
    todo lo demás siga midiéndose; una excepción tira la evaluación entera.
    """
    cache = _cache_reescrituras()
    if usar_cache and pregunta in cache:
        return cache[pregunta]

    if not config.hay_modelo():
        return pregunta
    try:
        consulta = (
            _reescritor()
            .invoke(
                [
                    {"role": "system", "content": INSTRUCCION_REESCRITURA},
                    {"role": "user", "content": pregunta},
                ]
            )
            .text.strip()
            .strip('"')
        )
    except Exception:
        return pregunta

    if not consulta:
        return pregunta
    cache[pregunta] = consulta
    _guardar_cache_reescrituras()
    return consulta


def con_reescritura(
    consulta: str,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
    k: int = config.K_POR_DEFECTO,
) -> list[dict]:
    """Reescritura + filtros de metadatos. Sin el híbrido."""
    return con_filtros(reescribir(consulta), ticker, fiscal_year, item, k)


# ---------------------------------------------------------------------------
# 5. Reranking con cross-encoder
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.MODELO_RERANKER)


def rerank(consulta: str, candidatos: list[dict], k: int) -> list[dict]:
    """Reordena los candidatos puntuando cada par (consulta, fragmento).

    **La diferencia con el bi-encoder.** El índice FAISS compara dos vectores
    calculados por separado: el fragmento se codificó sin saber nada de la
    consulta. Un cross-encoder mete consulta y fragmento juntos en el modelo y
    puntúa la pareja, así que puede captar relaciones que el vector del
    fragmento no podía anticipar. Es mucho más preciso y mucho más caro: no se
    puede aplicar a 1.749 fragmentos, solo a los pocos que el retriever ya ha
    preseleccionado.

    Por eso el patrón es en dos fases: recuperar barato y ancho (50
    candidatos), reordenar caro y estrecho (los 5 que se devuelven).

    Si el modelo no se puede cargar —sin red la primera vez, por ejemplo— se
    devuelven los candidatos en su orden original. Degradar es preferible a
    romper: el sistema sigue funcionando, solo que sin esta mejora.
    """
    if not candidatos:
        return []
    try:
        modelo = _reranker()
    except Exception:
        return candidatos[:k]

    pares = [(consulta, f["texto"]) for f in candidatos]
    puntuaciones = modelo.predict(pares)
    ordenados = sorted(
        zip(puntuaciones, candidatos), key=lambda par: -float(par[0])
    )
    salida = []
    for puntuacion, fragmento in ordenados[:k]:
        copia = dict(fragmento)
        copia["puntuacion_densa"] = fragmento["puntuacion"]
        copia["puntuacion"] = round(float(puntuacion), 4)
        salida.append(copia)
    return salida


# ---------------------------------------------------------------------------
# El retriever del sistema final
# ---------------------------------------------------------------------------
# Cuántos candidatos pasa el retriever barato al reranker caro. 50 sale de un
# compromiso: con 1.749 fragmentos y filtro de compañía y ejercicio ya
# aplicado, un documento tiene del orden de 150 fragmentos, así que 50 cubre
# un tercio del espacio filtrado; y 50 pares en un cross-encoder pequeño son
# unas décimas de segundo en CPU.
CANDIDATOS_PARA_RERANK = 50


def buscar(
    consulta: str,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
    k: int = config.K_POR_DEFECTO,
    estrategia: str = "final",
) -> list[dict]:
    """Punto de entrada único del retrieval. `estrategia` elige la variante.

    Tener una sola función con un selector, en vez de cinco funciones que las
    herramientas eligen por su cuenta, es lo que permite que el notebook 04
    mida las cinco con el mismo código y que la tabla compare el retriever y
    no la forma de llamarlo.
    """
    if estrategia == "denso_plano":
        return denso_plano(consulta, k=k)
    if estrategia == "filtros":
        return con_filtros(consulta, ticker, fiscal_year, item, k)
    if estrategia == "hibrido":
        return hibrido(consulta, ticker, fiscal_year, item, k)
    if estrategia == "reescritura":
        return con_reescritura(consulta, ticker, fiscal_year, item, k)
    if estrategia == "reescritura_hibrido":
        return hibrido(reescribir(consulta), ticker, fiscal_year, item, k)
    if estrategia == "final":
        consulta_en = reescribir(consulta)
        candidatos = hibrido(
            consulta_en, ticker, fiscal_year, item, k=CANDIDATOS_PARA_RERANK
        )
        return rerank(consulta_en, candidatos, k)
    raise ValueError(f"Estrategia de retrieval desconocida: {estrategia!r}")


ESTRATEGIAS = [
    "denso_plano",
    "filtros",
    "hibrido",
    "reescritura",
    "reescritura_hibrido",
    "final",
]

# Qué cuesta cada estrategia, para la columna de coste de la tabla del
# notebook 04. El coste de un retriever no es solo dinero: una dependencia
# más y 300 ms más de latencia también se pagan.
COSTE_ESTRATEGIA = {
    "denso_plano": "0 llamadas al LLM",
    "filtros": "0 llamadas al LLM",
    "hibrido": "0 llamadas, +1 índice léxico en memoria",
    "reescritura": "1 llamada al LLM por búsqueda",
    "reescritura_hibrido": "1 llamada + índice léxico",
    "final": "1 llamada + índice léxico + cross-encoder (50 pares)",
}


# ---------------------------------------------------------------------------
# La métrica: recall@k anclado al texto
# ---------------------------------------------------------------------------
_ESPACIOS = re.compile(r"\s+")


def normalizar(texto: str) -> str:
    """Espacios colapsados y minúsculas.

    Comparar cadenas de un PDF sin normalizar produce falsos negativos por
    saltos de línea y espacios dobles que no significan nada. Es la misma
    normalización que usa el evaluador de citas, y tiene que serlo: si el
    retriever y el evaluador midieran la coincidencia de forma distinta, un
    fragmento podría contar como recuperado y su cita como inventada.
    """
    return _ESPACIOS.sub(" ", texto).strip().lower()


def acierta(item_golden: dict, recuperados: list[dict]) -> bool:
    """¿Alguno de los fragmentos recuperados contiene el ancla entera?

    Es la primitiva de `recall@k`, y está copiada de `miax_s2.acierta` a
    propósito: si cada grupo escribiera su propia métrica, la tabla del
    informe dejaría de comparar nada con la de clase.

    **No depende del troceado.** La verdad del golden set es una frase del
    informe y no un `chunk_id`, porque en cuanto se cambia la ventana o el
    solape todos los identificadores son otros y el grupo que mejora el
    troceado saldría penalizado por haberlo mejorado. El notebook 04 cambia el
    troceado, así que esto no es hipotético.

    **Un fragmento del ejercicio equivocado NO cuenta**, aunque contenga el
    ancla. Los 10-K repiten factores de riesgo palabra por palabra de un año
    para otro: sin esta comprobación, recuperar el FY2024 puntuaría como si se
    hubiera encontrado el FY2025.
    """
    ancla = item_golden.get("ancla_texto")
    if not ancla:
        return False
    objetivo = normalizar(ancla)
    inicio, fin = item_golden.get("ancla_inicio"), item_golden.get("ancla_fin")

    for fragmento in recuperados:
        sin_metadatos = (
            fragmento.get("ticker") is None and fragmento.get("fiscal_year") is None
        )
        mismo_documento = fragmento.get("ticker") == item_golden.get("ticker") and int(
            fragmento.get("fiscal_year", -1)
        ) == int(item_golden.get("fiscal_year", -2))
        if not (sin_metadatos or mismo_documento):
            continue
        # 1. Por tramo exacto, si el fragmento trae desplazamientos.
        if (
            inicio is not None
            and fragmento.get("inicio_car") is not None
            and fragmento.get("item") == item_golden.get("item_esperado")
            and fragmento["inicio_car"] <= inicio
            and fragmento["fin_car"] >= fin
        ):
            return True
        # 2. Por texto normalizado. No le exige nada al troceador.
        if objetivo in normalizar(fragmento.get("texto", "")):
            return True
    return False


def recall_en_k(golden: list[dict], recuperados_por_id: dict) -> float:
    """`recall@k` sobre los ítems del golden set que llevan ancla."""
    con_ancla = [g for g in golden if g.get("ancla_texto")]
    if not con_ancla:
        return 0.0
    return sum(
        acierta(g, recuperados_por_id.get(g["id"], [])) for g in con_ancla
    ) / len(con_ancla)


def _a_tramos(texto: str, tokens_por_tramo: int, solape: int) -> list[tuple[int, int]]:
    """Ventanas `[inicio_car, fin_car)` de un texto, medidas en tokens.

    Se trocea por tokens y no por caracteres porque el límite que importa —lo
    que cabe en el contexto y lo que cuesta— se mide en tokens, no en letras.
    Los desplazamientos de salida sí van en caracteres, para que los
    fragmentos resultantes sigan siendo comparables con los anclas del golden
    set, que están expresados en caracteres.
    """
    codificador = corpus._codificador_tokens()
    ids = codificador.encode(texto)
    paso = max(1, tokens_por_tramo - solape)
    tramos: list[tuple[int, int]] = []
    posicion_caracter = 0
    for inicio in range(0, len(ids), paso):
        trozo = ids[inicio : inicio + tokens_por_tramo]
        if not trozo:
            break
        # El desplazamiento en caracteres se obtiene decodificando el prefijo:
        # es exacto y evita tener que confiar en una razón tokens/caracteres.
        inicio_car = len(codificador.decode(ids[:inicio]))
        fin_car = inicio_car + len(codificador.decode(trozo))
        tramos.append((inicio_car, min(fin_car, len(texto))))
        posicion_caracter = fin_car
        if inicio + tokens_por_tramo >= len(ids):
            break
    del posicion_caracter
    return tramos


def retrocear(tokens_por_tramo: int = 400, solape: int = 80) -> list[dict]:
    """Vuelve a trocear las 48 secciones con otra ventana y otro solape.

    Existe para poder responder a la pregunta que el enunciado plantea en el
    §4 —«¿es el troceado el que limita el recall, o es la búsqueda?»— sin
    responderla de oídas. El corpus que se entrega viene troceado con una
    ventana concreta; si el ancla cae partida entre dos fragmentos, ninguna
    mejora de la búsqueda la va a recuperar entera, y solo cambiando el
    troceado se puede saber.

    Es también el motivo por el que `acierta()` compara texto y no
    `chunk_id`: al retrocear, todos los identificadores son otros. Una métrica
    anclada al identificador daría recall cero aquí y penalizaría justo al que
    prueba la mejora.
    """
    nuevos: list[dict] = []
    for seccion in corpus.cargar_secciones():
        tramos = _a_tramos(seccion["texto"], tokens_por_tramo, solape)
        for posicion, (inicio, fin) in enumerate(tramos, 1):
            texto = seccion["texto"][inicio:fin]
            nuevos.append(
                {
                    "chunk_id": (
                        f"{seccion['ticker']}-{seccion['fiscal_year']}-"
                        f"{seccion['item']}-r{posicion:04d}"
                    ),
                    "ticker": seccion["ticker"],
                    "fiscal_year": int(seccion["fiscal_year"]),
                    "item": seccion["item"],
                    "posicion": posicion,
                    "texto": texto,
                    "n_tokens": corpus.contar_tokens(texto),
                    "contiene_tabla": "\t" in texto or texto.count("  ") > 20,
                    "inicio_car": inicio,
                    "fin_car": fin,
                }
            )
    return nuevos


# Qué prefijo pide cada modelo en la CONSULTA. No es opcional y no es el mismo
# para todos: los BGE en inglés piden esta frase exacta, y los de la familia
# MiniLM/MPNet no piden ninguno. Ponerle a MiniLM el prefijo de BGE le mete
# diez palabras de ruido en una consulta de seis, y compararlo así mediría el
# prefijo en vez de medir el modelo.
PREFIJOS_CONSULTA = {
    "BAAI/bge-small-en-v1.5": config.PREFIJO_CONSULTA_BGE,
    "BAAI/bge-base-en-v1.5": config.PREFIJO_CONSULTA_BGE,
    "BAAI/bge-large-en-v1.5": config.PREFIJO_CONSULTA_BGE,
}


def prefijo_de(modelo: str) -> str:
    return PREFIJOS_CONSULTA.get(modelo, "")


class _CodificadorOpenAI:
    """`text-embedding-3-small` con la misma interfaz que SentenceTransformer.

    Existe para que el bucle de medición del notebook 04 pueda tratar a los
    modelos locales y al de OpenAI exactamente igual, y la comparación sea de
    los modelos y no de dos rutas de código distintas.
    """

    def __init__(self, nombre: str = config.MODELO_EMBEDDINGS_OPENAI):
        from langchain_openai import OpenAIEmbeddings

        config.cargar_clave()
        self.nombre = nombre
        self._cliente = OpenAIEmbeddings(model=nombre)

    def encode(self, textos, batch_size: int = 256, **_):
        import numpy as np

        vectores: list[list[float]] = []
        for comienzo in range(0, len(textos), batch_size):
            vectores.extend(
                self._cliente.embed_documents(textos[comienzo : comienzo + batch_size])
            )
        matriz = np.asarray(vectores, dtype="float32")
        # OpenAI ya devuelve vectores normalizados, pero normalizar dos veces
        # no cuesta nada y garantiza que el producto interno sea el coseno.
        normas = np.linalg.norm(matriz, axis=1, keepdims=True)
        return matriz / np.clip(normas, 1e-12, None)


def crear_codificador(modelo: str):
    if modelo.startswith("text-embedding"):
        return _CodificadorOpenAI(modelo)
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(modelo)


@functools.lru_cache(maxsize=8)
def indice_de(
    tokens_por_tramo: int | None = None,
    solape: int = 80,
    modelo: str | None = None,
):
    """`(indice, fragmentos, codificador)` construido al vuelo.

    `tokens_por_tramo=None` usa el troceado que viene con el corpus; cualquier
    otro valor lo vuelve a trocear. En los dos casos se recalculan los vectores
    con el modelo indicado.

    No se guarda en disco: son entre 1.500 y 3.000 vectores, unos pocos
    megabytes y unos segundos de CPU, y versionar los índices intermedios de un
    barrido de hiperparámetros solo ensucia el repositorio. La caché en memoria
    evita reconstruirlo dentro del bucle de medición.
    """
    import faiss
    import numpy as np

    modelo = modelo or config.MODELO_EMBEDDINGS
    fragmentos = (
        corpus.cargar_chunks()
        if tokens_por_tramo is None
        else retrocear(tokens_por_tramo, solape)
    )
    codificador = crear_codificador(modelo)
    vectores = codificador.encode(
        [f["texto"] for f in fragmentos],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64,
    ).astype("float32")
    indice = faiss.IndexFlatIP(vectores.shape[1])
    indice.add(np.ascontiguousarray(vectores))
    return indice, fragmentos, codificador


def buscar_en_alterno(
    consulta: str,
    tokens_por_tramo: int | None = None,
    solape: int = 80,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
    k: int = config.K_POR_DEFECTO,
    modelo: str | None = None,
) -> list[dict]:
    """`con_filtros`, pero sobre un índice retroceado o con otro modelo."""
    modelo = modelo or config.MODELO_EMBEDDINGS
    indice, fragmentos, codificador = indice_de(tokens_por_tramo, solape, modelo)
    vector = codificador.encode(
        [prefijo_de(modelo) + consulta],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    puntuaciones, posiciones = indice.search(vector, indice.ntotal)

    salida: list[dict] = []
    for puntuacion, posicion in zip(puntuaciones[0], posiciones[0]):
        posicion = int(posicion)
        if posicion < 0:
            continue
        fragmento = fragmentos[posicion]
        if ticker and fragmento["ticker"] != ticker:
            continue
        if fiscal_year and fragmento["fiscal_year"] != int(fiscal_year):
            continue
        if item and fragmento["item"] != item:
            continue
        salida.append({**fragmento, "puntuacion": round(float(puntuacion), 4)})
        if len(salida) >= k:
            break
    return salida


def posicion_del_ancla(item_golden: dict, ordenados: list[dict]) -> int | None:
    """En qué puesto aparece el primer fragmento que contiene el ancla.

    `recall@5` dice sí o no; esto dice por cuánto. Un ancla en el puesto 7 y
    otra en el 1.400 fallan las dos y no son el mismo problema: la primera se
    arregla subiendo *k* o reordenando, y la segunda no.
    """
    for posicion, fragmento in enumerate(ordenados, 1):
        if acierta(item_golden, [fragmento]):
            return posicion
    return None


def medir_estrategia(
    golden: list[dict],
    estrategia: str,
    k: int = config.K_POR_DEFECTO,
    usar_metadatos_del_golden: bool = True,
) -> dict:
    """`recall@k` de una estrategia sobre las preguntas con ancla.

    `usar_metadatos_del_golden` decide de dónde salen los filtros. En el
    notebook 04 se pone a `True`, que es el techo: mide el retriever suponiendo
    que alguien le pasa el ticker, el ejercicio y la sección correctos. En
    producción nadie los pasa en una tabla, los tiene que sacar el agente de la
    pregunta, y por eso la evaluación de punta a punta del notebook 05 mide
    otra cosa distinta y más dura.
    """
    con_ancla = [g for g in golden if g.get("ancla_texto")]
    recuperados: dict[str, list[dict]] = {}
    for g in con_ancla:
        if usar_metadatos_del_golden:
            recuperados[g["id"]] = (
                buscar(
                    g["pregunta"],
                    g["ticker"],
                    g["fiscal_year"],
                    g.get("item_esperado"),
                    k=k,
                    estrategia=estrategia,
                )
                or []
            )
        else:
            recuperados[g["id"]] = (
                buscar(g["pregunta"], k=k, estrategia=estrategia) or []
            )
    fallan = [g["id"] for g in con_ancla if not acierta(g, recuperados[g["id"]])]
    return {
        "estrategia": estrategia,
        "recall": recall_en_k(con_ancla, recuperados),
        "evaluables": len(con_ancla),
        "fallan": fallan,
        "coste": COSTE_ESTRATEGIA.get(estrategia, ""),
        "recuperados": recuperados,
    }
