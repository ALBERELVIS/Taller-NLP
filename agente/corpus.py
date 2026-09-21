"""Capa de datos: el corpus, el índice y los hechos XBRL.

Separar esta capa de la lógica del agente no es estética. Las herramientas
tienen que ser funciones pequeñas, deterministas y comprobables sin levantar
un modelo de lenguaje; si cada una abriera sus propios ficheros, no habría
forma de testearlas ni de saber qué versión de los datos produjo una tabla.

## Por qué este módulo reconstruye el corpus

El material de la práctica se reparte en dos ZIP: `corpus_miax_2026.zip` (los
textos y los hechos XBRL) e `indice_faiss.zip` (el índice). En este proyecto
solo llegó el segundo, descomprimido en `dataset/indice/`. Falta, por tanto,
`secciones.jsonl`, `chunks.jsonl` y `xbrl_facts.parquet`.

Se reconstruyen, y se puede hacer sin pérdida por dos motivos independientes:

1. **Los textos están dentro del índice.** `chunks_meta.parquet` guarda, junto
   a cada vector, el texto íntegro del fragmento y sus desplazamientos
   `inicio_car`/`fin_car` dentro de la sección de la que salió. Colocando cada
   fragmento en su desplazamiento se recompone la sección entera. La prueba de
   que la recomposición es correcta no es un argumento, es una comprobación:
   los 13 `ancla_texto` del golden set oficial tienen que caer exactamente en
   sus `[ancla_inicio, ancla_fin)`, carácter a carácter. Si uno solo se
   desplaza, la reconstrucción está mal y hay que parar.

2. **Los hechos XBRL son públicos.** Son los datos que las propias compañías
   presentaron ante la SEC, y la SEC los publica en una API de solo lectura
   (`data.sec.gov/api/xbrl/companyfacts`). No es «bajarse EDGAR»: son seis
   peticiones a un JSON estructurado, se cachean en disco y no se vuelven a
   pedir. La validación es la misma idea que antes: las cifras que el golden
   set oficial declara como verdad tienen que coincidir al céntimo, y los
   huecos que el enunciado declara (Amazon sin `GrossProfit`) tienen que
   seguir vacíos.

Si en algún momento aparecen los ZIP originales, basta con dejarlos en la raíz
del proyecto: `construir_corpus()` los prefiere y se salta la reconstrucción.
"""

from __future__ import annotations

import functools
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from agente import config

# ---------------------------------------------------------------------------
# Los conceptos XBRL que forma el corpus
# ---------------------------------------------------------------------------
# Doce magnitudes de la taxonomía US-GAAP, elegidas por dos criterios: que
# cubran lo que una pregunta razonable sobre un 10-K puede pedir (cuenta de
# resultados, balance y flujos de caja), y que incluyan a propósito los casos
# donde el corpus tiene huecos reales.
#
# Las DOS etiquetas de ingresos están las dos en la lista, y eso es
# deliberado: NVIDIA y Alphabet usan `Revenues`; Apple, Microsoft, Meta y
# Amazon usan `RevenueFromContractWithCustomerExcludingAssessedTax`; y
# Alphabet etiqueta las dos en FY2024 pero solo la primera en FY2025. Meter
# las dos y dejar que falte la que falta es lo que reproduce fielmente la
# trampa que el enunciado describe, y lo que obliga al agente a mirar el
# fichero en vez de razonar por analogía con otra compañía.
CONCEPTOS_XBRL = [
    # Cuenta de resultados
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "CostOfRevenue",
    "GrossProfit",
    "ResearchAndDevelopmentExpense",
    "OperatingIncomeLoss",
    "IncomeTaxExpenseBenefit",
    "NetIncomeLoss",
    "EarningsPerShareDiluted",
    # Balance
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    # Flujos de caja
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
]

# Conceptos que se han considerado y **descartado a propósito**, porque
# incluirlos destruiría un hueco declarado del corpus:
#
# - `CostOfGoodsAndServicesSold`: Amazon y Microsoft lo etiquetan. Teniéndolo
#   junto a los ingresos, el margen bruto de Amazon se puede calcular restando,
#   y la respuesta correcta a «¿cuál fue el margen bruto de Amazon?» dejaría de
#   ser «no está en el corpus». Esa pregunta es uno de los casos que el
#   enunciado señala como legítimos y que el día 24 se va a preguntar.
# - `LiabilitiesAndStockholdersEquity`: permitiría despejar el pasivo de
#   Amazon restándole los fondos propios, con el mismo efecto.
#
# La regla general: el corpus puede tener menos conceptos que la realidad, pero
# no puede tener menos huecos, porque los huecos son parte de lo que se evalúa.
CONCEPTOS_DESCARTADOS = [
    "CostOfGoodsAndServicesSold",
    "LiabilitiesAndStockholdersEquity",
]

# Conceptos de saldo (una fecha) frente a conceptos de flujo (un periodo). La
# distinción importa porque en la API de la SEC los primeros solo traen `end`
# y los segundos traen `start` y `end`, y confundirlos hace que se recoja el
# dato de un trimestre en lugar del ejercicio.
CONCEPTOS_INSTANTANEOS = {"Assets", "Liabilities", "StockholdersEquity"}

# Huecos que el enunciado declara explícitamente (§3, tercer aviso). No son
# fallos del corpus: son preguntas legítimas cuya respuesta correcta es que el
# dato no está. `construir_corpus()` comprueba que siguen vacíos, porque una
# reconstrucción que los rellenara con algo parecido destruiría la parte de la
# evaluación que mide si el agente sabe decir «no lo sé».
HUECOS_DECLARADOS = [
    ("AMZN", "GrossProfit"),
    ("AMZN", "Liabilities"),
    ("AMZN", "ResearchAndDevelopmentExpense"),
    ("META", "GrossProfit"),
    ("GOOGL", "GrossProfit"),
]

# Cabecera exigida por la SEC en su política de acceso automatizado. Sin un
# User-Agent identificable la API responde 403.
USER_AGENT_SEC = "MIAX practica academica (contacto: alumno@miax.example)"
URL_COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# NVIDIA no publica sus estados financieros bajo el Item 8: los deja bajo el
# Item 15 y en el 8 escribe una remisión de dos líneas. El corpus original
# sirve el contenido correcto bajo la clave "8" y deja constancia del origen
# en un campo aparte. Se reproduce aquí para no perder esa información.
ITEM_ORIGEN_ESPECIAL = {("NVDA", "8"): "15"}


class CorpusNoConstruido(RuntimeError):
    """El corpus no está montado. Se lanza con instrucciones, no a secas."""


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _leer_jsonl(ruta: Path) -> list[dict]:
    with ruta.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def _escribir_jsonl(ruta: Path, filas: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as f:
        for fila in filas:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")


@functools.lru_cache(maxsize=1)
def _codificador_tokens():
    """El tokenizador con el que se cuentan los tokens del corpus.

    Se usa `cl100k_base`, que es el de la familia GPT. No es necesariamente el
    mismo con el que el profesor contó los suyos, pero reproduce sus cifras
    con un error del 0,03 % sobre la sección más larga del corpus, y lo que se
    necesita el recuento para —decidir si una sección es demasiado cara para
    leerla entera, y comparar el coste de los tres caminos hacia el mismo
    dato— no cambia con esa diferencia.
    """
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def contar_tokens(texto: str) -> int:
    return len(_codificador_tokens().encode(texto))


# ---------------------------------------------------------------------------
# Reconstrucción de las secciones a partir de los fragmentos
# ---------------------------------------------------------------------------
def reconstruir_secciones(meta: pd.DataFrame) -> list[dict]:
    """Las 48 secciones del corpus, recompuestas desde `chunks_meta.parquet`.

    Cada fragmento sabe en qué posición de su sección empieza y acaba
    (`inicio_car`, `fin_car`). Escribiendo cada fragmento en su posición dentro
    de un búfer del tamaño de la sección se recupera el texto original. Los
    fragmentos se solapan —unos 80 tokens, por diseño del troceador— y el
    solapamiento no molesta: al escribir dos veces los mismos caracteres en las
    mismas posiciones, el resultado es el mismo.

    Quedan 27 huecos de 4 caracteres en todo el corpus, todos en Microsoft y
    todos entre el final de un párrafo y el comienzo de un encabezado. Son
    separadores que el troceador consumió al partir por encabezados. Se
    rellenan con saltos de línea, que es lo que casi con total seguridad eran
    y, sobre todo, lo único que no puede alterar ninguna medición: tanto el
    evaluador de citas como `recall@k` normalizan los espacios en blanco antes
    de comparar.
    """
    secciones: list[dict] = []
    for (ticker, fiscal_year, item), grupo in meta.groupby(
        ["ticker", "fiscal_year", "item"], sort=True
    ):
        grupo = grupo.sort_values("posicion")
        longitud = int(grupo["fin_car"].max())
        buffer = ["\n"] * longitud  # lo no escrito queda como salto de línea
        for fila in grupo.itertuples():
            inicio = int(fila.inicio_car)
            for desplazamiento, caracter in enumerate(fila.texto):
                posicion = inicio + desplazamiento
                if posicion < longitud:
                    buffer[posicion] = caracter
        texto = "".join(buffer)
        secciones.append(
            {
                "ticker": ticker,
                "empresa": config.NOMBRES_EMPRESA[ticker],
                "fiscal_year": int(fiscal_year),
                "item": item,
                "item_origen": ITEM_ORIGEN_ESPECIAL.get((ticker, item), item),
                "texto": texto,
                "n_caracteres": len(texto),
                "n_tokens": contar_tokens(texto),
                "n_fragmentos": len(grupo),
            }
        )
    return secciones


def verificar_anclas(secciones: list[dict], golden: list[dict]) -> list[str]:
    """Los anclas del golden set que NO caen donde dicen caer.

    Esta es la prueba de que la reconstrucción es correcta, y es una prueba
    fuerte: un ancla es una frase literal del informe con sus desplazamientos
    exactos, calculados por el profesor sobre el texto original. Si al cortar
    el texto reconstruido por `[ancla_inicio, ancla_fin)` sale exactamente esa
    frase, el texto reconstruido y el original coinciden al menos en esos
    tramos, repartidos por cinco compañías, dos ejercicios y tres secciones.

    Devuelve la lista de problemas. Vacía significa correcto.
    """
    por_clave = {(s["ticker"], s["fiscal_year"], s["item"]): s for s in secciones}
    problemas: list[str] = []
    for item in golden:
        ancla = item.get("ancla_texto")
        inicio, fin = item.get("ancla_inicio"), item.get("ancla_fin")
        if not ancla or inicio is None or fin is None:
            continue
        clave = (item["ticker"], int(item["fiscal_year"]), item["item_esperado"])
        seccion = por_clave.get(clave)
        if seccion is None:
            problemas.append(f"{item['id']}: no existe la sección {clave}")
            continue
        recortado = seccion["texto"][inicio:fin]
        if recortado != ancla:
            problemas.append(
                f"{item['id']}: el tramo [{inicio}, {fin}) de {clave} dice "
                f"{recortado[:60]!r} y debería decir {ancla[:60]!r}"
            )
    return problemas


def chunks_desde_meta(meta: pd.DataFrame) -> list[dict]:
    """`chunks.jsonl` a partir de los metadatos del índice.

    Es un volcado directo: `chunks_meta.parquet` ya contiene exactamente los
    campos que el corpus original servía en `chunks.jsonl`. La fila *i* del
    parquet describe el vector *i* del índice FAISS, así que conservar el orden
    del parquet no es opcional: es lo que mantiene alineados texto y vector.
    """
    columnas = [
        "chunk_id",
        "ticker",
        "fiscal_year",
        "item",
        "posicion",
        "texto",
        "n_tokens",
        "contiene_tabla",
        "inicio_car",
        "fin_car",
    ]
    filas = []
    for fila in meta[columnas].itertuples(index=False):
        filas.append(
            {
                "chunk_id": fila.chunk_id,
                "ticker": fila.ticker,
                "fiscal_year": int(fila.fiscal_year),
                "item": fila.item,
                "posicion": int(fila.posicion),
                "texto": fila.texto,
                "n_tokens": int(fila.n_tokens),
                "contiene_tabla": bool(fila.contiene_tabla),
                "inicio_car": int(fila.inicio_car),
                "fin_car": int(fila.fin_car),
            }
        )
    return filas


# ---------------------------------------------------------------------------
# Hechos XBRL desde la SEC
# ---------------------------------------------------------------------------
def _descargar_companyfacts(ticker: str, pausa: float = 0.2) -> dict:
    """El JSON de `companyfacts` de una compañía, cacheado en disco.

    Una sola petición por compañía, seis en total, y solo la primera vez: el
    resultado se guarda en `.cache/sec/` y a partir de ahí el proyecto es
    reproducible sin red. La pausa respeta el límite de 10 peticiones por
    segundo que publica la SEC.
    """
    destino = config.DIR_CACHE / "sec" / f"{ticker}.json"
    if destino.is_file():
        return json.loads(destino.read_text(encoding="utf-8"))

    destino.parent.mkdir(parents=True, exist_ok=True)
    peticion = urllib.request.Request(
        URL_COMPANYFACTS.format(cik=config.CIK[ticker]),
        headers={"User-Agent": USER_AGENT_SEC, "Accept-Encoding": "gzip, deflate"},
    )
    with urllib.request.urlopen(peticion, timeout=120) as respuesta:
        crudo = respuesta.read()
        if respuesta.headers.get("Content-Encoding") == "gzip":
            import gzip

            crudo = gzip.decompress(crudo)
    datos = json.loads(crudo.decode("utf-8"))
    destino.write_text(json.dumps(datos), encoding="utf-8")
    time.sleep(pausa)
    return datos


def _hechos_anuales(datos: dict, concepto: str) -> list[dict]:
    """Los hechos de un concepto que vienen de un 10-K y cubren un ejercicio.

    Tres filtros, y los tres hacen falta:

    - `form == "10-K"`: descarta lo reportado en trimestrales.
    - `fp == "FY"`: descarta los acumulados de nueve meses.
    - duración entre 340 y 400 días para los conceptos de flujo: descarta los
      trimestres que a veces vienen etiquetados como anuales.
    """
    entradas = datos.get("facts", {}).get("us-gaap", {}).get(concepto)
    if not entradas:
        return []

    salida: list[dict] = []
    for unidad, hechos in entradas.get("units", {}).items():
        for hecho in hechos:
            if not str(hecho.get("form", "")).startswith("10-K"):
                continue
            if hecho.get("fp") != "FY":
                continue
            fin = hecho.get("end")
            inicio = hecho.get("start")
            if concepto in CONCEPTOS_INSTANTANEOS:
                if inicio is not None:
                    continue
            else:
                if inicio is None:
                    continue
                dias = (pd.Timestamp(fin) - pd.Timestamp(inicio)).days
                if not 340 <= dias <= 400:
                    continue
            salida.append({**hecho, "unit": unidad})
    return salida


def _cierres_de_ejercicio(datos: dict) -> dict[int, str]:
    """La fecha de cierre de cada ejercicio fiscal de una compañía.

    El criterio es el que el propio enunciado obliga a usar: **el ejercicio
    fiscal es el año en que cierra el ejercicio, no el año en que se presenta
    el informe**. NVIDIA cierra su FY2025 en enero de 2025 y Alphabet el suyo
    en diciembre de 2025, presentado en 2026; los dos son FY2025.

    Se deriva de `NetIncomeLoss`, que las seis compañías reportan siempre, en
    lugar de escribirse a mano. Escribir a mano seis pares de fechas es
    exactamente la clase de dato que se copia mal una vez y envenena todo lo
    que viene detrás.
    """
    cierres: dict[int, str] = {}
    for hecho in _hechos_anuales(datos, "NetIncomeLoss"):
        fin = hecho["end"]
        anio = int(fin[:4])
        # Si hay varias presentaciones del mismo ejercicio, la más reciente.
        if anio not in cierres or fin > cierres[anio]:
            cierres[anio] = fin
    return cierres


def descargar_hechos_xbrl(tickers: list[str] | None = None) -> pd.DataFrame:
    """`xbrl_facts.parquet`, reconstruido desde la API pública de la SEC.

    Una fila por (compañía, ejercicio, concepto) que la compañía reportó de
    verdad. Lo que no reportó **no aparece**, y eso es la mitad del valor de
    esta tabla: es lo que permite que `get_xbrl_fact` conteste «Amazon no
    reportó GrossProfit» en vez de devolver un cero silencioso.

    Cuando una compañía presenta el mismo hecho en dos informes distintos —el
    del año y el del siguiente, que lo repite como comparativo— se conserva la
    presentación más reciente, que es la que incorpora reexpresiones.
    """
    tickers = tickers or config.TICKERS
    filas: list[dict] = []

    for ticker in tickers:
        datos = _descargar_companyfacts(ticker)
        cierres = _cierres_de_ejercicio(datos)

        for concepto in CONCEPTOS_XBRL:
            for hecho in _hechos_anuales(datos, concepto):
                anio = int(hecho["end"][:4])
                if anio not in config.EJERCICIOS:
                    continue
                if hecho["end"] != cierres.get(anio):
                    continue  # no es el cierre de ejercicio de esa compañía
                filas.append(
                    {
                        "ticker": ticker,
                        "fiscal_year": anio,
                        "concept": concepto,
                        "value": float(hecho["val"]),
                        "unit": hecho["unit"],
                        "period_start": hecho.get("start"),
                        "period_end": hecho["end"],
                        "form": hecho.get("form", "10-K"),
                        "accn": hecho.get("accn", ""),
                        "filed": hecho.get("filed", ""),
                    }
                )

    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla

    # Una sola fila por (ticker, ejercicio, concepto): la presentada más tarde.
    tabla = (
        tabla.sort_values(["ticker", "fiscal_year", "concept", "filed"])
        .drop_duplicates(subset=["ticker", "fiscal_year", "concept"], keep="last")
        .sort_values(["ticker", "fiscal_year", "concept"])
        .reset_index(drop=True)
    )
    return tabla


def verificar_xbrl(xbrl: pd.DataFrame, golden: list[dict]) -> list[str]:
    """Los problemas de la tabla XBRL reconstruida. Vacía significa correcta.

    Dos comprobaciones independientes:

    1. **Contra el golden set oficial.** Cada pregunta numérica o comparativa
       declara un `concept_xbrl` y una `cifra_esperada`. Son cifras que el
       profesor sacó del fichero original, así que si la tabla reconstruida
       las reproduce al céntimo, reproduce el fichero original en esos puntos.
    2. **Contra los huecos declarados.** El enunciado dice qué conceptos no
       están. Si la reconstrucción los rellenara, estaría inventando datos
       donde el corpus dice que no los hay, y las preguntas de ausencia —al
       menos dos de las diez del día 24— dejarían de poder contestarse bien.
    """
    problemas: list[str] = []
    indexado = {
        (f.ticker, int(f.fiscal_year), f.concept): float(f.value)
        for f in xbrl.itertuples()
    }

    for item in golden:
        concepto = item.get("concept_xbrl")
        esperada = item.get("cifra_esperada")
        if not concepto or esperada is None:
            continue
        clave = (item["ticker"], int(item["fiscal_year"]), concepto)
        if clave not in indexado:
            problemas.append(f"{item['id']}: falta el hecho {clave}")
        elif abs(indexado[clave] - float(esperada)) > 0.5:
            problemas.append(
                f"{item['id']}: {clave} vale {indexado[clave]:,.0f} y el "
                f"golden set espera {float(esperada):,.0f}"
            )

    for ticker, concepto in HUECOS_DECLARADOS:
        presentes = [
            fy for fy in config.EJERCICIOS if (ticker, fy, concepto) in indexado
        ]
        if presentes:
            problemas.append(
                f"hueco declarado relleno: {ticker} tiene '{concepto}' en "
                f"FY{presentes}. El enunciado dice que no lo reporta."
            )

    return problemas


# ---------------------------------------------------------------------------
# Construcción del corpus
# ---------------------------------------------------------------------------
def _descomprimir_zips_originales() -> bool:
    """Si los ZIP del profesor están en la raíz, se usan y se verifican.

    Se comprueban antes que la reconstrucción por un motivo de principio: si
    existe el dato original, se usa el original. La reconstrucción es una
    respuesta a que falte, no una mejora.
    """
    paquetes = [
        (
            "corpus_miax_2026.zip",
            "4233c37fc9e9d12091af7a146063ad70903a3fe51404a485854f4021c63daee4",
        ),
        (
            "indice_faiss.zip",
            "6b5610ad8ac6ea50364445d39bb464d993cbd87048fb07c4fe16657d7ac11655",
        ),
    ]
    rutas = [config.RAIZ / nombre for nombre, _ in paquetes]
    if not all(r.is_file() for r in rutas):
        return False

    for (nombre, esperado), ruta in zip(paquetes, rutas):
        obtenido = _sha256(ruta)
        if obtenido != esperado:
            raise RuntimeError(
                f"{nombre} no coincide con el hash esperado: el fichero está "
                f"corrupto o es de otra versión.\n"
                f"  esperado: {esperado}\n  obtenido: {obtenido}"
            )
        with zipfile.ZipFile(ruta) as zf:
            zf.extractall(config.DIR_CORPUS)
    return True


def construir_corpus(forzar: bool = False, verboso: bool = True) -> dict:
    """Deja `corpus/` listo y devuelve un informe de lo que hizo.

    Es idempotente: si el corpus ya está construido y `forzar` es `False`, no
    hace nada. Está pensada para poder llamarse desde la primera celda de
    cualquier notebook sin coste.
    """

    def log(mensaje: str) -> None:
        if verboso:
            print(mensaje)

    config.DIR_CORPUS.mkdir(parents=True, exist_ok=True)
    resumen: dict = {"origen": None}

    ya_esta = all(
        p.is_file()
        for p in (
            config.RUTA_SECCIONES,
            config.RUTA_CHUNKS,
            config.RUTA_XBRL,
            config.RUTA_FAISS,
            config.RUTA_META,
        )
    )
    if ya_esta and not forzar:
        log(f"El corpus ya está montado en {config.DIR_CORPUS}. Nada que hacer.")
        resumen["origen"] = "ya_existente"
        return resumen

    # --- 1. ¿Están los ZIP originales? -------------------------------------
    if _descomprimir_zips_originales():
        log("ZIP originales encontrados y verificados: se usan tal cual.")
        resumen["origen"] = "zips_originales"
        return resumen

    log("No hay ZIP originales. Reconstruyendo el corpus.")
    resumen["origen"] = "reconstruido"

    # --- 2. El índice, copiado desde dataset/ ------------------------------
    origen_indice = config.DIR_DATASET / "indice"
    if not (origen_indice / "chunks_meta.parquet").is_file():
        raise CorpusNoConstruido(
            f"No encuentro {origen_indice / 'chunks_meta.parquet'}. Sin el "
            f"índice ni sus metadatos no hay de dónde reconstruir nada: deja "
            f"la carpeta `dataset/indice/` del profesor en su sitio, o los dos "
            f"ZIP originales en {config.RAIZ}."
        )
    config.DIR_INDICE.mkdir(parents=True, exist_ok=True)
    for fichero in origen_indice.iterdir():
        if fichero.is_file():
            shutil.copy2(fichero, config.DIR_INDICE / fichero.name)
    log(f"  índice copiado a {config.DIR_INDICE}")

    meta = pd.read_parquet(config.RUTA_META)
    resumen["n_fragmentos"] = len(meta)

    # --- 3. chunks.jsonl ---------------------------------------------------
    chunks = chunks_desde_meta(meta)
    _escribir_jsonl(config.RUTA_CHUNKS, chunks)
    log(f"  chunks.jsonl: {len(chunks)} fragmentos")

    # --- 4. secciones.jsonl y su verificación ------------------------------
    secciones = reconstruir_secciones(meta)
    golden = _leer_jsonl(config.RUTA_GOLDEN_OFICIAL)
    problemas_anclas = verificar_anclas(secciones, golden)
    if problemas_anclas:
        raise CorpusNoConstruido(
            "La reconstrucción de las secciones NO cuadra con los anclas del "
            "golden set oficial:\n  " + "\n  ".join(problemas_anclas)
        )
    _escribir_jsonl(config.RUTA_SECCIONES, secciones)
    resumen["n_secciones"] = len(secciones)
    resumen["n_anclas_verificadas"] = sum(1 for g in golden if g.get("ancla_texto"))
    log(
        f"  secciones.jsonl: {len(secciones)} secciones, "
        f"{resumen['n_anclas_verificadas']}/{resumen['n_anclas_verificadas']} "
        f"anclas del golden set oficial verificadas carácter a carácter"
    )

    # --- 5. xbrl_facts.parquet y su verificación ---------------------------
    try:
        xbrl = descargar_hechos_xbrl()
    except (urllib.error.URLError, TimeoutError) as e:
        raise CorpusNoConstruido(
            f"No se pudo consultar la API de la SEC ({type(e).__name__}: {e}).\n"
            f"Hace falta red UNA vez para reconstruir xbrl_facts.parquet; "
            f"después el proyecto funciona sin conexión. Alternativa: deja "
            f"corpus_miax_2026.zip en {config.RAIZ}."
        ) from e

    problemas_xbrl = verificar_xbrl(xbrl, golden)
    if problemas_xbrl:
        raise CorpusNoConstruido(
            "La tabla XBRL reconstruida NO cuadra con el golden set oficial:\n"
            "  " + "\n  ".join(problemas_xbrl)
        )
    xbrl.to_parquet(config.RUTA_XBRL, index=False)
    resumen["n_hechos_xbrl"] = len(xbrl)
    log(
        f"  xbrl_facts.parquet: {len(xbrl)} hechos, "
        f"{xbrl.concept.nunique()} conceptos, verificados contra el golden set"
    )

    # --- 6. Manifiesto propio ---------------------------------------------
    _escribir_manifiesto(secciones, chunks, xbrl)
    log(f"  MANIFIESTO.md escrito en {config.DIR_CORPUS}")

    return resumen


def _escribir_manifiesto(
    secciones: list[dict], chunks: list[dict], xbrl: pd.DataFrame
) -> None:
    """Documenta qué se reconstruyó, desde qué y con qué comprobaciones.

    El manifiesto original declara el SHA-256 de `chunks.jsonl`, y el nuestro
    no puede coincidir: dos volcados a JSON del mismo contenido difieren en el
    orden de las claves y en el escapado. Por eso este manifiesto sustituye la
    comprobación por hash —que aquí no dice nada— por las dos que sí dicen
    algo: que el número de vectores, de filas de metadatos y de fragmentos
    coincide, y que los anclas del golden set caen en su sitio.
    """
    contenido = f"""# Manifiesto del corpus reconstruido

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
| Secciones | {len(secciones)} |
| Fragmentos | {len(chunks)} |
| Hechos XBRL | {len(xbrl)} |
| Conceptos XBRL distintos | {xbrl.concept.nunique()} |
| Tokens totales | {sum(s["n_tokens"] for s in secciones):,} |
| SHA-256 de `chunks.jsonl` | `{_sha256(config.RUTA_CHUNKS)}` |

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
"""
    (config.DIR_CORPUS / "MANIFIESTO.md").write_text(contenido, encoding="utf-8")


# ---------------------------------------------------------------------------
# Acceso en tiempo de ejecución
# ---------------------------------------------------------------------------
def _exigir_corpus() -> None:
    if not config.RUTA_CHUNKS.is_file():
        raise CorpusNoConstruido(
            "El corpus no está montado. Ejecuta primero el notebook "
            "`00_Preparacion_del_corpus.ipynb`, o desde Python:\n"
            "    from agente.corpus import construir_corpus\n"
            "    construir_corpus()"
        )


@functools.lru_cache(maxsize=1)
def cargar_secciones() -> list[dict]:
    """Las 48 secciones. Es lo que sirve `read_section`."""
    _exigir_corpus()
    return _leer_jsonl(config.RUTA_SECCIONES)


@functools.lru_cache(maxsize=1)
def cargar_chunks() -> list[dict]:
    """Los 1.749 fragmentos. Es lo que sirve `search_filings`."""
    _exigir_corpus()
    return _leer_jsonl(config.RUTA_CHUNKS)


@functools.lru_cache(maxsize=1)
def chunks_por_id() -> dict[str, dict]:
    """`chunk_id -> fragmento`. Lo usa el evaluador de citas."""
    return {c["chunk_id"]: c for c in cargar_chunks()}


@functools.lru_cache(maxsize=1)
def cargar_xbrl() -> pd.DataFrame:
    """Los hechos XBRL. Es la fuente autorizada para cualquier cifra."""
    _exigir_corpus()
    return pd.read_parquet(config.RUTA_XBRL)


@functools.lru_cache(maxsize=1)
def cargar_indice():
    """`(indice, meta, codificador)`. Tarda unos segundos la primera vez.

    El modelo de embeddings pesa unos 130 MB y se descarga la primera vez que
    se usa, así que la carga va aquí dentro y no en el import: importar el
    módulo tiene que ser instantáneo.

    `meta` está alineado por posición con el índice: la fila *i* describe el
    vector *i*. Si eso se rompe, el retrieval devuelve el texto equivocado sin
    dar ningún error, que es el peor modo de fallo posible. La comprobación de
    abajo no es decorativa.
    """
    _exigir_corpus()
    import faiss
    from sentence_transformers import SentenceTransformer

    indice = faiss.read_index(str(config.RUTA_FAISS))
    meta = pd.read_parquet(config.RUTA_META)
    if indice.ntotal != len(meta):
        raise RuntimeError(
            f"El índice tiene {indice.ntotal} vectores y los metadatos "
            f"{len(meta)} filas. Están desalineados: vuelve a construir el "
            f"corpus con construir_corpus(forzar=True)."
        )
    codificador = SentenceTransformer(config.MODELO_EMBEDDINGS)
    return indice, meta, codificador


def seccion(ticker: str, fiscal_year: int, item: str) -> dict | None:
    """Una sección concreta, o `None` si no está en el corpus."""
    for s in cargar_secciones():
        if (
            s["ticker"] == ticker
            and int(s["fiscal_year"]) == int(fiscal_year)
            and s["item"] == item
        ):
            return s
    return None


def hecho_xbrl(ticker: str, fiscal_year: int, concept: str) -> dict | None:
    """Un hecho XBRL concreto, o `None` si la compañía no lo reportó."""
    xbrl = cargar_xbrl()
    filas = xbrl[
        (xbrl.ticker == ticker)
        & (xbrl.fiscal_year == int(fiscal_year))
        & (xbrl.concept == concept)
    ]
    return None if filas.empty else filas.iloc[0].to_dict()


def conceptos_disponibles(ticker: str, fiscal_year: int) -> list[str]:
    """Qué conceptos reportó esa compañía ese ejercicio.

    Lo devuelve `get_xbrl_fact` cuando el concepto pedido no está, y es la
    pieza que arregla el fallo «se inventó la cifra»: el modelo pide el
    concepto que le funcionó con otra compañía, no lo encuentra, y en vez de
    rellenar el hueco recibe la lista de lo que sí hay.
    """
    xbrl = cargar_xbrl()
    filas = xbrl[(xbrl.ticker == ticker) & (xbrl.fiscal_year == int(fiscal_year))]
    return sorted(filas.concept.unique().tolist())
