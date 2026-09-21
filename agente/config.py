"""Configuración central: rutas, clave de API, modelo y precios.

Este módulo existe para que ninguna de las decisiones globales del proyecto
esté escrita dos veces. Todo lo que un notebook necesita saber sobre *dónde*
están los datos, *qué* modelo se usa y *cuánto* cuesta una llamada sale de
aquí, y solo de aquí.

Tres decisiones se toman en este fichero, y las tres afectan al resultado de
la práctica:

1. **De dónde sale la clave de OpenAI.** Nunca del código. Primero de la
   variable de entorno `OPENAI_API_KEY`, y si no está, del fichero
   `api_key.txt` de la raíz del proyecto, que `.gitignore` excluye.
2. **Qué modelo hace de cerebro.** No se fija a ciegas: se prueba una lista de
   candidatos en orden de preferencia y se usa el primero al que la cuenta
   tenga acceso. El resultado se cachea en disco para que dos ejecuciones
   consecutivas no gasten una llamada cada una en averiguar lo mismo.
3. **Cuánto cuesta cada llamada.** El enunciado pide coste medio por pregunta
   como columna de primer nivel de la tabla del informe. La API de OpenAI
   devuelve tokens, no dólares, así que el precio hay que ponerlo nosotros y
   dejar por escrito de cuándo es.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

# Ruido de la consola que no aporta nada y tapa la salida que sí importa. Se
# silencia aquí, en el único módulo que importan todos los demás, en vez de
# repetir tres líneas mágicas en la primera celda de cada notebook.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
# `RAIZ` se calcula desde la posición de este fichero y no desde el directorio
# de trabajo. Es lo que permite que `responder()` funcione igual invocada desde
# un notebook de la raíz, desde `python -m agente.interfaz` o desde un test:
# el directorio de trabajo cambia, la posición del módulo no.
RAIZ = Path(__file__).resolve().parent.parent

RUTA_CLAVE = RAIZ / "api_key.txt"

# El corpus reconstruido por `00_Preparacion_del_corpus.ipynb`. Se llama
# `corpus/` y vive en la raíz a propósito: es exactamente donde lo buscan
# `miax_s1.py` y `miax_s2.py`, que se reparten en clase y que no vamos a
# modificar. Ver `CANDIDATOS_CORPUS` en esos dos módulos.
DIR_CORPUS = RAIZ / "corpus"
DIR_INDICE = DIR_CORPUS / "indice"

# El material que entrega el profesor, tal y como llegó. No se toca.
DIR_DATASET = RAIZ / "dataset"

RUTA_SECCIONES = DIR_CORPUS / "secciones.jsonl"
RUTA_CHUNKS = DIR_CORPUS / "chunks.jsonl"
RUTA_XBRL = DIR_CORPUS / "xbrl_facts.parquet"
RUTA_FAISS = DIR_INDICE / "corpus.faiss"
RUTA_META = DIR_INDICE / "chunks_meta.parquet"

RUTA_GOLDEN_OFICIAL = RAIZ / "golden_set.jsonl"
RUTA_GOLDEN_PROPIO = RAIZ / "golden_set_propio.jsonl"
RUTA_GOLDEN_AUSENCIAS = RAIZ / "golden_set_ausencias.jsonl"
RUTA_HOLDOUT_SIMULADO = RAIZ / "holdout_simulado.jsonl"

DIR_RESULTADOS = RAIZ / "resultados"
DIR_INFORME = RAIZ / "informe"
DIR_CACHE = RAIZ / ".cache"

# ---------------------------------------------------------------------------
# El índice y su contrato
# ---------------------------------------------------------------------------
# Estas tres constantes NO son configurables: las fija el índice FAISS que se
# entrega con la práctica, documentado en `dataset/indice/MANIFEST.md`.
#
# El prefijo es el detalle que más recall cuesta si se olvida. Los modelos BGE
# piden un prefijo en la CONSULTA y no en los fragmentos indexados. Omitirlo no
# da ningún error: simplemente recupera peor. Como el índice que se entrega se
# construyó con esa convención, cambiarla por nuestra cuenta invalidaría la
# comparación con el baseline de clase.
MODELO_EMBEDDINGS = "BAAI/bge-small-en-v1.5"
DIMENSION_EMBEDDINGS = 384
PREFIJO_CONSULTA_BGE = "Represent this sentence for searching relevant passages: "

# Cross-encoder del experimento de reranking (notebook 04). Es pequeño
# (~280 MB) y corre en CPU en segundos para 50 pares, que es todo lo que le
# pedimos.
MODELO_RERANKER = "BAAI/bge-reranker-base"

# Modelo de embeddings de OpenAI, solo para el experimento comparativo del
# notebook 04. No entra en el sistema final: ver la justificación allí.
MODELO_EMBEDDINGS_OPENAI = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# El corpus, declarado
# ---------------------------------------------------------------------------
TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA"]
EJERCICIOS = [2024, 2025]
ITEMS = ["1A", "7", "7A", "8"]

NOMBRES_EMPRESA = {
    "AAPL": "Apple Inc.",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms, Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
}

# Código CIK de la SEC, necesario para reconstruir `xbrl_facts.parquet` desde
# `data.sec.gov`. Son identificadores públicos y estables.
CIK = {
    "AAPL": "0000320193",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
}

DESCRIPCION_ITEMS = {
    "1A": "Risk Factors — los riesgos que la compañía declara",
    "7": "MD&A — la dirección explicando sus propios resultados",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
}

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
# Lista de candidatos en orden de preferencia. La razón del orden:
#
# - `gpt-5-mini` razona lo suficiente para enrutar entre cuatro herramientas y
#   descomponer una pregunta comparativa en dos consultas, que es lo que de
#   verdad se evalúa, a una fracción del precio del modelo grande.
# - `gpt-4.1-mini` y `gpt-4o-mini` son las alternativas si la cuenta no tiene
#   acceso a la familia 5. Soportan tool calling y salida estructurada.
# - `gpt-5` va el último: es el mejor, y también el más caro y el más lento.
#   Con ~120 invocaciones entre baseline, intermedio y final sobre dos golden
#   sets, la diferencia de coste no es despreciable.
#
# Se prueba en tiempo de ejecución en lugar de fijarse porque no sabemos a qué
# modelos da acceso la cuenta que ejecute esto. Un `KeyError` el día 24 no es
# un riesgo aceptable.
MODELOS_CANDIDATOS = [
    "gpt-5-mini",
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-5",
]

# USD por millón de tokens (entrada, salida). Consultado el 21/09/2026 en
# https://platform.openai.com/docs/pricing
#
# REVISAR LA VÍSPERA DE LA PRESENTACIÓN: OpenAI cambia precios sin avisar, y
# la columna de coste del informe deja de significar nada si el precio que se
# usó no es el que estaba vigente. La fecha de consulta se imprime junto a la
# tabla precisamente por eso.
FECHA_PRECIOS = "2026-09-21"
PRECIOS_OPENAI = {
    "gpt-5":            (1.25, 10.00),
    "gpt-5-mini":       (0.25,  2.00),
    "gpt-5-nano":       (0.05,  0.40),
    "gpt-4.1":          (2.00,  8.00),
    "gpt-4.1-mini":     (0.40,  1.60),
    "gpt-4.1-nano":     (0.10,  0.40),
    "gpt-4o":           (2.50, 10.00),
    "gpt-4o-mini":      (0.15,  0.60),
    "text-embedding-3-small": (0.02, 0.00),
}

# La familia gpt-5 solo admite `temperature=1` (el valor por defecto): fijar 0
# devuelve un 400. Para todo lo demás, `temperature=0` es obligatorio en
# cualquier cosa que se vaya a evaluar. Con temperatura alta, dos ejecuciones
# de la misma pregunta dan métricas distintas y no se sabe si el sistema
# mejoró o si hubo suerte.
FAMILIAS_SIN_TEMPERATURA = ("gpt-5",)

# ---------------------------------------------------------------------------
# Tolerancia del guardrail y de los evaluadores
# ---------------------------------------------------------------------------
# 1 % relativo. La tolerancia existe porque redondear 281.724 millones a
# «281.700 millones» no es inventarse un número; decir 250.000 sí lo es.
#
# El valor está documentado aquí y no repetido en cada notebook para que el
# informe pueda afirmar que middleware y evaluador usan exactamente el mismo
# criterio. Si fuesen dos constantes distintas, una respuesta podría pasar el
# guardrail y suspender el evaluador, y la tabla no significaría nada.
TOLERANCIA_CIFRA = 0.01

# k por defecto del retrieval. 5 es el valor del enunciado y el del baseline de
# clase; se mantiene para que la comparación baseline-contra-final compare el
# sistema y no el tamaño de la ventana de recuperación.
K_POR_DEFECTO = 5

# Límites del agente. `run_limit` cuenta por invocación.
LIMITE_LLAMADAS_HERRAMIENTA = 8
LIMITE_LLAMADAS_MODELO = 10

# Identificador del grupo, usado en el campo `autor` del golden set propio.
AUTOR_GOLDEN = "grupo-nlp"


# ---------------------------------------------------------------------------
# Clave de API
# ---------------------------------------------------------------------------
class ClaveNoEncontrada(RuntimeError):
    """No hay clave de OpenAI. Se lanza con instrucciones, no a secas."""


# Marcador del fichero de ejemplo. Si la clave sigue siendo esto, es que nadie
# la ha puesto, y es mejor decirlo ahora que fallar con un 401 a mitad de una
# evaluación de veinte preguntas.
_PLACEHOLDER = "PEGA-AQUI-TU-CLAVE-DE-OPENAI"


def cargar_clave(obligatoria: bool = True) -> str | None:
    """Deja la clave de OpenAI en el entorno y la devuelve.

    Orden de búsqueda, de más a menos prioritario:

    1. La variable de entorno `OPENAI_API_KEY`, si ya está puesta. Es lo que
       permite ejecutar esto en un CI o en Colab sin tocar ficheros.
    2. El fichero `api_key.txt` de la raíz del proyecto.

    La clave no aparece escrita en ningún notebook ni en ningún módulo, que es
    un requisito explícito del §5 del enunciado. `api_key.txt` está en
    `.gitignore`, así que no puede subirse al repositorio por descuido.

    Args:
        obligatoria: si es `False`, devuelve `None` en vez de lanzar cuando no
            hay clave. Lo usan las celdas de datos, que funcionan sin modelo.
    """
    del_entorno = os.environ.get("OPENAI_API_KEY", "").strip()
    if del_entorno and _PLACEHOLDER not in del_entorno:
        return del_entorno

    if RUTA_CLAVE.is_file():
        del_fichero = RUTA_CLAVE.read_text(encoding="utf-8").strip()
        # Un fichero con varias líneas o con `OPENAI_API_KEY=...` es el error
        # de copia-y-pega más común. Se tolera en vez de fallar.
        del_fichero = del_fichero.splitlines()[0].strip() if del_fichero else ""
        if del_fichero.upper().startswith("OPENAI_API_KEY"):
            del_fichero = del_fichero.split("=", 1)[-1].strip().strip("'\"")
        if del_fichero and _PLACEHOLDER not in del_fichero:
            os.environ["OPENAI_API_KEY"] = del_fichero
            return del_fichero

    if not obligatoria:
        return None
    raise ClaveNoEncontrada(
        f"No hay clave de OpenAI.\n"
        f"  Abre {RUTA_CLAVE} y sustituye su contenido por tu clave, en una\n"
        f"  sola línea y sin comillas. El fichero está en .gitignore, así que\n"
        f"  no se sube al repositorio.\n"
        f"  Alternativa: define la variable de entorno OPENAI_API_KEY."
    )


def hay_clave() -> bool:
    """`True` si hay una clave utilizable. No lanza."""
    return cargar_clave(obligatoria=False) is not None


@functools.lru_cache(maxsize=1)
def hay_modelo() -> bool:
    """`True` si además de clave hay un modelo que de verdad responde.

    Tener clave y poder llamar al modelo no son lo mismo, y confundirlos
    cuesta caro. Una clave válida sobre una cuenta sin crédito devuelve un
    `429 - You have no credits remaining`; una clave de una organización sin
    acceso a la familia 5 devuelve un `404`. En los dos casos `hay_clave()`
    dice `True` y la primera invocación real revienta, normalmente a mitad de
    una evaluación de veinte preguntas y después de haber pagado las primeras.

    Esta función es la que deben mirar los notebooks antes de decidir si
    ejecutan un bloque que llama al modelo. Cuesta una llamada de cuatro
    tokens la primera vez y está cacheada en memoria para el resto del
    proceso.

    Devuelve `False` en vez de lanzar, a propósito: un notebook sin modelo
    tiene que poder recorrerse entero —el corpus, el retrieval, las métricas
    de recuperación y los evaluadores se miden sin gastar un céntimo—, y solo
    los bloques que necesitan el agente deben quedarse en blanco.
    """
    if not hay_clave():
        return False
    try:
        modelo_por_defecto()
        return True
    except Exception:
        return False


def motivo_sin_modelo() -> str:
    """Por qué no hay modelo, en una frase que diga qué hacer."""
    if not hay_clave():
        return (
            f"No hay clave de OpenAI. Pégala en {RUTA_CLAVE}, en una sola "
            f"línea y sin comillas."
        )
    try:
        modelo_por_defecto()
    except Exception as e:
        return f"Hay clave, pero ningún modelo responde:\n{e}"
    return "Hay modelo."


# ---------------------------------------------------------------------------
# Selección de modelo
# ---------------------------------------------------------------------------
RUTA_MODELO_DETECTADO = DIR_CACHE / "modelo_detectado.json"


def _admite_temperatura_cero(modelo: str) -> bool:
    return not modelo.startswith(FAMILIAS_SIN_TEMPERATURA)


def crear_modelo(nombre: str | None = None, **extra):
    """Un `BaseChatModel` de OpenAI listo para usar.

    Envuelve `init_chat_model` para concentrar en un punto las dos cosas que
    hay que acordarse de hacer siempre: cargar la clave antes, y fijar
    `temperature=0` solo en las familias que lo admiten.
    """
    from langchain.chat_models import init_chat_model

    cargar_clave()
    nombre = nombre or modelo_por_defecto()
    parametros = dict(extra)
    if _admite_temperatura_cero(nombre):
        parametros.setdefault("temperature", 0)
    return init_chat_model(f"openai:{nombre}", **parametros)


@functools.lru_cache(maxsize=1)
def modelo_por_defecto(forzar_deteccion: bool = False) -> str:
    """El primer modelo de `MODELOS_CANDIDATOS` al que la cuenta tiene acceso.

    Por qué se detecta en vez de fijarse: el repositorio se ejecuta sobre una
    cuenta de OpenAI cuyo catálogo no controlamos, y el acceso a los modelos
    más nuevos depende del nivel de la cuenta. Fijar `gpt-5-mini` a ciegas
    convierte un problema de permisos en un `NotFoundError` a mitad de la
    evaluación.

    El resultado se guarda en `.cache/modelo_detectado.json` para que las
    ejecuciones siguientes no gasten una llamada en redescubrir lo mismo. La
    caché está en `.gitignore`: depende de la cuenta, no del código.

    Se puede saltar todo esto fijando la variable de entorno `MIAX_MODELO`.
    """
    forzado = os.environ.get("MIAX_MODELO", "").strip()
    if forzado:
        return forzado

    if not forzar_deteccion and RUTA_MODELO_DETECTADO.is_file():
        try:
            guardado = json.loads(RUTA_MODELO_DETECTADO.read_text("utf-8"))
            if guardado.get("modelo") in PRECIOS_OPENAI:
                return guardado["modelo"]
        except Exception:
            pass  # caché corrupta: se vuelve a detectar

    cargar_clave()
    from langchain.chat_models import init_chat_model

    errores: list[str] = []
    for candidato in MODELOS_CANDIDATOS:
        try:
            parametros = {"max_tokens": 4}
            if _admite_temperatura_cero(candidato):
                parametros["temperature"] = 0
            modelo = init_chat_model(f"openai:{candidato}", **parametros)
            modelo.invoke("ok")
            DIR_CACHE.mkdir(exist_ok=True)
            RUTA_MODELO_DETECTADO.write_text(
                json.dumps({"modelo": candidato}, indent=2), encoding="utf-8"
            )
            return candidato
        except Exception as e:  # sin acceso, sin cuota o nombre retirado
            errores.append(f"  {candidato}: {type(e).__name__}: {str(e)[:120]}")

    raise RuntimeError(
        "Ninguno de los modelos candidatos respondió:\n"
        + "\n".join(errores)
        + "\nFija uno a mano con la variable de entorno MIAX_MODELO."
    )


def precio_de(modelo: str) -> tuple[float, float]:
    """`(USD por millón de tokens de entrada, de salida)`.

    Devuelve `(0, 0)` para un modelo sin precio conocido en vez de lanzar: una
    tabla con la columna de coste a cero es un problema visible; una
    excepción a mitad de una evaluación de veinte preguntas tira el trabajo.
    """
    return PRECIOS_OPENAI.get(modelo, (0.0, 0.0))


def resumen_configuracion() -> dict:
    """Lo que hay que registrar junto a cualquier resultado.

    Un número sin la configuración que lo produjo no es reproducible. Esto se
    vuelca en los ficheros de resultados y se imprime en el informe.
    """
    modelo = modelo_por_defecto() if hay_modelo() else "(sin modelo disponible)"
    entrada, salida = precio_de(modelo)
    return {
        "modelo": modelo,
        "temperatura": 0 if _admite_temperatura_cero(modelo) else 1,
        "modelo_embeddings": MODELO_EMBEDDINGS,
        "k": K_POR_DEFECTO,
        "tolerancia_cifra": TOLERANCIA_CIFRA,
        "limite_llamadas_herramienta": LIMITE_LLAMADAS_HERRAMIENTA,
        "limite_llamadas_modelo": LIMITE_LLAMADAS_MODELO,
        "precio_entrada_usd_por_millon": entrada,
        "precio_salida_usd_por_millon": salida,
        "fecha_precios": FECHA_PRECIOS,
    }
