"""Agente investigador sobre informes 10-K de la SEC.

Práctica de *LLMs aplicados a Finanzas* — MIAX.

El paquete está separado de los notebooks a propósito. Los notebooks cuentan
la historia y justifican las decisiones; el paquete es lo que se importa desde
un clon limpio para ejecutar las diez preguntas ciegas del día 24 sin abrir
ningún notebook, que es lo que pide el §5 del enunciado:

    from agente import responder, evaluar

    responder("¿Cuál fue el revenue de NVIDIA en FY2025?")
    evaluar("holdout.jsonl")

Mapa del paquete, en el orden en que dependen unos de otros:

    config.py        rutas, clave de API, modelo, precios, tolerancias
    corpus.py        carga de secciones, chunks, XBRL e índice FAISS
    esquema.py       RespuestaFinanciera, el contrato §7 del enunciado
    retrieval.py     denso, filtros, BM25+RRF, reescritura, reranking
    herramientas.py  las cuatro herramientas con firma contractual
    middleware.py    guardrail de cifras contra XBRL y límites de llamadas
    agente.py        construcción del agente en sus tres perfiles
    evaluadores.py   cita, cifra y trayectoria
    interfaz.py      responder() y evaluar()
    informe.py       tablas y PDF
"""

from __future__ import annotations

__version__ = "1.0.0"

# Importación perezosa: `import agente` tiene que ser instantáneo y no puede
# exigir que el corpus esté montado ni que haya clave de API. Cargar FAISS y el
# codificador de embeddings son varios segundos, y hay notebooks (el 00, el de
# preparación del corpus) que importan este paquete *antes* de que el corpus
# exista.
__all__ = ["responder", "evaluar", "config"]


def __getattr__(nombre: str):
    # `importlib.import_module` y no `from agente import ...`: esta segunda
    # forma vuelve a pasar por este mismo `__getattr__` mientras el submódulo
    # no esté registrado como atributo del paquete, y recursiona sin fondo.
    import importlib

    if nombre in ("responder", "evaluar"):
        return getattr(importlib.import_module("agente.interfaz"), nombre)
    if nombre == "config":
        return importlib.import_module("agente.config")
    raise AttributeError(f"module 'agente' has no attribute '{nombre}'")
