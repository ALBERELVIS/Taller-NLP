"""Construcción de golden sets: anclas, cifras y volcado a JSONL.

Estas funciones estaban escritas dentro de `03_Golden_set_propio.ipynb`. Se han
movido aquí cuando hizo falta construir un segundo conjunto —el hold-out
simulado del notebook 07— porque la alternativa era copiarlas, y dos copias de
un validador o de un extractor de anclas se separan en cuanto alguien toca una.
El notebook 03 sigue contando el razonamiento paso a paso; lo que hace ahora es
importar de aquí en lugar de definirlo.

## La idea que gobierna todo el módulo

**Nada que pueda leerse del corpus se escribe a mano.** Ni una cifra, ni un
desplazamiento, ni un `chunk_id`. Lo único que se escribe es la pregunta, el
emisor, el ejercicio y un *marcador*: las primeras palabras de la frase del
informe que contiene la respuesta.

El motivo es que una errata en un ancla —un espacio de más, unas comillas
tipográficas cambiadas— **no da ningún error**. Hace que la pregunta sea
imposible de acertar y hunde el `recall` medido por un motivo que no tiene nada
que ver con el retriever. Un golden set con tres erratas mide el golden set, no
el sistema.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

from agente import config, corpus, esquema

MAXIMO_PALABRAS = esquema.MAXIMO_PALABRAS_ANCLA


@functools.lru_cache(maxsize=1)
def _secciones() -> dict:
    return {(s["ticker"], s["fiscal_year"], s["item"]): s for s in corpus.cargar_secciones()}


@functools.lru_cache(maxsize=1)
def _chunks() -> list[dict]:
    return corpus.cargar_chunks()


def resolver_ancla(
    ticker: str,
    fiscal_year: int,
    item: str,
    marcador: str,
    fin_marcador: str | None = None,
) -> dict:
    """Localiza la frase que contiene el marcador y devuelve sus desplazamientos.

    El ancla va desde el comienzo del marcador hasta el final de la frase en
    que aparece. Empezar en el marcador y no al principio de la frase es
    deliberado: los marcadores se eligen en el arranque de la frase, y así el
    ancla queda estable aunque el párrafo anterior cambie.

    `fin_marcador` corta antes del final de la frase. Hace falta porque los
    10-K están llenos de frases de sesenta palabras con seis subordinadas
    enumerativas, y el validador oficial limita el ancla a cuarenta. El corte
    sigue siendo texto literal del informe: se recorta la frase, no se
    reescribe.

    Lanza si el marcador no es único en la sección. Un ancla ambigua contaría
    como acierto un fragmento de otro sitio del documento, que es precisamente
    lo que la métrica tiene que distinguir.
    """
    seccion = _secciones()[(ticker, fiscal_year, item)]
    texto = seccion["texto"]

    apariciones = texto.count(marcador)
    if apariciones != 1:
        raise ValueError(
            f"{ticker} FY{fiscal_year} item {item}: el marcador {marcador[:50]!r} "
            f"aparece {apariciones} veces. Un ancla ambigua contaría como "
            f"acierto un fragmento de otro sitio del documento."
        )

    inicio = texto.index(marcador)
    # Final del ancla: lo que llegue antes de estas dos cosas.
    #   - el final de la frase: un punto seguido de espacio, salto o fin;
    #   - el final del párrafo: una línea en blanco.
    # Hace falta el segundo corte porque varias secciones del Item 1A están
    # escritas como listas de viñetas separadas por punto y coma, sin puntos.
    # Sin él, el ancla se comería la lista entera.
    if fin_marcador:
        posicion = texto.find(fin_marcador, inicio)
        if posicion < 0:
            raise ValueError(
                f"{ticker} FY{fiscal_year} item {item}: el marcador de fin "
                f"{fin_marcador!r} no aparece detrás del de inicio."
            )
        fin = posicion + len(fin_marcador)
    else:
        fin = inicio + len(marcador)
        while fin < len(texto):
            if texto[fin] == "." and (fin + 1 >= len(texto) or texto[fin + 1] in " \n\t"):
                fin += 1
                break
            if texto.startswith("\n\n", fin):
                break
            fin += 1

    ancla = texto[inicio:fin].rstrip()
    fin = inicio + len(ancla)

    if len(ancla.split()) > MAXIMO_PALABRAS:
        raise ValueError(
            f"{ticker} FY{fiscal_year} item {item}: el ancla sale de "
            f"{len(ancla.split())} palabras, y el máximo son {MAXIMO_PALABRAS}. "
            f"Acorta el marcador, usa fin_marcador o elige otra frase."
        )
    if texto[inicio:fin] != ancla:
        raise AssertionError("El recorte por desplazamientos no devuelve el ancla.")

    # El fragmento del índice que contiene el ancla ENTERA. Si el ancla queda
    # partida entre dos fragmentos no hay ninguno, y eso no es un error del
    # golden set: es un límite del troceado, y conviene saberlo.
    candidatos = [
        c for c in _chunks()
        if c["ticker"] == ticker
        and c["fiscal_year"] == fiscal_year
        and c["item"] == item
        and c["inicio_car"] <= inicio
        and c["fin_car"] >= fin
    ]
    return {
        "ancla_texto": ancla,
        "ancla_inicio": inicio,
        "ancla_fin": fin,
        "chunk_id_esperado": candidatos[0]["chunk_id"] if candidatos else None,
    }


def cifra_de(ticker: str, fiscal_year: int, concepto: str) -> tuple[float, str]:
    """La cifra y la unidad, leídas del corpus. Nunca escritas a mano."""
    hecho = corpus.hecho_xbrl(ticker, fiscal_year, concepto)
    if hecho is None:
        raise ValueError(
            f"{ticker} no reporta '{concepto}' en FY{fiscal_year}. Una pregunta "
            f"numérica sobre un concepto que no existe le exige al agente algo "
            f"imposible: la tabla mediría un error nuestro, no del sistema."
        )
    return float(hecho["value"]), str(hecho["unit"])


def en_millones(valor: float) -> str:
    return f"{valor / 1e6:,.0f} millones de dólares".replace(",", ".")


def construir(spec: dict) -> dict:
    """Una entrada completa del golden set a partir de su especificación.

    La especificación solo lleva lo que un humano tiene que decidir: la
    pregunta, el emisor, el ejercicio, la familia, el concepto XBRL si lo hay y
    el marcador del ancla si lo hay. Todo lo demás —cifras, unidades,
    desplazamientos, `chunk_id` y herramienta esperada— lo rellena este código
    leyendo el corpus.
    """
    familia = spec["familia"]
    item = dict(
        id=spec["id"],
        pregunta=spec["pregunta"],
        familia=familia,
        ticker=spec["ticker"],
        fiscal_year=spec["fiscal_year"],
        respuesta_esperada=spec.get("respuesta_esperada"),
        cifra_esperada=None,
        unidad=None,
        concept_xbrl=spec.get("concepto"),
        item_esperado=spec.get("item"),
        ancla_texto=None,
        ancla_inicio=None,
        ancla_fin=None,
        chunk_id_esperado=None,
        herramienta_esperada=[],
        cifra_anterior_esperada=None,
        fiscal_year_anterior=spec.get("fiscal_year_anterior"),
        autor=spec.get("autor", config.AUTOR_GOLDEN),
    )

    if spec.get("concepto"):
        valor, unidad = cifra_de(spec["ticker"], spec["fiscal_year"], spec["concepto"])
        item["cifra_esperada"] = valor
        item["unidad"] = unidad
    if spec.get("fiscal_year_anterior"):
        anterior, _ = cifra_de(
            spec["ticker"], spec["fiscal_year_anterior"], spec["concepto"]
        )
        item["cifra_anterior_esperada"] = anterior
    if spec.get("marcador"):
        item.update(
            resolver_ancla(
                spec["ticker"],
                spec["fiscal_year"],
                spec["item"],
                spec["marcador"],
                spec.get("fin_marcador"),
            )
        )

    # La herramienta esperada se deduce de la familia, no se escribe. Es la
    # política de enrutado del sistema puesta por escrito una sola vez:
    #   - una cifra sale de get_xbrl_fact, siempre;
    #   - una explicación sale de search_filings;
    #   - una comparativa necesita las dos, y get_xbrl_fact dos veces.
    if familia == "numerica":
        item["herramienta_esperada"] = ["get_xbrl_fact"]
    elif familia == "extractiva":
        item["herramienta_esperada"] = ["search_filings"]
    else:
        item["herramienta_esperada"] = ["get_xbrl_fact", "search_filings"]

    # Las comparativas y las numéricas llevan la respuesta esperada rematada
    # con las cifras exactas, para que quien revise el fichero pueda
    # comprobarlo sin abrir el parquet.
    if familia == "comparativa":
        item["respuesta_esperada"] += (
            f" [FY{spec['fiscal_year_anterior']}: "
            f"{item['cifra_anterior_esperada']:,.0f} → "
            f"FY{spec['fiscal_year']}: {item['cifra_esperada']:,.0f} "
            f"{item['unidad']}]"
        )
    elif familia == "numerica":
        formato = ",.2f" if item["unidad"] == "USD/shares" else ",.0f"
        item["respuesta_esperada"] = (
            f"{item['cifra_esperada']:{formato}} {item['unidad']} "
            f"({spec['ticker']} FY{spec['fiscal_year']}, {spec['concepto']})"
        )
    return item


def escribir(ruta: Path, filas: list[dict], campos: list[str] | None = None) -> None:
    """Vuelca a JSONL con las claves en orden canónico.

    El orden no es cosmética: un fichero donde cada línea lleva las claves en un
    orden distinto es ilegible en un diff, y un golden set es un fichero que se
    revisa a mano y que se versiona.
    """
    campos = campos or esquema.CAMPOS_GOLDEN
    ruta = Path(ruta)
    with ruta.open("w", encoding="utf-8") as f:
        for fila in filas:
            ordenada = {c: fila.get(c) for c in campos if c in fila}
            ordenada.update({k: v for k, v in fila.items() if k not in campos})
            f.write(json.dumps(ordenada, ensure_ascii=False) + "\n")
    print(f"{ruta.name}: {len(filas)} preguntas, {ruta.stat().st_size / 1024:.1f} KB")
