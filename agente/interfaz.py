"""La interfaz pública: `responder()` y `evaluar()`.

Es el único punto del proyecto que el §5 del enunciado convierte en contrato
operativo: el día 24 se clona el repositorio, se pone la clave y se ejecutan
diez preguntas que nadie ha visto, en veinte minutos y sin abrir un notebook.

    from agente import responder, evaluar

    responder("¿Cuál fue el revenue de NVIDIA en FY2025?")
    evaluar("holdout.jsonl")

O desde la terminal, sin escribir Python:

    python -m agente.interfaz responder "¿Cuál fue el revenue de NVDA en FY2025?"
    python -m agente.interfaz evaluar holdout.jsonl

## Tres decisiones de diseño de este módulo

**`evaluar()` acepta una ruta o una lista.** El enunciado la especifica con una
ruta a JSONL, y así se comporta. Pero los notebooks necesitan evaluar
subconjuntos —las cinco preguntas duras, las de ausencia— sin escribir ficheros
temporales, así que también acepta una lista de dicts. Es una ampliación
compatible: cualquier llamada que pase una ruta sigue funcionando igual.

**Nada se cachea entre preguntas.** Cada pregunta estrena `thread_id`. Si
compartieran hilo, la pregunta quince vería lo que se consultó en las catorce
anteriores y podría acertar por contagio sin llamar a ninguna herramienta. La
evaluación mediría la conversación en lugar de la pregunta.

**Un fallo en una pregunta no tira la evaluación.** Con veinte minutos de
margen y sin nadie delante, una excepción en la pregunta tres no puede costar
las diecisiete restantes. Se captura, se registra en la fila como error y se
sigue.
"""

from __future__ import annotations

import functools
import json
import sys
import time
from pathlib import Path

import pandas as pd

from agente import config
from agente.agente import (
    coste_de,
    construir_agente,
    herramientas_usadas,
    hubo_correccion,
    tokens_de,
)
from agente.evaluadores import EVALUADORES, acierta_familia, cita_fundamentada


# ---------------------------------------------------------------------------
# responder
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def _agente(perfil: str, modelo: str | None):
    """El agente compilado, reutilizado entre preguntas.

    Se cachea porque compilar el grafo y crear el cliente del modelo no es
    gratis, y hacerlo veinte veces seguidas añadiría segundos a la evaluación
    sin cambiar ningún resultado. El `InMemorySaver` que lleva dentro separa
    las conversaciones por `thread_id`, así que reutilizar el agente no mezcla
    el contexto de dos preguntas.
    """
    return construir_agente(perfil=perfil, modelo=modelo)


def responder(
    pregunta: str,
    thread_id: str | None = None,
    perfil: str = "final",
    modelo: str | None = None,
) -> dict:
    """Responde una pregunta sobre los 10-K del corpus.

    Args:
        pregunta: la pregunta, en español o en inglés.
        thread_id: hilo de conversación. Dos llamadas con el mismo `thread_id`
            comparten memoria, lo que permite preguntas de seguimiento
            («¿y el año anterior?»). Si es `None`, cada llamada estrena hilo.
        perfil: `"baseline"`, `"filtros"` o `"final"`.
        modelo: nombre del modelo de OpenAI; `None` para autodetectar.

    Devuelve el estado completo de la invocación —incluida la trayectoria, que
    los evaluadores necesitan— más cuatro campos medidos:
    `latencia_s`, `coste_usd`, `tokens_entrada`/`tokens_salida` y `modelo`.

    Se devuelve el estado entero y no solo la respuesta a propósito. Un
    `responder()` que devolviera únicamente el texto obligaría a ejecutar el
    agente dos veces para poder evaluar la trayectoria, y a pagar dos veces.
    """
    agente = _agente(perfil, modelo)
    hilo = thread_id or f"pregunta-{time.time_ns()}"

    comienzo = time.perf_counter()
    estado = agente.invoke(
        {"messages": [{"role": "user", "content": pregunta}]},
        config={"configurable": {"thread_id": hilo}},
    )
    latencia = time.perf_counter() - comienzo

    nombre_modelo = modelo or config.modelo_por_defecto()
    entrada, salida = tokens_de(estado)
    return {
        **estado,
        "pregunta": pregunta,
        "thread_id": hilo,
        "perfil": perfil,
        "modelo": nombre_modelo,
        "latencia_s": latencia,
        "coste_usd": coste_de(estado, nombre_modelo),
        "tokens_entrada": entrada,
        "tokens_salida": salida,
    }


# ---------------------------------------------------------------------------
# evaluar
# ---------------------------------------------------------------------------
def cargar_golden(origen) -> list[dict]:
    """Un golden set, desde una ruta a JSONL o desde una lista ya cargada."""
    if isinstance(origen, (list, tuple)):
        return list(origen)
    ruta = Path(origen)
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No encuentro el golden set en {ruta.resolve()}. Pasa la ruta a "
            f"un fichero JSONL con una pregunta por línea."
        )
    with ruta.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def evaluar(
    ruta_jsonl,
    perfil: str = "final",
    modelo: str | None = None,
    guardar_en: str | Path | None = None,
    verboso: bool = True,
) -> pd.DataFrame:
    """Ejecuta el agente sobre un golden set y devuelve la tabla de resultados.

    Una fila por pregunta, con:

    - los tres evaluadores obligatorios: `cita`, `cifra`, `trayectoria`;
    - el acierto por familia, `acierto`;
    - coste, latencia, tokens y número de llamadas a herramienta;
    - si intervino el guardrail numérico;
    - lo que respondió el agente, para poder mirarlo cuando algo falle.

    `guardar_en` escribe dos ficheros con el mismo nombre base: un `.csv` con
    la tabla y un `.jsonl` con la trayectoria completa de cada pregunta. La
    trayectoria se guarda porque una tabla de aciertos dice qué falló y no dice
    por qué, y sin el porqué no se puede arreglar nada.
    """
    items = cargar_golden(ruta_jsonl)
    filas: list[dict] = []
    crudos: list[dict] = []

    if verboso:
        print(
            f"Evaluando {len(items)} preguntas · perfil '{perfil}' · "
            f"modelo {modelo or config.modelo_por_defecto()}"
        )

    for numero, item in enumerate(items, 1):
        identificador = item.get("id", f"q{numero:03d}")
        try:
            resultado = responder(
                item["pregunta"],
                thread_id=f"eval-{perfil}-{identificador}",
                perfil=perfil,
                modelo=modelo,
            )
            error = None
        except Exception as e:  # una pregunta rota no puede tirar las otras 19
            resultado = {"messages": [], "latencia_s": 0.0, "coste_usd": 0.0}
            error = f"{type(e).__name__}: {e}"

        respuesta = resultado.get("structured_response")
        fila = _puntuar(item, resultado, error)
        filas.append(fila)
        crudos.append(
            {
                "id": identificador,
                "pregunta": item.get("pregunta"),
                "perfil": perfil,
                "error": error,
                "respuesta_estructurada": (
                    respuesta.model_dump() if respuesta is not None else None
                ),
                "trayectoria": _trayectoria_serializable(resultado),
                **{
                    k: fila[k]
                    for k in (
                        "coste_usd",
                        "latencia_s",
                        "llamadas",
                        "guardrail",
                        "tokens_entrada",
                        "tokens_salida",
                        "herramientas",
                    )
                },
            }
        )

        if verboso:
            marca = "ERROR" if error else _marca(fila)
            print(
                f"  [{numero:2d}/{len(items)}] {identificador:10s} {marca}  "
                f"{fila['latencia_s']:5.1f}s  ${fila['coste_usd']:.4f}  "
                f"{fila['llamadas']} llamadas"
            )

    tabla = pd.DataFrame(filas)

    if guardar_en:
        destino = Path(guardar_en)
        destino.parent.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(destino.with_suffix(".csv"), index=False, encoding="utf-8")
        with destino.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
            for crudo in crudos:
                f.write(json.dumps(crudo, ensure_ascii=False, default=str) + "\n")
        if verboso:
            print(f"\nGuardado en {destino.with_suffix('.csv')} y .jsonl")

    if verboso and not tabla.empty:
        print()
        print(formatear_resumen(resumir(tabla, perfil)))

    return tabla


def _puntuar(item: dict, resultado: dict, error: str | None) -> dict:
    """Una fila de la tabla: los evaluadores más el coste, para una pregunta.

    Está separada de `evaluar()` a propósito, y es lo que permite que
    `recalcular()` exista. Invocar al agente cuesta dinero y minutos; aplicarle
    los evaluadores a una respuesta ya guardada no cuesta nada. Mientras las
    dos cosas estuvieran en la misma función, cambiar la definición de un
    evaluador obligaba a volver a pagar todas las invocaciones para actualizar
    la tabla, y esa fricción empuja a no corregir un evaluador que se ha
    quedado corto.
    """
    respuesta = resultado.get("structured_response")
    usadas = herramientas_usadas(resultado)
    return {
        "id": item.get("id"),
        "pregunta": item.get("pregunta"),
        "familia": item.get("familia"),
        "ticker": item.get("ticker"),
        "fiscal_year": item.get("fiscal_year"),
        "respuesta_esperada": item.get("respuesta_esperada"),
        "acierto": None if error else acierta_familia(item, resultado),
        **{
            nombre: (None if error else evaluador(item, resultado))
            for nombre, evaluador in EVALUADORES.items()
        },
        # No es uno de los tres del §4: va aparte porque es la condición que
        # distingue «citó algo real» de «citó el sitio donde está la
        # respuesta», y sin ella el acierto extractivo no discrimina entre
        # configuraciones de retrieval.
        "fundamentada": None if error else cita_fundamentada(item, resultado),
        "llamadas": len(usadas),
        "herramientas": "|".join(usadas),
        "guardrail": None if error else hubo_correccion(resultado),
        "coste_usd": resultado.get("coste_usd", 0.0),
        "latencia_s": resultado.get("latencia_s", 0.0),
        "tokens_entrada": resultado.get("tokens_entrada", 0),
        "tokens_salida": resultado.get("tokens_salida", 0),
        "fuente": getattr(respuesta, "fuente", None),
        # El número que afirma el agente. Se llama distinto de `cifra` a
        # propósito: `cifra` es el evaluador (True/False/None). Si se usara
        # el mismo nombre, el número pisaría el evaluador y la columna del
        # informe dejaría de medir si la cifra es correcta.
        "cifra_afirmada": getattr(respuesta, "cifra", None),
        "cifra_esperada": item.get("cifra_esperada"),
        "chunk_id": getattr(respuesta, "chunk_id", None),
        "respuesta": getattr(respuesta, "respuesta", None),
        "error": error,
    }


class _MensajeGuardado:
    """Lo mínimo de un mensaje de LangChain para releer una trayectoria.

    `herramientas_usadas()` solo mira `tool_calls`, así que no hace falta
    reconstruir un `AIMessage` de verdad. Reconstruirlo sería además peor: la
    trayectoria guardada tiene el contenido recortado a 4.000 caracteres, y un
    objeto que se parece a un mensaje real pero no lo es invita a usarlo para
    cosas para las que no sirve.
    """

    __slots__ = ("tool_calls", "content", "type")

    def __init__(self, entrada: dict):
        self.tool_calls = [
            {"name": c["nombre"], "args": c.get("args", {}), "id": ""}
            for c in entrada.get("tool_calls", [])
        ]
        self.content = entrada.get("contenido", "")
        self.type = {"AIMessage": "ai", "ToolMessage": "tool",
                     "HumanMessage": "human"}.get(entrada.get("tipo"), "ai")


def recalcular(ruta_jsonl, golden, perfil: str | None = None):
    """Vuelve a puntuar una evaluación ya ejecutada, leyéndola de disco.

    Sirve para cuando cambia la definición de un evaluador: las respuestas del
    agente son las mismas, lo que cambia es cómo se puntúan, y volver a
    invocarlo costaría dinero y no aportaría nada.

    `golden` es la ruta al golden set o la lista de ítems correspondiente. Los
    ítems se emparejan con las respuestas por `id`.

    Lo que **no** puede recalcular: `guardrail`, porque si el verificador
    intervino o no es un hecho de la ejecución y no del evaluador. Se conserva
    el valor que se registró entonces.
    """
    from agente.esquema import RespuestaFinanciera

    items = {i["id"]: i for i in cargar_golden(golden)}
    crudos = [
        json.loads(linea)
        for linea in Path(ruta_jsonl).open(encoding="utf-8")
        if linea.strip()
    ]

    filas = []
    for crudo in crudos:
        item = items.get(crudo["id"])
        if item is None:
            continue
        datos = crudo.get("respuesta_estructurada")
        resultado = {
            "structured_response": RespuestaFinanciera(**datos) if datos else None,
            "messages": [_MensajeGuardado(m) for m in crudo.get("trayectoria", [])],
            "coste_usd": crudo.get("coste_usd", 0.0),
            "latencia_s": crudo.get("latencia_s", 0.0),
        }
        fila = _puntuar(item, resultado, crudo.get("error"))
        for clave in ("llamadas", "guardrail", "tokens_entrada", "tokens_salida",
                      "coste_usd", "latencia_s", "herramientas"):
            if crudo.get(clave) is not None:
                fila[clave] = crudo[clave]
        filas.append(fila)

    return pd.DataFrame(filas)


def _marca(fila: dict) -> str:
    def simbolo(v):
        return "·" if v is None else ("OK" if v else "no")

    return (
        f"acierto={simbolo(fila['acierto']):2s} cita={simbolo(fila['cita']):2s} "
        f"cifra={simbolo(fila['cifra']):2s} tray={simbolo(fila['trayectoria']):2s}"
    )


def _trayectoria_serializable(resultado) -> list[dict]:
    """La trayectoria en JSON plano, para poder releerla sin LangChain."""
    salida = []
    for mensaje in resultado.get("messages", []) if isinstance(resultado, dict) else []:
        entrada = {
            "tipo": type(mensaje).__name__,
            "contenido": str(getattr(mensaje, "content", ""))[:4000],
        }
        llamadas = getattr(mensaje, "tool_calls", None)
        if llamadas:
            entrada["tool_calls"] = [
                {"nombre": c["name"], "args": c["args"]} for c in llamadas
            ]
        salida.append(entrada)
    return salida


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
def _media(serie) -> float | None:
    """Media ignorando los `None`. `None` si no hay ningún caso aplicable.

    Es la razón de ser del tercer estado de los evaluadores: promediar
    tratando los `None` como ceros diría más sobre la composición del golden
    set que sobre el sistema.
    """
    validos = [v for v in serie if v is not None]
    return sum(bool(v) for v in validos) / len(validos) if validos else None


def resumir(tabla: pd.DataFrame, nombre: str = "sistema") -> dict:
    """Una fila de la tabla comparativa del informe."""
    if tabla.empty:
        return {"sistema": nombre}
    resumen = {
        "sistema": nombre,
        "n": len(tabla),
        "acierto": _media(tabla["acierto"]),
        "cita": _media(tabla["cita"]),
        "cifra": _media(tabla["cifra"]),
        "trayectoria": _media(tabla["trayectoria"]),
        "fundamentada": _media(tabla["fundamentada"]) if "fundamentada" in tabla else None,
    }
    for familia in ("extractiva", "numerica", "comparativa"):
        subconjunto = tabla[tabla.familia == familia]
        resumen[f"acierto_{familia}"] = (
            _media(subconjunto["acierto"]) if len(subconjunto) else None
        )
    resumen.update(
        {
            "coste_medio_usd": float(tabla["coste_usd"].mean()),
            "coste_total_usd": float(tabla["coste_usd"].sum()),
            "latencia_media_s": float(tabla["latencia_s"].mean()),
            "llamadas_medias": float(tabla["llamadas"].mean()),
            "guardrail_saltó": int(tabla["guardrail"].fillna(False).sum()),
            "errores": int(tabla["error"].notna().sum()),
        }
    )
    return resumen


def formatear_resumen(resumen: dict) -> str:
    def porcentaje(v):
        return "  n/a" if v is None else f"{100 * v:5.1f}%"

    return (
        f"  acierto {porcentaje(resumen.get('acierto'))} · "
        f"cita {porcentaje(resumen.get('cita'))} · "
        f"cifra {porcentaje(resumen.get('cifra'))} · "
        f"trayectoria {porcentaje(resumen.get('trayectoria'))}\n"
        f"  coste medio ${resumen.get('coste_medio_usd', 0):.4f}/pregunta · "
        f"total ${resumen.get('coste_total_usd', 0):.3f} · "
        f"latencia media {resumen.get('latencia_media_s', 0):.1f} s · "
        f"{resumen.get('llamadas_medias', 0):.1f} llamadas/pregunta"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli(argumentos: list[str]) -> int:
    uso = (
        "Uso:\n"
        '  python -m agente.interfaz responder "¿pregunta?" [--perfil final]\n'
        "  python -m agente.interfaz evaluar holdout.jsonl [--perfil final] "
        "[--guardar resultados/holdout]\n"
    )
    if not argumentos or argumentos[0] in ("-h", "--help"):
        print(uso)
        return 0

    orden, *resto = argumentos
    opciones: dict[str, str] = {}
    posicionales: list[str] = []
    i = 0
    while i < len(resto):
        if resto[i].startswith("--"):
            opciones[resto[i][2:]] = resto[i + 1] if i + 1 < len(resto) else ""
            i += 2
        else:
            posicionales.append(resto[i])
            i += 1

    perfil = opciones.get("perfil", "final")

    if orden == "responder":
        if not posicionales:
            print(uso)
            return 1
        resultado = responder(posicionales[0], perfil=perfil)
        respuesta = resultado.get("structured_response")
        if respuesta is None:
            print("El agente no devolvió respuesta estructurada.")
            return 1
        print(json.dumps(respuesta.model_dump(), ensure_ascii=False, indent=2))
        print(
            f"\n[{resultado['latencia_s']:.1f} s · "
            f"${resultado['coste_usd']:.4f} · "
            f"{len(herramientas_usadas(resultado))} llamadas · "
            f"modelo {resultado['modelo']}]"
        )
        return 0

    if orden == "evaluar":
        if not posicionales:
            print(uso)
            return 1
        destino = opciones.get(
            "guardar", str(config.DIR_RESULTADOS / f"evaluacion_{perfil}")
        )
        evaluar(posicionales[0], perfil=perfil, guardar_en=destino)
        return 0

    print(uso)
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
