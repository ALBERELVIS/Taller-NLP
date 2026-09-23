"""Regenera `demo_traza.json` a partir de una ejecución real del agente.

`miax_s1.demo_apertura()` reproduce una ejecución grabada en `demo_traza.json`
para que la primera celda de la sesión 1 pueda enseñar el agente terminado
antes de que haya nada instalado. El fichero que se repartió en clase es
**provisional**: sus salidas de herramienta son reales, pero la prosa final es
de ejemplo, le falta la clave `metadatos.fecha` que la propia función lee, y
por eso `demo_apertura()` termina en un `KeyError`.

El propio `miax_s1.py` dice qué hacer con eso:

    [AVISO: traza PROVISIONAL. Las salidas de herramienta son reales, la prosa
     final es de ejemplo. Regenérala con `python modulos/generar_traza_demo.py`
     en cuanto haya clave.]

Ese script no venía con el material. Esto es ese script.

Uso:

    python modulos/generar_traza_demo.py                 # perfil final
    python modulos/generar_traza_demo.py --perfil baseline
    python modulos/generar_traza_demo.py --pregunta "..."

Requisitos: `api_key.txt` con la clave, y el corpus montado (notebook 00).

Qué se conserva y qué se cambia. El formato del JSON es el que espera
`miax_s1._imprimir_paso`, sin tocar una clave: `pregunta`, `trayectoria` con
pasos de tipo `herramienta` o `razonamiento`, `respuesta` con los campos del
esquema, y `metadatos`. Lo único que cambia respecto al fichero repartido es
que el contenido sale de una invocación de verdad y que `metadatos` lleva la
`fecha` que la función necesita para imprimirse.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from agente import config
from agente.agente import cronometrar, herramientas_usadas
from agente.interfaz import responder

RUTA = AQUI / "demo_traza.json"

# La pregunta de la demo de apertura. Se mantiene la del fichero repartido si
# está, para que la clase vea exactamente lo mismo que enseña el profesor.
PREGUNTA_POR_DEFECTO = (
    "¿Qué riesgos relacionados con la inteligencia artificial añadió Microsoft "
    "en su 10-K de FY2025, y cuánto crecieron sus ingresos ese ejercicio?"
)

# Cuántos caracteres del resultado de cada herramienta se guardan. La demo
# imprime solo las dos primeras líneas, pero guardar el resultado entero
# convertiría un fichero de 8 KB en uno de 200 KB por una sección larga.
MAXIMO_RESULTADO = 600


def _pregunta_original() -> str:
    """La pregunta del fichero repartido, si existe."""
    if RUTA.is_file():
        try:
            return json.loads(RUTA.read_text(encoding="utf-8"))["pregunta"]
        except Exception:
            pass
    return PREGUNTA_POR_DEFECTO


def trayectoria_de(estado: dict) -> list[dict]:
    """Los mensajes de una invocación, en el formato que espera la demo.

    Se recorren los mensajes en orden y se emparejan las peticiones de
    herramienta con su resultado por `tool_call_id`. Emparejar por posición
    sería más corto y estaría mal: cuando el modelo pide dos herramientas en
    la misma vuelta —lo habitual en una comparativa— los resultados pueden
    llegar en otro orden.
    """
    resultados = {
        getattr(m, "tool_call_id", None): str(getattr(m, "content", ""))
        for m in estado["messages"]
        if getattr(m, "type", None) == "tool"
    }

    pasos: list[dict] = []
    for mensaje in estado["messages"]:
        for llamada in getattr(mensaje, "tool_calls", None) or []:
            pasos.append({
                "tipo": "herramienta",
                "herramienta": llamada["name"],
                "argumentos": llamada["args"],
                "resultado": resultados.get(llamada["id"], "")[:MAXIMO_RESULTADO],
            })
        texto = (getattr(mensaje, "text", "") or "").strip()
        if getattr(mensaje, "type", None) == "ai" and texto:
            pasos.append({"tipo": "razonamiento", "texto": texto[:300]})
    return pasos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pregunta", default=None)
    parser.add_argument("--perfil", default="final",
                        choices=["baseline", "filtros", "final"])
    argumentos = parser.parse_args()

    if not config.hay_clave():
        print(f"No hay clave de OpenAI. Pégala en {config.RUTA_CLAVE} y "
              f"vuelve a intentarlo.")
        return 1

    pregunta = argumentos.pregunta or _pregunta_original()
    print(f"Pregunta: {pregunta}")
    print(f"Perfil:   {argumentos.perfil}")
    print("Ejecutando…")

    estado, segundos = cronometrar(
        responder, pregunta, thread_id="demo-apertura",
        perfil=argumentos.perfil,
    )
    respuesta = estado["structured_response"]

    traza = {
        "pregunta": pregunta,
        "trayectoria": trayectoria_de(estado),
        "respuesta": respuesta.model_dump(),
        "metadatos": {
            "origen": "ejecucion_real",
            "fecha": date.today().isoformat(),
            "modelo": estado["modelo"],
            "perfil": argumentos.perfil,
            "llamadas_herramienta": len(herramientas_usadas(estado)),
            "latencia_s": round(segundos, 2),
            "coste_usd": round(estado["coste_usd"], 6),
        },
    }

    RUTA.write_text(json.dumps(traza, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n{RUTA.name} regenerado: "
          f"{len(traza['trayectoria'])} pasos, "
          f"{traza['metadatos']['llamadas_herramienta']} llamadas, "
          f"{segundos:.1f} s, {estado['coste_usd'] * 100:.2f} ¢")
    print(f"\nRESPUESTA\n{respuesta.respuesta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
