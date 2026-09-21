"""El contrato de salida del agente y el del golden set.

## Por qué el agente no devuelve prosa

Un agente que contesta en prosa obliga a que alguien lea la respuesta para
saber si es correcta. Con 20 preguntas propias, 20 oficiales, tres
configuraciones del sistema y 10 preguntas ciegas que hay que ejecutar en 20
minutos, eso no escala.

La salida estructurada hace tres cosas a la vez, y las tres son requisitos del
enunciado y no comodidades:

1. **Convierte la cita en un requisito estructural.** «Por favor, cita la
   fuente» es una súplica en el prompt que el modelo cumple a veces. Un campo
   obligatorio del esquema no se puede omitir sin que la validación falle.
2. **Permite evaluar leyendo campos en vez de parseando prosa.** Los tres
   evaluadores del §4 leen `cifra`, `chunk_id` y `cita`; ninguno tiene que
   adivinar dónde está el número dentro de un párrafo.
3. **Da al guardrail algo concreto que verificar.** El middleware que contrasta
   contra XBRL necesita saber qué cifra se afirma, de qué compañía y de qué
   ejercicio. Sin esos tres campos no puede consultar el hecho correcto.

Los campos del §7 del enunciado están literales: no se quita ni se renombra
ninguno. Los añadidos van al final, tienen valor por defecto y ninguna parte
de la evaluación depende de ellos, así que el esquema sigue siendo compatible
con cualquier evaluador que espere el contrato original.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from agente import config

AUTOR_POR_DEFECTO = config.AUTOR_GOLDEN


class RespuestaFinanciera(BaseModel):
    """Respuesta trazable a una pregunta sobre informes 10-K."""

    # --- Contrato §7 del enunciado. Literal, no se toca. -------------------
    respuesta: str = Field(description="Respuesta en prosa, breve y directa")
    cifra: float | None = Field(
        default=None,
        description=(
            "Valor numérico en unidades ABSOLUTAS, si la pregunta pide uno. "
            "Ejemplo: 60922000000.0, no 60922 ni '60.922 millones'. Si la "
            "pregunta pide una variación entre dos ejercicios, pon aquí el "
            "valor del ejercicio más reciente."
        ),
    )
    unidad: str | None = Field(
        default=None,
        description="USD, shares, USD/shares, porcentaje…",
    )
    ticker: str | None = Field(
        default=None,
        description="Símbolo bursátil de la compañía: NVDA, MSFT, AAPL, GOOGL, META o AMZN",
    )
    ejercicio: int | None = Field(
        default=None,
        description=(
            "Ejercicio fiscal al que se refiere la cifra. Si la pregunta "
            "compara dos, el más reciente de los dos."
        ),
    )
    fuente: Literal["xbrl", "texto", "ambas", "ninguna"] = Field(
        description=(
            "De dónde sale el dato. 'xbrl' si viene de get_xbrl_fact, 'texto' "
            "si viene de search_filings o read_section, 'ambas' si hacen "
            "falta las dos, y 'ninguna' si el dato NO está en el corpus. "
            "Usar 'ninguna' cuando no está es la respuesta correcta, no un "
            "fracaso."
        )
    )
    cita: str | None = Field(
        default=None,
        description=(
            "Texto LITERAL del informe que respalda la respuesta, copiado "
            "palabra por palabra del fragmento recuperado. Una frase, no un "
            "resumen ni una traducción."
        ),
    )
    chunk_id: str | None = Field(
        default=None,
        description=(
            "Identificador del fragmento del que se ha copiado la cita, tal "
            "y como aparece entre corchetes al principio del fragmento. "
            "UNO SOLO: el del fragmento de donde sale la frase del campo "
            "'cita'. Ejemplo: 'MSFT-2025-1A-0009'."
        ),
    )

    @field_validator("chunk_id", mode="before")
    @classmethod
    def _un_solo_chunk_id(cls, valor):
        """Se queda con el primer identificador cuando el modelo pone varios.

        El campo es uno, y el prompt lo dice, pero cuando la respuesta se apoya
        en dos fragmentos el modelo tiende a rellenarlo con
        `'MSFT-2025-1A-0017; MSFT-2025-1A-0021'`. No es una alucinación: los dos
        identificadores existen y los dos respaldan la respuesta.

        Sin normalizar, esa cadena no se encuentra en el índice de fragmentos y
        `cita_correcta` la cuenta como cita inventada, que es justo lo contrario
        de lo que pasó. El evaluador estaría midiendo el formato del campo en
        lugar de la veracidad de la fuente.

        Se conserva el primero porque el campo `cita` lleva **una** frase, y el
        contrato es que esa frase salga de ese fragmento. Si el modelo puso
        primero el fragmento equivocado, la tercera comprobación de
        `cita_correcta` —que el texto citado esté de verdad ahí— lo detecta
        igual. La normalización no perdona nada: solo evita castigar un
        separador.
        """
        if not isinstance(valor, str):
            return valor
        encontrados = re.findall(r"[A-Z]{1,6}-\d{4}-[0-9A-Z]{1,3}-\d{3,5}", valor)
        if encontrados:
            return encontrados[0]
        return valor.strip() or None

    # --- Añadidos. Opcionales, con valor por defecto, y nada depende de ---
    # --- ellos: el esquema sigue cumpliendo el contrato original.       ---
    cifra_anterior: float | None = Field(
        default=None,
        description=(
            "Solo en preguntas comparativas: el valor del ejercicio ANTERIOR, "
            "en unidades absolutas. Permite comprobar que el agente consultó "
            "de verdad los dos ejercicios en lugar de estimar la variación."
        ),
    )
    ejercicio_anterior: int | None = Field(
        default=None,
        description="Solo en comparativas: a qué ejercicio corresponde cifra_anterior.",
    )
    concepto_xbrl: str | None = Field(
        default=None,
        description=(
            "Concepto us-gaap consultado, si se usó get_xbrl_fact. Ejemplo: "
            "'Revenues'. Sirve para auditar que se pidió el concepto correcto "
            "para esa compañía y no el de otra."
        ),
    )


class ItemGolden(BaseModel):
    """Una pregunta del golden set, con su respuesta conocida.

    Replica el esquema del §7 del enunciado. Existe como modelo de Pydantic —y
    no solo como dict— porque escribir 20 preguntas a mano en JSONL con doce
    campos cada una garantiza erratas, y es mejor que salten al construirlas
    que al ejecutar la evaluación.
    """

    id: str
    pregunta: str
    familia: Literal["extractiva", "numerica", "comparativa"]
    ticker: str
    fiscal_year: int
    respuesta_esperada: str
    cifra_esperada: float | None = None
    unidad: str | None = None
    concept_xbrl: str | None = None
    item_esperado: str | None = None
    ancla_texto: str | None = None
    ancla_inicio: int | None = None
    ancla_fin: int | None = None
    chunk_id_esperado: str | None = None
    herramienta_esperada: list[str] = Field(default_factory=list)
    autor: str = AUTOR_POR_DEFECTO

    # --- Añadidos, todos opcionales ---------------------------------------
    # El validador oficial ignora las claves que no conoce, así que estos
    # campos no rompen la compatibilidad con un golden set del esquema base.
    cifra_anterior_esperada: float | None = Field(
        default=None,
        description=(
            "Solo en comparativas: el valor del ejercicio anterior. Sin este "
            "campo no se puede distinguir un agente que consultó los dos "
            "ejercicios de uno que consultó uno y estimó el otro, que es "
            "justo lo que la familia comparativa está puesta para detectar."
        ),
    )
    fiscal_year_anterior: int | None = None
    respuesta_esperada_es_ausencia: bool = Field(
        default=False,
        description=(
            "Marca las preguntas cuya respuesta correcta es que el dato NO "
            "está en el corpus. Van en un fichero aparte: el validador "
            "oficial exige `cifra_esperada` en la familia numérica y una "
            "pregunta sin respuesta, por definición, no la tiene."
        ),
    )


# Orden canónico de las claves al volcar a JSONL. No es cosmética: un fichero
# donde cada línea lleva las claves en un orden distinto es ilegible en un
# diff, y el golden set es un fichero que se revisa a mano.
CAMPOS_GOLDEN = [
    "id",
    "pregunta",
    "familia",
    "ticker",
    "fiscal_year",
    "respuesta_esperada",
    "cifra_esperada",
    "unidad",
    "concept_xbrl",
    "item_esperado",
    "ancla_texto",
    "ancla_inicio",
    "ancla_fin",
    "chunk_id_esperado",
    "herramienta_esperada",
    "cifra_anterior_esperada",
    "fiscal_year_anterior",
    "autor",
]

FAMILIAS = {"extractiva", "numerica", "comparativa"}

# Palabras máximas de un ancla. El límite lo pone el validador oficial y tiene
# su razón: un ancla de tres párrafos la recupera cualquier retriever, así que
# no mide la calidad de la recuperación sino el tamaño de la ventana.
MAXIMO_PALABRAS_ANCLA = 40


def validar_golden(preguntas: list[dict], exigir_20: bool = True) -> list[str]:
    """Los problemas de un golden set. Lista vacía significa correcto.

    Es el validador de la celda 32 del notebook de la sesión 1, movido aquí
    para que el notebook que construye el golden set y el que lo usa apliquen
    exactamente el mismo criterio. Duplicar un validador es garantizar que las
    dos copias se separen.

    Sobre las tres comprobaciones que van más allá de mirar que los campos
    existan:

    - **El concepto XBRL tiene que existir para esa compañía y ese
      ejercicio.** Es la que atrapa el error de escribir una pregunta
      numérica sobre Alphabet usando el concepto de ingresos de Apple. Sin
      ella, el golden set exigiría al agente algo imposible y la tabla
      mediría un error nuestro.
    - **El ancla no puede pasar de 40 palabras.** Un ancla larga la recupera
      cualquier cosa.
    - **Un ancla tiene que caer donde dice.** Se comprueba contra el texto
      reconstruido, no contra la buena voluntad de quien la escribió.
    """
    from agente import corpus

    problemas: list[str] = []
    vistos: set[str] = set()
    xbrl = corpus.cargar_xbrl()
    secciones = {
        (s["ticker"], s["fiscal_year"], s["item"]): s for s in corpus.cargar_secciones()
    }

    for numero, p in enumerate(preguntas, 1):
        pid = p.get("id", f"#{numero}")
        faltan = [c for c in ("id", "pregunta", "familia", "ticker", "fiscal_year") if not p.get(c)]
        if faltan:
            problemas.append(f"{pid}: faltan los campos {faltan}")
            continue
        if p["id"] in vistos:
            problemas.append(f"{pid}: id repetido")
        vistos.add(p["id"])

        if p["familia"] not in FAMILIAS:
            problemas.append(f"{pid}: familia '{p['familia']}' no válida")
        if p["ticker"] not in config.TICKERS:
            problemas.append(f"{pid}: {p['ticker']} no está en el corpus")
        if int(p["fiscal_year"]) not in config.EJERCICIOS:
            problemas.append(f"{pid}: FY{p['fiscal_year']} no está en el corpus")

        if p["familia"] in {"numerica", "comparativa"} and not p.get(
            "respuesta_esperada_es_ausencia"
        ):
            if p.get("cifra_esperada") is None:
                problemas.append(f"{pid}: numérica sin cifra_esperada")
            concepto = p.get("concept_xbrl")
            hay = xbrl[
                (xbrl.ticker == p["ticker"])
                & (xbrl.fiscal_year == int(p["fiscal_year"]))
                & (xbrl.concept == concepto)
            ]
            if concepto and hay.empty:
                problemas.append(
                    f"{pid}: {p['ticker']} no reporta '{concepto}' en "
                    f"FY{p['fiscal_year']}. El concepto se mira en "
                    f"xbrl_facts.parquet, nunca por analogía con otra compañía."
                )
            elif concepto and p.get("cifra_esperada") is not None:
                real = float(hay.iloc[0].value)
                if abs(real - float(p["cifra_esperada"])) > 0.5:
                    problemas.append(
                        f"{pid}: cifra_esperada {float(p['cifra_esperada']):,.0f} "
                        f"no coincide con el XBRL, que dice {real:,.0f}"
                    )

        if p["familia"] in {"extractiva", "comparativa"}:
            ancla = p.get("ancla_texto")
            if not ancla:
                problemas.append(f"{pid}: extractiva sin ancla_texto")
            elif len(ancla.split()) > MAXIMO_PALABRAS_ANCLA:
                problemas.append(
                    f"{pid}: ancla de {len(ancla.split())} palabras. Una "
                    f"frase. Así no medís vuestro retrieval, medís vuestro "
                    f"tamaño de ventana."
                )
            else:
                clave = (p["ticker"], int(p["fiscal_year"]), p.get("item_esperado"))
                seccion = secciones.get(clave)
                if seccion is None:
                    problemas.append(f"{pid}: no existe la sección {clave}")
                elif p.get("ancla_inicio") is not None:
                    recorte = seccion["texto"][p["ancla_inicio"] : p["ancla_fin"]]
                    if recorte != ancla:
                        problemas.append(
                            f"{pid}: el ancla no cae en [{p['ancla_inicio']}, "
                            f"{p['ancla_fin']}) de {clave}"
                        )
                elif ancla not in seccion["texto"]:
                    problemas.append(f"{pid}: el ancla no aparece en {clave}")

        if not p.get("herramienta_esperada"):
            problemas.append(f"{pid}: sin herramienta_esperada")

    if exigir_20:
        if len(preguntas) != 20:
            problemas.append(f"hacen falta 20 preguntas, hay {len(preguntas)}")
        n_comparativas = sum(p.get("familia") == "comparativa" for p in preguntas)
        if n_comparativas < 6:
            problemas.append(f"hacen falta 6 comparativas, hay {n_comparativas}")
    return problemas
