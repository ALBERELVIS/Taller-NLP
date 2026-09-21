"""Las cuatro herramientas del agente. Su interfaz pública.

## Las firmas son contrato

Los nombres y los parámetros del §7 del enunciado no se cambian: las diez
preguntas ciegas del día 24 se ejecutan contra ellos y el evaluador de
trayectoria busca exactamente estos nombres. Lo que sí se puede hacer, y se
hace, es reimplementar el cuerpo y **añadir parámetros con valor por
defecto**: una llamada que no los pase se comporta igual que antes.

## El docstring es comportamiento, no documentación

El modelo no ve el cuerpo de la función. Ve el nombre, el esquema de
argumentos y el docstring entero, y con eso decide si llama y con qué. Es el
único sitio del programa donde reescribir un comentario cambia lo que hace el
sistema.

De ahí salen las reglas que se han seguido al escribirlos:

1. **Decir cuándo llamarla, no solo qué hace.** «Es la fuente autorizada para
   cualquier cifra; úsala SIEMPRE en lugar de leer un número del texto» es una
   instrucción de enrutado, no una descripción.
2. **Decir cuándo NO llamarla.** El docstring de `search_filings` dice que no
   se use para cifras. Esa frase vale más que las otras cinco, porque el fallo
   caro no es no llamar a la herramienta buena, es llamar a la mala.
3. **Meter el vocabulario dentro.** Si `item` espera `'1A'`, `'7'`, `'7A'` u
   `'8'`, hay que enumerarlos: el modelo no puede adivinar un vocabulario que
   no ha visto. Lo mismo con los conceptos us-gaap.
4. **Decir qué hacer cuando no hay resultado.** Un resultado vacío sin
   instrucción hace que el modelo repita la misma llamada. Aquí, cada camino
   sin resultado devuelve texto que dice qué probar a continuación.
5. **Diferenciarlos entre sí.** Cuatro docstrings que digan «busca información
   en los informes» son cuatro herramientas indistinguibles. Cada uno nombra
   explícitamente a su alternativa y dice cuándo preferirla.

## La asimetría que el agente tiene que entender

`get_xbrl_fact` es barata, exacta y determinista. `search_filings` es cara,
difusa y aproximada. `read_section` es carísima. Elegir bien entre ellas es el
trabajo del agente, y es lo que se evalúa: acertar una cifra leyéndola de la
prosa cuenta como fallo aunque el número salga bien, porque ese camino no
generaliza al día en que la tabla venga partida —y el 41 % de los fragmentos
de este corpus llevan una tabla partida dentro—.
"""

from __future__ import annotations

from langchain.tools import tool

from agente import config, corpus, retrieval

# Umbral a partir del cual `read_section` avisa de lo que va a costar. La
# sección más larga del corpus son 34.751 tokens; la media, unos 13.500. El
# aviso no bloquea la lectura: informa al modelo del precio para que pueda
# decidir, que es distinto de impedírselo.
UMBRAL_SECCION_CARA = 15_000


# ---------------------------------------------------------------------------
# 1. list_available
# ---------------------------------------------------------------------------
@tool
def list_available() -> str:
    """Lista qué compañías, ejercicios fiscales y secciones existen en el corpus.

    ÚSALA SIEMPRE antes de afirmar que un dato no existe, y antes de cualquier
    otra herramienta si no estás seguro de que la compañía o el ejercicio por
    los que te preguntan están en el corpus. El corpus es cerrado y pequeño: si
    algo no aparece en esta lista, no está, y la respuesta correcta es decirlo
    con fuente="ninguna" en lugar de estimarlo.

    No necesita argumentos y es gratis: no cuesta ni una llamada al modelo ni
    una búsqueda.

    Devuelve, por compañía: su nombre, los ejercicios fiscales disponibles y
    las secciones del 10-K que se han cargado.

    AVISO IMPORTANTE sobre el ejercicio fiscal: `fiscal_year` es el año en que
    CIERRA el ejercicio, no el año en que se presentó el informe. Las seis
    compañías cierran en cuatro meses distintos. El FY2025 de NVIDIA cerró en
    enero de 2025 y el de Alphabet en diciembre de 2025. Fíate del campo, nunca
    de la fecha de presentación.
    """
    secciones = corpus.cargar_secciones()
    xbrl = corpus.cargar_xbrl()

    por_ticker: dict[str, dict] = {}
    for s in secciones:
        entrada = por_ticker.setdefault(
            s["ticker"],
            {"empresa": s["empresa"], "ejercicios": set(), "items": set()},
        )
        entrada["ejercicios"].add(int(s["fiscal_year"]))
        entrada["items"].add(s["item"])

    lineas = ["COMPAÑÍAS Y EJERCICIOS DISPONIBLES", ""]
    for ticker, datos in sorted(por_ticker.items()):
        cierres = (
            xbrl[xbrl.ticker == ticker]
            .groupby("fiscal_year")["period_end"]
            .max()
            .to_dict()
        )
        detalle = ", ".join(
            f"FY{fy} (cierra {cierres.get(fy, '?')})"
            for fy in sorted(datos["ejercicios"])
        )
        lineas.append(f"{ticker} — {datos['empresa']}")
        lineas.append(f"    ejercicios: {detalle}")
        lineas.append(f"    secciones : {sorted(datos['items'])}")

    lineas += [
        "",
        "SECCIONES DEL 10-K",
        "    1A  Risk Factors — los riesgos que la compañía declara",
        "    7   MD&A — la dirección explicando sus propios resultados",
        "    7A  Market Risk — exposición a tipos, divisa y precios",
        "    8   Financial Statements — estados financieros y sus notas",
        "",
        "CONCEPTOS XBRL DISPONIBLES, POR COMPAÑÍA Y EJERCICIO",
        "(el concepto NO es el mismo en todas las compañías: compruébalo aquí",
        " antes de llamar a get_xbrl_fact, nunca lo deduzcas por analogía)",
        "",
    ]
    for ticker in sorted(por_ticker):
        for fy in sorted(por_ticker[ticker]["ejercicios"]):
            disponibles = corpus.conceptos_disponibles(ticker, fy)
            lineas.append(f"    {ticker} FY{fy}: {', '.join(disponibles)}")

    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# 2. get_xbrl_fact
# ---------------------------------------------------------------------------
@tool
def get_xbrl_fact(ticker: str, fiscal_year: int, concept: str) -> str:
    """Devuelve el valor EXACTO de una magnitud financiera tal y como la compañía la reportó en XBRL.

    Es la FUENTE AUTORIZADA para cualquier cifra. Úsala SIEMPRE en lugar de
    leer un número del texto del informe, incluso cuando el número aparezca
    claramente en un fragmento que ya has recuperado. Un número leído de la
    prosa puede venir de una tabla partida por el troceador, de un dato
    trimestral o de un comparativo de otro ejercicio; este no.

    Para una pregunta que compare dos ejercicios, LLÁMALA DOS VECES, una por
    ejercicio, y calcula tú la diferencia. No estimes la variación.

    Args:
        ticker: Símbolo bursátil. Uno de: NVDA, MSFT, AAPL, GOOGL, META, AMZN.
        fiscal_year: Ejercicio fiscal en que CIERRA el ejercicio: 2024 o 2025.
        concept: Concepto de la taxonomía US-GAAP. Los disponibles son
            'Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax',
            'CostOfRevenue', 'GrossProfit', 'ResearchAndDevelopmentExpense',
            'OperatingIncomeLoss', 'IncomeTaxExpenseBenefit', 'NetIncomeLoss',
            'EarningsPerShareDiluted', 'Assets', 'Liabilities',
            'StockholdersEquity',
            'NetCashProvidedByUsedInOperatingActivities' y
            'PaymentsToAcquirePropertyPlantAndEquipment'.

    AVISO CRÍTICO sobre los ingresos: el concepto NO es el mismo en todas las
    compañías. NVIDIA usa 'Revenues'. Apple, Microsoft, Meta y Amazon usan
    'RevenueFromContractWithCustomerExcludingAssessedTax'. Alphabet etiqueta
    los dos en FY2024 pero solo 'Revenues' en FY2025. NUNCA razones por
    analogía con otra compañía: si no estás seguro, llama primero a
    list_available, que enumera los conceptos disponibles de cada una.

    Devuelve el valor con su unidad y la fecha de cierre del ejercicio. Si esa
    compañía NO reportó ese concepto en ese ejercicio, lo dice explícitamente y
    enumera los que sí reportó. Cuando eso ocurra, no es un error que haya que
    rodear buscando el número en el texto: puede ser que la compañía
    sencillamente no publique esa magnitud, y entonces la respuesta correcta es
    decir que no está en el corpus, con fuente="ninguna".
    """
    ticker = (ticker or "").strip().upper()
    concept = (concept or "").strip()
    try:
        fiscal_year = int(fiscal_year)
    except (TypeError, ValueError):
        return (
            f"'{fiscal_year}' no es un ejercicio válido. Usa 2024 o 2025 "
            f"(el año en que CIERRA el ejercicio)."
        )

    if ticker not in config.TICKERS:
        return (
            f"'{ticker}' no está en el corpus. Las compañías disponibles son "
            f"{', '.join(config.TICKERS)}. Usa list_available para ver el "
            f"detalle. No estimes el dato: si la compañía no está, la "
            f"respuesta correcta es que no está en el corpus."
        )
    if fiscal_year not in config.EJERCICIOS:
        return (
            f"FY{fiscal_year} no está en el corpus. Los ejercicios "
            f"disponibles son {config.EJERCICIOS}. No extrapoles a partir de "
            f"otros años."
        )

    hecho = corpus.hecho_xbrl(ticker, fiscal_year, concept)
    if hecho is None:
        disponibles = corpus.conceptos_disponibles(ticker, fiscal_year)
        return (
            f"{ticker} NO reportó '{concept}' en FY{fiscal_year}.\n"
            f"Conceptos que SÍ reportó: {', '.join(disponibles)}.\n"
            f"Si el concepto que buscas no está en esa lista, la compañía no "
            f"publica esa magnitud en us-gaap y la respuesta correcta es que "
            f"el dato no está en el corpus (fuente=\"ninguna\"). No lo "
            f"busques en el texto ni lo deduzcas de otras cifras."
        )

    valor = hecho["value"]
    unidad = hecho["unit"]
    formateado = f"{valor:,.2f}" if unidad == "USD/shares" else f"{valor:,.0f}"
    return (
        f"{ticker} FY{fiscal_year} · {concept} = {formateado} {unidad} "
        f"(cierre de ejercicio {hecho['period_end']}, según el "
        f"{hecho['form']}). Fuente: XBRL, valor exacto reportado."
    )


# ---------------------------------------------------------------------------
# 3. search_filings
# ---------------------------------------------------------------------------
def _buscar_y_formatear(
    query: str,
    ticker: str | None,
    fiscal_year: int | None,
    item: str | None,
    k: int,
    estrategia: str,
) -> str:
    """El cuerpo de `search_filings`, sin el decorador.

    Está separado porque la estrategia de retrieval **no puede ser un
    argumento que vea el modelo**: es una decisión de configuración del
    sistema. Si el modelo pudiera elegirla, dos ejecuciones del mismo perfil
    usarían retrievers distintos y la comparación baseline contra final
    dejaría de medir el sistema. Así, el esquema que viaja al modelo es
    exactamente el del contrato del §7 y la estrategia la fija quien monta el
    agente.
    """
    if not (query or "").strip():
        return "La consulta está vacía. Escribe qué quieres buscar, en inglés."

    ticker = (
        ticker.strip().upper() if isinstance(ticker, str) and ticker.strip() else None
    )
    item = item.strip() if isinstance(item, str) and item.strip() else None
    if fiscal_year is not None:
        try:
            fiscal_year = int(fiscal_year)
        except (TypeError, ValueError):
            fiscal_year = None

    # Un filtro con un valor que no existe devolvería cero resultados sin decir
    # por qué, y el modelo repetiría la misma búsqueda. Se avisa.
    if ticker and ticker not in config.TICKERS:
        return (
            f"'{ticker}' no está en el corpus. Compañías disponibles: "
            f"{', '.join(config.TICKERS)}."
        )
    if fiscal_year is not None and fiscal_year not in config.EJERCICIOS:
        return (
            f"FY{fiscal_year} no está en el corpus. Ejercicios disponibles: "
            f"{config.EJERCICIOS}."
        )
    if item and item not in config.ITEMS:
        return (
            f"'{item}' no es una sección válida. Usa '1A' (riesgos), '7' "
            f"(MD&A), '7A' (riesgo de mercado) u '8' (estados financieros)."
        )

    try:
        k = max(1, min(int(k), 20))
    except (TypeError, ValueError):
        k = config.K_POR_DEFECTO

    fragmentos = retrieval.buscar(
        query,
        ticker=ticker,
        fiscal_year=fiscal_year,
        item=item,
        k=k,
        estrategia=estrategia,
    )
    return retrieval.formatear_fragmentos(fragmentos)


DOCSTRING_SEARCH_FILINGS = """Busca fragmentos de texto relevantes en los informes 10-K del corpus.

    Úsala para preguntas CUALITATIVAS: riesgos declarados, estrategia,
    litigios, controles de exportación, y sobre todo para las EXPLICACIONES
    que da la dirección de por qué una magnitud subió o bajó.

    NO la uses para obtener cifras. Para cualquier número está get_xbrl_fact,
    que es exacto y mil veces más barato. En una pregunta comparativa el reparto
    correcto es: las dos cifras con get_xbrl_fact, y esta herramienta solo para
    la frase que explica la variación.

    Args:
        query: Qué buscar. ESCRÍBELA EN INGLÉS: el corpus está en inglés y el
            modelo de búsqueda es monolingüe inglés. Usa el vocabulario del
            propio informe ("net sales", "gross margin percentage",
            "provision for income taxes"), no la traducción literal de la
            pregunta.
        ticker: Filtra por compañía. PÁSALO SIEMPRE que la pregunta nombre una:
            sin él compiten los fragmentos de las otras cinco.
        fiscal_year: Filtra por ejercicio, 2024 o 2025. PÁSALO SIEMPRE que la
            pregunta lo mencione: los 10-K repiten sus factores de riesgo casi
            palabra por palabra de un año para otro, y sin este filtro lo más
            parecido a tu consulta puede ser el ejercicio equivocado.
        item: Filtra por sección. '1A' riesgos, '7' MD&A (explicaciones de la
            dirección sobre los resultados), '7A' riesgo de mercado (divisa,
            tipos de interés), '8' estados financieros y sus notas. Pásalo
            cuando sepas dónde vive la respuesta: el Item 7A son solo 37
            fragmentos de 1.749 y sin filtro no compite.
        k: Cuántos fragmentos devolver. 5 por defecto; sube a 10 si la primera
            búsqueda no ha traído lo que buscabas.

    Devuelve k fragmentos, cada uno precedido de su chunk_id entre corchetes.
    COPIA ESE chunk_id en el campo chunk_id de tu respuesta y copia la frase
    exacta que te sirve en el campo cita, palabra por palabra y sin traducir:
    la cita se verifica automáticamente contra el texto original.

    Si no encuentras nada útil, no repitas la misma consulta: quita el filtro
    de item, sube k, o reformula con otras palabras del informe."""


@tool("search_filings", description=DOCSTRING_SEARCH_FILINGS)
def search_filings(
    query: str,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
    k: int = 5,
) -> str:
    return _buscar_y_formatear(query, ticker, fiscal_year, item, k, "final")


# ---------------------------------------------------------------------------
# 4. read_section
# ---------------------------------------------------------------------------
@tool
def read_section(ticker: str, fiscal_year: int, item: str) -> str:
    """Devuelve el TEXTO COMPLETO de una sección de un 10-K.

    Es la herramienta CARA: puede devolver más de treinta mil tokens de una
    sola vez, y esos tokens se vuelven a pagar en CADA vuelta siguiente del
    razonamiento, no solo en la primera.

    Úsala solo como último recurso, y solo si se cumplen las dos condiciones:
    ya has llamado a search_filings al menos una vez sobre esa misma sección, y
    los fragmentos que ha devuelto son insuficientes porque necesitas el
    contexto completo (por ejemplo, para enumerar TODOS los riesgos de una
    sección, no uno concreto).

    NO la uses para buscar un dato concreto: para eso está search_filings.
    NO la uses nunca para obtener una cifra: para eso está get_xbrl_fact.

    Args:
        ticker: Símbolo bursátil: NVDA, MSFT, AAPL, GOOGL, META o AMZN.
        fiscal_year: Ejercicio fiscal, 2024 o 2025.
        item: '1A' riesgos, '7' MD&A, '7A' riesgo de mercado,
            '8' estados financieros.

    Devuelve el texto de la sección precedido de su tamaño en tokens.
    """
    ticker = (ticker or "").strip().upper()
    item = (item or "").strip()
    try:
        fiscal_year = int(fiscal_year)
    except (TypeError, ValueError):
        return f"'{fiscal_year}' no es un ejercicio válido. Usa 2024 o 2025."

    seccion = corpus.seccion(ticker, fiscal_year, item)
    if seccion is None:
        return (
            f"No hay Item {item} de {ticker} FY{fiscal_year} en el corpus. "
            f"Usa list_available para ver qué compañías, ejercicios y "
            f"secciones existen."
        )

    cabecera = (
        f"[{ticker} FY{fiscal_year} Item {item} — "
        f"{config.DESCRIPCION_ITEMS.get(item, '')} — "
        f"{seccion['n_tokens']:,} tokens]"
    )
    if seccion["n_tokens"] > UMBRAL_SECCION_CARA:
        cabecera += (
            f"\n[AVISO: esta sección es muy larga. Si solo necesitabas un "
            f"dato concreto, search_filings habría bastado y habría costado "
            f"unas {seccion['n_tokens'] // 2000} veces menos.]"
        )
    return f"{cabecera}\n\n{seccion['texto']}"


# ---------------------------------------------------------------------------
# El cinturón completo
# ---------------------------------------------------------------------------
HERRAMIENTAS = [list_available, get_xbrl_fact, search_filings, read_section]
POR_NOMBRE = {h.name: h for h in HERRAMIENTAS}

# Las cuatro del contrato, por si algún evaluador necesita comprobar que no se
# ha renombrado ninguna.
NOMBRES_CONTRACTUALES = [
    "list_available",
    "get_xbrl_fact",
    "search_filings",
    "read_section",
]


def construir_herramientas(estrategia_retrieval: str = "final") -> list:
    """Las cuatro herramientas, con `search_filings` atado a una estrategia.

    El agente no debe elegir la estrategia de retrieval: es una decisión de
    configuración del sistema, no del modelo. Exponerla como argumento de la
    herramienta y dejar que el modelo la rellene haría que dos ejecuciones del
    mismo perfil usaran retrievers distintos, y la comparación baseline contra
    final dejaría de medir el sistema.

    Por eso el perfil se fija aquí, cerrando el argumento por defecto, y el
    esquema que ve el modelo sigue siendo el del contrato.
    """
    if estrategia_retrieval not in retrieval.ESTRATEGIAS:
        raise ValueError(
            f"Estrategia {estrategia_retrieval!r} desconocida. "
            f"Disponibles: {retrieval.ESTRATEGIAS}"
        )
    if estrategia_retrieval == "final":
        return HERRAMIENTAS

    # El nombre y la descripción son contrato: el evaluador de trayectoria
    # busca exactamente 'search_filings', y el modelo tiene que ver el mismo
    # docstring en los tres perfiles. Si el baseline tuviera una descripción
    # peor, la comparación mediría el docstring y no el retriever.
    @tool("search_filings", description=DOCSTRING_SEARCH_FILINGS)
    def search_filings_perfilada(
        query: str,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        item: str | None = None,
        k: int = 5,
    ) -> str:
        return _buscar_y_formatear(
            query, ticker, fiscal_year, item, k, estrategia_retrieval
        )

    return [list_available, get_xbrl_fact, search_filings_perfilada, read_section]
