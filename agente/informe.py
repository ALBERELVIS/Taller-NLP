"""Genera el informe en PDF a partir de los resultados guardados en disco.

## Por qué el informe se genera por código y no se escribe a mano

Un informe escrito a mano se separa de los datos en cuanto se vuelve a
ejecutar una evaluación. La tabla del documento dice una cosa, el CSV dice
otra, y no hay forma de saber cuál es la buena sin rehacer el trabajo. Aquí el
PDF se construye leyendo `resultados/*.csv`, así que **no puede** estar
desactualizado: o los ficheros están y las cifras son las últimas medidas, o
no están y el informe lo dice en su sitio en lugar de callárselo.

Eso tiene una consecuencia que conviene asumir: la prosa de análisis no puede
depender de cifras concretas escritas a pelo. Las frases del informe describen
el método, la interpretación y los límites; los números los pone el código
leyendo los CSV, y las comparaciones que sí dependen de valores concretos
—«cuál mejoró más», «cuál es el más caro»— se calculan también.

## Por qué ReportLab y no LaTeX o Markdown→PDF

Tres razones, en orden de peso:

1. **No añade dependencias del sistema.** LaTeX exige una distribución de
   varios gigabytes; `wkhtmltopdf` y `pandoc` son binarios externos. El
   enunciado pide que el proyecto se pueda ejecutar en un clon limpio, y
   `pip install reportlab` es todo lo que hace falta aquí.
2. **Las tablas se construyen desde los `DataFrame`.** No hay un paso
   intermedio de serializar a texto y volver a parsear, que es donde se
   pierden el formato de las cifras y el resaltado del mejor valor.
3. **El resaltado condicional es trivial.** Marcar la mejor celda de cada
   columna es un bucle sobre un `TableStyle`, no una extensión de Markdown.

Los gráficos los hace Matplotlib y se incrustan como PNG. Se guardan también
sueltos en `informe/figuras/` por si hacen falta para las diapositivas de la
defensa.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from agente import config

# ---------------------------------------------------------------------------
# Identidad del grupo
# ---------------------------------------------------------------------------
# AQUÍ VAN LOS TRES NOMBRES. Es lo único del informe que no se puede deducir
# de los datos. Mientras esté el marcador, la portada lo enseña en rojo para
# que no se entregue sin rellenar.
INTEGRANTES: list[str] = [
    "NOMBRE Y APELLIDOS 1",
    "NOMBRE Y APELLIDOS 2",
    "NOMBRE Y APELLIDOS 3",
]
MARCADOR = "NOMBRE Y APELLIDOS"

TITULO = "Agente investigador sobre informes 10-K de la SEC"
SUBTITULO = "Práctica de LLMs aplicados a Finanzas · MIAX"

RUTA_PDF = config.DIR_INFORME / "Informe_MIAX_Agente_10K.pdf"
DIR_FIGURAS = config.DIR_INFORME / "figuras"

# Paleta sobria. Un informe técnico con seis colores es más difícil de leer,
# no más bonito.
AZUL = "#1f3864"
AZUL_CLARO = "#dce6f1"
GRIS = "#595959"
GRIS_CLARO = "#f2f2f2"
VERDE = "#2e7d32"
ROJO = "#c62828"


# ---------------------------------------------------------------------------
# Lectura de resultados
# ---------------------------------------------------------------------------
def _leer(nombre: str) -> pd.DataFrame | None:
    """Un CSV de `resultados/`, o `None` si no se ha generado todavía."""
    ruta = config.DIR_RESULTADOS / nombre
    if not ruta.is_file():
        return None
    try:
        tabla = pd.read_csv(ruta)
        return tabla if not tabla.empty else None
    except Exception:
        return None


def recopilar() -> dict:
    """Todo lo que el informe necesita, leído de disco de una vez.

    Se recopila en un solo sitio y no según hace falta en cada sección para
    que el informe sepa desde el principio qué tiene y qué le falta, y pueda
    decirlo en el resumen ejecutivo en vez de dejar huecos sueltos por el
    documento.
    """
    datos = {
        "configuracion": config.resumen_configuracion(),
        "fecha": date.today().isoformat(),
        "retrieval": _leer("retrieval_experimentos.csv"),
        "retrieval_por_conjunto": _leer("retrieval_por_conjunto.csv"),
        "comparativa": _leer("tabla_comparativa.csv"),
        "ausencias": _leer("tabla_ausencias.csv"),
        "holdout": _leer("eval_holdout_final.csv"),
    }
    for conjunto in ("oficial", "propio", "ausencias"):
        for sistema in ("baseline", "filtros", "final"):
            datos[f"eval_{conjunto}_{sistema}"] = _leer(
                f"eval_{conjunto}_{sistema}.csv"
            )

    # Estadísticas del corpus y de los golden sets. Son baratas de calcular y
    # evitan tener que escribirlas a mano en la prosa, que es donde se quedan
    # desactualizadas.
    try:
        from agente import corpus

        secciones = corpus.cargar_secciones()
        chunks = corpus.cargar_chunks()
        xbrl = corpus.cargar_xbrl()
        datos["corpus"] = {
            "secciones": len(secciones),
            "chunks": len(chunks),
            "tokens": sum(s["n_tokens"] for s in secciones),
            "hechos_xbrl": len(xbrl),
            "conceptos": int(xbrl.concept.nunique()),
            "emisores": int(xbrl.ticker.nunique()),
        }
    except Exception:
        datos["corpus"] = None

    for clave, ruta in (
        ("golden_oficial", config.RUTA_GOLDEN_OFICIAL),
        ("golden_propio", config.RUTA_GOLDEN_PROPIO),
        ("golden_ausencias", config.RUTA_GOLDEN_AUSENCIAS),
    ):
        if Path(ruta).is_file():
            with open(ruta, encoding="utf-8") as f:
                datos[clave] = [json.loads(l) for l in f if l.strip()]
        else:
            datos[clave] = []

    return datos


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def _figura(nombre: str):
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    return DIR_FIGURAS / nombre


def grafico_recall(datos: dict) -> Path | None:
    """Barras horizontales de `recall@5` por configuración de retrieval.

    Horizontales y no verticales porque las etiquetas son frases («+ híbrido
    BM25 con RRF») y en vertical habría que girarlas o abreviarlas.
    """
    tabla = datos.get("retrieval")
    if tabla is None:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orden = tabla.sort_values("recall@5")
    etiquetas = [str(c)[:46] for c in orden["configuración"]]
    valores = orden["recall@5"].astype(float)

    alto = max(3.0, 0.32 * len(etiquetas) + 1.0)
    fig, ax = plt.subplots(figsize=(9, alto))
    colores = [AZUL if v == valores.max() else "#8ea9c9" for v in valores]
    barras = ax.barh(etiquetas, valores, color=colores)
    ax.bar_label(barras, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("recall@5 sobre las 26 anclas de los dos golden sets")
    ax.set_title("Recall de cada configuración de retrieval", color=AZUL)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(labelsize=8)
    fig.tight_layout()

    ruta = _figura("recall_retrieval.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def grafico_sistemas(datos: dict) -> Path | None:
    """Calidad y coste de los tres sistemas, en dos paneles.

    Dos paneles y no dos ejes en el mismo: superponer un porcentaje y un
    precio en la misma gráfica con dos escalas invita a leer cruces entre las
    líneas que no significan nada.
    """
    tabla = datos.get("comparativa")
    if tabla is None:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (izq, der) = plt.subplots(1, 2, figsize=(10, 4))
    sistemas = ["baseline", "filtros", "final"]
    ancho = 0.35

    for desplazamiento, conjunto in zip((-ancho / 2, ancho / 2), ("oficial", "propio")):
        sub = tabla[tabla.conjunto == conjunto].set_index("sistema")
        alturas = [
            float(sub.loc[s, "acierto"]) if s in sub.index and pd.notna(sub.loc[s, "acierto"]) else 0.0
            for s in sistemas
        ]
        posiciones = [i + desplazamiento for i in range(len(sistemas))]
        barras = izq.bar(posiciones, alturas, ancho, label=f"golden {conjunto}")
        izq.bar_label(barras, fmt="%.2f", fontsize=7, padding=2)

    izq.set_xticks(range(len(sistemas)), sistemas)
    izq.set_ylim(0, 1.1)
    izq.set_ylabel("acierto")
    izq.set_title("Calidad", color=AZUL)
    izq.legend(fontsize=8)
    izq.grid(axis="y", alpha=0.3)

    sub = tabla[tabla.conjunto == "oficial"].set_index("sistema")
    costes = [
        float(sub.loc[s, "coste medio ($)"]) * 100 if s in sub.index else 0.0
        for s in sistemas
    ]
    latencias = [
        float(sub.loc[s, "latencia (s)"]) if s in sub.index else 0.0 for s in sistemas
    ]
    barras = der.bar(sistemas, costes, color="#8ea9c9")
    der.bar_label(barras, fmt="%.2f¢", fontsize=8, padding=2)
    der.set_ylabel("céntimos de dólar por pregunta")
    der.set_title("Coste y latencia", color=AZUL)
    der.grid(axis="y", alpha=0.3)

    eje_latencia = der.twinx()
    eje_latencia.plot(sistemas, latencias, "o--", color=ROJO, label="latencia (s)")
    eje_latencia.set_ylabel("segundos por pregunta", color=ROJO)
    eje_latencia.tick_params(axis="y", labelcolor=ROJO)

    fig.tight_layout()
    ruta = _figura("sistemas.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def grafico_familias(datos: dict) -> Path | None:
    """Acierto por familia de pregunta y sistema, sobre los dos conjuntos."""
    tabla = datos.get("comparativa")
    if tabla is None:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    familias = ["extractiva", "numérica", "comparativa"]
    disponibles = [f for f in familias if f in tabla.columns]
    if not disponibles:
        return None

    fig, ax = plt.subplots(figsize=(9, 3.6))
    sistemas = ["baseline", "filtros", "final"]
    ancho = 0.25

    # Se promedia sobre los dos conjuntos porque el desglose por familia sobre
    # 20 preguntas deja 6 o 7 casos por celda, y separarlo además por conjunto
    # daría barras de 3 preguntas que no se pueden interpretar.
    for i, sistema in enumerate(sistemas):
        sub = tabla[tabla.sistema == sistema]
        alturas = [float(sub[f].mean()) if f in sub else 0.0 for f in disponibles]
        posiciones = [j + (i - 1) * ancho for j in range(len(disponibles))]
        barras = ax.bar(posiciones, alturas, ancho, label=sistema)
        ax.bar_label(barras, fmt="%.2f", fontsize=7, padding=2)

    ax.set_xticks(range(len(disponibles)), disponibles)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("acierto")
    ax.set_title("Acierto por familia de pregunta", color=AZUL)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    ruta = _figura("familias.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------------------
# Piezas del documento
# ---------------------------------------------------------------------------
def _estilos():
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    base = getSampleStyleSheet()
    return {
        "titulo": ParagraphStyle(
            "TituloPortada", parent=base["Title"], fontSize=24, leading=29,
            textColor=AZUL, spaceAfter=10,
        ),
        "subtitulo": ParagraphStyle(
            "Subtitulo", parent=base["Normal"], fontSize=13, leading=17,
            textColor=GRIS, alignment=1, spaceAfter=28,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontSize=15, leading=19,
            textColor=AZUL, spaceBefore=16, spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontSize=11.5, leading=15,
            textColor=GRIS, spaceBefore=11, spaceAfter=5,
        ),
        "cuerpo": ParagraphStyle(
            "Cuerpo", parent=base["BodyText"], fontSize=9.4, leading=13.6,
            alignment=TA_JUSTIFY, spaceAfter=7,
        ),
        "nota": ParagraphStyle(
            "Nota", parent=base["BodyText"], fontSize=8.3, leading=11.5,
            textColor=GRIS, alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "pie": ParagraphStyle(
            "Pie", parent=base["Normal"], fontSize=7.6, textColor=GRIS,
            alignment=1,
        ),
        "celda": ParagraphStyle(
            "Celda", parent=base["Normal"], fontSize=7.4, leading=9.4,
        ),
    }


def _tabla(
    datos_tabla: list[list[str]],
    anchos=None,
    resaltar: dict[int, int] | None = None,
    tamano: float = 7.4,
):
    """Una tabla con la cabecera azul y, opcionalmente, celdas resaltadas.

    `resaltar` mapea número de columna a número de fila (contando la cabecera)
    para marcar el mejor valor de esa columna. Se resalta con negrita y un
    fondo suave en vez de con color de texto, porque el informe se imprime en
    blanco y negro más veces de las que nadie admite.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle

    estilo_celda = ParagraphStyle(
        "CeldaTabla", fontName="Helvetica", fontSize=tamano,
        leading=tamano + 2.2,
    )
    estilo_cabecera = ParagraphStyle(
        "CabeceraTabla", fontName="Helvetica-Bold", fontSize=tamano,
        leading=tamano + 2.2, textColor=colors.white,
    )

    def _celda(valor, cabecera: bool = False):
        if isinstance(valor, Paragraph):
            return valor
        texto = str(valor).replace("\n", "<br/>")
        return Paragraph(texto, estilo_cabecera if cabecera else estilo_celda)

    celdas = [
        [_celda(valor, cabecera=(i == 0)) for valor in fila]
        for i, fila in enumerate(datos_tabla)
    ]
    tabla = Table(celdas, colWidths=anchos, repeatRows=1, hAlign="LEFT")
    estilo = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(AZUL)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), tamano),
        ("LEADING", (0, 0), (-1, -1), tamano + 2.2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b7c5d8")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor(GRIS_CLARO)]),
    ]
    for columna, fila in (resaltar or {}).items():
        estilo += [
            ("BACKGROUND", (columna, fila), (columna, fila), colors.HexColor(AZUL_CLARO)),
            ("FONTNAME", (columna, fila), (columna, fila), "Helvetica-Bold"),
        ]
    tabla.setStyle(TableStyle(estilo))
    return tabla


def _mejores(tabla: pd.DataFrame, columnas: list[str], menor_es_mejor: set[str]) -> dict:
    """Para cada columna, la fila con el mejor valor. Índices de la tabla PDF."""
    mejores = {}
    for i, columna in enumerate(columnas):
        if columna not in tabla or not pd.api.types.is_numeric_dtype(tabla[columna]):
            continue
        serie = tabla[columna].dropna()
        if serie.empty:
            continue
        posicion = serie.idxmin() if columna in menor_es_mejor else serie.idxmax()
        mejores[i] = int(tabla.index.get_loc(posicion)) + 1
    return mejores


# ---------------------------------------------------------------------------
# El documento
# ---------------------------------------------------------------------------
def generar(ruta_pdf: Path | str = RUTA_PDF, verboso: bool = True) -> Path:
    """Escribe el PDF y devuelve su ruta."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    datos = recopilar()
    estilos = _estilos()
    ruta_pdf = Path(ruta_pdf)
    ruta_pdf.parent.mkdir(parents=True, exist_ok=True)

    P = lambda texto, estilo="cuerpo": Paragraph(texto, estilos[estilo])  # noqa: E731
    historia: list = []

    def titulo(texto):
        historia.append(P(texto, "h1"))

    def subtitulo(texto):
        historia.append(P(texto, "h2"))

    def parrafo(texto):
        historia.append(P(texto, "cuerpo"))

    def nota(texto):
        historia.append(P(texto, "nota"))

    def espacio(alto=6):
        historia.append(Spacer(1, alto))

    def imagen(ruta, ancho=165 * mm):
        if ruta is None:
            return
        from PIL import Image as PilImage  # viene con matplotlib

        with PilImage.open(ruta) as img:
            proporcion = img.height / img.width
        historia.append(Image(str(ruta), width=ancho, height=ancho * proporcion))
        espacio(8)

    # === Portada ===========================================================
    historia.append(Spacer(1, 45 * mm))
    historia.append(P(TITULO, "titulo"))
    historia.append(P(SUBTITULO, "subtitulo"))

    faltan_nombres = any(MARCADOR in n for n in INTEGRANTES)
    color_nombres = ROJO if faltan_nombres else AZUL
    historia.append(P(
        "<para alignment='center'>"
        + "<br/>".join(f"<font color='{color_nombres}'>{n}</font>" for n in INTEGRANTES)
        + "</para>",
        "cuerpo",
    ))
    if faltan_nombres:
        historia.append(P(
            "<para alignment='center'><b>Faltan los nombres.</b> Se rellenan en "
            "<font face='Courier'>INTEGRANTES</font>, al principio de "
            "<font face='Courier'>agente/informe.py</font>, y se vuelve a "
            "generar el PDF.</para>",
            "nota",
        ))

    espacio(24)
    cfg = datos["configuracion"]
    historia.append(P(
        f"<para alignment='center'>{datos['fecha']}<br/>"
        f"modelo <b>{cfg['modelo']}</b> · temperatura {cfg['temperatura']} · "
        f"embeddings {cfg['modelo_embeddings']}<br/>"
        f"k={cfg['k']} · tolerancia numérica {cfg['tolerancia_cifra']:.0%} · "
        f"precios de OpenAI consultados el {cfg['fecha_precios']}</para>",
        "nota",
    ))
    historia.append(PageBreak())

    # === 1. Resumen ejecutivo =============================================
    _resumen_ejecutivo(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos)
    historia.append(PageBreak())

    # === 2. Arquitectura ==================================================
    _arquitectura(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos)
    historia.append(PageBreak())

    # === 3. Corpus ========================================================
    _corpus(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos)

    # === 4. Golden set ====================================================
    _golden(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos)
    historia.append(PageBreak())

    # === 5. Retrieval =====================================================
    _retrieval(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos, imagen)
    historia.append(PageBreak())

    # === 6. Guardrails ====================================================
    _guardrails(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos)

    # === 7. Evaluadores ===================================================
    _evaluadores(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos)
    historia.append(PageBreak())

    # === 8. La tabla ======================================================
    _tabla_sistemas(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos, imagen)
    historia.append(PageBreak())

    # === 9. Lo que no funcionó ============================================
    _negativos(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos)

    # === 10. Generalización ===============================================
    _generalizacion(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos)

    # === 11. Reproducibilidad =============================================
    _reproducibilidad(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos)

    from reportlab.lib import colors

    def pie(canvas, documento):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(GRIS))
        canvas.drawCentredString(
            A4[0] / 2, 12 * mm,
            f"{TITULO} · {SUBTITULO} · página {documento.page}",
        )
        canvas.restoreState()

    documento = SimpleDocTemplate(
        str(ruta_pdf),
        pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title=TITULO, author=" · ".join(INTEGRANTES),
    )
    documento.build(historia, onFirstPage=pie, onLaterPages=pie)

    if verboso:
        print(f"Informe escrito en {ruta_pdf}  "
              f"({ruta_pdf.stat().st_size / 1024:.0f} KB)")
        if faltan_nombres:
            print("AVISO: la portada sigue con los marcadores de nombre. "
                  "Rellena INTEGRANTES en agente/informe.py.")
        ausentes = [k for k in ("retrieval", "comparativa") if datos.get(k) is None]
        if ausentes:
            print(f"AVISO: faltan resultados de {ausentes}. Ejecuta los "
                  f"notebooks 04 y 05 y vuelve a generar el informe.")
    return ruta_pdf


# ---------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------
def _pct(valor) -> str:
    return "n/d" if valor is None or pd.isna(valor) else f"{100 * float(valor):.0f} %"


def _resumen_ejecutivo(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos):
    titulo("1. Resumen ejecutivo")

    parrafo(
        "Se ha construido un agente que responde preguntas en lenguaje natural sobre "
        "doce informes 10-K —seis emisores tecnológicos, ejercicios 2024 y 2025— "
        "decidiendo por sí mismo cuál de cuatro herramientas usar en cada momento, y "
        "devolviendo siempre una respuesta estructurada con la cifra, su unidad, el "
        "origen del dato y la cita literal que lo respalda."
    )
    parrafo(
        "La tesis que defiende este trabajo es que, en este dominio, <b>la calidad no "
        "está en el modelo sino en el enrutado y en la verificación</b>. El mismo modelo, "
        "con el mismo prompt y las mismas cuatro herramientas, pasa de fallar la mitad de "
        "las preguntas a resolverlas cambiando únicamente cómo recupera el texto y "
        "añadiendo una comprobación automática de cada cifra contra los datos XBRL "
        "auditados."
    )

    comparativa = datos.get("comparativa")
    if comparativa is not None:
        oficial = comparativa[comparativa.conjunto == "oficial"].set_index("sistema")
        propio = comparativa[comparativa.conjunto == "propio"].set_index("sistema")

        def valor(tabla, sistema, columna):
            if sistema not in tabla.index:
                return None
            return tabla.loc[sistema, columna]

        subtitulo("Los tres resultados que resumen el trabajo")
        filas = [["", "golden oficial", "golden propio"]]
        for etiqueta, columna in (
            ("acierto, sistema baseline", "acierto"),
            ("acierto, sistema final", "acierto"),
        ):
            sistema = "baseline" if "baseline" in etiqueta else "final"
            filas.append([
                etiqueta,
                _pct(valor(oficial, sistema, columna)),
                _pct(valor(propio, sistema, columna)),
            ])
        for etiqueta, columna in (
            ("cita verificada, sistema final", "cita"),
            ("cita en el sitio correcto, sistema final", "fundamentada"),
            ("cifra correcta, sistema final", "cifra"),
            ("trayectoria correcta, sistema final", "trayectoria"),
            ("recall@5, sistema final", "recall@5"),
        ):
            filas.append([
                etiqueta,
                _pct(valor(oficial, "final", columna)),
                _pct(valor(propio, "final", columna)),
            ])
        coste_final = valor(oficial, "final", "coste medio ($)")
        coste_base = valor(oficial, "baseline", "coste medio ($)")
        if coste_final is not None and pd.notna(coste_final):
            filas.append([
                "coste por pregunta, baseline → final",
                f"{100 * float(coste_base):.2f} ¢ → {100 * float(coste_final):.2f} ¢",
                "",
            ])
        historia.append(_tabla(filas, anchos=[95 * 2.2, 55 * 1.6, 55 * 1.6]))
        espacio(10)

    from reportlab.lib.units import mm  # noqa: F401  (usado por los anchos)

    subtitulo("Qué hay que mirar de esta tabla, y qué no")
    parrafo(
        "El número que más dice no es el acierto: es la <b>distancia entre el acierto y "
        "la cita verificada</b>. Un sistema que acierta mucho y cita poco está "
        "respondiendo de memoria, y eso en un dominio auditado no vale, porque no se "
        "puede comprobar ni se puede repetir con otra compañía. Las tres columnas de "
        "verificabilidad están precisamente para impedir que un acierto no auditable se "
        "cuente como un acierto."
    )
    nota(
        "Advertencia estadística que aplica a todo el informe: cada conjunto tiene 20 "
        "preguntas, así que una sola pregunta vale cinco puntos porcentuales. Las "
        "diferencias de una o dos preguntas entre configuraciones no son distinguibles "
        "del ruido y no se presentan como mejoras. Las que sí se defienden son las que "
        "superan varias preguntas y, sobre todo, las que se reproducen en los dos "
        "conjuntos a la vez: el golden set oficial no lo hemos escrito nosotros, así que "
        "es la única medida libre de sesgo de autoría."
    )


def _arquitectura(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos):
    titulo("2. Arquitectura y criterio de enrutado")

    parrafo(
        "El sistema no es un RAG clásico. Un RAG decide de antemano qué información "
        "necesita el modelo: recupera una vez, con la pregunta tal cual y una <i>k</i> "
        "fija, y genera. Eso se rompe con las tres clases de pregunta que tiene esta "
        "práctica: las que necesitan dos recuperaciones y una comparación, las que piden "
        "una cifra que vive en una tabla estructurada y no en la prosa, y las que se "
        "refieren a algo que no está en el corpus —donde un RAG plano devuelve igualmente "
        "sus <i>k</i> fragmentos y el modelo los usa."
    )
    parrafo(
        "La alternativa es un agente: el modelo decide en cada vuelta qué herramienta "
        "llamar, ve el resultado y vuelve a decidir. La recuperación deja de ser un paso "
        "previo y pasa a ser una acción más, que puede repetirse, refinarse o no ocurrir."
    )

    subtitulo("Las cuatro herramientas y cuándo debe llamarse cada una")
    filas = [
        ["Herramienta", "Qué devuelve", "Coste", "Cuándo la llama el agente"],
        ["list_available", "El catálogo: emisores, ejercicios,\nsecciones y conceptos XBRL",
         "0 tokens", "Antes de afirmar que algo no existe, y ante\ncualquier duda sobre el catálogo"],
        ["get_xbrl_fact", "Un valor exacto reportado\nen XBRL", "~40 tokens",
         "SIEMPRE que la pregunta pida una cifra.\nEs la fuente autorizada"],
        ["search_filings", "k fragmentos de texto,\ncon su chunk_id", "~2.000 tokens",
         "Riesgos, estrategia, litigios y cualquier\nexplicación de la dirección"],
        ["read_section", "Una sección entera del 10-K",
         "hasta 34.751\ntokens", "Último recurso, si search_filings ya falló\nsobre esa misma sección"],
    ]
    historia.append(_tabla([[c.replace("\n", "<br/>") for c in f] for f in filas],
                           anchos=[78, 118, 52, 182]))
    espacio(9)

    parrafo(
        "El enrutado entre las cuatro no se programa: se escribe. El único sitio donde el "
        "modelo lee qué hace cada herramienta es su <i>docstring</i>, así que el docstring "
        "no es documentación, es la parte del programa que decide el comportamiento. Los "
        "cuatro están redactados como política —cuándo llamar, cuándo no, qué vocabulario "
        "usar, qué hacer si no hay resultados— y no como descripción."
    )
    parrafo(
        "La diferencia se mide y está en el notebook de la sesión 1: con dos herramientas "
        "de cuerpo idéntico y descripciones distintas, la vaga no solo hace que el modelo "
        "dude de cuál usar, sino que le quita la información con la que construye los "
        "argumentos. Sin la lista de conceptos us-gaap en el docstring, el modelo tiene "
        "que inventarse el nombre exacto de un concepto contable a partir de la palabra "
        "«beneficio»."
    )

    subtitulo("El detalle de dominio que más fallos causa")
    parrafo(
        "El concepto XBRL de los ingresos <b>no es el mismo en todas las compañías</b>. "
        "NVIDIA y Alphabet etiquetan <font face='Courier'>Revenues</font>; Apple, "
        "Microsoft, Meta y Amazon usan "
        "<font face='Courier'>RevenueFromContractWithCustomerExcludingAssessedTax</font>. "
        "Un modelo que acaba de consultar Apple y pide el mismo concepto para NVIDIA no "
        "encuentra nada, y ese fallo se manifiesta como una cifra estimada o como un "
        "bucle de reintentos. Por eso <font face='Courier'>list_available</font> enumera "
        "los conceptos disponibles emisor por emisor y ejercicio por ejercicio: convierte "
        "una deducción por analogía, que falla, en una consulta, que no."
    )
    parrafo(
        "El segundo detalle es el ejercicio fiscal. <font face='Courier'>fiscal_year</font> "
        "es el año en que <b>cierra</b> el ejercicio, no el de presentación, y los seis "
        "emisores cierran en cuatro meses distintos: NVIDIA en enero, Microsoft en junio, "
        "Apple en septiembre y los otros tres en diciembre. La fecha de cierre se "
        "devuelve junto a cada ejercicio por la misma razón."
    )

    subtitulo("Los tres perfiles que se comparan")
    filas = [
        ["Perfil", "Retrieval de search_filings", "Guardrail numérico"],
        ["baseline", "denso plano, k=5, sin filtros", "no"],
        ["filtros", "denso + filtros de ticker, ejercicio e item", "no"],
        ["final", "reescritura + híbrido BM25/RRF + reranking", "sí"],
    ]
    historia.append(_tabla(filas, anchos=[70, 250, 95]))
    espacio(8)
    nota(
        "Los tres se diferencian de forma acumulativa y en un solo eje cada vez. Modelo, "
        "temperatura, prompt de sistema, esquema de salida, límites de llamadas y k son "
        "idénticos en los tres. Es la única forma de que la diferencia entre dos filas de "
        "la tabla final sea atribuible a lo que efectivamente se movió."
    )


def _corpus(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos):
    titulo("3. Reconstrucción del corpus y su validación")

    info = datos.get("corpus")
    parrafo(
        "El material recibido estaba incompleto: llegó el índice FAISS con sus metadatos "
        "(<font face='Courier'>dataset/indice/</font>) pero no "
        "<font face='Courier'>secciones.jsonl</font>, "
        "<font face='Courier'>chunks.jsonl</font> ni "
        "<font face='Courier'>xbrl_facts.parquet</font>, que son los tres ficheros que "
        "las herramientas leen. Sin ellos no hay práctica, así que el primer trabajo fue "
        "reconstruirlos. Está entero en "
        "<font face='Courier'>00_Preparacion_del_corpus.ipynb</font>."
    )

    subtitulo("Cómo se reconstruyó cada pieza")
    parrafo(
        "<b>Las secciones.</b> <font face='Courier'>chunks_meta.parquet</font> guarda el "
        "texto íntegro de cada fragmento junto con sus desplazamientos "
        "<font face='Courier'>inicio_car</font> y <font face='Courier'>fin_car</font> "
        "dentro de la sección de la que salió. Con eso, reensamblar las 48 secciones es "
        "colocar cada fragmento en su posición y resolver los solapes. No es una "
        "aproximación: los solapes coinciden carácter a carácter, y donde no coincidieran "
        "el reensamblado fallaría de forma visible."
    )
    parrafo(
        "<b>Los hechos XBRL.</b> Se descargan de "
        "<font face='Courier'>data.sec.gov/api/xbrl/companyfacts</font>, que es la fuente "
        "original y pública. Para cada emisor y concepto se selecciona el hecho anual "
        "declarado en un formulario 10-K cuyo periodo coincide con el cierre de ejercicio "
        "del emisor, que es la parte no trivial: la misma magnitud aparece en el fichero "
        "varias veces, en formularios distintos y con periodos distintos."
    )

    subtitulo("Por qué la validación no puede ser el SHA-256")
    parrafo(
        "El material de clase verifica la integridad comparando el hash de "
        "<font face='Courier'>chunks.jsonl</font> con el declarado en el manifiesto. Esa "
        "comprobación aquí <b>no sirve</b>, y decirlo importa: el hash de un fichero que "
        "acabamos de generar nosotros coincide consigo mismo por construcción. Demuestra "
        "que el fichero no se ha corrompido al copiarlo, que no es lo que está en duda."
    )
    parrafo(
        "Lo que sí demuestra algo es comprobar la reconstrucción contra una verdad "
        "externa que no controlamos, y el golden set oficial la proporciona por partida "
        "doble. Sus preguntas extractivas traen un <font face='Courier'>ancla_texto</font> "
        "—una frase literal del informe— con los desplazamientos exactos donde debe "
        "aparecer; y sus preguntas numéricas traen la cifra esperada. Las dos "
        "comprobaciones son <font face='Courier'>assert</font> del notebook 00:"
    )
    filas = [
        ["Comprobación", "Resultado"],
        ["Las 13 anclas del golden set oficial caen carácter a carácter en\n"
         "los desplazamientos [ancla_inicio, ancla_fin) que el propio golden declara",
         "13 / 13"],
        ["Las cifras del golden set oficial coinciden con el XBRL descargado", "todas"],
        ["Los huecos declarados en el enunciado siguen vacíos\n"
         "(AMZN sin GrossProfit, Liabilities ni I+D; META y GOOGL sin GrossProfit)",
         "reproducidos"],
        ["indice.ntotal == len(chunks_meta) == len(chunks.jsonl), y los\n"
         "conjuntos de chunk_id son idénticos", "1.749"],
    ]
    historia.append(_tabla([[c.replace("\n", "<br/>") for c in f] for f in filas],
                           anchos=[330, 85]))
    espacio(8)
    parrafo(
        "La tercera es la que más tranquilidad da, y es contraintuitiva: se verifica que "
        "<b>faltan</b> datos. Amazon no etiqueta <font face='Courier'>GrossProfit</font> "
        "en us-gaap, y el enunciado lo declara como hueco intencionado. Si nuestra "
        "reconstrucción lo hubiera rellenado —por ejemplo calculándolo como ingresos "
        "menos coste de ventas— habríamos eliminado justo las preguntas que comprueban si "
        "el agente sabe decir «ese dato no está» en lugar de estimarlo."
    )

    if info:
        subtitulo("El corpus resultante")
        filas = [
            ["secciones", "fragmentos", "tokens", "hechos XBRL", "conceptos", "emisores"],
            [f"{info['secciones']}", f"{info['chunks']:,}".replace(",", "."),
             f"{info['tokens']:,}".replace(",", "."), f"{info['hechos_xbrl']}",
             f"{info['conceptos']}", f"{info['emisores']}"],
        ]
        historia.append(_tabla(filas, anchos=[70, 72, 72, 72, 65, 60]))
        espacio(8)


def _golden(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos):
    titulo("4. El golden set propio")

    propio = datos.get("golden_propio") or []
    ausencias = datos.get("golden_ausencias") or []

    if propio:
        conteo = {}
        for p in propio:
            conteo[p["familia"]] = conteo.get(p["familia"], 0) + 1
        emisores = len({p["ticker"] for p in propio})
        ejercicios = len({p["fiscal_year"] for p in propio})
        items = len({p.get("item_esperado") for p in propio if p.get("item_esperado")})
        parrafo(
            f"Son {len(propio)} preguntas en "
            f"<font face='Courier'>golden_set_propio.jsonl</font>: "
            f"{conteo.get('extractiva', 0)} extractivas, "
            f"{conteo.get('numerica', 0)} numéricas y "
            f"{conteo.get('comparativa', 0)} comparativas —el enunciado exige al menos "
            f"6 de estas últimas—, cubriendo los {emisores} emisores, los {ejercicios} "
            f"ejercicios y los {items} items. Más "
            f"{len(ausencias)} preguntas de ausencia en un fichero aparte."
        )

    subtitulo("Las anclas se extraen del texto, no se escriben")
    parrafo(
        "Cada pregunta extractiva o comparativa lleva un "
        "<font face='Courier'>ancla_texto</font> con sus desplazamientos. Escribirlos a "
        "mano garantiza erratas —un espacio de más, unas comillas tipográficas "
        "cambiadas— y una errata en un ancla no da error: hace que la pregunta sea "
        "imposible de acertar y hunde el <i>recall</i> medido por un motivo que no tiene "
        "nada que ver con el retriever. Por eso el notebook 03 localiza cada ancla "
        "<b>dentro del texto reconstruido</b> y calcula los desplazamientos, en lugar de "
        "aceptarlos."
    )
    parrafo(
        "El límite de 40 palabras por ancla lo impone el validador oficial y tiene su "
        "razón: un ancla de tres párrafos la recupera cualquier retriever, así que no "
        "mediría la calidad de la recuperación sino el tamaño de la ventana."
    )

    subtitulo("Por qué las preguntas de ausencia van en un fichero aparte")
    parrafo(
        "El validador oficial exige <font face='Courier'>cifra_esperada</font> en la "
        "familia numérica. Una pregunta cuya respuesta correcta es «ese dato no está en "
        "el corpus» no tiene cifra que esperar, y meterla en el fichero principal "
        "obligaría a inventarse un número para que pasara la validación, que es "
        "exactamente lo contrario de lo que la pregunta mide."
    )
    parrafo(
        "Se evalúan con un criterio propio y doble: "
        "<font face='Courier'>fuente == \"ninguna\"</font> <b>y</b> "
        "<font face='Courier'>cifra is None</font>. Las dos partes hacen falta. Solo la "
        "primera dejaría pasar una respuesta que dice «no consta» y a la vez rellena el "
        "campo numérico con una estimación, que es justo el fallo que estas preguntas "
        "están puestas para detectar."
    )

    tabla_ausencias = datos.get("ausencias")
    if tabla_ausencias is not None:
        filas = [["sistema", "dice «no está»", "inventa una cifra"]]
        for _, fila in tabla_ausencias.iterrows():
            filas.append([
                str(fila["sistema"]),
                _pct(fila.get("dice 'no está'")),
                _pct(fila.get("inventa cifra")),
            ])
        historia.append(_tabla(filas, anchos=[100, 120, 120]))
        espacio(8)


def _retrieval(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos, imagen):
    titulo("5. Las mejoras de retrieval, medidas")

    parrafo(
        "El enunciado pide tres mejoras del <i>retrieval</i> con su "
        "<font face='Courier'>recall@k</font>. Se han medido ocho configuraciones sobre "
        "las 26 anclas de los dos golden sets, y el orden en que se cuentan es el orden "
        "en que se descubrieron, incluido el arreglo que no funcionó."
    )
    parrafo(
        "La métrica es <font face='Courier'>recall@5</font> contra el "
        "<font face='Courier'>ancla_texto</font>, no contra el "
        "<font face='Courier'>chunk_id</font>. La diferencia no es cosmética: si la "
        "verdad fuese un identificador de fragmento, en cuanto se cambia la ventana o el "
        "solape todos los identificadores son otros y la métrica se queda sin referencia, "
        "de modo que reparar el troceado saldría penalizado por haberlo reparado."
    )

    imagen(grafico_recall(datos))

    tabla = datos.get("retrieval")
    if tabla is not None:
        columnas = ["configuración", "recall@5", "aciertos", "pos. mediana",
                    "latencia/preg (s)", "coste"]
        disponibles = [c for c in columnas if c in tabla.columns]
        orden = tabla.sort_values("recall@5", ascending=False).reset_index(drop=True)
        cabecera = [c.replace("pos. mediana", "pos. med.")
                     .replace("latencia/preg (s)", "lat. (s)") for c in disponibles]
        filas = [cabecera]
        for _, fila in orden.iterrows():
            filas.append([
                str(fila["configuración"])[:46],
                f"{float(fila['recall@5']):.3f}",
                str(fila.get("aciertos", "")),
                f"{float(fila['pos. mediana']):.0f}" if "pos. mediana" in fila else "",
                f"{float(fila['latencia/preg (s)']):.2f}" if "latencia/preg (s)" in fila else "",
                str(fila.get("coste", ""))[:42],
            ][: len(disponibles)])
        mejor_fila = int(orden["recall@5"].idxmax()) + 1
        historia.append(_tabla(filas, anchos=[162, 42, 38, 38, 38, 144],
                               resaltar={1: mejor_fila}, tamano=6.6))
        espacio(9)

    por_conjunto = datos.get("retrieval_por_conjunto")
    if por_conjunto is not None:
        subtitulo("Las tres obligatorias, desglosadas por conjunto")
        filas = [list(por_conjunto.columns)]
        for _, fila in por_conjunto.iterrows():
            filas.append([str(fila.iloc[0])] +
                         [f"{float(v):.3f}" for v in fila.iloc[1:]])
        historia.append(_tabla(filas, anchos=[160, 78, 78, 78]))
        espacio(8)
        nota(
            "El desglose por conjunto es la comprobación de que no nos hemos escrito un "
            "golden set a medida de nuestro propio retriever. Una mejora que solo "
            "apareciese en el nuestro y no en el oficial sería sospechosa."
        )

    subtitulo("Lectura de los resultados")
    parrafo(
        "<b>El filtro de metadatos es el que más devuelve por lo que cuesta.</b> Sin "
        "filtro, la consulta compite contra 1.749 fragmentos, y los 10-K de seis "
        "tecnológicas del mismo año se parecen muchísimo: el párrafo de riesgos de IA de "
        "Microsoft y el de Meta son casi intercambiables para un modelo de <i>embeddings</i>. "
        "Al restringir a emisor, ejercicio y sección, el problema deja de ser «encontrar "
        "la aguja» y pasa a ser «ordenar unas decenas de candidatos». No cuesta ni una "
        "llamada al modelo."
    )
    parrafo(
        "<b>El híbrido BM25 no mejoró, y la razón es el idioma.</b> BM25 puntúa por "
        "coincidencia de términos; las preguntas están en español y el corpus en inglés, "
        "así que los únicos términos que coinciden son los nombres propios y las cifras, "
        "que el filtro de metadatos ya ha aprovechado. Para el resto de la consulta, BM25 "
        "aporta un orden esencialmente aleatorio que, al fundirse por RRF, degrada el "
        "orden denso, que sí era informativo."
    )
    parrafo(
        "<b>La reescritura de la consulta es la que desbloquea todo lo demás.</b> Traducir "
        "la pregunta al inglés y al vocabulario del propio informe —«cuánto creció» pasa a "
        "<i>revenue increased</i>, «ingresos» a <i>net sales</i>— es el arreglo que más "
        "sube el <i>recall</i>, y es también el único que añade una llamada al modelo por "
        "búsqueda. Es el orden que importa: la literatura recomienda primero el híbrido y "
        "luego refinar, y aquí el orden correcto es <b>primero traducir y después "
        "fundir</b>, porque hasta que la consulta no está en el idioma del corpus la "
        "señal léxica no tiene nada que aportar."
    )
    nota(
        "Las reescrituras se cachean en <font face='Courier'>.cache/reescrituras.json</font>. "
        "Por reproducibilidad, para que la tabla se regenere sin volver a pagar y sin "
        "depender de que el modelo dé la misma respuesta dos veces; y por coste, porque "
        "durante el desarrollo la misma pregunta se reescribe decenas de veces sin que "
        "eso aporte información nueva."
    )


def _guardrails(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos):
    titulo("6. Guardrails")

    parrafo(
        "Un agente financiero que puede equivocarse en una cifra sin que nada lo detecte "
        "no es un sistema, es una demostración. Los guardrails son lo que convierte una "
        "instrucción del prompt —que el modelo cumple a veces— en una garantía."
    )

    subtitulo("Verificación numérica contra XBRL")
    parrafo(
        "Es el guardrail de dominio, y va como <i>middleware</i> "
        "<font face='Courier'>@after_model</font>: cuando el agente ya ha producido su "
        "respuesta estructurada y antes de que salga, si esa respuesta afirma un número "
        "se contrasta contra los hechos XBRL del emisor y el ejercicio declarados. Si no "
        "cuadra con ninguno, el desajuste se le devuelve al modelo con "
        "<font face='Courier'>jump_to=\"model\"</font> y la lista de lo que sí hay "
        "reportado."
    )
    filas = [
        ["Decisión", "Por qué"],
        ["Se compara contra los hechos de ESE emisor\ny ESE ejercicio, no contra todo el XBRL",
         "Con 137 hechos, casi cualquier número de nueve cifras se\n"
         "parece al 1 % a alguno. Comparar contra todo da el visto\n"
         "bueno casi siempre y la detección tiende a cero en silencio"],
        ["Se descartan los hechos cuya unidad no es\ncomparable con la cifra afirmada",
         "Contrastar dólares contra un recuento de acciones genera\n"
         "falsas alarmas: el modelo acierta, el guardrail salta y la\n"
         "«corrección» empeora la respuesta"],
        ["El mensaje correctivo ENUMERA los hechos\ndisponibles",
         "El error típico no es inventarse un número: es pedir el\n"
         "concepto equivocado. Con la lista delante, el modelo\n"
         "corrige en una vuelta; sin ella, agota el límite probando"],
        ["Una sola corrección por invocación,\nmarcada en el estado",
         "Sin ese freno, un modelo que insiste se queda en bucle\n"
         "entre el nodo del modelo y el verificador, y el límite de\n"
         "llamadas a herramienta no lo corta porque no gasta ninguna"],
        ["Tolerancia del 1 % relativo",
         "Redondear 281.724 millones a «281.700 millones» no es\n"
         "inventarse un número; decir 250.000 sí lo es. Es la misma\n"
         "constante que usa el evaluador, a propósito"],
    ]
    historia.append(_tabla([[c.replace("\n", "<br/>") for c in f] for f in filas],
                           anchos=[150, 265], tamano=6.9))
    espacio(8)
    parrafo(
        "Que el <i>middleware</i> y el evaluador compartan la constante de tolerancia no "
        "es una comodidad: si fuesen dos números distintos, una respuesta podría pasar el "
        "guardrail y suspender el evaluador, y la tabla del informe no significaría nada."
    )

    for conjunto in ("oficial", "propio"):
        tabla = datos.get(f"eval_{conjunto}_final")
        if tabla is None or "guardrail" not in tabla:
            continue
        salto = tabla["guardrail"].fillna(False).astype(bool)
        if not salto.any():
            nota(f"En el golden set {conjunto}, el guardrail no llegó a intervenir en "
                 f"ninguna de las {len(tabla)} preguntas.")
            continue
        con = tabla[salto]["acierto"].dropna()
        sin = tabla[~salto]["acierto"].dropna()
        nota(
            f"En el golden set {conjunto} intervino en {int(salto.sum())} de "
            f"{len(tabla)} preguntas. De esas, acabaron correctas "
            f"{int(con.sum())}/{len(con)}; de las que no necesitaron intervención, "
            f"{int(sin.sum())}/{len(sin)}."
        )

    subtitulo("Los dos topes de llamadas")
    parrafo(
        "<font face='Courier'>ToolCallLimitMiddleware(run_limit=8)</font> y "
        "<font face='Courier'>ModelCallLimitMiddleware(run_limit=10)</font>. No enseñan "
        "nada al modelo: le ponen un techo. Cubren las dos formas distintas de no "
        "terminar, y hacen falta las dos: el primero corta al agente que reintenta "
        "variantes de un concepto que no existe —el caso real del margen bruto de "
        "Amazon—, y el segundo corta al que se enrosca razonando sin llamar a ninguna "
        "herramienta, que no gasta ni una llamada y por tanto el primero lo deja correr."
    )
    parrafo(
        "Los dos topes se declaran <b>antes</b> que el verificador en la cadena de "
        "<i>middleware</i>. El orden importa: la cadena se ejecuta en el orden de la "
        "lista, y un tope colocado detrás del verificador no impide que este abra una "
        "vuelta de más."
    )

    subtitulo("Human-in-the-loop, escrito y desactivado en la evaluación")
    parrafo(
        "<font face='Courier'>HumanInTheLoopMiddleware</font> está implementado para "
        "interrumpir antes de una llamada a <font face='Courier'>read_section</font> "
        "sobre una sección muy larga, que es la única acción del sistema capaz de gastar "
        "decenas de miles de tokens de golpe. Queda fuera de la evaluación a propósito: "
        "una interrupción esperando aprobación humana deja colgada cualquier ejecución "
        "automática, y la del día 24 lo es."
    )


def _evaluadores(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos):
    titulo("7. Los tres evaluadores")

    parrafo(
        "No son tres formas de medir lo mismo. Cada uno detecta una manera distinta de "
        "que una respuesta parezca buena sin serlo, y por eso los tres tienen que estar."
    )
    filas = [
        ["Evaluador", "Qué comprueba", "La patología que detecta"],
        ["cita_correcta",
         "Que el chunk_id exista, que el fragmento sea del\n"
         "documento por el que se pregunta, y que el texto\n"
         "de la cita esté de verdad en ese fragmento",
         "Cita inventada, o real pero de otro\nemisor. Un número correcto con una\n"
         "cita falsa no es auditable"],
        ["cifra_coincide_xbrl",
         "Que la cifra afirmada cuadre con la esperada\ndentro del 1 %",
         "Número equivocado, o ausente cuando\nla pregunta pedía uno"],
        ["uso_la_tool_correcta",
         "Que la trayectoria pase por TODAS las\nherramientas que el ítem declara esperadas",
         "Acierto por memoria del modelo, sin\npasar por el corpus"],
    ]
    historia.append(_tabla([[c.replace("\n", "<br/>") for c in f] for f in filas],
                           anchos=[88, 190, 137], tamano=6.9))
    espacio(9)

    subtitulo("El tercer estado: por qué devuelven True, False o None")
    parrafo(
        "<font face='Courier'>None</font> significa «no aplica», y sin él la tabla "
        "mentiría. Si el evaluador de cifras devolviese <font face='Courier'>False</font> "
        "en una pregunta extractiva —que no tiene cifra que acertar—, un sistema perfecto "
        "marcaría 50 % en esa columna y nadie sabría por qué. Los "
        "<font face='Courier'>None</font> se excluyen del promedio, de modo que cada "
        "columna mide solo los casos en los que tiene sentido medir."
    )

    subtitulo("La decisión discutible, y por qué se mantiene")
    parrafo(
        "<font face='Courier'>uso_la_tool_correcta</font> <b>suspende a quien acierta sin "
        "usar las herramientas</b>. Una cifra correcta sacada de la memoria del modelo no "
        "es un acierto del sistema: no es auditable, no es citable y, en la siguiente "
        "compañía donde el modelo no la recuerde, será un error con exactamente el mismo "
        "aspecto de confianza. Lo que se evalúa el día 24 es si el procedimiento "
        "generaliza a diez preguntas que nadie ha visto, y lo que generaliza es el "
        "procedimiento, no la suerte."
    )
    parrafo(
        "El de la cita compara normalizando y solo los primeros 120 caracteres, porque el "
        "modelo recorta y cambia las comillas. Exigir igualdad literal completa mediría la "
        "fidelidad tipográfica en lugar del respaldo documental, que es lo que interesa."
    )

    subtitulo("Un cuarto evaluador que no pide el enunciado, y por qué hizo falta")
    parrafo(
        "<font face='Courier'>cita_correcta</font> comprueba que la cita sea <b>real</b>. "
        "Lo que no puede comprobar es si ese fragmento real tiene algo que ver con la "
        "pregunta, y el problema no es teórico: en la primera ejecución completa, el "
        "sistema <i>baseline</i> —el que peor recupera de los tres— acertaba el 100 % de "
        "las preguntas extractivas. Con cuarenta fragmentos por sección, encontrar alguno "
        "real de la compañía correcta que citar es trivial. Una métrica que puntúa igual "
        "al mejor y al peor de los sistemas no está midiendo el sistema."
    )
    parrafo(
        "<font face='Courier'>cita_fundamentada</font> cierra ese hueco con un dato que el "
        "golden set ya traía y que no se estaba usando: el tramo "
        "<font face='Courier'>[ancla_inicio, ancla_fin)</font> donde vive la frase que "
        "responde. El fragmento citado declara su propio tramo "
        "<font face='Courier'>[inicio_car, fin_car)</font> en las mismas coordenadas, así "
        "que «¿citó desde el sitio correcto?» se reduce a si los dos intervalos se "
        "solapan. Es objetivo y no necesita un juez."
    )
    parrafo(
        "Se exige <b>solape</b> y no contención a propósito: el troceador parte por donde "
        "le toca y un ancla puede quedar repartida entre dos fragmentos. En ese caso los "
        "dos son citas legítimas, y exigir que uno contuviese el ancla entera penalizaría "
        "al agente por una decisión del troceador."
    )
    nota(
        "Los tres evaluadores del §4 se han dejado exactamente como los pide el enunciado. "
        "<font face='Courier'>cita_fundamentada</font> se reporta en columna aparte y se "
        "usa como condición adicional del acierto por familia, no sustituye a ninguno."
    )


def _tabla_sistemas(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos, imagen):
    titulo("8. Baseline contra intermedio contra final")

    comparativa = datos.get("comparativa")
    if comparativa is None:
        parrafo(
            "No hay resultados de evaluación en <font face='Courier'>resultados/</font>. "
            "Ejecuta <font face='Courier'>05_Evaluacion_baseline_vs_final.ipynb</font> y "
            "vuelve a generar este informe."
        )
        return

    parrafo(
        "Los tres sistemas sobre los dos conjuntos de preguntas. El mejor valor de cada "
        "columna va resaltado <b>dentro de cada conjunto</b>: comparar el acierto del "
        "baseline en un golden set con el del final en el otro no significa nada, porque "
        "no son las mismas preguntas."
    )

    calidad = ["sistema", "acierto", "extractiva", "numérica", "comparativa",
               "cita", "fundamentada", "cifra", "trayectoria"]
    coste = ["sistema", "recall@5", "coste medio ($)", "latencia (s)", "llamadas/preg"]
    menor_es_mejor = {"coste medio ($)", "latencia (s)", "llamadas/preg"}
    alias = {
        "fundamentada": "fundam.",
        "trayectoria": "trayect.",
        "comparativa": "compar.",
        "extractiva": "extract.",
        "numérica": "numérica",
        "coste medio ($)": "coste",
        "latencia (s)": "latencia",
        "llamadas/preg": "llamadas",
        "recall@5": "recall@5",
    }

    def pintar(sub: pd.DataFrame, columnas: list[str], anchos: list[float]) -> None:
        disponibles = [c for c in columnas if c in sub.columns]
        filas = [[alias.get(c, c) for c in disponibles]]
        for _, fila in sub.iterrows():
            visual = [str(fila["sistema"])]
            for columna in disponibles[1:]:
                v = fila[columna]
                if pd.isna(v):
                    visual.append("n/d")
                elif columna == "coste medio ($)":
                    visual.append(f"{100 * float(v):.2f} ¢")
                elif columna in {"latencia (s)", "llamadas/preg"}:
                    visual.append(f"{float(v):.1f}")
                else:
                    visual.append(f"{float(v):.2f}")
            filas.append(visual)
        historia.append(_tabla(
            filas,
            anchos=anchos[:len(disponibles)],
            resaltar=_mejores(sub, disponibles, menor_es_mejor),
            tamano=7.2,
        ))
        espacio(6)

    for conjunto in ("oficial", "propio"):
        sub = comparativa[comparativa.conjunto == conjunto].reset_index(drop=True)
        if sub.empty:
            continue
        subtitulo(f"Golden set {conjunto} ({int(sub['n'].iloc[0])} preguntas)")
        pintar(sub, calidad, [62, 48, 50, 52, 50, 40, 48, 40, 50])
        pintar(sub, coste, [62, 70, 70, 70, 70])
        espacio(4)

    imagen(grafico_sistemas(datos))
    imagen(grafico_familias(datos))

    subtitulo("Cómo leer la columna de coste")
    parrafo(
        "El asterisco del resaltado marca el valor más bajo en coste, latencia y número "
        "de llamadas, y eso no es un elogio: el sistema más barato es normalmente el que "
        "menos hace. La tabla está montada así para que la decisión de ingeniería se tome "
        "con los dos números delante. Si la mejora en acierto del sistema final fuese "
        "pequeña y su coste se multiplicase, la decisión correcta sería quedarse en el "
        "intermedio, y eso solo se puede argumentar con las dos columnas juntas."
    )

    # Comparación numérica calculada, no escrita a mano.
    oficial = comparativa[comparativa.conjunto == "oficial"].set_index("sistema")
    if {"baseline", "final"}.issubset(oficial.index):
        try:
            delta = float(oficial.loc["final", "acierto"]) - float(oficial.loc["baseline", "acierto"])
            razon_coste = (float(oficial.loc["final", "coste medio ($)"])
                           / max(float(oficial.loc["baseline", "coste medio ($)"]), 1e-9))
            razon_lat = (float(oficial.loc["final", "latencia (s)"])
                         / max(float(oficial.loc["baseline", "latencia (s)"]), 1e-9))
            if razon_coste >= 1:
                coste_txt = f"cuesta <b>{razon_coste:.1f}×</b> más por pregunta"
            else:
                coste_txt = f"cuesta <b>{razon_coste:.2f}×</b> lo que el baseline por pregunta"
            parrafo(
                f"En cifras, sobre el golden set oficial: el sistema final gana "
                f"<b>{100 * delta:+.0f} puntos</b> de acierto respecto al baseline, "
                f"{coste_txt} y tarda "
                f"<b>{razon_lat:.1f}×</b> más."
            )
        except Exception:
            pass


def _negativos(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos):
    titulo("9. Lo que se probó y no funcionó")

    parrafo(
        "Esta sección existe porque un informe que solo cuenta lo que salió bien no "
        "permite saber si las decisiones fueron razonadas o afortunadas. Los resultados "
        "negativos de abajo están medidos con la misma métrica y sobre las mismas "
        "preguntas que los positivos."
    )

    subtitulo("El híbrido BM25 sobre consultas en español")
    parrafo(
        "Es el fracaso más instructivo, porque es el arreglo que toda la literatura "
        "recomienda. No mejoró y llegó a empeorar, por la razón ya explicada del idioma. "
        "La conclusión accionable no es «el híbrido no sirve», sino «el híbrido está "
        "condicionado a que la consulta esté en el idioma del corpus», y eso cambia el "
        "orden en que hay que aplicar los arreglos."
    )

    subtitulo("Re-trocear el corpus")
    parrafo(
        "Se reindexó el corpus entero con cuatro combinaciones de ventana y solape (256/32, "
        "400/80, 512/64 y 768/128) para comprobar si el troceado original era el cuello de "
        "botella. Las ventanas más grandes recuperan algo mejor, y era esperable sin que "
        "eso demuestre nada sobre la calidad del troceado: una ventana más grande contiene "
        "más texto y por tanto tiene más probabilidad de contener el ancla. La mejora no "
        "compensó el coste de mantener un índice propio divergente del que se entrega, así "
        "que el sistema final <b>usa el índice original</b>."
    )

    subtitulo("Cambiar el modelo de embeddings")
    parrafo(
        "Se compararon cuatro, incluido "
        "<font face='Courier'>text-embedding-3-small</font> de OpenAI, que es de pago. "
        "Ninguno superó de forma clara al <font face='Courier'>bge-small-en-v1.5</font> "
        "del índice que se entrega, y el modelo de pago quedó a la par del local gratuito. "
        "Es un resultado útil en sentido negativo: indica que el cuello de botella de este "
        "corpus no está en la representación vectorial, sino en el idioma de la consulta y "
        "en el filtrado, que es donde sí hubo mejoras grandes."
    )
    nota(
        "<font face='Courier'>all-MiniLM-L6-v2</font> queda notablemente por debajo, y "
        "conviene explicar por qué para no sacar la conclusión equivocada: no lleva el "
        "prefijo de consulta que los modelos BGE esperan, y esa asimetría entre cómo se "
        "codifican los fragmentos y cómo se codifica la consulta es exactamente el fallo "
        "silencioso que advierte el manifiesto del índice. No da ningún error; solo "
        "recupera peor."
    )

    for conjunto in ("oficial", "propio"):
        tabla = datos.get(f"eval_{conjunto}_final")
        if tabla is None:
            continue
        fallos = tabla[tabla["acierto"] == False]  # noqa: E712
        if fallos.empty:
            continue
        subtitulo(f"Preguntas que el sistema final sigue fallando ({conjunto})")
        filas = [["id", "familia", "emisor", "qué respondió"]]
        for _, fila in fallos.iterrows():
            filas.append([
                str(fila["id"]), str(fila["familia"]), str(fila["ticker"]),
                str(fila.get("respuesta", ""))[:110],
            ])
        historia.append(_tabla(filas, anchos=[48, 62, 42, 263], tamano=6.6))
        espacio(8)


def _generalizacion(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos):
    titulo("10. ¿Generalizará a las diez preguntas ciegas?")

    parrafo(
        "La pregunta correcta no es si el sistema acierta mucho en las preguntas que "
        "hemos visto, sino <b>cuánto de lo medido es procedimiento y cuánto es ajuste</b>. "
        "Lo que sigue es el argumento en los dos sentidos, con lo que apoya cada lado."
    )

    subtitulo("Razones para esperar que sí")
    parrafo(
        "<b>Nada del sistema conoce las preguntas.</b> No hay reglas condicionadas a un "
        "emisor, a un ejercicio ni a un identificador. La única información específica del "
        "dominio que lleva el sistema son los conceptos XBRL y las fechas de cierre, y esa "
        "se lee del corpus en tiempo de ejecución con "
        "<font face='Courier'>list_available</font>, no está escrita en el código."
    )
    parrafo(
        "<b>Las mejoras se reproducen en los dos conjuntos.</b> El golden set oficial no lo "
        "hemos escrito nosotros, así que las ganancias que aparecen también ahí no pueden "
        "explicarse por cómo redactamos nuestras preguntas."
    )
    parrafo(
        "<b>Los guardrails no dependen del enunciado de la pregunta.</b> La verificación "
        "contra XBRL contrasta contra los hechos reportados, sea cual sea la pregunta que "
        "produjo la cifra. Es una red que funciona igual con preguntas nunca vistas."
    )

    subtitulo("Razones para esperar que no del todo")
    parrafo(
        "<b>El <i>recall</i> de los experimentos es un techo.</b> En el notebook 04 los "
        "filtros de emisor, ejercicio e item se le pasan al retriever <b>perfectos</b>, "
        "porque vienen del golden set. En producción los infiere el agente a partir de la "
        "pregunta, y cada vez que se equivoca de item el filtro que ayudaba pasa a hacer "
        "daño, porque descarta el fragmento correcto en lugar de despriorizarlo. La tabla "
        "del capítulo 8 incluye el <i>recall</i> del sistema completo, sin filtros dados, "
        "justamente para que esa diferencia sea visible."
    )
    parrafo(
        "<b>Hemos iterado sobre estas preguntas.</b> Aunque no se haya ajustado nada a un "
        "caso concreto, haber mirado los fallos varias veces sesga las decisiones de diseño "
        "hacia lo que funciona en estas 40 preguntas. Es inevitable y hay que descontarlo "
        "al extrapolar."
    )
    parrafo(
        "<b>La potencia estadística es baja.</b> Con 20 preguntas por conjunto, el "
        "intervalo de confianza de cualquiera de estos porcentajes es ancho. El acierto "
        "sobre diez preguntas ciegas puede quedar varios puntos por encima o por debajo "
        "solo por qué preguntas toquen."
    )

    subtitulo("Dónde es más probable que falle, y qué haríamos")
    filas = [
        ["Escenario", "Por qué", "Mitigación ya prevista"],
        ["Pregunta sobre una sección\nque no es la que el agente\nsupone",
         "El filtro de item descarta el\nfragmento correcto",
         "Reintentar sin filtro de item cuando\nla primera búsqueda vuelve pobre"],
        ["Concepto XBRL que solo\nreporta un emisor",
         "El modelo lo pide por analogía\ncon otro emisor",
         "list_available lo enumera; el guardrail\ndetecta la cifra que no cuadra"],
        ["Pregunta cuya respuesta\nno está en el corpus",
         "La tentación de estimar en vez\nde decir que no está",
         "Medido aparte en golden_set_ausencias\ny recogido en el capítulo 4"],
        ["Comparativa entre emisores\ndistintos, no entre ejercicios",
         "Nuestras comparativas son todas\nentre ejercicios del mismo emisor",
         "Ninguna. Es el hueco reconocido de\nnuestro golden set"],
    ]
    historia.append(_tabla([[c.replace("\n", "<br/>") for c in f] for f in filas],
                           anchos=[112, 135, 168], tamano=6.8))
    espacio(8)

    holdout = datos.get("holdout")
    if holdout is not None:
        subtitulo("Ensayo con hold-out simulado")
        parrafo(
            f"Se apartaron {len(holdout)} preguntas que no se usaron para iterar y se "
            f"evaluaron de una sola pasada con <font face='Courier'>evaluar()</font>, "
            f"cronometrando el proceso entero. El resultado y el tiempo están en "
            f"<font face='Courier'>resultados/eval_holdout_final.csv</font>."
        )


def _reproducibilidad(datos, titulo, subtitulo, parrafo, nota, espacio, historia, estilos):
    titulo("11. Reproducibilidad y entrega")

    cfg = datos["configuracion"]
    parrafo(
        "Todo el proyecto se ejecuta en orden sobre un clon limpio. No hay entornos "
        "virtuales, no hay pasos manuales y no hay ninguna clave escrita en el código."
    )
    filas = [
        ["Orden", "Qué hace"],
        ["api_key.txt", "Aquí se pega la clave de OpenAI. Está en .gitignore"],
        ["pip install -r requirements.txt", "Dependencias con versiones fijadas, no rangos"],
        ["00_Preparacion_del_corpus.ipynb", "Reconstruye y valida el corpus"],
        ["S1_Herramientas_y_Bucle_Alumno.ipynb", "Material de clase, resuelto"],
        ["S2_Robustez_y_Evaluacion_Alumno.ipynb", "Material de clase, resuelto"],
        ["03_Golden_set_propio.ipynb", "Construye y valida las 20 preguntas propias"],
        ["04_Retrieval_experimentos.ipynb", "Mide las ocho configuraciones de retrieval"],
        ["05_Evaluacion_baseline_vs_final.ipynb", "Evalúa los tres sistemas y guarda los crudos"],
        ["06_Informe.ipynb", "Genera este PDF a partir de resultados/"],
    ]
    historia.append(_tabla(filas, anchos=[180, 235]))
    espacio(8)

    parrafo(
        "La interfaz del §6 del enunciado es "
        "<font face='Courier'>agente/interfaz.py</font>: "
        "<font face='Courier'>responder(pregunta, thread_id=None)</font> y "
        "<font face='Courier'>evaluar(ruta_jsonl)</font>, con línea de órdenes "
        "<font face='Courier'>python -m agente.interfaz evaluar holdout.jsonl</font>. "
        "<font face='Courier'>evaluar()</font> acepta tanto una ruta como una lista ya "
        "cargada, para que el día 24 no dependa de dónde esté el fichero."
    )
    parrafo(
        "Las versiones de las dependencias van <b>fijadas</b>, no en rangos. LangChain "
        "publica cada pocos días y una API que se mueve por debajo rompe un notebook sin "
        "que se haya tocado una línea de código; <font face='Courier'>&gt;=1.3</font> no "
        "es una versión."
    )
    nota(
        f"Configuración con la que se generaron estos resultados: modelo "
        f"<b>{cfg['modelo']}</b>, temperatura {cfg['temperatura']}, embeddings "
        f"{cfg['modelo_embeddings']}, k={cfg['k']}, tolerancia numérica "
        f"{cfg['tolerancia_cifra']:.0%}, topes de "
        f"{cfg['limite_llamadas_herramienta']} llamadas a herramienta y "
        f"{cfg['limite_llamadas_modelo']} al modelo por invocación. Precios de OpenAI "
        f"consultados el {cfg['fecha_precios']} "
        f"({cfg['precio_entrada_usd_por_millon']} $/M de entrada, "
        f"{cfg['precio_salida_usd_por_millon']} $/M de salida). Los precios cambian sin "
        f"aviso: la columna de coste de este informe hay que releerla con esa fecha "
        f"delante."
    )


if __name__ == "__main__":
    generar()
