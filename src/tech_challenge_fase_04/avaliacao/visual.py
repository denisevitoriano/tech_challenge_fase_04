"""Paleta e estilo dos gráficos do projeto.

Centraliza as cores em um lugar só para que os notebooks de exploração e de
avaliação — e as figuras exportadas para o relatório e o vídeo — formem um
sistema visual coerente.

A paleta categórica é usada **em ordem fixa** (série 1 sempre azul, série 2
sempre laranja). Cor identifica a entidade, nunca a posição no ranking: se um
gráfico filtra séries, as que sobram mantêm suas cores.

O uso é limitado a três séries por gráfico — é o máximo que se mantém
distinguível para daltonismo em todos os pares simultaneamente. Além disso,
agrupar em "outros" ou separar em gráficos menores.
"""

from __future__ import annotations

# Slots categóricos, em ordem de uso obrigatória.
AZUL = "#2a78d6"
LARANJA = "#eb6834"
AQUA = "#1baf7a"
CATEGORICA = [AZUL, LARANJA, AQUA]

# Escala sequencial (magnitude contínua), do claro ao escuro.
SEQUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95"]

# Tinta e cromo do gráfico.
SUPERFICIE = "#fcfcfb"
TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_APAGADA = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"


def aplicar_estilo() -> None:
    """Configura o matplotlib com o estilo do projeto.

    Chame uma vez no início de cada notebook, antes de plotar.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            # Grade e eixos recuam; os dados é que devem chamar atenção.
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRADE,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.edgecolor": EIXO,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": TINTA_SECUNDARIA,
            "axes.titlecolor": TINTA_PRIMARIA,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "xtick.color": TINTA_APAGADA,
            "ytick.color": TINTA_APAGADA,
            "xtick.labelcolor": TINTA_SECUNDARIA,
            "ytick.labelcolor": TINTA_SECUNDARIA,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICA),
        }
    )


def rotular_barras(eixo, formato: str = "{:.0f}", deslocamento: float = 3.0) -> None:
    """Escreve o valor no topo de cada barra.

    Rótulo direto substitui a consulta ao eixo e garante que a informação não
    dependa só da cor — necessário para os tons de contraste mais baixo.
    """
    for barra in eixo.patches:
        altura = barra.get_height()
        eixo.annotate(
            formato.format(altura),
            (barra.get_x() + barra.get_width() / 2, altura),
            xytext=(0, deslocamento),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TINTA_SECUNDARIA,
        )
