"""Converte o wikitext bruto do Wikilivros em receitas estruturadas.

Entrada:  ``data/raw/receitas_wikilivros.jsonl``   (uma página por linha)
Saída:    ``data/processed/receitas.jsonl``        (uma receita por linha)

Uma página pode conter **várias** receitas — o livro usa o padrão
``== Nome do prato - 1 ==``, ``== Nome do prato - 2 ==`` para variações da
mesma preparação. Cada variação vira um registro independente, o que amplia
o corpus de forma legítima (são textos realmente distintos).

O formato interno predominante é::

    == Bombocado de mandioca - 1 ==
    ==='''Ingredientes e Preparo:'''===

    * 1 lata de leite condensado --
    * 6 ovos;

    :'''Preparo:'''

    # Misture bem e bata todos os ingredientes.
    # Asse em banho-maria por 40 minutos.

Ou seja: ingredientes e modo de preparo convivem sob um único cabeçalho,
separados por um marcador em negrito. O parser trata esse caso e também as
variantes em que ``Ingredientes`` e ``Preparo`` são cabeçalhos próprios.

Uso::

    uv run python -m tech_challenge_fase_04.dados.preparo
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

ORIGEM_PADRAO = Path("data/raw/receitas_wikilivros.jsonl")
DESTINO_PADRAO = Path("data/processed/receitas.jsonl")

# Páginas que são índices gerados por template, não receitas.
MARCAS_DE_INDICE = ("{{Livro de receitas/Lista", "ESTA LISTA É GERADA")

# Cabeçalhos que descrevem uma *parte* da receita corrente, e não uma nova
# variação. Usado para decidir se um cabeçalho `== ... ==` inicia outra receita.
SECAO_INTERNA = re.compile(
    r"(?i)^\s*(ingredientes?|modo de (preparo|fazer|preparar)|preparo|prepara[çc][ãa]o"
    r"|ingredientes? e (modo de )?(preparo|prepara[çc][ãa]o)"
    r"|massa|recheio|cobertura|calda|molho|creme|glac[êe]"
    r"|para (a|o|os|as) .+|dicas?|observa[çc][õo]es?|variantes?"
    r"|refer[êe]ncias?|liga[çc][õo]es externas|ver tamb[ée]m|notas?)\s*:?\s*$"
)

# Marcador que separa ingredientes de modo de preparo dentro de uma mesma seção.
MARCADOR_PREPARO = re.compile(
    r"(?im)^[:;*#]?\s*'{2,}\s*(?:modo de\s+)?(?:preparo|preparar|fazer|prepara[çc][ãa]o)\s*:?\s*'{2,}"
)

# Categorias no padrão "Ovo, receitas com" são etiquetas de ingrediente, não
# classificação do prato — descartadas na escolha da categoria.
CATEGORIA_DE_INGREDIENTE = re.compile(r"(?i),\s*receitas? com\s*$")

# Sufixo " - 3" nos títulos de variação.
SUFIXO_VARIACAO = re.compile(r"\s*[-–]\s*\d+\s*$")

# Cabeçalhos genéricos ("Receita 2") não nomeiam o prato — cai-se no título da página.
TITULO_GENERICO = re.compile(r"(?i)^\s*(receitas?|varia[çc][ãa]o|modo)\s*\d*\s*$")

# Templates que envolvem *texto útil*: {{w|abacaxi}} vira "abacaxi". Precisam ser
# resolvidos antes da remoção genérica de templates, que descarta o conteúdo.
TEMPLATE_DE_TEXTO = re.compile(r"(?i)\{\{\s*(?:w|wikipedia|wikt|wikcionário)\s*\|([^{}]*?)\}\}")

# Sobras da limpeza: item sem substantivo, só quantidade e preposição ("2 de").
INGREDIENTE_VAZIO = re.compile(r"(?i)^[\d\s/,.()]*(?:de|da|do|das|dos|e|a|o|à)?[\d\s/,.()]*$")

logger = logging.getLogger(__name__)


@dataclass
class Receita:
    titulo: str
    categoria: str
    ingredientes: list[str]
    preparo: list[str]
    fonte_titulo: str
    fonte_url: str


# --------------------------------------------------------------------------- #
# Limpeza de wikitext
# --------------------------------------------------------------------------- #


def _remover_templates(texto: str) -> str:
    """Remove ``{{...}}``, respeitando aninhamento, varrendo o texto uma vez."""
    saida: list[str] = []
    profundidade = 0
    i = 0
    while i < len(texto):
        if texto.startswith("{{", i):
            profundidade += 1
            i += 2
        elif texto.startswith("}}", i) and profundidade:
            profundidade -= 1
            i += 2
        else:
            if not profundidade:
                saida.append(texto[i])
            i += 1
    return "".join(saida)


def limpar(texto: str) -> str:
    """Reduz wikitext a texto corrido legível."""
    texto = re.sub(r"<!--.*?-->", "", texto, flags=re.DOTALL)
    texto = re.sub(r"<ref[^>]*/>", "", texto)
    texto = re.sub(r"<ref[^>]*>.*?</ref>", "", texto, flags=re.DOTALL)
    # {{w|alvo|rótulo}} exibe o rótulo; {{w|alvo}} exibe o próprio alvo.
    texto = TEMPLATE_DE_TEXTO.sub(lambda m: m.group(1).split("|")[-1].strip(), texto)
    texto = _remover_templates(texto)
    # Imagens e arquivos precisam sair antes dos links comuns.
    texto = re.sub(
        r"\[\[(?:Image|Imagem|File|Ficheiro|Arquivo):[^\]]*\]\]", "", texto, flags=re.IGNORECASE
    )
    texto = re.sub(r"\[\[Categoria:[^\]]*\]\]", "", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", texto)  # [[alvo|rótulo]]
    texto = re.sub(r"\[\[([^\]]*)\]\]", r"\1", texto)  # [[alvo]]
    texto = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", texto)  # link externo rotulado
    texto = re.sub(r"\[https?://\S+\]", "", texto)
    texto = re.sub(r"<[^>]+>", "", texto)  # HTML solto
    texto = re.sub(r"'{2,}", "", texto)  # negrito/itálico
    texto = texto.replace("&nbsp;", " ").replace("&amp;", "&")
    texto = re.sub(r"__[A-Z]+__", "", texto)  # __NOTOC__, __NOEDITSECTION__
    return texto


def _normalizar_item(linha: str) -> str:
    """Limpa um item de lista: marcadores, pontuação de fim e espaços."""
    linha = re.sub(r"^[*#:;]+\s*", "", linha)
    linha = limpar(linha).strip()
    # O livro usa " --" para marcar o primeiro ingrediente e ";" para separar.
    linha = re.sub(r"\s*--+\s*$", "", linha)
    linha = re.sub(r"\s*[;,]\s*$", "", linha)
    return re.sub(r"\s+", " ", linha).strip(" .")


# --------------------------------------------------------------------------- #
# Extração
# --------------------------------------------------------------------------- #


def _categoria(wikitext: str) -> str:
    """Escolhe a categoria mais descritiva do prato, ignorando tags de ingrediente."""
    brutas = re.findall(r"\[\[Categoria:\s*([^\]|]+)", wikitext, flags=re.IGNORECASE)
    for bruta in brutas:
        nome = bruta.strip()
        if nome.startswith("{{") or not nome:
            continue
        if CATEGORIA_DE_INGREDIENTE.search(nome):
            continue
        return nome
    return "Geral"


def _blocos_de_variacao(corpo: str, titulo_pagina: str) -> list[tuple[str, str]]:
    """Divide o corpo em ``(título, texto)`` por variação de receita.

    Um cabeçalho de nível 2 inicia nova variação, exceto quando nomeia uma parte
    da receita (``== Ingredientes ==``), caso em que pertence à variação atual.
    """
    partes = re.split(r"(?m)^==(?!=)\s*(.+?)\s*==\s*$", corpo)

    # split com grupo de captura: [texto_antes, cab1, texto1, cab2, texto2, ...]
    blocos: list[tuple[str, str]] = []
    preambulo = partes[0]

    for i in range(1, len(partes), 2):
        cabecalho = re.sub(r"'{2,}", "", partes[i]).strip().rstrip(":")
        texto = partes[i + 1]

        if SECAO_INTERNA.match(cabecalho) and blocos:
            # Parte da receita anterior — reanexa mantendo o cabeçalho.
            anterior_titulo, anterior_texto = blocos[-1]
            blocos[-1] = (anterior_titulo, f"{anterior_texto}\n=== {cabecalho} ===\n{texto}")
        elif SECAO_INTERNA.match(cabecalho):
            blocos.append((titulo_pagina, f"=== {cabecalho} ===\n{texto}"))
        else:
            blocos.append((cabecalho, texto))

    if not blocos:
        blocos = [(titulo_pagina, preambulo)]

    return blocos


def _dividir_ingredientes_preparo(texto: str) -> tuple[list[str], list[str]]:
    """Separa a lista de ingredientes das etapas de preparo dentro de uma variação."""
    marcador = MARCADOR_PREPARO.search(texto)
    if marcador:
        zona_ingredientes, zona_preparo = texto[: marcador.start()], texto[marcador.end() :]
    else:
        # Sem marcador em negrito: tenta um cabeçalho dedicado de preparo.
        cabecalho = re.search(
            r"(?im)^=+\s*'*\s*(?:modo de\s+)?(?:preparo|preparar|fazer|prepara[çc][ãa]o)\s*:?\s*'*\s*=+\s*$",
            texto,
        )
        if cabecalho:
            zona_ingredientes, zona_preparo = texto[: cabecalho.start()], texto[cabecalho.end() :]
        else:
            # Último recurso: a primeira linha numerada abre o preparo.
            numerada = re.search(r"(?m)^#", texto)
            if numerada:
                zona_ingredientes, zona_preparo = (
                    texto[: numerada.start()],
                    texto[numerada.start() :],
                )
            else:
                return [], []

    ingredientes = [
        limpo
        for linha in zona_ingredientes.splitlines()
        # Item de lista, ou linha solta terminada em ";" / "--" (padrão do livro).
        if re.match(r"^\s*[*#]", linha) or re.search(r"[;]\s*$|--\s*$", linha)
        if (limpo := _normalizar_item(linha)) and len(limpo) > 3
        if not INGREDIENTE_VAZIO.match(limpo)
    ]

    preparo = [
        limpo
        for linha in zona_preparo.splitlines()
        if re.match(r"^\s*[*#]", linha) or (linha.strip() and not linha.startswith("="))
        if (limpo := _normalizar_item(linha)) and len(limpo) > 10
    ]

    return ingredientes, preparo


def extrair(registro: dict[str, str]) -> Iterator[Receita]:
    """Emite todas as receitas contidas em uma página."""
    wikitext = registro["wikitext"]
    if any(marca in wikitext for marca in MARCAS_DE_INDICE):
        return

    titulo_pagina = registro["titulo"].removeprefix("Livro de receitas/").strip()
    categoria = _categoria(wikitext)

    # A limpeza de categorias/templates acontece por bloco; aqui só tiramos ruído
    # que atrapalha a divisão em seções.
    corpo = re.sub(r"(?m)^\{\{Navegação\|.*$", "", wikitext)
    corpo = re.sub(r"\[\[Categoria:[^\]]*\]\]", "", corpo, flags=re.IGNORECASE)

    for titulo_bruto, texto in _blocos_de_variacao(corpo, titulo_pagina):
        ingredientes, preparo = _dividir_ingredientes_preparo(texto)
        if len(ingredientes) < 2 or len(preparo) < 1:
            continue

        titulo = SUFIXO_VARIACAO.sub("", limpar(titulo_bruto)).strip()
        if not titulo or TITULO_GENERICO.match(titulo):
            titulo = titulo_pagina

        yield Receita(
            titulo=titulo,
            categoria=categoria,
            ingredientes=ingredientes,
            preparo=preparo,
            fonte_titulo=registro["titulo"],
            fonte_url=registro["url"],
        )


def processar(origem: Path = ORIGEM_PADRAO, destino: Path = DESTINO_PADRAO) -> int:
    """Lê o JSONL bruto, extrai as receitas e grava o JSONL processado."""
    destino.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with origem.open(encoding="utf-8") as entrada, destino.open("w", encoding="utf-8") as saida:
        for linha in entrada:
            for receita in extrair(json.loads(linha)):
                saida.write(json.dumps(asdict(receita), ensure_ascii=False) + "\n")
                total += 1

    logger.info("Extraídas %d receitas para %s.", total, destino)
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    total = processar()
    print(f"\n{total} receitas extraídas em {DESTINO_PADRAO}")


if __name__ == "__main__":
    main()
