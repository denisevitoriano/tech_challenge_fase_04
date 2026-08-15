"""Monta o corpus de treino a partir das receitas estruturadas.

Entrada: ``data/processed/receitas.jsonl``
Saída:   ``data/processed/treino.jsonl`` e ``data/processed/validacao.jsonl``

Formato do exemplo
------------------
Cada receita é serializada como um bloco de texto delimitado por tokens
especiais, para que o modelo aprenda tanto a estrutura quanto onde parar::

    <|receita|>
    MODO: por-titulo
    TÍTULO: Bolo de fubá cremoso
    CATEGORIA: Bolos
    INGREDIENTES:
    - 3 ovos
    - 2 xícaras (chá) de fubá
    PREPARO:
    1. Bata os ovos com o açúcar.
    2. Asse em forno médio por 40 minutos.
    <|fim|>

Dois modos, uma receita
-----------------------
O playground precisa atender dois usos: "tenho um título, me dê a receita" e
"tenho estes ingredientes, me dê a receita". São ordenações diferentes dos
mesmos campos, então cada receita entra no corpus **duas vezes**, com a tag
``MODO:`` indicando a ordem. Na geração, o prompt fixa o modo e o modelo
completa os campos restantes.

Isso não é vazamento: o alvo de predição muda junto com a ordem dos campos.

Split treino/validação
----------------------
O split é feito por **página de origem**, não por receita. Variações da mesma
página (``Bolo de fubá - 1``, ``- 2``, ``- 3``) são quase-duplicatas; separá-las
entre treino e validação inflaria artificialmente a qualidade da perplexidade.

Uso::

    uv run python -m tech_challenge_fase_04.dados.corpus
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

ORIGEM_PADRAO = Path("data/processed/receitas.jsonl")
DESTINO_TREINO = Path("data/processed/treino.jsonl")
DESTINO_VALIDACAO = Path("data/processed/validacao.jsonl")

INICIO = "<|receita|>"
FIM = "<|fim|>"
TOKENS_ESPECIAIS = [INICIO, FIM]

MODO_TITULO = "por-titulo"
MODO_INGREDIENTES = "por-ingredientes"

PROPORCAO_VALIDACAO = 0.1
SEMENTE = 42

logger = logging.getLogger(__name__)


def _bloco_ingredientes(ingredientes: list[str]) -> str:
    linhas = "\n".join(f"- {item}" for item in ingredientes)
    return f"INGREDIENTES:\n{linhas}"


def _bloco_preparo(preparo: list[str]) -> str:
    linhas = "\n".join(f"{n}. {passo}" for n, passo in enumerate(preparo, start=1))
    return f"PREPARO:\n{linhas}"


def formatar(receita: dict[str, Any], modo: str) -> str:
    """Serializa uma receita no formato de treino, na ordem pedida pelo modo."""
    titulo = f"TÍTULO: {receita['titulo']}"
    categoria = f"CATEGORIA: {receita['categoria']}"
    ingredientes = _bloco_ingredientes(receita["ingredientes"])
    preparo = _bloco_preparo(receita["preparo"])

    # CATEGORIA precede TÍTULO para que o modo "surpreenda-me" possa condicionar
    # a geração a uma categoria e deixar o próprio modelo inventar o título.
    if modo == MODO_TITULO:
        campos = [categoria, titulo, ingredientes, preparo]
    elif modo == MODO_INGREDIENTES:
        campos = [ingredientes, categoria, titulo, preparo]
    else:
        raise ValueError(f"modo desconhecido: {modo!r}")

    corpo = "\n".join([f"MODO: {modo}", *campos])
    return f"{INICIO}\n{corpo}\n{FIM}"


def prefixo_por_titulo(titulo: str, categoria: str) -> str:
    """Prompt para *título → receita*. O modelo completa a partir de ``INGREDIENTES:``."""
    return (
        f"{INICIO}\nMODO: {MODO_TITULO}\nCATEGORIA: {categoria}\nTÍTULO: {titulo}\nINGREDIENTES:\n"
    )


def prefixo_por_ingredientes(ingredientes: list[str]) -> str:
    """Prompt para *ingredientes → receita*. O modelo completa a partir de ``CATEGORIA:``."""
    linhas = "\n".join(f"- {item.strip()}" for item in ingredientes if item.strip())
    return f"{INICIO}\nMODO: {MODO_INGREDIENTES}\nINGREDIENTES:\n{linhas}\nCATEGORIA:"


def prefixo_livre(categoria: str) -> str:
    """Prompt do modo "surpreenda-me": o modelo inventa o título dentro da categoria."""
    return f"{INICIO}\nMODO: {MODO_TITULO}\nCATEGORIA: {categoria}\nTÍTULO:"


def construir(
    origem: Path = ORIGEM_PADRAO,
    destino_treino: Path = DESTINO_TREINO,
    destino_validacao: Path = DESTINO_VALIDACAO,
) -> tuple[int, int]:
    """Gera os arquivos de treino e validação. Devolve ``(n_treino, n_validacao)``."""
    receitas = [json.loads(linha) for linha in origem.open(encoding="utf-8")]

    # Agrupa por página de origem para que variações fiquem do mesmo lado do split.
    paginas = sorted({r["fonte_titulo"] for r in receitas})
    random.Random(SEMENTE).shuffle(paginas)
    corte = int(len(paginas) * PROPORCAO_VALIDACAO)
    paginas_validacao = set(paginas[:corte])

    conjuntos: dict[str, list[dict[str, str]]] = {"treino": [], "validacao": []}
    for receita in receitas:
        alvo = "validacao" if receita["fonte_titulo"] in paginas_validacao else "treino"
        for modo in (MODO_TITULO, MODO_INGREDIENTES):
            conjuntos[alvo].append(
                {
                    "texto": formatar(receita, modo),
                    "modo": modo,
                    "titulo": receita["titulo"],
                    "categoria": receita["categoria"],
                }
            )

    for nome, caminho in (("treino", destino_treino), ("validacao", destino_validacao)):
        caminho.parent.mkdir(parents=True, exist_ok=True)
        with caminho.open("w", encoding="utf-8") as arquivo:
            for exemplo in conjuntos[nome]:
                arquivo.write(json.dumps(exemplo, ensure_ascii=False) + "\n")

    n_treino, n_validacao = len(conjuntos["treino"]), len(conjuntos["validacao"])
    logger.info(
        "Corpus: %d exemplos de treino, %d de validação (%d páginas reservadas).",
        n_treino,
        n_validacao,
        len(paginas_validacao),
    )
    return n_treino, n_validacao


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    n_treino, n_validacao = construir()
    print(f"\ntreino:    {n_treino:5d} exemplos -> {DESTINO_TREINO}")
    print(f"validação: {n_validacao:5d} exemplos -> {DESTINO_VALIDACAO}")


if __name__ == "__main__":
    main()
