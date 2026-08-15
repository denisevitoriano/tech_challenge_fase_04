"""Coleta o corpus de receitas do Wikilivros (pt.wikibooks.org).

O livro colaborativo "Livro de receitas" reúne ~2.100 páginas em português,
sob licença CC-BY-SA 3.0. Cada receita é uma subpágina no padrão
``Livro de receitas/<Nome do prato>``.

A coleta é feita pela API MediaWiki em duas etapas:

1. ``list=allpages`` com ``apprefix`` — descobre os títulos existentes;
2. ``prop=revisions&rvprop=content`` — baixa o wikitext bruto de cada página.

O wikitext **cru** é salvo sem tratamento em ``data/raw/``. Toda limpeza fica
em :mod:`tech_challenge_fase_04.dados.preparo`, para que o dado original
permaneça somente-leitura e o pipeline seja reproduzível a partir dele.

Uso::

    uv run python -m tech_challenge_fase_04.dados.coleta
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

API = "https://pt.wikibooks.org/w/api.php"
PREFIXO = "Livro de receitas/"

# A Wikimedia bloqueia requisições sem User-Agent identificável (HTTP 403).
# A política pede nome do projeto e forma de contato.
USER_AGENT = (
    "TechChallengeFase04/0.1 (https://github.com/denisevitoriano; "
    "denisevitoriano@gmail.com) uso educacional"
)

# A API aceita até 50 títulos por requisição para usuários anônimos.
LOTE = 50

# Pausa entre requisições. A API não exige, mas é boa prática com serviço público.
PAUSA_S = 0.2

DESTINO_PADRAO = Path("data/raw/receitas_wikilivros.jsonl")

logger = logging.getLogger(__name__)


def _consultar(parametros: dict[str, str]) -> dict[str, Any]:
    """Faz uma chamada à API MediaWiki, com retentativa em erro transitório."""
    # formatversion=2 devolve `query.pages` como lista e o wikitext na chave
    # "content"; a versão 1 devolve um dicionário indexado por pageid e usa "*".
    parametros = {**parametros, "format": "json", "formatversion": "2"}
    url = f"{API}?{urllib.parse.urlencode(parametros)}"
    requisicao = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for tentativa in range(4):
        try:
            with urllib.request.urlopen(requisicao, timeout=30) as resposta:
                return json.load(resposta)
        except (urllib.error.URLError, TimeoutError) as erro:
            if tentativa == 3:
                raise
            espera = 2**tentativa
            logger.warning("Falha na API (%s). Nova tentativa em %ss.", erro, espera)
            time.sleep(espera)

    raise RuntimeError("inalcançável")  # pragma: no cover


def listar_titulos() -> list[str]:
    """Devolve todos os títulos de subpáginas de ``Livro de receitas/``.

    A API pagina o resultado em blocos de 500; o campo ``continue`` da resposta
    carrega o ponto de retomada até a listagem terminar.
    """
    titulos: list[str] = []
    continuacao: dict[str, str] = {}

    while True:
        resposta = _consultar(
            {
                "action": "query",
                "list": "allpages",
                "apprefix": PREFIXO,
                "aplimit": "500",
                **continuacao,
            }
        )
        titulos.extend(p["title"] for p in resposta["query"]["allpages"])

        if "continue" not in resposta:
            break
        continuacao = resposta["continue"]

    logger.info("Encontrados %d títulos com o prefixo %r.", len(titulos), PREFIXO)
    return titulos


def baixar_wikitext(titulos: list[str]) -> Iterator[dict[str, str]]:
    """Baixa o wikitext de cada título, em lotes, e emite um registro por página.

    Páginas sem revisão (apagadas entre a listagem e a leitura) são puladas.
    """
    for inicio in range(0, len(titulos), LOTE):
        lote = titulos[inicio : inicio + LOTE]
        resposta = _consultar(
            {
                "action": "query",
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "titles": "|".join(lote),
            }
        )

        for pagina in resposta["query"]["pages"]:
            try:
                wikitext = pagina["revisions"][0]["slots"]["main"]["content"]
            except (KeyError, IndexError):
                logger.debug("Sem revisão utilizável: %s", pagina.get("title"))
                continue

            yield {
                "titulo": pagina["title"],
                "url": "https://pt.wikibooks.org/wiki/"
                + urllib.parse.quote(pagina["title"].replace(" ", "_")),
                "wikitext": wikitext,
            }

        logger.info("Baixadas %d/%d páginas.", min(inicio + LOTE, len(titulos)), len(titulos))
        time.sleep(PAUSA_S)


def coletar(destino: Path = DESTINO_PADRAO) -> int:
    """Executa a coleta completa e grava um JSONL. Devolve o total de registros."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    titulos = listar_titulos()

    total = 0
    with destino.open("w", encoding="utf-8") as arquivo:
        for registro in baixar_wikitext(titulos):
            arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
            total += 1

    logger.info("Gravados %d registros em %s.", total, destino)
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    total = coletar()
    print(f"\n{total} páginas coletadas em {DESTINO_PADRAO}")
    print("Licença do conteúdo: CC-BY-SA 3.0 (Wikilivros)")


if __name__ == "__main__":
    main()
