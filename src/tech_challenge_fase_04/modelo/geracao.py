"""Geração de receitas com o modelo ajustado.

Concentra tudo o que o playground e a avaliação precisam: carregar o modelo,
montar o prompt de cada modo, amostrar e devolver a receita já estruturada.

Os três modos espelham os prefixos definidos em
:mod:`tech_challenge_fase_04.dados.corpus`:

``por_titulo``
    O usuário dá o nome do prato e a categoria; o modelo escreve ingredientes
    e preparo.
``por_ingredientes``
    O usuário lista o que tem em casa; o modelo batiza o prato e escreve o
    preparo.
``livre``
    Só a categoria. O modelo inventa o prato inteiro — é o modo que melhor
    demonstra originalidade.

Parâmetros de amostragem
------------------------
``temperature`` achata ou aguça a distribuição; ``top_p`` corta a cauda por
massa acumulada; ``top_k`` corta por posição. Valores altos aumentam
diversidade e reduzem coerência — o notebook de avaliação mede esse
compromisso explicitamente.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tech_challenge_fase_04.dados.corpus import (
    FIM,
    INICIO,
    prefixo_livre,
    prefixo_por_ingredientes,
    prefixo_por_titulo,
)
from tech_challenge_fase_04.modelo import MODELO_HUB


@dataclass
class ReceitaGerada:
    """Resultado de uma geração, já separado em campos."""

    titulo: str = ""
    categoria: str = ""
    ingredientes: list[str] = field(default_factory=list)
    preparo: list[str] = field(default_factory=list)
    texto_bruto: str = ""

    @property
    def bem_formada(self) -> bool:
        """Tem os quatro campos e conteúdo suficiente para valer como receita."""
        return bool(self.titulo) and len(self.ingredientes) >= 2 and len(self.preparo) >= 1


def dispositivo() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=2)
def carregar(caminho: str = MODELO_HUB, device: str | None = None):
    """Carrega modelo e tokenizador, com cache para não recarregar a cada chamada.

    ``caminho`` aceita tanto um diretório local quanto um repositório do Hub —
    o ``from_pretrained`` resolve os dois. O padrão é o modelo publicado, para
    que funcione sem treino local.
    """
    device = device or dispositivo()

    try:
        tokenizador = AutoTokenizer.from_pretrained(caminho)
        modelo = AutoModelForCausalLM.from_pretrained(caminho).to(device)
    except OSError as erro:
        # O erro cru do transformers não distingue "diretório inexistente" de
        # "repositório errado no Hub"; a mensagem abaixo cobre os dois casos.
        local = Path(caminho)
        if not local.is_dir() and local.parent.exists():
            dica = (
                f"O diretório {caminho!r} não existe. Rode o treino antes "
                "(`uv run python -m tech_challenge_fase_04.modelo.treino`) "
                f"ou use o modelo publicado ({MODELO_HUB!r})."
            )
        else:
            dica = f"Confira o nome do repositório {caminho!r} no Hub e a conexão de rede."
        raise FileNotFoundError(f"Não foi possível carregar o modelo. {dica}") from erro

    modelo.eval()
    return modelo, tokenizador, device


# --------------------------------------------------------------------------- #
# Parsing da saída
# --------------------------------------------------------------------------- #

# `[ \t]*` e não `\s*`: `\s` casa com "\n" e faria o cabeçalho engolir a primeira
# linha da lista que vem logo abaixo (o primeiro ingrediente, o primeiro passo).
_CAMPO = re.compile(r"(?m)^(MODO|CATEGORIA|TÍTULO|INGREDIENTES|PREPARO):[ \t]*(.*)$")


def interpretar(texto: str) -> ReceitaGerada:
    """Converte o texto gerado de volta em campos estruturados.

    Tolerante a truncamento: se o modelo parar no meio do preparo, devolve o
    que já veio em vez de falhar. A propriedade ``bem_formada`` é quem decide
    se o resultado é aproveitável.
    """
    corpo = texto.split(FIM)[0].replace(INICIO, "").strip()

    receita = ReceitaGerada(texto_bruto=corpo)
    marcadores = list(_CAMPO.finditer(corpo))

    for indice, marcador in enumerate(marcadores):
        campo, resto_da_linha = marcador.group(1), marcador.group(2).strip()
        fim = marcadores[indice + 1].start() if indice + 1 < len(marcadores) else len(corpo)
        bloco = corpo[marcador.end() : fim]

        if campo == "TÍTULO":
            receita.titulo = resto_da_linha
        elif campo == "CATEGORIA":
            receita.categoria = resto_da_linha
        elif campo == "INGREDIENTES":
            receita.ingredientes = [
                item.lstrip("-").strip()
                for linha in bloco.splitlines()
                if (item := linha.strip()).startswith("-") and len(item) > 2
            ]
        elif campo == "PREPARO":
            receita.preparo = [
                re.sub(r"^\d+\.\s*", "", passo).strip()
                for linha in bloco.splitlines()
                if (passo := linha.strip()) and re.match(r"^\d+\.", passo)
            ]

    return receita


# --------------------------------------------------------------------------- #
# Geração
# --------------------------------------------------------------------------- #


def gerar_de_prompt(
    prompt: str,
    *,
    n: int = 1,
    temperature: float = 0.9,
    top_p: float = 0.92,
    top_k: int = 50,
    repetition_penalty: float = 1.15,
    max_new_tokens: int = 320,
    seed: int | None = None,
    caminho: str = MODELO_HUB,
) -> list[ReceitaGerada]:
    """Amostra ``n`` continuações para um prompt já montado."""
    modelo, tokenizador, device = carregar(caminho)

    if seed is not None:
        torch.manual_seed(seed)

    entradas = tokenizador(prompt, return_tensors="pt").to(device)
    id_fim = tokenizador.convert_tokens_to_ids(FIM)

    with torch.no_grad():
        saidas = modelo.generate(
            **entradas,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            eos_token_id=id_fim,
            pad_token_id=tokenizador.pad_token_id,
        )

    return [interpretar(tokenizador.decode(saida, skip_special_tokens=False)) for saida in saidas]


def por_titulo(titulo: str, categoria: str = "Geral", **parametros) -> list[ReceitaGerada]:
    """Gera a receita de um prato nomeado pelo usuário."""
    return gerar_de_prompt(prefixo_por_titulo(titulo, categoria), **parametros)


def por_ingredientes(ingredientes: list[str], **parametros) -> list[ReceitaGerada]:
    """Inventa uma receita a partir de uma lista de ingredientes."""
    return gerar_de_prompt(prefixo_por_ingredientes(ingredientes), **parametros)


def livre(categoria: str = "Geral", **parametros) -> list[ReceitaGerada]:
    """Modo "surpreenda-me": o modelo inventa prato e receita dentro da categoria."""
    return gerar_de_prompt(prefixo_livre(categoria), **parametros)


def formatar_para_leitura(receita: ReceitaGerada) -> str:
    """Renderiza a receita em markdown, para o playground e para o notebook."""
    partes = [f"## {receita.titulo or 'Sem título'}"]
    if receita.categoria:
        partes.append(f"*{receita.categoria}*")
    if receita.ingredientes:
        partes.append("\n**Ingredientes**\n")
        partes.extend(f"- {item}" for item in receita.ingredientes)
    if receita.preparo:
        partes.append("\n**Modo de preparo**\n")
        partes.extend(f"{n}. {passo}" for n, passo in enumerate(receita.preparo, start=1))
    return "\n".join(partes)
