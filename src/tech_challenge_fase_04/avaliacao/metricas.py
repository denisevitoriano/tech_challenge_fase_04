"""Métricas para avaliar o gerador de receitas.

A prova pede para "avaliar a qualidade e originalidade do conteúdo gerado".
Isso é decomposto aqui em quatro eixos independentes — nenhum deles sozinho
conta a história inteira:

Qualidade (:func:`perplexidade`)
    Quão bem o modelo prevê receitas que nunca viu. Comparar o modelo base
    com o ajustado quantifica o efeito do fine-tuning.

Originalidade (:func:`taxa_de_copia`, :func:`maior_trecho_copiado`)
    Um modelo que decorou o corpus teria perplexidade ótima e originalidade
    nula. Estas métricas medem o quanto do texto gerado é literalmente
    recortado do treino.

Diversidade (:func:`distinct_n`, :func:`repeticao_entre_amostras`)
    Um modelo pode gerar texto original e ainda assim produzir sempre a
    *mesma* receita. Aqui se mede a variedade entre amostras.

Estrutura (:func:`validade_estrutural`)
    Fração das gerações que saem com título, ingredientes e preparo
    utilizáveis. É a métrica mais próxima da experiência real do playground.

Qualidade e originalidade puxam em direções opostas: baixar a temperatura
melhora a coerência e aumenta a cópia. O notebook de avaliação mapeia essa
fronteira em vez de reportar um número único.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

import torch

PALAVRA = re.compile(r"\w+", re.UNICODE)


def tokenizar_palavras(texto: str) -> list[str]:
    """Tokenização simples por palavra, usada nas métricas de n-grama."""
    return PALAVRA.findall(texto.lower())


def ngramas(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# --------------------------------------------------------------------------- #
# Qualidade
# --------------------------------------------------------------------------- #


def perplexidade(
    modelo,
    tokenizador,
    textos: Iterable[str],
    *,
    max_length: int = 512,
    device: str | None = None,
) -> float:
    """Perplexidade média por token, ponderada pelo tamanho de cada texto.

    A ponderação importa: fazer a média das perplexidades de cada texto daria
    peso igual a uma receita de 50 e a uma de 500 tokens. O correto é somar as
    log-verossimilhanças e dividir pelo total de tokens previstos.
    """
    device = device or next(modelo.parameters()).device
    modelo.eval()

    soma_log_verossimilhanca = 0.0
    total_tokens = 0

    with torch.no_grad():
        for texto in textos:
            ids = tokenizador(
                texto, return_tensors="pt", truncation=True, max_length=max_length
            ).input_ids.to(device)

            # Um único token não produz nenhuma predição.
            if ids.shape[1] < 2:
                continue

            perda = modelo(input_ids=ids, labels=ids).loss.item()
            previstos = ids.shape[1] - 1  # o primeiro token não é previsto
            soma_log_verossimilhanca += perda * previstos
            total_tokens += previstos

    if not total_tokens:
        return float("nan")
    return math.exp(soma_log_verossimilhanca / total_tokens)


# --------------------------------------------------------------------------- #
# Originalidade
# --------------------------------------------------------------------------- #


def indexar_ngramas(corpus: Iterable[str], n: int = 5) -> set[tuple[str, ...]]:
    """Constrói o conjunto de n-gramas do corpus de treino, para busca de cópia."""
    indice: set[tuple[str, ...]] = set()
    for texto in corpus:
        indice.update(ngramas(tokenizar_palavras(texto), n))
    return indice


def taxa_de_copia(texto: str, indice: set[tuple[str, ...]], n: int = 5) -> float:
    """Fração dos n-gramas do texto gerado que já existem no corpus de treino.

    Perto de 1 indica recorte literal; perto de 0, composição nova. Algum
    overlap é esperado e desejável — "leve ao forno por 30 minutos" é a forma
    natural de dizer aquilo em português.
    """
    grams = ngramas(tokenizar_palavras(texto), n)
    if not grams:
        return 0.0
    return sum(1 for g in grams if g in indice) / len(grams)


def maior_trecho_copiado(texto: str, corpus_tokens: list[list[str]]) -> int:
    """Comprimento (em palavras) do maior trecho contíguo presente no treino.

    Complementa :func:`taxa_de_copia`: um texto pode ter overlap baixo no
    agregado e ainda assim conter um parágrafo inteiro decorado.
    """
    tokens = tokenizar_palavras(texto)
    if not tokens:
        return 0

    # Busca binária no comprimento: se nenhum trecho de tamanho k foi copiado,
    # nenhum trecho maior foi. Evita varrer todos os tamanhos possíveis.
    indices_por_tamanho: dict[int, set[tuple[str, ...]]] = {}

    def existe(k: int) -> bool:
        if k not in indices_por_tamanho:
            indice: set[tuple[str, ...]] = set()
            for doc in corpus_tokens:
                indice.update(ngramas(doc, k))
            indices_por_tamanho[k] = indice
        return any(g in indices_por_tamanho[k] for g in ngramas(tokens, k))

    baixo, alto = 0, min(len(tokens), 60)
    while baixo < alto:
        meio = (baixo + alto + 1) // 2
        if existe(meio):
            baixo = meio
        else:
            alto = meio - 1
    return baixo


# --------------------------------------------------------------------------- #
# Diversidade
# --------------------------------------------------------------------------- #


def distinct_n(textos: Sequence[str], n: int = 2) -> float:
    """Razão entre n-gramas únicos e n-gramas totais no conjunto gerado.

    Métrica clássica de diversidade lexical (Li et al., 2016). Valor baixo
    denuncia um modelo preso em muletas — "misture bem", "leve ao forno".
    """
    todos: list[tuple[str, ...]] = []
    for texto in textos:
        todos.extend(ngramas(tokenizar_palavras(texto), n))
    if not todos:
        return 0.0
    return len(set(todos)) / len(todos)


def repeticao_entre_amostras(textos: Sequence[str]) -> float:
    """Similaridade de Jaccard média entre todos os pares de gerações.

    Alternativa barata ao self-BLEU: 0 significa amostras sem vocabulário em
    comum, 1 significa que o modelo gerou sempre a mesma coisa.
    """
    conjuntos = [set(tokenizar_palavras(texto)) for texto in textos]
    conjuntos = [c for c in conjuntos if c]
    if len(conjuntos) < 2:
        return 0.0

    soma = 0.0
    pares = 0
    for i in range(len(conjuntos)):
        for j in range(i + 1, len(conjuntos)):
            uniao = conjuntos[i] | conjuntos[j]
            soma += len(conjuntos[i] & conjuntos[j]) / len(uniao)
            pares += 1
    return soma / pares


# --------------------------------------------------------------------------- #
# Estrutura
# --------------------------------------------------------------------------- #


def validade_estrutural(receitas: Sequence) -> dict[str, float]:
    """Resume quão bem-formadas saíram as gerações.

    Recebe objetos ``ReceitaGerada`` (de :mod:`~tech_challenge_fase_04.modelo.geracao`).
    """
    if not receitas:
        return {"taxa_bem_formada": 0.0, "media_ingredientes": 0.0, "media_passos": 0.0}

    bem_formadas = [r for r in receitas if r.bem_formada]
    return {
        "taxa_bem_formada": len(bem_formadas) / len(receitas),
        "media_ingredientes": sum(len(r.ingredientes) for r in receitas) / len(receitas),
        "media_passos": sum(len(r.preparo) for r in receitas) / len(receitas),
        "taxa_titulo_repetido": _taxa_titulo_repetido(receitas),
    }


def _taxa_titulo_repetido(receitas: Sequence) -> float:
    """Fração de gerações cujo título não é único no conjunto."""
    titulos = [r.titulo.strip().lower() for r in receitas if r.titulo.strip()]
    if not titulos:
        return 0.0
    contagem = Counter(titulos)
    repetidos = sum(q for q in contagem.values() if q > 1)
    return repetidos / len(titulos)
