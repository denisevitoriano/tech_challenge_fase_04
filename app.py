"""Playground do gerador de receitas — Tech Challenge Fase 04.

Aplicação Streamlit para testar o poder generativo do GPT-2 ajustado.
Roda local (`uv run streamlit run app.py`) e no Streamlit Community Cloud.

Funciona sem configuração: na falta de um modelo treinado localmente, carrega o
publicado no Hugging Face Hub. Ver :func:`origem_do_modelo` para a ordem de
precedência completa.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import streamlit as st

# O projeto usa layout `src/`, e o Streamlit Cloud roda o app sem instalar o
# pacote. Colocar `src/` no path evita depender de `pip install -e .` no deploy.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tech_challenge_fase_04.avaliacao.metricas import indexar_ngramas, taxa_de_copia
from tech_challenge_fase_04.modelo import DIRETORIO_MODELO, MODELO_HUB
from tech_challenge_fase_04.modelo.geracao import (
    ReceitaGerada,
    formatar_para_leitura,
    livre,
    por_ingredientes,
    por_titulo,
)

CORPUS_TREINO = Path("data/processed/treino.jsonl")
RECEITAS = Path("data/processed/receitas.jsonl")

st.set_page_config(page_title="Cozinheiro Artificial", page_icon="🍲", layout="wide")


# --------------------------------------------------------------------------- #
# Carregamento com cache
# --------------------------------------------------------------------------- #


def origem_do_modelo() -> str:
    """De onde carregar o modelo, na ordem de precedência.

    1. variável de ambiente ``MODELO_HF`` — útil para testar outro modelo sem
       tocar em arquivo nenhum;
    2. secret ``MODELO_HF`` — o override do Streamlit Cloud;
    3. o diretório local de treino, se existir — quem acabou de treinar quer
       testar o próprio resultado, não o que está publicado;
    4. o modelo no Hub.

    O passo 4 é o que faz o app funcionar sem nenhuma configuração: o deploy
    sobe direto e quem clonar o repositório roda com um comando só.
    """
    if do_ambiente := os.environ.get("MODELO_HF"):
        return do_ambiente

    try:
        if "MODELO_HF" in st.secrets:
            return st.secrets["MODELO_HF"]
    except FileNotFoundError:
        pass  # sem secrets.toml em execução local

    if Path(DIRETORIO_MODELO).is_dir():
        return DIRETORIO_MODELO

    return MODELO_HUB


@st.cache_resource(show_spinner="Carregando o modelo...")
def indice_de_copia() -> set[tuple[str, ...]] | None:
    """Índice de 5-gramas do treino, para medir originalidade em tempo real."""
    if not CORPUS_TREINO.exists():
        return None
    textos = (json.loads(linha)["texto"] for linha in CORPUS_TREINO.open(encoding="utf-8"))
    return indexar_ngramas(textos, n=5)


def diagnosticar(receita: ReceitaGerada, modo: str) -> str | None:
    """Aviso a mostrar quando a geração saiu problemática, ou ``None``.

    Não usa ``receita.bem_formada`` de propósito. Aquela propriedade é a métrica
    da avaliação, e exige dois ingredientes — critério que faz sentido para medir
    o modelo, mas não para avisar o usuário: no modo "Pelos ingredientes" a lista
    vem de quem digitou, e reclamar dela seria culpar o modelo por uma escolha
    de quem está usando o app.
    """
    if not receita.preparo:
        return (
            "O modelo não chegou a escrever o modo de preparo. "
            "Aumente o tamanho máximo em tokens, ou gere de novo."
        )
    if not receita.titulo:
        return "O modelo não nomeou o prato. Vale gerar de novo."
    if len(receita.ingredientes) < 2 and modo != "Pelos ingredientes":
        return (
            "O modelo listou um ingrediente só. Gerar de novo costuma resolver — "
            "receitas curtas são uma limitação conhecida deste modelo."
        )
    return None


@st.cache_data
def categorias_disponiveis() -> list[str]:
    """Categorias vistas no treino, ordenadas por frequência."""
    if not RECEITAS.exists():
        return ["Geral"]
    contagem: dict[str, int] = {}
    for linha in RECEITAS.open(encoding="utf-8"):
        categoria = json.loads(linha)["categoria"]
        contagem[categoria] = contagem.get(categoria, 0) + 1
    return [c for c, _ in sorted(contagem.items(), key=lambda x: -x[1])]


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #

st.title("🍲 Cozinheiro Artificial")
st.caption(
    "GPT-2 em português ajustado em 2.506 receitas do Wikilivros. "
    "Tudo o que aparece abaixo é inventado pelo modelo — confira antes de cozinhar."
)

with st.sidebar:
    st.header("Parâmetros de geração")
    st.caption("Controlam o compromisso entre coerência e criatividade.")

    temperature = st.slider(
        "Temperatura",
        0.1,
        1.5,
        1.1,
        0.05,
        help="Baixa: previsível e repetitivo. Alta: criativo e caótico.",
    )
    top_p = st.slider(
        "top-p (nucleus)",
        0.1,
        1.0,
        0.92,
        0.01,
        help="Considera só os tokens mais prováveis que somem esta massa.",
    )
    top_k = st.slider(
        "top-k",
        0,
        200,
        50,
        5,
        help="Limita o sorteio aos k tokens mais prováveis. 0 desativa.",
    )
    penalidade = st.slider(
        "Penalidade de repetição",
        1.0,
        1.6,
        1.15,
        0.05,
        help="Acima de 1 desencoraja repetir o que já foi escrito.",
    )
    max_tokens = st.slider("Tamanho máximo (tokens)", 80, 512, 320, 20)
    amostras = st.slider("Quantas receitas gerar", 1, 4, 1)

    usar_semente = st.checkbox("Fixar semente (reprodutível)", value=False)
    semente = st.number_input("Semente", 0, 10**6, 42, disabled=not usar_semente)

    st.divider()
    st.caption(f"Modelo: `{origem_do_modelo()}`")

modo = st.radio(
    "Como você quer gerar?",
    ["Pelo título", "Pelos ingredientes", "Surpreenda-me"],
    horizontal=True,
)

categorias = categorias_disponiveis()
parametros = {
    "n": amostras,
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "repetition_penalty": penalidade,
    "max_new_tokens": max_tokens,
    "seed": int(semente) if usar_semente else None,
    "caminho": origem_do_modelo(),
}

receitas: list[ReceitaGerada] = []

if modo == "Pelo título":
    coluna_titulo, coluna_categoria = st.columns([3, 2])
    titulo = coluna_titulo.text_input("Nome do prato", "Bolo de banana com canela")
    categoria = coluna_categoria.selectbox("Categoria", categorias)
    if st.button("Gerar receita", type="primary"):
        with st.spinner("Cozinhando..."):
            receitas = por_titulo(titulo, categoria, **parametros)

elif modo == "Pelos ingredientes":
    texto_ingredientes = st.text_area(
        "O que você tem em casa? (um ingrediente por linha)",
        "2 ovos\n1 xícara de farinha de trigo\n1 lata de leite condensado",
        height=140,
    )
    if st.button("Gerar receita", type="primary"):
        itens = [linha for linha in texto_ingredientes.splitlines() if linha.strip()]
        with st.spinner("Cozinhando..."):
            receitas = por_ingredientes(itens, **parametros)

else:
    categoria = st.selectbox("Categoria", categorias)
    if st.button("Inventar um prato", type="primary"):
        if not usar_semente:
            parametros["seed"] = random.randint(0, 10**6)
        with st.spinner("Inventando..."):
            receitas = livre(categoria, **parametros)


# --------------------------------------------------------------------------- #
# Resultados
# --------------------------------------------------------------------------- #

if receitas:
    indice = indice_de_copia()

    for numero, receita in enumerate(receitas, start=1):
        if len(receitas) > 1:
            st.divider()
            st.subheader(f"Receita {numero}")

        if aviso := diagnosticar(receita, modo):
            st.warning(aviso)

        coluna_receita, coluna_metricas = st.columns([3, 1])
        coluna_receita.markdown(formatar_para_leitura(receita))

        with coluna_metricas:
            st.metric("Ingredientes", len(receita.ingredientes))
            st.metric("Etapas", len(receita.preparo))

            if indice is not None:
                copia = taxa_de_copia(receita.texto_bruto, indice, n=5)
                st.metric(
                    "Trechos vindos do treino",
                    f"{copia:.0%}",
                    help=(
                        "Fração das sequências de 5 palavras que já existiam no "
                        "corpus. Algum overlap é natural — são as formas fixas da "
                        "língua. Perto de 100% indicaria memorização."
                    ),
                )

        with st.expander("Texto bruto gerado pelo modelo"):
            st.code(receita.texto_bruto, language=None)

    st.download_button(
        "Baixar receitas (.md)",
        "\n\n---\n\n".join(formatar_para_leitura(r) for r in receitas),
        file_name="receitas_geradas.md",
        mime="text/markdown",
    )
