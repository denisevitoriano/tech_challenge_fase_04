"""Testes da montagem do corpus de treino e do split treino/validação."""

import json

import pytest

from tech_challenge_fase_04.dados.corpus import (
    FIM,
    INICIO,
    MODO_INGREDIENTES,
    MODO_TITULO,
    construir,
    formatar,
    prefixo_livre,
    prefixo_por_ingredientes,
    prefixo_por_titulo,
)

RECEITA = {
    "titulo": "Bolo de fubá",
    "categoria": "Bolos",
    "ingredientes": ["3 ovos", "2 xícaras de fubá"],
    "preparo": ["Bata os ovos", "Asse por 40 minutos"],
    "fonte_titulo": "Livro de receitas/Bolo de fubá",
    "fonte_url": "",
}


class TestFormatar:
    def test_delimita_o_exemplo_com_os_tokens_especiais(self):
        texto = formatar(RECEITA, MODO_TITULO)
        assert texto.startswith(INICIO)
        assert texto.endswith(FIM)

    def test_modo_titulo_poe_categoria_antes_do_titulo(self):
        # A ordem permite condicionar o modo "surpreenda-me" por categoria.
        texto = formatar(RECEITA, MODO_TITULO)
        assert texto.index("CATEGORIA:") < texto.index("TÍTULO:")
        assert texto.index("TÍTULO:") < texto.index("INGREDIENTES:")

    def test_modo_ingredientes_poe_ingredientes_primeiro(self):
        texto = formatar(RECEITA, MODO_INGREDIENTES)
        assert texto.index("INGREDIENTES:") < texto.index("TÍTULO:")

    def test_preparo_sai_numerado(self):
        texto = formatar(RECEITA, MODO_TITULO)
        assert "1. Bata os ovos" in texto
        assert "2. Asse por 40 minutos" in texto

    def test_modo_desconhecido_falha_alto(self):
        with pytest.raises(ValueError, match="modo desconhecido"):
            formatar(RECEITA, "por-telepatia")


class TestPrefixos:
    def test_prefixo_por_titulo_termina_pronto_para_ingredientes(self):
        assert prefixo_por_titulo("Bolo", "Bolos").endswith("INGREDIENTES:\n")

    def test_prefixo_por_ingredientes_termina_pronto_para_categoria(self):
        assert prefixo_por_ingredientes(["3 ovos"]).endswith("CATEGORIA:")

    def test_prefixo_por_ingredientes_descarta_linhas_vazias(self):
        prefixo = prefixo_por_ingredientes(["3 ovos", "  ", "", "1 xícara de leite"])
        assert prefixo.count("- ") == 2

    def test_prefixo_livre_deixa_o_titulo_em_aberto(self):
        prefixo = prefixo_livre("Doces")
        assert "CATEGORIA: Doces" in prefixo
        assert prefixo.endswith("TÍTULO:")

    def test_prefixos_usam_o_mesmo_formato_do_treino(self):
        # O prompt precisa ser prefixo literal do exemplo de treino, senão o
        # modelo vê uma distribuição diferente da que aprendeu.
        treino = formatar(RECEITA, MODO_TITULO)
        assert treino.startswith(prefixo_por_titulo(RECEITA["titulo"], RECEITA["categoria"]))


class TestSplit:
    @pytest.fixture
    def corpus(self, tmp_path):
        """Duas páginas, uma delas com três variações do mesmo prato."""
        receitas = []
        for pagina in range(20):
            for variacao in range(3):
                receitas.append(
                    {
                        **RECEITA,
                        "titulo": f"Prato {pagina} v{variacao}",
                        "fonte_titulo": f"Livro de receitas/Prato {pagina}",
                    }
                )

        origem = tmp_path / "receitas.jsonl"
        origem.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in receitas), encoding="utf-8"
        )
        treino, validacao = tmp_path / "treino.jsonl", tmp_path / "validacao.jsonl"
        construir(origem, treino, validacao)
        return treino, validacao

    def _paginas(self, caminho, receitas_por_titulo):
        titulos = [json.loads(l)["titulo"] for l in caminho.open(encoding="utf-8")]
        return {receitas_por_titulo[t] for t in titulos}

    def test_cada_receita_entra_nos_dois_modos(self, corpus):
        treino, validacao = corpus
        total = sum(1 for _ in treino.open(encoding="utf-8"))
        total += sum(1 for _ in validacao.open(encoding="utf-8"))
        assert total == 20 * 3 * 2  # páginas × variações × modos

    def test_variacoes_da_mesma_pagina_nao_se_dividem(self, corpus):
        """O ponto do split por página: evitar quase-duplicatas entre os conjuntos."""
        treino, validacao = corpus

        def paginas(caminho):
            return {
                json.loads(linha)["titulo"].rsplit(" v", 1)[0]
                for linha in caminho.open(encoding="utf-8")
            }

        assert not (paginas(treino) & paginas(validacao))

    def test_validacao_recebe_aproximadamente_a_proporcao_pedida(self, corpus):
        treino, validacao = corpus
        n_treino = sum(1 for _ in treino.open(encoding="utf-8"))
        n_validacao = sum(1 for _ in validacao.open(encoding="utf-8"))
        assert 0.05 <= n_validacao / (n_treino + n_validacao) <= 0.20
