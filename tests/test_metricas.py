"""Testes das métricas de originalidade, diversidade e estrutura.

A perplexidade fica de fora: exige baixar um modelo, e o que ela faz de
não-trivial (ponderar pelo número de tokens) é verificado no notebook de
avaliação, onde há modelo carregado.
"""

from dataclasses import dataclass, field

import pytest

from tech_challenge_fase_04.avaliacao.metricas import (
    distinct_n,
    indexar_ngramas,
    maior_trecho_copiado,
    ngramas,
    repeticao_entre_amostras,
    taxa_de_copia,
    tokenizar_palavras,
    validade_estrutural,
)

CORPUS = ["bata os ovos com o açúcar até formar um creme homogêneo e reserve"]


@dataclass
class ReceitaFalsa:
    """Dublê de `ReceitaGerada` — evita depender do módulo de geração."""

    titulo: str = "Bolo"
    ingredientes: list[str] = field(default_factory=lambda: ["a", "b"])
    preparo: list[str] = field(default_factory=lambda: ["passo"])

    @property
    def bem_formada(self) -> bool:
        return bool(self.titulo) and len(self.ingredientes) >= 2 and len(self.preparo) >= 1


class TestTokenizacao:
    def test_separa_por_palavra_e_normaliza_caixa(self):
        assert tokenizar_palavras("Bata os OVOS!") == ["bata", "os", "ovos"]

    def test_preserva_acentos(self):
        assert "açúcar" in tokenizar_palavras("com açúcar")

    def test_ngramas_deslizam_de_um_em_um(self):
        assert ngramas(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]

    def test_ngrama_maior_que_o_texto_devolve_vazio(self):
        assert ngramas(["a"], 3) == []


class TestOriginalidade:
    def test_texto_identico_ao_corpus_tem_copia_total(self):
        indice = indexar_ngramas(CORPUS, n=5)
        assert taxa_de_copia(CORPUS[0], indice, n=5) == 1.0

    def test_texto_sem_relacao_nao_tem_copia(self):
        indice = indexar_ngramas(CORPUS, n=5)
        assert taxa_de_copia("frite a cebola no azeite quente com alho", indice, n=5) == 0.0

    def test_texto_curto_demais_nao_gera_divisao_por_zero(self):
        indice = indexar_ngramas(CORPUS, n=5)
        assert taxa_de_copia("bata os", indice, n=5) == 0.0

    def test_maior_trecho_copiado_encontra_a_extensao_certa(self):
        corpus_tokens = [tokenizar_palavras(CORPUS[0])]
        # Seis palavras do corpus, cercadas de texto novo dos dois lados.
        texto = "primeiro bata os ovos com o açúcar depois adicione a farinha peneirada"
        assert maior_trecho_copiado(texto, corpus_tokens) == 6

    def test_maior_trecho_copiado_em_texto_novo_e_pequeno(self):
        corpus_tokens = [tokenizar_palavras(CORPUS[0])]
        assert maior_trecho_copiado("frite a cebola no azeite", corpus_tokens) == 0

    def test_maior_trecho_copiado_aceita_texto_vazio(self):
        assert maior_trecho_copiado("", [tokenizar_palavras(CORPUS[0])]) == 0


class TestDiversidade:
    def test_texto_sem_repeticao_tem_distinct_maximo(self):
        assert distinct_n(["a b c d"], n=2) == 1.0

    def test_amostras_identicas_derrubam_o_distinct(self):
        assert distinct_n(["a b", "a b", "a b"], n=2) == pytest.approx(1 / 3)

    def test_distinct_de_lista_vazia_e_zero(self):
        assert distinct_n([], n=2) == 0.0

    def test_amostras_identicas_tem_repeticao_total(self):
        assert repeticao_entre_amostras(["bata os ovos", "bata os ovos"]) == 1.0

    def test_amostras_sem_vocabulario_comum_nao_se_repetem(self):
        assert repeticao_entre_amostras(["bata os ovos", "frite a cebola"]) == 0.0

    def test_uma_amostra_so_nao_tem_par_para_comparar(self):
        assert repeticao_entre_amostras(["bata os ovos"]) == 0.0


class TestEstrutura:
    def test_todas_bem_formadas(self):
        resumo = validade_estrutural([ReceitaFalsa(), ReceitaFalsa()])
        assert resumo["taxa_bem_formada"] == 1.0

    def test_metade_bem_formada(self):
        resumo = validade_estrutural([ReceitaFalsa(), ReceitaFalsa(preparo=[])])
        assert resumo["taxa_bem_formada"] == 0.5

    def test_lista_vazia_nao_divide_por_zero(self):
        assert validade_estrutural([])["taxa_bem_formada"] == 0.0

    def test_detecta_titulos_repetidos(self):
        # Sinal de modelo preso: inventa sempre o mesmo prato.
        resumo = validade_estrutural(
            [ReceitaFalsa(titulo="Bolo"), ReceitaFalsa(titulo="Bolo"), ReceitaFalsa(titulo="Torta")]
        )
        assert resumo["taxa_titulo_repetido"] == pytest.approx(2 / 3)

    def test_titulos_todos_distintos(self):
        resumo = validade_estrutural([ReceitaFalsa(titulo="A"), ReceitaFalsa(titulo="B")])
        assert resumo["taxa_titulo_repetido"] == 0.0
