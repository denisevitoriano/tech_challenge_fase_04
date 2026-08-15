"""Testes da extração de receitas a partir do wikitext do Wikilivros."""

from tech_challenge_fase_04.dados.preparo import extrair, limpar

PAGINA = {
    "titulo": "Livro de receitas/Bolo de fubá",
    "url": "https://pt.wikibooks.org/wiki/Livro_de_receitas/Bolo_de_fubá",
    "wikitext": """{{Navegação|[[../|<<< Índice]]|[[../Bolos/]]}}

Existem várias '''maneiras de preparar:'''

== Bolo de fubá - 1 ==
==='''Ingredientes e Preparo:'''===

* 3 {{w|ovos}} --
* 2 xícaras (chá) de {{w|fubá}};
* 1 pitada de sal.

:'''Preparo:'''

# Bata os ovos com o açúcar até formar um creme.
# Asse em forno médio por 40 minutos.

== Bolo de fubá - 2 ==
=== Ingredientes ===
* 4 ovos;
* 1 xícara de leite;
=== Modo de preparar ===
# Misture tudo no liquidificador.
# Leve ao forno.

{{AutoCat}}
[[Categoria:Bolos|{{SUBPAGENAME}}]]
[[Categoria:Ovo, receitas com|{{SUBPAGENAME}}]]
""",
}


class TestLimpar:
    def test_resolve_template_de_texto_preservando_a_palavra(self):
        # {{w|abacaxi}} é link para a Wikipédia: o conteúdo deve sobreviver.
        assert limpar("1 {{w|abacaxi}} médio") == "1 abacaxi médio"

    def test_template_de_texto_com_rotulo_usa_o_rotulo(self):
        assert limpar("{{w|Açúcar mascavo|açúcar}}") == "açúcar"

    def test_remove_templates_comuns_por_inteiro(self):
        assert limpar("texto {{AutoCat}} fim") == "texto  fim"

    def test_remove_templates_aninhados(self):
        assert limpar("a {{x|{{y}}|z}} b") == "a  b"

    def test_wikilink_com_rotulo_mantem_o_rotulo(self):
        assert limpar("[[w:Cravo-da-índia|Cravo-da-índia]] a gosto") == "Cravo-da-índia a gosto"

    def test_wikilink_simples_mantem_o_alvo(self):
        assert limpar("[[canela]] em pó") == "canela em pó"

    def test_remove_negrito_e_italico(self):
        assert limpar("'''Preparo:''' misture") == "Preparo: misture"


class TestExtrair:
    def test_separa_cada_variacao_da_pagina(self):
        receitas = list(extrair(PAGINA))
        assert len(receitas) == 2

    def test_remove_o_sufixo_numerico_do_titulo(self):
        receitas = list(extrair(PAGINA))
        assert all(r.titulo == "Bolo de fubá" for r in receitas)

    def test_ingredientes_saem_limpos_e_completos(self):
        primeira = next(iter(extrair(PAGINA)))
        assert primeira.ingredientes == [
            "3 ovos",
            "2 xícaras (chá) de fubá",
            "1 pitada de sal",
        ]

    def test_preparo_preserva_a_ordem_das_etapas(self):
        primeira = next(iter(extrair(PAGINA)))
        assert primeira.preparo == [
            "Bata os ovos com o açúcar até formar um creme",
            "Asse em forno médio por 40 minutos",
        ]

    def test_variacao_com_cabecalhos_separados_tambem_e_extraida(self):
        segunda = list(extrair(PAGINA))[1]
        assert segunda.ingredientes == ["4 ovos", "1 xícara de leite"]
        assert segunda.preparo == ["Misture tudo no liquidificador", "Leve ao forno"]

    def test_categoria_ignora_etiqueta_de_ingrediente(self):
        # "Ovo, receitas com" é etiqueta de ingrediente; "Bolos" classifica o prato.
        primeira = next(iter(extrair(PAGINA)))
        assert primeira.categoria == "Bolos"

    def test_guarda_a_procedencia_para_o_split_por_pagina(self):
        for receita in extrair(PAGINA):
            assert receita.fonte_titulo == "Livro de receitas/Bolo de fubá"

    def test_pagina_de_indice_nao_gera_receita(self):
        indice = {
            "titulo": "Livro de receitas/Feijão em geral",
            "url": "",
            "wikitext": "{{Livro de receitas/Lista|imagem=x}}\n{{AutoCat}}",
        }
        assert list(extrair(indice)) == []

    def test_pagina_sem_preparo_e_descartada(self):
        incompleta = {
            "titulo": "Livro de receitas/Vazia",
            "url": "",
            "wikitext": "== Vazia ==\n=== Ingredientes ===\n* 1 ovo;\n* 2 xícaras de leite;",
        }
        assert list(extrair(incompleta)) == []
