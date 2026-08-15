"""Testes do parsing da saída do modelo.

O modelo devolve texto solto; é `interpretar` que o transforma em campos. Como
a saída de um modelo é imprevisível, o requisito não é só acertar o caso feliz —
é **nunca explodir** diante de saída truncada ou malformada.
"""

from tech_challenge_fase_04.dados.corpus import MODO_INGREDIENTES, MODO_TITULO, formatar
from tech_challenge_fase_04.modelo.geracao import formatar_para_leitura, interpretar

RECEITA = {
    "titulo": "Bolo de fubá",
    "categoria": "Bolos",
    "ingredientes": ["3 ovos", "2 xícaras de fubá", "1 pitada de sal"],
    "preparo": ["Bata os ovos", "Asse por 40 minutos"],
}


class TestRoundTrip:
    """`formatar` e `interpretar` são inversas — se não forem, o playground mente."""

    def test_modo_titulo_recupera_todos_os_campos(self):
        recuperada = interpretar(formatar(RECEITA, MODO_TITULO))
        assert recuperada.titulo == RECEITA["titulo"]
        assert recuperada.categoria == RECEITA["categoria"]
        assert recuperada.ingredientes == RECEITA["ingredientes"]
        assert recuperada.preparo == RECEITA["preparo"]

    def test_modo_ingredientes_recupera_todos_os_campos(self):
        recuperada = interpretar(formatar(RECEITA, MODO_INGREDIENTES))
        assert recuperada.ingredientes == RECEITA["ingredientes"]
        assert recuperada.preparo == RECEITA["preparo"]

    def test_nao_perde_o_primeiro_item_das_listas(self):
        """Regressão: `\\s*` no regex de campo engolia a quebra de linha e
        fazia o cabeçalho absorver o primeiro ingrediente e o primeiro passo."""
        recuperada = interpretar(formatar(RECEITA, MODO_TITULO))
        assert recuperada.ingredientes[0] == "3 ovos"
        assert recuperada.preparo[0] == "Bata os ovos"

    def test_ignora_o_que_vier_depois_do_token_de_fim(self):
        texto = formatar(RECEITA, MODO_TITULO) + "\n<|receita|>\nTÍTULO: Outra coisa"
        assert interpretar(texto).titulo == "Bolo de fubá"


class TestBemFormada:
    def test_receita_completa_e_bem_formada(self):
        assert interpretar(formatar(RECEITA, MODO_TITULO)).bem_formada

    def test_sem_preparo_nao_e_bem_formada(self):
        texto = "TÍTULO: X\nINGREDIENTES:\n- 3 ovos\n- 1 leite\nPREPARO:\n"
        assert not interpretar(texto).bem_formada

    def test_com_um_unico_ingrediente_nao_e_bem_formada(self):
        texto = "TÍTULO: X\nINGREDIENTES:\n- 3 ovos\nPREPARO:\n1. Bata"
        assert not interpretar(texto).bem_formada

    def test_sem_titulo_nao_e_bem_formada(self):
        texto = "INGREDIENTES:\n- 3 ovos\n- 1 leite\nPREPARO:\n1. Bata"
        assert not interpretar(texto).bem_formada


class TestEntradasDegeneradas:
    """Saída de modelo é imprevisível: nada aqui pode levantar exceção."""

    def test_texto_vazio(self):
        assert not interpretar("").bem_formada

    def test_texto_sem_nenhuma_estrutura(self):
        assert not interpretar("blablabla receita gostosa de bolo").bem_formada

    def test_geracao_truncada_no_meio_do_preparo(self):
        texto = "TÍTULO: X\nCATEGORIA: Y\nINGREDIENTES:\n- 3 ovos\n- 1 leite\nPREPARO:\n1. Bata os"
        recuperada = interpretar(texto)
        assert recuperada.ingredientes == ["3 ovos", "1 leite"]
        assert recuperada.preparo == ["Bata os"]

    def test_campos_fora_de_ordem(self):
        texto = "PREPARO:\n1. Bata\nTÍTULO: X\nINGREDIENTES:\n- 3 ovos\n- 1 leite"
        recuperada = interpretar(texto)
        assert recuperada.titulo == "X"
        assert recuperada.preparo == ["Bata"]

    def test_formatar_para_leitura_aceita_receita_vazia(self):
        assert "Sem título" in formatar_para_leitura(interpretar(""))
