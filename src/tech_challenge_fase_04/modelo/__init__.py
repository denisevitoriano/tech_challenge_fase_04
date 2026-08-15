"""Treino, geração e avaliação do modelo generativo de receitas."""

# Modelo pré-treinado que serve de ponto de partida para o ajuste.
MODELO_BASE = "pierreguillou/gpt2-small-portuguese"

# Onde o treino grava o modelo ajustado.
DIRETORIO_MODELO = "modelos/receitas-gpt2-pt"

# O mesmo modelo publicado no Hugging Face Hub. São ~500 MB, que não cabem no
# git — é daqui que o Streamlit Cloud carrega.
MODELO_HUB = "denisevitoriano/receitas-gpt2-pt"
