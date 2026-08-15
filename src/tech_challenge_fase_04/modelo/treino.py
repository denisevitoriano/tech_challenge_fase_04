"""Fine-tuning do ``pierreguillou/gpt2-small-portuguese`` no corpus de receitas.

O modelo base é um GPT-2 small (124M parâmetros) pré-treinado em português.
Ele já sabe a língua, mas não conhece o *formato* de uma receita nem o
vocabulário culinário — é exatamente isso que o ajuste ensina.

Decisões de treino
------------------
``max_length=512``
    Cobre 97% dos exemplos do corpus. Subir para 768 cobriria 99,8%, mas
    encareceria cada passo em ~50% para ganhar 3% de dados.

Tokens especiais
    ``<|receita|>`` e ``<|fim|>`` delimitam o exemplo. O ``<|fim|>`` vira o
    ``eos_token_id`` da configuração de geração, de modo que o modelo aprende
    a encerrar a receita sozinho — sem isso, o playground geraria texto até
    estourar o limite de tokens.

``<|pad|>`` separado do ``<|endoftext|>``
    Se o padding usasse o mesmo id do fim de texto, o *collator* mascararia
    também os tokens de fim reais, e o modelo nunca aprenderia a parar.

Batch efetivo 16 (4 × 4 passos de acumulação)
    Batch 4 é o que cabe com folga em 8 GB de RAM unificada; a acumulação
    recupera a estabilidade de gradiente de um batch maior.

Uso::

    uv run python -m tech_challenge_fase_04.modelo.treino
    uv run python -m tech_challenge_fase_04.modelo.treino --epocas 4 --lr 3e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from tech_challenge_fase_04.dados.corpus import FIM, INICIO
from tech_challenge_fase_04.modelo import DIRETORIO_MODELO, MODELO_BASE

TREINO = Path("data/processed/treino.jsonl")
VALIDACAO = Path("data/processed/validacao.jsonl")

TOKEN_PADDING = "<|pad|>"
MAX_LENGTH = 512

logger = logging.getLogger(__name__)


def _dispositivo() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def carregar_jsonl(caminho: Path) -> Dataset:
    linhas = [json.loads(linha) for linha in caminho.open(encoding="utf-8")]
    return Dataset.from_list([{"texto": linha["texto"]} for linha in linhas])


def preparar_tokenizador(base: str = MODELO_BASE):
    """Carrega o tokenizador e registra os tokens especiais do formato."""
    tokenizador = AutoTokenizer.from_pretrained(base)
    tokenizador.add_special_tokens(
        {
            "additional_special_tokens": [INICIO, FIM],
            "pad_token": TOKEN_PADDING,
        }
    )
    return tokenizador


def treinar(
    epocas: float = 3.0,
    lr: float = 5e-5,
    batch: int = 4,
    acumulacao: int = 4,
    saida: str = DIRETORIO_MODELO,
    fp16: bool | None = None,
) -> dict[str, float]:
    """Executa o fine-tuning e salva o modelo. Devolve as métricas finais.

    ``fp16`` só é aplicado em GPU NVIDIA; em CPU e MPS a precisão mista ou não
    existe ou degrada o desempenho. Deixe ``None`` para decidir pelo hardware.
    """
    dispositivo = _dispositivo()
    if fp16 is None:
        fp16 = dispositivo == "cuda"
    logger.info("Dispositivo: %s | fp16: %s", dispositivo, fp16)

    tokenizador = preparar_tokenizador()
    modelo = AutoModelForCausalLM.from_pretrained(MODELO_BASE)
    # Os tokens novos exigem linhas novas na matriz de embeddings.
    modelo.resize_token_embeddings(len(tokenizador))

    id_fim = tokenizador.convert_tokens_to_ids(FIM)
    modelo.generation_config.eos_token_id = id_fim
    modelo.generation_config.pad_token_id = tokenizador.pad_token_id
    modelo.config.pad_token_id = tokenizador.pad_token_id

    def tokenizar(lote: dict[str, list[str]]) -> dict:
        return tokenizador(lote["texto"], truncation=True, max_length=MAX_LENGTH)

    treino = carregar_jsonl(TREINO).map(tokenizar, batched=True, remove_columns=["texto"])
    validacao = carregar_jsonl(VALIDACAO).map(tokenizar, batched=True, remove_columns=["texto"])
    logger.info("Exemplos: %d de treino, %d de validação.", len(treino), len(validacao))

    # O transformers 5.x removeu `warmup_ratio`; o equivalente é calcular os
    # passos a partir do total planejado (5% do treino em aquecimento).
    passos_por_epoca = math.ceil(len(treino) / (batch * acumulacao))
    warmup = max(10, int(passos_por_epoca * epocas * 0.05))

    argumentos = TrainingArguments(
        output_dir=f"{saida}/checkpoints",
        num_train_epochs=epocas,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=acumulacao,
        learning_rate=lr,
        warmup_steps=warmup,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        report_to="none",
        fp16=fp16,
        seed=42,
    )

    treinador = Trainer(
        model=modelo,
        args=argumentos,
        train_dataset=treino,
        eval_dataset=validacao,
        data_collator=DataCollatorForLanguageModeling(tokenizador, mlm=False),
    )

    treinador.train()
    metricas = treinador.evaluate()
    metricas["perplexidade"] = math.exp(metricas["eval_loss"])
    logger.info(
        "Validação final — loss %.4f | perplexidade %.2f",
        metricas["eval_loss"],
        metricas["perplexidade"],
    )

    destino = Path(saida)
    destino.mkdir(parents=True, exist_ok=True)
    treinador.save_model(str(destino))
    tokenizador.save_pretrained(str(destino))
    (destino / "metricas_treino.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return metricas


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning do gerador de receitas.")
    parser.add_argument("--epocas", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--acumulacao", type=int, default=4)
    parser.add_argument("--saida", default=DIRETORIO_MODELO)
    parser.add_argument("--fp16", action="store_true", default=None)
    argumentos = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metricas = treinar(
        epocas=argumentos.epocas,
        lr=argumentos.lr,
        batch=argumentos.batch,
        acumulacao=argumentos.acumulacao,
        saida=argumentos.saida,
        fp16=argumentos.fp16,
    )
    print(f"\nModelo salvo em {argumentos.saida}")
    print(f"perplexidade de validação: {metricas['perplexidade']:.2f}")


if __name__ == "__main__":
    main()
