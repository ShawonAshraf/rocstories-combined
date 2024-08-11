import os

import wandb
from datasets import load_dataset
from fire import Fire
from loguru import logger

from eval.evaluate import evaluate
from eval.prediction import batch_predict


def run(dataset_size: int, model_name: str):
    # prepare wandb run
    logger.info("Preparing Wandb Init")
    wandb.init(
        project="llm-sentence-ordering",
        config={"model_name": model_name, "dataset_size": dataset_size},
        name=model_name,
    )

    logger.info(f"Model: {model_name} :: Dataset Size: {dataset_size}")

    logger.info("Loading dataset")
    ds = load_dataset(os.getenv("HF_HUB_DATASET_NAME"))
    sampled_ds = ds["train"].select(range(dataset_size))

    gold = [data["gold_order"] for data in sampled_ds]
    llm_pred_results = batch_predict(sampled_ds)

    logger.info("Running Prediction")
    predicted = [p.predicted_order for p in llm_pred_results]

    logger.info("Evaluating")
    eval_results = evaluate(gold, predicted)

    logger.info("Logging results")
    wandb.log(eval_results)
    wandb.finish()


if __name__ == "__main__":
    Fire(run)
