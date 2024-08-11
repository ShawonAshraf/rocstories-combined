from typing import List

from datasets import Dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from scipy.stats import kendalltau
from tqdm.auto import tqdm

load_dotenv(override=True)


class PredictionResult(BaseModel):
    predicted_order: List[int] = Field(
        "A list of integers in the range 0 upto 4, which contains the serial numbers of sentences in the correct order."
    )


def format_inputs(ds: Dataset) -> List[str]:
    formatted_inputs = []
    for _, data in tqdm(enumerate(ds)):
        ins = ""
        for idx, sent in enumerate(data["shuffled_sentences"]):
            ins += f"{idx}. {sent}\n"
        formatted_inputs.append(ins)

    return formatted_inputs


def overlapping_accuracy(gold, predicted):
    assert len(gold) == len(predicted)

    overlaps = 0
    for idx in range(len(gold)):
        if gold[idx] == predicted[idx]:
            overlaps += 1

    return overlaps / len(gold)


def evaluate(gold: List[int], predicted: List[int]):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
    tau = kendalltau(x=gold, y=predicted)
    acc = overlapping_accuracy(gold, predicted)

    return {
        "tau_stat": tau.statistic,
        "tau_p": tau.pvalue,
        "correlation": tau.correlation,
        "overlap_accuracy": acc,
    }
