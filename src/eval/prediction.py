import os
from typing import List

import instructor
from datasets import Dataset
from openai import OpenAI
from tqdm.auto import tqdm

from eval.utils import PredictionResult, format_inputs


def predict(client: OpenAI, inp: str, model_name: str) -> PredictionResult:
    system_message = """
    You are given 5 sentences from a story in a shuffled order. 
    Each sentence has a serial number assoiciated with it. 
    You can find the serial number at the beginning of each sentence, for example: 0. <sentence>.
    The serial numbers start from 0 and ends at 4.
    Your task is to predict the correct order of sentences which would resemble the original story and then return the serial numbers as a list of integers, which contain the serial numbers.

    #IMPORTANT
    There are exactly five sentences. 
    """

    res = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": inp},
        ],
        temperature=0.0,
        response_model=PredictionResult,
    )

    return res


def batch_predict(ds: Dataset) -> List[PredictionResult]:
    client = instructor.from_openai(
        OpenAI(api_key="token-abc123", base_url=os.getenv("VLLM_ENDPOINT_URL"))
    )

    predicted = list()
    formatted_inputs = format_inputs(ds)

    for _, inp in tqdm(enumerate(formatted_inputs)):
        res = predict(client, inp)
        predicted.append(res)

    return predicted
