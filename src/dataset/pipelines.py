import os
import random
from typing import List

import polars as pl
from datasets import Dataset
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

load_dotenv(override=True)


class Story(BaseModel):
    """
    Pydantic model representing a story with its title, sentences, shuffled sentences, and the gold order of sentences.

    Attributes:
        title (str): The title of the story.
        sentences (List[str]): The sentences in the story.
        shuffled_sentences (List[str]): The shuffled sentences in the story.
        gold_order (List[int]): The indexes of the sentences from shuffled_sentences. To be used as a gold label for prediction.
    """

    title: str = Field(..., description="The title of the story")
    sentences: List[str] = Field(..., description="The sentences in the story")
    shuffled_sentences: List[str] = Field(
        ..., description="The shuffled sentences in the story"
    )
    gold_order: List[int] = Field(
        ...,
        description="The indexes of the sentences from shuffled_sentences. To be used as a gold label for prediction.",
    )


def read_data_from_csv() -> pl.DataFrame:
    """
    Reads data from CSV files specified in environment variables and concatenates them into a single DataFrame.

    This function reads two CSV files whose paths are defined by the environment variables `DF16_PATH` and `DF17_PATH`.
    It uses the Polars library to read the CSV files into DataFrames and then concatenates these DataFrames into a single DataFrame.

    Returns:
        pl.DataFrame: A concatenated DataFrame containing the data from both CSV files.
    """

    assert os.getenv("DF16_PATH") is not None, "DF16_PATH is not set"
    assert os.getenv("DF17_PATH") is not None, "DF17_PATH is not set"

    logger.info(f"Reading data from {os.getenv('DF16_PATH')}")
    df16 = pl.read_csv(os.getenv("DF16_PATH"))
    logger.info(f"Reading data from {os.getenv('DF17_PATH')}")
    df17 = pl.read_csv(os.getenv("DF17_PATH"))
    return pl.concat([df16, df17])


def drop_columns(
    df: pl.DataFrame, column_names: List[str] = ["storyid"]
) -> pl.DataFrame:
    logger.info(f"Dropping columns: {column_names}")
    df = df.drop(column_names)
    return df


def create_dataset(df: pl.DataFrame) -> List[Story]:
    """
    Creates a dataset from a Polars DataFrame.

    This function processes each row of the DataFrame to create a list of `Story` objects. Each `Story` object contains the title, sentences, shuffled sentences, and the gold order of sentences.

    Args:
        df (pl.DataFrame): The DataFrame containing the story data. Each row should represent a story with the first column as the title and the subsequent columns as sentences.

    Returns:
        List[Story]: A list of `Story` objects created from the DataFrame.
    """
    dataset = []
    total_rows = len(df.rows())

    for _, row in tqdm(enumerate(df.rows()), desc="Creating dataset", total=total_rows):
        title = row[0]

        # returns a tuple, convert to list
        sentences = list(row[1:])

        # copy and shuffle the order of sentences
        shuffled_sentences = sentences.copy()
        random.shuffle(shuffled_sentences)

        # indexes of the original sentences in shuffled sentences
        gold_order = [shuffled_sentences.index(sentence) for sentence in sentences]

        dataset.append(
            Story(
                title=title,
                sentences=sentences,
                shuffled_sentences=shuffled_sentences,
                gold_order=gold_order,
            )
        )
    return dataset


def convert_to_hf_dataset(dataset: List[Story]) -> Dataset:
    """
    Converts a list of `Story` objects to a Hugging Face `Dataset` object.

    Args:
        dataset (List[Story]): A list of `Story` objects to be converted to a Hugging Face `Dataset` object.

    Returns:
        Dataset: A Hugging Face `Dataset` object containing the processed data.
    """
    # convert to list of dicts
    dataset = [story.dict() for story in dataset]
    return Dataset.from_list(dataset)


def pipeline() -> Dataset:
    """
    Executes the data processing pipeline.

    This function orchestrates the entire data processing pipeline by sequentially applying
    a series of functions to transform the data. The pipeline reads data from a CSV file,
    drops specified columns, creates a dataset of `Story` objects, and converts it into a
    Hugging Face `Dataset` object.

    Returns:
        Dataset: A Hugging Face `Dataset` object containing the processed data.
    """
    df = read_data_from_csv()
    df = drop_columns(df)
    dataset = create_dataset(df)
    dataset = convert_to_hf_dataset(dataset)

    # push to hf hub
    hub_dataset_name = os.getenv("HF_HUB_DATASET_NAME")
    assert hub_dataset_name is not None, "HF_HUB_DATASET_NAME is not set"
    dataset.push_to_hub(
        hub_dataset_name, private=os.getenv("PRIVATE_DATASET") == "true"
    )
    return dataset
