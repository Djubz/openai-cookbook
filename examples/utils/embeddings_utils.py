```python
import textwrap as tr
from typing import List, Optional

import matplotlib.pyplot as plt
import plotly.express as px
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, precision_recall_curve

from openai import OpenAI
import numpy as np
import pandas as pd

client = OpenAI(max_retries=5)


def get_embedding(text: str, model="text-embedding-3-small", **kwargs) -> List[float]:
    # Substitute newlines, which may adversely impact performance.
    text = text.replace("\n", " ")

    response = client.embeddings.create(input=[text], model=model, **kwargs)

    return response.data[0].embedding


async def aget_embedding(
    text: str, model="text-embedding-3-small", **kwargs
) -> List[float]:
    # Substitute newlines, which may adversely impact performance.
    text = text.replace("\n", " ")

    return (await client.embeddings.create(input=[text], model=model, **kwargs))[
        "data"
    ][0]["embedding"]


def get_embeddings(
    list_of_text: List[str], model="text-embedding-3-small", **kwargs
) -> List[List[float]]:
    assert len(list_of_text) <= 2048, "The batch size must not exceed 2048."

    # Substitute newlines, which may adversely impact performance.
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = client.embeddings.create(input=list_of_text, model=model, **kwargs).data
    return [d.embedding for d in data]


async def aget_embeddings(
    list_of_text: List[str], model="text-embedding-3-small", **kwargs
) -> List[List[float]]:
    assert len(list_of_text) <= 2048, "The batch size must not exceed 2048."

    # Substitute newlines, which may adversely impact performance.
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = (
        await client.embeddings.create(input=list_of_text, model=model, **kwargs)
    ).data
    return [d.embedding for d in data]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def plot_multiclass_precision_recall(
    y_score, y_true_untransformed, class_list, classifier_name
):
    """
    Generates a Precision-Recall plot for a multiclass scenario. It visualizes average precision-recall, individual class precision-recall, and reference F1 contours.

    The code has been slightly altered but is primarily based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    """
    n_classes = len(class_list)
```
