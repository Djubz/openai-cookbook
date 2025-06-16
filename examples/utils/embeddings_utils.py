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


def obtain_embedding(text: str, model="text-embedding-3-small", **kwargs) -> List[float]:
    # Replace newlines, as they might negatively affect performance.
    text = text.replace("\n", " ")

    response = client.embeddings.create(input=[text], model=model, **kwargs)

    return response.data[0].embedding


async def aobtain_embedding(
    text: str, model="text-embedding-3-small", **kwargs
) -> List[float]:
    # Replace newlines, as they might negatively affect performance.
    text = text.replace("\n", " ")

    return (await client.embeddings.create(input=[text], model=model, **kwargs))[
        "data"
    ][0]["embedding"]


def obtain_embeddings(
    texts: List[str], model="text-embedding-3-small", **kwargs
) -> List[List[float]]:
    assert len(texts) <= 2048, "The maximum batch size allowed is 2048."

    # Replace newlines, as they might negatively affect performance.
    texts = [text.replace("\n", " ") for text in texts]

    data = client.embeddings.create(input=texts, model=model, **kwargs).data
    return [d.embedding for d in data]


async def aobtain_embeddings(
    texts: List[str], model="text-embedding-3-small", **kwargs
) -> List[List[float]]:
    assert len(texts) <= 2048, "The maximum batch size allowed is 2048."

    # Replace newlines, as they might negatively affect performance.
    texts = [text.replace("\n", " ") for text in texts]

    data = (
        await client.embeddings.create(input=texts, model=model, **kwargs)
    ).data
    return [d.embedding for d in data]


def calculate_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```
