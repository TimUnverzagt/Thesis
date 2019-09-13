# General modules
import numpy as np
from typing import NamedTuple
from typing import List


# Docs as named tuples
class TokenizedDoc(NamedTuple):
    tokenized_words: List[str]
    tokenized_sents: List[List[str]]


class EmbeddedDoc(NamedTuple):
    embedded_words: np.ndarray
    embedded_sents: List[np.ndarray]
