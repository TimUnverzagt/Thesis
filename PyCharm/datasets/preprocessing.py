# General modules
import numpy as np
from typing import NamedTuple
from typing import List
from nltk.tokenize import sent_tokenize, word_tokenize


def embed(tok_text, vocabulary):
    embedding = []
    idx_for_unrec = len(vocabulary)
    for word in tok_text:
        if word in vocabulary:
            embedding.append(vocabulary[word])
        else:
            continue
            # embedding.append(idx_for_unrec)
    return embedding


def tokenize(text, lower=False, head_stripper=None):
    tok_text = text
    if head_stripper is not None:
        tok_text = strip_head(text=tok_text, stripper=head_stripper)
    if lower:
        tok_text = tok_text.lower()
    tok_text = word_tokenize(tok_text)
    return tok_text


def unify_length(tok_doc, target_length, padding='zero'):
    doc_len = len(tok_doc)
    if doc_len > target_length:
        edited_doc = tok_doc[0:target_length]
    elif doc_len < target_length:
        edited_doc = tok_doc
        for j in range(doc_len, target_length):
            if padding == 'zero':
                edited_doc.append(0)
            else:
                print('Padding identifier not recognized while unifying length of documents')
    else:
        edited_doc = tok_doc
    return edited_doc


def strip_head(text, stripper):
    split_text = text.split(stripper)
    truncated_text = ""
    for i in range(len(split_text)):
        if i == 0:
            continue
        truncated_text += split_text[i]
    return truncated_text