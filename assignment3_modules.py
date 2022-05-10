import collections

import numpy as np
from difflib import SequenceMatcher


def filter_text(text, alphabet):
    return ''.join(l for l in text.lower() if l in alphabet)

def get_freqs(text):
    freqs = collections.Counter(text)
    freqs = {k: v / sum(freqs.values()) for k, v in freqs.items()}
    freqs = dict(sorted(freqs.items(), key=lambda x: x[1], reverse=True))
    return freqs

def unigram_encode_decode(text, mapping):
    return ''.join([mapping[l] for l in text])

def get_encode_mapping(orig_freqs):
    return dict(zip(orig_freqs, np.random.permutation(list(orig_freqs))))

def get_decode_mapping(orig_freqs, text_freqs):
    return dict(zip(text_freqs, orig_freqs))

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()  # метрика близости текстов в диапазоне  [0, 1]

def accept(current, new):
    if new > current:
        return True
    return np.random.rand() < np.exp(new - current)