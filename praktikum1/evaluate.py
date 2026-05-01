import nltk
import datasets
from nltk.corpus import stopwords
import re
import string
import numpy as np
import torch
import torch.nn as nn

def main():
    # IMDB-Datensatz laden
    imdb = datasets.load_dataset("stanfordnlp/imdb")
    print(imdb)

    # Stopwords herunterladen und laden
    nltk.download("stopwords")
    stop_en = stopwords.words("english")

    # preprocess_text anwenden
    clean_imdb = imdb.map(lambda example: preprocess_text(example, stop_en))

    # vocabulary auf train set bauen
    w_to_id, id_to_w, freq = build_vocab(clean_imdb['train'], k=10000)

    # Wörter in id's umwandeln
    final_imdb = clean_imdb.map(lambda example: add_ids(example, w_to_id, max_len=256))

    # Nur mit ids, attention_ids und label weiterarbeiten
    final_imdb = final_imdb.remove_columns(["text", "words"])

    imdb_torch = final_imdb.with_format("torch")

    # Modellparameter laden
    checkpoint = torch.load('model_v1.pt')
    # richitge Dimension durch Länge von w
    embed_dim = checkpoint["w"].shape[0]
    # Struktur nochmal bauen
    embd = nn.Embedding(len(w_to_id), embed_dim)
    embd.weight = nn.Parameter(checkpoint["embd_weight"])
    w = checkpoint['w']
    b = checkpoint['b']

    test_correct = 0
    test_total = 0

    for sample in imdb_torch["test"]:
        _ids, _att, _label = sample["ids"], sample["attention_ids"], sample["label"]

        _embds = embd(_ids)
        _sum = _embds * _att[:, None]
        _sum = _sum.sum(axis=0) / _att.sum()

        y_hat = logreg(_sum, w, b)

        pred = 1 if y_hat.item() >= 0.5 else 0
        if pred == _label.item():
            test_correct += 1
        test_total += 1

        test_acc = test_correct / test_total

        print(f"test accuracy: {test_acc}")


def preprocess_text(example, stopwords):
  """
  Preprocess the text by:
  1. Remove html tags.
  2. Convert to lowercase.
  3. Remove punctuation.
  4. Split into words.
  5. Remove stopwords.
  6. Remove single-character words that are just letters
     (but keep, e.g. "$"! -> `.isalpha()`).

  Args:
      example: A dictionary containing a 'text' field.
      stopwords: list of stopwords.

  Returns:
      Dictionary with the processed text as 'words'.
  """
  text = example["text"]
  # 1
  text = re.sub(r"<[^>]+>", " ", text)
  # 2
  text = text.lower()
  # 3
  text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
  # 4
  words = text.split()
  # 5
  words = [word for word in words if word not in stopwords]
  # 6
  words = [word for word in words if not (len(word) == 1 and word.isalpha())]

  return {"words": words}

def build_vocab(dataset, k=10000):
    """
    Build a vocabulary of the k most frequent words.

    Args:
      dataset: The dataset containing tokenized texts
      k: Number of most frequent words to keep

    Returns:
      word_to_id: Dictionary mapping words to IDs
      id_to_word: Dictionary mapping IDs to words
  """
    freq = {}
    for example in dataset:
       for word in example["words"]:
            if word not in freq:
             freq[word] = 1
            else:
             freq[word] += 1

    # absteigend nach Häufigkeit sortieren
    sorted_words = sorted(freq.items(), key=lambda item: item[1], reverse=True)
    most_kfreq_words = sorted_words[:k]

    w_to_id = {"<UNK>": 0}
    id_to_w = {0: "<UNK>"}

    current_id = 1
    for word, count in most_kfreq_words:
       w_to_id[word] = current_id
       id_to_w[current_id] = word
       current_id += 1

    return w_to_id, id_to_w, freq

def add_ids(example, word2id, max_len=256):
    """
    Convert words to ids. Cut sentences off at `max_len`.
    Fill short sentences up to `max_len`, using `<UNK>`.
    Use the feature `attention_ids` to indicate whether an id
    is filled up or not.

    Args:
      example: Dicionarty containg 'words'.
      word2id: Mapping words to ids.
      max_len: The maximum length of a sentence of ids.

    Returns:
      Dictionary with 'ids' and `attention_ids`.
    """
    ids = []
    atts = []
    for i, w in enumerate(example['words']):
        if i >= max_len:
            break
        if w in word2id:
            ids.append(word2id[w])
        else:
            ids.append(0)
        atts.append(1)
    
    if i < max_len:
        ids.extend([0]*(max_len - i - 1))
        atts.extend([0]*(max_len - i - 1))

    example['ids'] = ids
    example['attention_ids'] = atts
    return example

def logit(x, w, b):
    return x @ w + b


def logreg(x, w, b):
    return torch.sigmoid(logit(x, w, b))


def bce(y_hat, y, eps=1e-7):
    return -(y * torch.log(y_hat + eps) + (1 - y) * torch.log(1 - y_hat + eps))


def my_grad(x, y, y_hat):
    w_grad = -(y - y_hat) * x
    b_grad = -(y - y_hat)
    return w_grad, b_grad


if __name__ == "__main__":
    main() 