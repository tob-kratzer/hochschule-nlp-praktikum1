import nltk
import datasets
from nltk.corpus import stopwords
import re
import string
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LogReg(nn.Module):
    def __init__(self, my_dim, vocab_size):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, my_dim)
        self.linear = nn.Linear(my_dim, 1)

        self.my_dim = my_dim
        self.vocab_size = vocab_size

    def forward(self, ids, att_ids):
        embeds = self.embd(ids)
        embeds = embeds * att_ids[:, :, None]
        sentences = embeds.sum(dim=1) / att_ids.sum(dim=1, keepdim=True)
        return torch.sigmoid(self.linear(sentences)).squeeze(1)


def main():
    imdb = datasets.load_dataset("stanfordnlp/imdb")
    print(imdb)

    nltk.download("stopwords")
    stop_en = stopwords.words("english")

    clean_imdb = imdb.map(lambda example: preprocess_text(example, stop_en))

    w_to_id, id_to_w, freq = build_vocab(clean_imdb["train"], k=10000)

    final_imdb = clean_imdb.map(lambda example: add_ids(example, w_to_id, max_len=256))

    final_imdb = final_imdb.remove_columns(["text", "words"])

    imdb_torch = final_imdb.with_format("torch")

    testset = DataLoader(imdb_torch["test"], batch_size=4, shuffle=False, num_workers=0, drop_last=False)

    embed_dim = 16

    logreg = LogReg(my_dim=embed_dim, vocab_size=len(w_to_id))
    logreg.load_state_dict(torch.load("model_v1.pt"))
    logreg.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in testset:
            ids, att, labels = batch['ids'], batch['attention_ids'], batch['label']
            preds = logreg(ids, att)
            predicted = (preds >= 0.5).long().squeeze()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f'Test accuracy: {correct/total:.4f}')


def preprocess_text(example, stopwords):

  text = example["text"]
  text = re.sub(r"<[^>]+>", " ", text)
  text = text.lower()
  text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
  words = text.split()
  words = [word for word in words if word not in stopwords]
  words = [word for word in words if not (len(word) == 1 and word.isalpha())]

  return {"words": words}


def build_vocab(dataset, k=10000):

    freq = {}
    for example in dataset:
       for word in example["words"]:
            if word not in freq:
             freq[word] = 1
            else:
             freq[word] += 1

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


if __name__ == "__main__":
    main()