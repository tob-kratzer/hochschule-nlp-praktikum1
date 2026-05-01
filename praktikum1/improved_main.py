import nltk
import datasets
from nltk.corpus import stopwords
import re
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math


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

    trainset = DataLoader(imdb_torch["train"], batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    testset = DataLoader(imdb_torch["test"], batch_size=4, shuffle=False, num_workers=0, drop_last=False)

    embed_dim = 16
    lr = 0.01
    epochs = 10

    logreg = LogReg(my_dim=embed_dim, vocab_size=len(w_to_id))
    my_opti = optim.SGD(params=logreg.parameters(), lr=lr, momentum=0.9)

    writer = SummaryWriter("runs/experiment1")
    global_step = 0
    T = 100
    prev_params = {}

    for k in range(epochs):
        train_bce_all = []
        train_correct = 0
        train_total = 0

        for batch in trainset:
            _ids = batch["ids"]
            _att = batch["attention_ids"]
            _label = batch["label"].float()

            y_hat = logreg(_ids, _att)

            loss = F.binary_cross_entropy(y_hat, _label)
            train_bce_all.append(loss.item())

            prediction = (y_hat >= 0.5).int()
            train_correct += (prediction == _label.int()).sum().item()
            train_total += _label.shape[0]

            loss.backward()

            # Loss loggen
            writer.add_scalar("loss/train", loss.item(), global_step)

            # Gradienten Normen loggen
            for name, param in logreg.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.detach().norm(2).item()
                    writer.add_scalar(f"grad_norm/{name}", grad_norm, global_step)

            my_opti.step()

            # weights loggen
            if global_step % T == 0:
                for name, param in logreg.named_parameters():
                    p = param.detach()
                    std = p.std().item()
                    if std > 0:
                        writer.add_scalar(f"weight_log_std/{name}", math.log(std), global_step)

                    if name in prev_params:
                        delta_std = (p - prev_params[name]).std().item()
                        if delta_std > 0:
                            writer.add_scalar(f"weight_log_change_std/{name}", math.log(delta_std), global_step)

                    prev_params[name] = p.clone()

            my_opti.zero_grad()
            global_step += 1

        train_loss = np.mean(train_bce_all)
        train_acc = train_correct / train_total

        test_bce_all = []
        test_correct = 0
        test_total = 0

        for batch in testset:
            _ids = batch["ids"]
            _att = batch["attention_ids"]
            _label = batch["label"].float()

            y_hat = logreg(_ids, _att)

            loss = F.binary_cross_entropy(y_hat, _label)
            test_bce_all.append(loss.item())

            pred = (y_hat >= 0.5).int()
            test_correct += (pred == _label.int()).sum().item()
            test_total += _label.shape[0]

        test_loss = np.mean(test_bce_all)
        test_acc = test_correct / test_total

        print(f"k: {k+1}")
        print(f"train loss: {train_loss}, train accuracy: {train_acc}")
        print(f"test loss: {test_loss}, test accuracy: {test_acc}")

    torch.save(logreg.state_dict(), "model_v1.pt")
    writer.close()


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
    