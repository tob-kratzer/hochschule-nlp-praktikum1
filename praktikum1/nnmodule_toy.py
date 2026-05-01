import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class LogReg(nn.Module):
    def __init__(self, my_dim):
        super().__init__()
        self.linear = nn.Linear(my_dim, 1)
        self.my_dim = my_dim

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    

class MyDataset(Dataset):
    def __init__(self, inputs, targets):
        super(MyDataset, self).__init__()
        assert len(inputs) == len(targets)
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
    

def main():
    # Device auswählen
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device:", device)

    # Modell erstellen
    logreg = LogReg(my_dim=16)
    logreg = logreg.to(device)
    # Optimizer für SGD erstellen
    my_opti = optim.SGD(params=logreg.parameters(), lr=0.01, momentum=0.9)

    # Zufällige Daten erstlln
    x = torch.randn(8, logreg.my_dim)
    y = torch.randint(low=0, high=2, size=(8, 1), dtype=torch.float32)

    # Dataset aus Zufallsdaten bauen
    myds = MyDataset(inputs=x, targets=y)
    mydl = DataLoader(myds, batch_size=2, shuffle=True)

    epochs = 100

    for k in range(epochs):
        bce_all = []
        # Iteration über batches aus DataLoader
        for xs, ys in mydl:
            # input batch auf selbes device wie Modell verschieben
            xs = xs.to(device)
            ys = ys.to(device)
            # Modell auf Batch anwenden
            ys_hat = logreg(xs)
            loss = F.binary_cross_entropy(ys_hat, ys)
            # Gradient berechnen
            loss.backward()
            # Parameter des Modells aktualisieren
            my_opti.step()
            my_opti.zero_grad()

            bce_all.append(loss.item())

        if k % 10 == 0:
            print(np.mean(bce_all))


if __name__ == "__main__":
    main()