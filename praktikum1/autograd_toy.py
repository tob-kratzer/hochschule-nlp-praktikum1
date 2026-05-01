import numpy as np
import torch
from torch import autograd

my_dim = 16
my_lr = 0.001
eps = 1e-7
epochs = 10

# Jetzt ohne requires_grad=False
w = torch.randn(my_dim, requires_grad=True)
b = torch.randn(1, requires_grad=True)

def logit(x, w, b):
    return x @ w + b

def logreg(x, w, b):
    return torch.sigmoid(logit(x, w, b))

def bce(y_hat, y):
    return -(y * torch.log(y_hat + eps) + (1 - y) * torch.log(1 - y_hat + eps))

def my_grad(x, y, y_hat):
    w_grad = -(y - y_hat) * x
    b_grad = -(y - y_hat)
    return w_grad, b_grad

# Zufällige Testdaten erzeugen
x = torch.randn(8, my_dim)
y = torch.randint(low=0, high=2, size=(8,), dtype=torch.float32)

# Ein Beispiel auswählen
xs = x[0]
ys = y[0]

y_hat = logreg(xs, w, b)
bce_ = bce(y_hat, ys)

# Gradienten vergleichen
w_grad_hand, b_grad_hand = my_grad(xs, ys, y_hat)
w_grad_auto, b_grad_auto = autograd.grad(bce_, inputs=(w, b))

print("w_grad_hand:", w_grad_hand[:5])
print("w_grad_auto:", w_grad_auto[:5])
print("b_grad_hand:", b_grad_hand)
print("b_grad_auto:", b_grad_auto)

for k in range(epochs):
    bce_all = []
    for i, (xs, ys) in enumerate(zip(x, y)):
        y_hat = logreg(xs, w, b)
        bce_ = bce(y_hat, ys)
        bce_all.append(bce_.item())
        w_grad, b_grad = autograd.grad(bce_, inputs=(w, b))

        with torch.no_grad():
            w -= my_lr * w_grad
            b -= my_lr * b_grad

    train_loss = np.mean(bce_all)
    print(f"k: {k+1}")
    print(f"train loss: {train_loss}")