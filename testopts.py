import torch
from torch import autograd
import numpy as np
from adanonconv import *


def loss(w):
    return loss_closure(w)()

def loss_closure(w):
    # x = autograd.Variable(torch.FloatTensor([1]))
    x = autograd.Variable(torch.randn(5))+10
    # x = autograd.Variable(torch.FloatTensor([1,1,1,1,1]))
    # x = autograd.Variable(torch.FloatTensor([1]))
    y = 10
    s = 1.0
    # s = np.abs(np.random.normal(2,1))
    def closure():
        return 0.5 * (w.dot(x)-y).abs() * s
    return closure

def batchLoss(w, N):
    accum =0
    for _ in range(N):
        accum += loss(w)
    return accum/N

# [1,2,3,4,5]
w = autograd.Variable(torch.FloatTensor(np.zeros(5)), requires_grad=True)
# w = autograd.Variable(torch.FloatTensor([10]), requires_grad=True)
# opt = torch.optim.Adagrad([w], 1)
# opt = AdaNonConvL([w], 0.00001, True)
opt = MetaLROptimizer([w], unbiased = False)
# opt = MyAdaGrad([w],50)
for _ in range(100):
    closure = loss_closure(w)
    l = closure()
    print('loss: ',l)
    opt.zero_grad()
    l.backward()
    opt.step(closure)

print('loss: ',batchLoss(w, 10000))
print('norm: ',(w.data- torch.FloatTensor([1,2,3,4,5])).norm(2))