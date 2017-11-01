import torch
import numpy as np
from torch.optim import Optimizer

EPSILON = 1e-8

class FreeRex(Optimizer):
    """Implements FreeRex algorithm 
        http://proceedings.mlr.press/v65/cutkosky17a.html.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate analogue (specifies a point on optimal regret frontier) (default: 1/sqrt(5))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1.0/np.sqrt(5.0), weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(FreeRex, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['one_over_eta_sq'] = p.data.new().resize_as_(p.data).fill_(EPSILON)
                state['grad_sum'] = p.data.new().resize_as_(p.data).zero_()
                state['beta'] = p.data.new().resize_as_(p.data).zero_()
                state['scaling'] = 1.0#p.data.new().resize_as_(p.data).fill_(1.0)
                state['max_grad'] = p.data.new().resize_as_(p.data).fill_(EPSILON)
                state['max_l2'] = EPSILON

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients ")
                    grad = grad.add(group['weight_decay'], p.data)

                if p.grad.data.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices().numpy()
                    grad_values = grad._values()
                    size = torch.Size([x for x in grad.size()])

                    sparse_one_over_eta_sq = state['one_over_eta_sq'][grad_indices]
                    sparse_max_grad = state['max_grad'][grad_indices]
                    sparse_grad_sum = state['grad_sum'][grad_indices]


                    sparse_max_grad = torch.max(sparse_max_grad, torch.abs(grad_values))
                    


                    sparse_grad_sum.add_(grad_values)


                    sparse_one_over_eta_sq = torch.max(sparse_one_over_eta_sq + 2*grad_values.pow(2), sparse_max_grad * torch.abs(sparse_grad_sum))

                    state['max_grad'][grad_indices] = sparse_max_grad
                    state['grad_sum'][grad_indices] = sparse_grad_sum
                    state['one_over_eta_sq'][grad_indices] = sparse_one_over_eta_sq
                    p.data[grad_indices] = -torch.sign(sparse_grad_sum) * (torch.exp(torch.abs(sparse_grad_sum)*group['lr']/(torch.sqrt(sparse_one_over_eta_sq))) - 1.0)

                else:
                    state['max_grad'] = torch.max(state['max_grad'], torch.abs(grad))

                    state['max_l2'] = max(state['max_l2'], grad.norm(2))

                    state['scaling'] = min(state['scaling'], state['max_l2']/torch.sum(state['max_grad']))

                    state['grad_sum'].add_(grad)

                    state['one_over_eta_sq'] = torch.max(state['one_over_eta_sq'] + 2*grad.pow(2), state['max_grad'] * torch.abs(state['grad_sum']))

                    p.data = -torch.sign(state['grad_sum']) * (torch.exp(torch.abs(state['grad_sum'])*group['lr']/(torch.sqrt(state['one_over_eta_sq']))) - 1.0) * state['scaling']#/np.sqrt(state['max_l1'])

        return loss