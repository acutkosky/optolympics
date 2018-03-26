import torch
import numpy as np
from torch.optim import Optimizer

EPSILON = 1e-8

def element_wise_clip(tensor, low, high):
    return torch.min(torch.max(tensor, low), high)

def spherical_clip(tensor, norm, radius):
    return tensor*radius/norm

class BettingReduction(Optimizer):
    """Implements multi-stage reduction from exp-concave optimization
    to online linear optimization to stochastic critical point convergence.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate analogue (specifies a point on optimal regret frontier) (default: 1/sqrt(5))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, smoothness=0.01, lr=1.0, maximum_gradient=1.0):
        defaults = dict(lr=lr,
                        smoothness=smoothness,
                        maximum_gradient=maximum_gradient)
        super(BettingReduction, self).__init__(params, defaults)
        self.smoothness = smoothness
        self.ons_lr = 2.0/(2.0-np.log(3.0))
        self.lr = lr

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['maximum_gradient'] = p.data.new().resize_as_(p.data).fill_(maximum_gradient)
                state['ons_sum_gradient_squared'] = p.data.new().resize_as_(p.data).fill_(1.0)
                state['average_iterate_unweighted'] = p.data.clone()
                state['average_iterate_grad_weighted'] = p.data.clone()
                state['sum_gradient_squared'] = p.data.new().resize_as_(p.data).fill_(maximum_gradient**2)
                state['bet'] = p.data.new().resize_as_(p.data).fill_(0)
                state['center'] = p.data.clone()
                state['wealth'] = p.data.new().resize_as_(p.data).fill_(lr)

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

                regularized_grad = grad + 2 * self.smoothness * \
                    (p.data - state['average_iterate_unweighted'])

                state['sum_gradient_squared'] += regularized_grad**2

                #this is going to be slow because it probably
                #allocates and frees -state['maximum_gradient'].
                #we'll clean it up later if things are looking good.
                clipped_grad = element_wise_clip(regularized_grad,
                                                -state['maximum_gradient'],
                                                state['maximum_gradient'])

                state['maximum_gradient'] = \
                    torch.max(state['maximum_gradient'],
                              torch.abs(regularized_grad))


                state['wealth'] -= clipped_grad * \
                            (p.data - state['center'] \
                                - state['average_iterate_grad_weighted'])

                ons_grad = clipped_grad/(1.0 - clipped_grad * state['bet'])

                state['ons_sum_gradient_squared'] += ons_grad**2


                #perform ONS update on betting fraction
                state['bet'] -= self.ons_lr *\
                                ons_grad/state['ons_sum_gradient_squared'] 
                #same efficiency issue here
                state['bet'] = element_wise_clip(state['bet'],
                                        -0.5/state['maximum_gradient'],
                                         0.5/state['maximum_gradient'])

                #prediction comes from betting fraction, offset by
                #average weighted iterate ('momentum term')
                p.data = state['center'] + state['wealth'] * state['bet'] +\
                                state['average_iterate_grad_weighted']

                state['average_iterate_grad_weighted'] = \
                    state['average_iterate_grad_weighted'] +\
                    (p.data - state['average_iterate_grad_weighted']) *\
                    (regularized_grad**2)/state['sum_gradient_squared']

                state['average_iterate_unweighted'] = \
                    state['average_iterate_unweighted'] +\
                    (p.data - state['average_iterate_unweighted']) / \
                    (state['step'] + 1)

        return loss



class Direction_Finder:
    def __init__(self, tensor):
        self.current = tensor.new().resize_as_(tensor).fill_(0)
        self.grad_sum_sq = EPSILON

    def old_value(self):
        return self.current
    def update(self, gradient):
        self.grad_sum_sq += torch.norm(gradient)**2
        self.current -= gradient/np.sqrt(2 * self.grad_sum_sq)
        norm = torch.norm(self.current)
        if norm>1:
            self.current = spherical_clip(self.current, norm, 1)
        return self.current



class BettingReductionSphere(Optimizer):
    """Implements multi-stage reduction from exp-concave optimization
    to online linear optimization to stochastic critical point convergence.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate analogue (specifies a point on optimal regret frontier) (default: 1/sqrt(5))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, smoothness=0.01, lr=1.0, maximum_gradient=1.0):
        defaults = dict(lr=lr,
                        smoothness=smoothness,
                        maximum_gradient=maximum_gradient)
        super(BettingReductionSphere, self).__init__(params, defaults)
        self.smoothness = smoothness
        self.ons_lr = 2.0/(2.0-np.log(3.0))
        self.lr = lr

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['maximum_gradient'] = maximum_gradient
                state['ons_sum_gradient_squared'] = 1.0
                state['average_iterate_unweighted'] = p.data.clone()
                state['average_iterate_grad_weighted'] = p.data.clone()
                state['sum_gradient_squared'] = maximum_gradient**2
                state['1d_prediction'] = 0
                state['bet'] = 0
                state['center'] = p.data.clone()
                state['wealth'] = lr
                state['direction_finder'] = Direction_Finder(p.data)

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
                regularized_grad = grad + 2 * self.smoothness * \
                    (p.data - state['average_iterate_unweighted'])

                grad_norm = torch.norm(regularized_grad)

                state['sum_gradient_squared'] += grad_norm**2

                old_direction = state['direction_finder'].old_value()

                oned_grad = torch.sum(old_direction * regularized_grad)
                direction = state['direction_finder'].update(regularized_grad)

                clipped_grad = np.clip(oned_grad, 
                                        -state['maximum_gradient'],
                                        state['maximum_gradient'])


                state['maximum_gradient'] = max(state['maximum_gradient'],
                                                np.abs(oned_grad))


                state['wealth'] -= clipped_grad * state['1d_prediction']

                ons_grad = clipped_grad/(1.0 - clipped_grad * state['bet'])

                state['ons_sum_gradient_squared'] += ons_grad**2


                #perform ONS update on betting fraction
                state['bet'] -= self.ons_lr *\
                                ons_grad/state['ons_sum_gradient_squared'] 
                #same efficiency issue here
                state['bet'] = np.clip(state['bet'],
                                        -0.5/state['maximum_gradient'],
                                         0.5/state['maximum_gradient'])

                state['1d_prediction'] = state['wealth'] * state['bet']

                #prediction comes from betting fraction, offset by
                #average weighted iterate ('momentum term')
                p.data = state['center'] + state['1d_prediction'] * \
                                direction + \
                                state['average_iterate_grad_weighted']

                state['average_iterate_grad_weighted'] = \
                    state['average_iterate_grad_weighted'] +\
                    (p.data - state['average_iterate_grad_weighted']) *\
                    (grad_norm**2)/state['sum_gradient_squared']

                state['average_iterate_unweighted'] = \
                    state['average_iterate_unweighted'] +\
                    (p.data - state['average_iterate_unweighted']) / \
                    (state['step'] + 1)
                print("wealth: ",state['wealth'])
                print("maximum_gradient", state['maximum_gradient'])


        return loss
