import torch
import numpy as np
import oloalgs
from torch.optim import Optimizer

EPSILON = 1e-8

class MyAdaGrad(Optimizer):
    """Implements an optimization scheme that adapts the learning rate
    over time using an sub-optimizer."""

    def __init__(self, params, eta=1.0):
        defaults = dict(eta=eta)
        super(MyAdaGrad, self).__init__(params, defaults)
        self.eta = eta

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['grad_squared_sum'] = EPSILON


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

                state['grad_squared_sum'] += grad.norm(2)**2
                offset = -self.eta * grad/np.sqrt(state['grad_squared_sum'])
                p.data += offset




        return loss


class MetaLROptimizer(Optimizer):
    """Implements an optimization scheme that adapts the learning rate
    over time using an sub-optimizer."""

    def __init__(self, params, epsilon=1.0, unbiased = True):
        defaults = dict(epsilon=epsilon, unbiased = unbiased)
        super(MetaLROptimizer, self).__init__(params, defaults)
        self.do_update = not unbiased#False
        self.unbiased = unbiased

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['last_grad'] = p.data.new().resize_as_(p.data).zero_()
                state['last_loss'] = 0
                state['eta_optimizer'] = oloalgs.parabolic_bounded_optimizer()
                # state['eta_optimizer'] = oloalgs.ONSCoinBetting1D(positive_only=True, domain = [0.000001, 10000])
                state['offset'] = EPSILON
                state['eta_grad_lin_part'] = 0
                state['do_update'] = True
                state['grad_squared_sum'] = 1.0#EPSILON
                state['L'] = 0
                state['eta'] = 0


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

                state['last_loss'] = loss

                state['step'] += 1
                state['current_grad'] = grad

                # state['grad_squared_sum'] = state['grad_squared_sum'] + grad.norm(2)**2

                if(self.do_update):
                    grad_product = torch.sum(p.grad.data * state['last_grad'])
                    last_grad_norm_sq = state['last_grad'].norm(2)**2

                    state['grad_squared_sum'] = 1.0*(state['grad_squared_sum']) + state['last_grad'].norm(2)**2
                    # state['grad_squared_sum'] = max(1.0, state['last_grad'].norm(2)**2)
                    state['eta_grad_lin_part'] = -grad_product/np.sqrt(state['grad_squared_sum'])
                    eta_grad_quad_part_est = 0.5 * state['L'] * state['last_grad'].norm(2)**2/state['grad_squared_sum']
                    # eta_grad_est = state['eta_grad_lin_part'] + 2 * state['eta'] * eta_grad_quad_part_est
                    eta_grad_est = np.array([state['eta_grad_lin_part'],eta_grad_quad_part_est])

                    state['eta_optimizer'].hint(eta_grad_est)

                    eta = state['eta_optimizer'].get_prediction()[0]
                    state['eta'] = eta
                    # eta = 0.5
                    print('using eta: ',eta)
                    state['offset'] = -eta * state['last_grad']/np.sqrt(state['grad_squared_sum'])
                    p.data += state['offset']
                    print('grad: ',state['last_grad'])
                    print('grad_squared_sum: ',state['grad_squared_sum'])
                else:
                    state['last_grad'] = grad


                if(self.unbiased):
                    self.do_update = not self.do_update


        if(not self.do_update or not self.unbiased):
            loss = closure()
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if p.grad is None:
                        continue

                    Lt = 2 * ( loss.data[0] - state['last_loss'].data[0] - (state['offset']*(state['current_grad'])).sum() )/(0.000001+state['offset'].norm(2)**2)
                    Lt= max(Lt, 0)
                    print('loss difference: ', loss.data[0] - state['last_loss'].data[0])
                    state['L'] = max(Lt,state['L'])
                    # print('loss: ', loss.data[0])
                    # print('last loss: ', state['last_loss'].data[0])
                    # print('offset: ', state['offset'])
                    # print('grad: ', state['current_grad'])
                    # print('param: ',p.data)
                    eta = state['eta']
                    print('Lt: ', Lt)
                    print('offset norm sq: ',state['offset'].norm(2)**2)
                    print('last grad norm sq: ', state['last_grad'].norm(2)**2)
                    eta_grad_quad_part = 0.5 * Lt * state['last_grad'].norm(2)**2/state['grad_squared_sum']

                    # eta_grad = state['eta_grad_lin_part'] + 2 * eta *eta_grad_quad_part
                    eta_grad = np.array([state['eta_grad_lin_part'], eta_grad_quad_part])
                    print('eta grad lin part: ', state['eta_grad_lin_part'])
                    print('eta grad quad part: ', eta_grad_quad_part)
                    print('eta grad: ', eta_grad)

                    state['eta_optimizer'].update(eta_grad)

                    eta = state['eta_optimizer'].get_prediction()
                    print('eta',eta)
                    state['last_grad'] = state['current_grad']


        return loss


class AdaNonConv(Optimizer):
    """Implements an adaptive step size for smooth non-convex optimization

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate analogue
    """

    def __init__(self, params, lr=1.0, unbiased = True):
        defaults = dict(L=1.0/lr, unbiased=unbiased)
        super(AdaNonConv, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['old_grad'] = p.data.new().resize_as_(p.data).zero_()
                state['grad_norm_sq'] = 0
                state['old_grad_norm_sq'] = 0
                state['eta'] = 0.001
                state['eta_grad_sum'] = 0
                state['eta_mu_sum'] = 0.1#EPSILON
                state['max_grad'] = EPSILON
                state['do_update'] = True
                state['grad_squared_sum'] = EPSILON

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
                grad_product = torch.sum(p.grad.data * state['old_grad'])
                old_grad_norm_sq = state['old_grad_norm_sq']
                state['old_grad_norm_sq'] = state['grad_norm_sq']

                state['old_grad'] = grad
                state['grad_norm_sq'] = grad.norm(2)**2
                state['grad_squared_sum'] += state['grad_norm_sq']
                state['max_grad'] = max(state['max_grad'], np.sqrt(state['old_grad_norm_sq']))

                if(state['do_update']):
                    state['eta_grad_sum'] += grad_product/np.sqrt(state['grad_squared_sum'])
                    # state['eta_mu_sum'] += state['eta'] * group['L'] * old_grad_norm_sq
                    state['eta_mu_sum'] += group['L'] * old_grad_norm_sq/state['grad_squared_sum']
                    # state['eta'] = state['eta_grad_sum']/state['eta_mu_sum']
                    state['eta'] = min(max(state['eta_grad_sum']/state['eta_mu_sum'],0.00001),1.0/np.sqrt(state['grad_norm_sq']))
                    p.data -= state['eta'] * grad/np.sqrt(state['grad_squared_sum'])


                if(group['unbiased']):
                    state['do_update'] = not state['do_update']


        return loss


class AdaNonConvL(Optimizer):
    """Implements an adaptive step size for smooth non-convex optimization

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate analogue
    """

    def __init__(self, params, lr=1.0, unbiased = True):
        defaults = dict(L=0.1, unbiased=unbiased)
        super(AdaNonConvL, self).__init__(params, defaults)
        self.do_update = True
        self.unbiased = unbiased

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['old_grad'] = p.data.new().resize_as_(p.data).zero_()
                state['grad_norm_sq'] = 0
                state['old_grad_norm_sq'] = 0
                state['eta'] = 0.001
                state['eta_grad_sum'] = 0
                state['eta_mu_sum'] = 1.0+EPSILON
                state['max_grad'] = EPSILON
                state['grad_squared_sum'] = EPSILON

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

                state['old_loss'] = loss

                state['step'] += 1
                grad_product = torch.sum(p.grad.data * state['old_grad'])
                old_grad_norm_sq = state['old_grad_norm_sq']

                if(self.do_update):
                    state['eta_grad_sum'] = grad_product/np.sqrt(state['grad_squared_sum']) + state['eta_grad_sum']
                    state['eta_grad'] = -grad_product
                    state['offset'] = -state['eta'] * state['old_grad']/np.sqrt(state['grad_squared_sum'])

                    p.data -= state['eta'] * state['old_grad']/np.sqrt(state['grad_squared_sum'])


                state['old_grad_norm_sq'] = state['grad_norm_sq']
                state['grad_norm_sq'] = grad.norm(2)**2
                state['grad_squared_sum'] += state['grad_norm_sq']

                state['old_grad'] = grad
                state['grad_norm_sq'] = grad.norm(2)**2
                state['max_grad'] = max(state['max_grad'], np.sqrt(state['old_grad_norm_sq']))
                if(self.unbiased):
                    self.do_update = not self.do_update


        if(not self.do_update):
            loss = closure()
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if p.grad is None:
                        continue

                    Lt = 2 * ( loss.data[0] - state['old_loss'].data[0] - (state['offset']*(p.grad.data)).sum() )/(0.0001+state['offset'].norm(2)**2)
                    print('Lt: ', Lt)
                    group['L'] = max(Lt, 0.9*group['L'])
                    print(group['L'])

                    state['eta_mu_sum'] = group['L'] * state['old_grad_norm_sq']/state['grad_squared_sum'] + state['eta_mu_sum']
                    state['eta'] = state['eta_grad_sum']/state['eta_mu_sum']
                    # state['eta'] = min(min(max(state['eta'],0.00001/np.sqrt(state['grad_squared_sum'])),100/np.sqrt(state['grad_squared_sum'])),1.0/state['max_grad'])
                    state['eta'] = min(max(state['eta'],0.0000),1000)

                    # state['eta'] = min(max((state['eta_grad_sum']/state['eta_mu_sum']),0.00001/np.sqrt(state['grad_squared_sum'])),100/np.sqrt(state['grad_squared_sum']))
                    print('eta',state['eta'])


        return loss




class AdaNonConvLPerC(Optimizer):
    """Implements an adaptive step size for smooth non-convex optimization

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate analogue
    """

    def __init__(self, params, lr=1.0, unbiased = True):
        defaults = dict(L=0.1, unbiased=unbiased)
        super(AdaNonConvLPerC, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['old_grad'] = p.data.new().resize_as_(p.data).zero_()
                state['grad_norm_sq'] = p.data.new().resize_as_(p.data).zero_()
                state['old_grad_norm_sq'] = p.data.new().resize_as_(p.data).zero_()
                state['eta'] = p.data.new().resize_as_(p.data).zero_()+0.001
                state['eta_grad_sum'] = p.data.new().resize_as_(p.data).zero_()
                state['eta_mu_sum'] = 1.0+p.data.new().resize_as_(p.data).zero_()
                state['max_grad'] = EPSILON + p.data.new().resize_as_(p.data).zero_()
                state['do_update'] = True
                state['grad_squared_sum'] = EPSILON + p.data.new().resize_as_(p.data).zero_()
                state['L'] = group['L'] + p.data.new().resize_as_(p.data).zero_()

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

                state['old_loss'] = loss

                state['step'] += 1
                grad_product = torch.sum(p.grad.data * state['old_grad'])
                old_grad_norm_sq = state['old_grad_norm_sq']

                if(state['do_update']):
                    state['eta_grad_sum'] += grad_product/np.sqrt(state['grad_squared_sum'])
                    state['eta_grad'] = -grad_product
                    state['offset'] = -state['eta'] * state['old_grad']/np.sqrt(state['grad_squared_sum'])

                    p.data -= state['eta'] * state['old_grad']/np.sqrt(state['grad_squared_sum'])


                state['old_grad_norm_sq'] = state['grad_norm_sq']
                state['grad_norm_sq'] = grad*grad#.norm(2)**2
                state['grad_squared_sum'] += state['grad_norm_sq']

                state['old_grad'] = grad
                state['grad_norm_sq'] = grad*grad#.norm(2)**2
                state['max_grad'] = torch.max(state['max_grad'], np.sqrt(state['old_grad_norm_sq']))
                if(group['unbiased']):
                    state['do_update'] = not state['do_update']

        loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                Lt = 2 * ( loss.data[0] - state['old_loss'].data[0] - (state['offset']*(p.grad.data)).sum() )/(0.0000000001+state['offset']**2)
                print('Lt: ', Lt)
                state['L'] = torch.max(Lt, state['L'])
                print(state['L'])

                state['eta_mu_sum'] += state['L'] * state['old_grad_norm_sq']/state['grad_squared_sum']
                state['eta'] = state['eta_grad_sum']/state['eta_mu_sum']
                # state['eta'] = min(min(max(state['eta'],0.00001/np.sqrt(state['grad_squared_sum'])),100/np.sqrt(state['grad_squared_sum'])),1.0/state['max_grad'])
                state['eta'] = torch.clamp(state['eta'], 0.1, 10)
                #min(max(state['eta'],0.00001),100)

                # state['eta'] = min(max((state['eta_grad_sum']/state['eta_mu_sum']),0.00001/np.sqrt(state['grad_squared_sum'])),100/np.sqrt(state['grad_squared_sum']))
                print('eta',state['eta'])





        return loss





class AdaNonConvAdaL(Optimizer):
    """Implements an adaptive step size for smooth non-convex optimization

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate analogue
    """

    def __init__(self, params, lr=1.0, unbiased = True):
        defaults = dict(L=0.1, unbiased=unbiased)
        super(AdaNonConvAdaL, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['old_grad'] = p.data.new().resize_as_(p.data).zero_()
                state['grad_norm_sq'] = 0
                state['old_grad_norm_sq'] = 0
                state['eta'] = 0.001
                state['eta_grad_sum'] = 0
                state['eta_mu_sum'] = 1.0+EPSILON
                state['max_grad'] = EPSILON
                state['do_update'] = True
                state['grad_squared_sum'] = EPSILON
                state['L'] = group['L']
                state['L_grad_sq_sum'] = EPSILON
                state['L_grad_sum'] = 0

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

                state['old_loss'] = loss

                state['step'] += 1
                grad_product = torch.sum(p.grad.data * state['old_grad'])
                old_grad_norm_sq = state['old_grad_norm_sq']

                if(state['do_update']):
                    state['eta_grad_sum'] += grad_product/np.sqrt(state['grad_squared_sum'])
                    state['eta_grad'] = -grad_product
                    state['offset'] = -state['eta'] * state['old_grad']/np.sqrt(state['grad_squared_sum'])

                    p.data -= state['eta'] * state['old_grad']/np.sqrt(state['grad_squared_sum'])


                state['old_grad_norm_sq'] = state['grad_norm_sq']
                state['grad_norm_sq'] = grad.norm(2)**2
                state['grad_squared_sum'] += state['grad_norm_sq']

                state['old_grad'] = grad
                state['grad_norm_sq'] = grad.norm(2)**2
                state['max_grad'] = max(state['max_grad'], np.sqrt(state['old_grad_norm_sq']))
                if(group['unbiased']):
                    state['do_update'] = not state['do_update']

        loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                Lt = 2 * ( loss.data[0] - state['old_loss'].data[0] - (state['offset']*(p.grad.data)).sum() )/(0.0000000001+state['offset'].norm(2)**2)
                # print('Lt: ', Lt)
                # group['L'] = max(Lt, 0.001)#group['L'])
                # print(group['L'])

                L_loss = loss.data[0] - state['old_loss'].data[0] - (state['offset']*(p.grad.data)).sum() - state['L'] * state['offset'].norm(2)**2
                L_grad = -state['offset'].norm(2) * L_loss
                # if(L_loss > 0):
                #     L_grad = -state['offset'].norm(2)**2
                # else:
                #     L_grad = 0.1*state['offset'].norm(2)**2

                state['L_grad_sq_sum'] += L_grad**2
                state['L_grad_sum'] += L_grad

                state['L'] = max(-np.sign(state['L_grad_sum'])*(np.exp(np.abs(state['L_grad_sum'])/(np.sqrt(state['L_grad_sq_sum']))) - 1.0),0.0001)

                # state['L'] -= L_grad/np.sqrt(state['L_grad_sq_sum'])


                state['eta_mu_sum'] += group['L'] * state['old_grad_norm_sq']/state['grad_squared_sum']
                state['eta'] = state['eta_grad_sum']/state['eta_mu_sum']
                # state['eta'] = min(min(max(state['eta'],0.00001/np.sqrt(state['grad_squared_sum'])),100/np.sqrt(state['grad_squared_sum'])),1.0/state['max_grad'])
                state['eta'] = min(max(state['eta'],0.00001),100)

                # state['eta'] = min(max((state['eta_grad_sum']/state['eta_mu_sum']),0.00001/np.sqrt(state['grad_squared_sum'])),100/np.sqrt(state['grad_squared_sum']))
                print('eta',state['eta'])


        return loss



