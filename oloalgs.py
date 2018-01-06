import numpy as np

EPSILON = 1e-8


def lpnorm(x, p=2):
    flat = np.reshape(x, np.size(x))
    return np.linalg.norm(flat, p)

def dot(x,y):
    flatx = np.reshape(x, np.size(x))
    flaty = np.reshape(y, np.size(y))
    return np.sum(x*y)

def lp_projector(x, d=1, p=2):
    norm = lpnorm(x, p)
    if(norm <= d):
        return x, np.zeros(shape=x.shape)

    projection = x/norm*d
    gradient = np.power(projection, p-1)*np.sign(x)
    return projection, gradient

def dual_lp(p):
    if(p==1):
        q = float('inf')
    else:
        q = 1.0/(1.0-1.0/p)
    return q

class OLOalgorithm:
    """base class for deriving online linear optimization algorithms."""

    def __init__(self, origin, verbose=False):
        self.origin = origin
        self.prediction = origin
        self.has_updated = False
        self.verbose = verbose

    def update(self, gradient):
        if(self.has_updated):
            raise RunTimeError('Called update twice for the same prediction!')
        self.has_updated = True

    def hint(self, hint):
        return
        # raise NotImplementedError

    def get_prediction(self):
        self.has_updated = False
        return self.prediction

class SOLO(OLOalgorithm):
    def __init__(self, origin=0, G=EPSILON,verbose=False):
        super(SOLO, self).__init__(origin,verbose)
        self.origin = origin
        self.G = G
        self.sum_grad_squares = 0
        self.sum_gradients = 0

    def update(self, gradient):
        super(SOLO, self).update(gradient)
        self.sum_grad_squares += gradient**2
        self.sum_gradients += gradient
        self.prediction = -self.sum_gradients/np.sqrt(self.G + self.sum_grad_squares)

class FreeRex(OLOalgorithm):
    def __init__(self, origin=0, lr=1.0,verbose=False):
        super(FreeRex, self).__init__(origin,verbose)
        self.one_over_eta_sq = 0
        self.grad_sum = 0
        self.max_grad = 0
        self.lr = lr

    def update(self, gradient):
        super(FreeRex, self).update(gradient)
        if(self.verbose):
            print('freerex gradient: ', gradient)
            print('freerex prediction: ',self.prediction)
            print('freerex maxgrad: ', self.max_grad)
        if(gradient != 0):
            self.max_grad = max(self.max_grad, np.abs(gradient))
            self.grad_sum += gradient
            self.one_over_eta_sq = max(self.one_over_eta_sq + 2*gradient**2, self.max_grad*np.abs(self.grad_sum))
            self.prediction = self.origin - np.sign(self.grad_sum) * (np.exp(np.abs(self.grad_sum)*self.lr/np.sqrt(self.one_over_eta_sq))-1.0)

class ONSCoinBetting1D(OLOalgorithm):
    """Uses ONS to adapt a betting fraction for 1-dimensional OLO"""

    def __init__(self, origin=0, G=1, epsilon=1, positive_only = False, domain=None, verbose=False):
        super(ONSCoinBetting1D, self).__init__(origin,verbose)

        self.beta = 0
        self.initial_beta_regularizer = G*0.25
        self.sum_beta_strong_convex_params = 0
        self.sum_beta_gradients = 0
        self.epsilon = epsilon
        self.reward = 0
        self.G = G
        self.initial_wealth = epsilon * G
        self.positive_only = 1.0
        self.sumgradsquares = 0.0
        self.domain = domain
        self.resets = 0
        self.since_last_reset = 0
        self.steps = 0
        self.last_gradient = None
        self.numpos = 0

        if(positive_only):
            self.positive_only = 0.0

    def hint(self, hint):
        self.G = max(abs(hint), EPSILON)



    def update(self, gradient):
        super(ONSCoinBetting1D, self).update(gradient)
        self.last_gradient = gradient

    def get_prediction(self):
        if self.last_gradient != None:
            self.do_update()

        return super(ONSCoinBetting1D, self).get_prediction()



    def do_update(self):
        gradient = self.last_gradient
        self.last_gradient = None


        clipped_gradient = np.clip(gradient,-self.G, self.G)
        if(self.domain != None):
            if self.prediction == self.domain[0] and clipped_gradient > 0:
                clipped_gradient = 0
            if self.prediction == self.domain[1] and clipped_gradient < 0:
                clipped_gradient = 0
        clipped_gradient = clipped_gradient/self.G

        self.reward += -clipped_gradient*self.prediction
        if(self.verbose):
            print('ONS1D clipped grad: ', clipped_gradient)
            print('ONS1D prediction: ', self.prediction)       
            print('ONS1D reward: ', self.reward)
            print('ONS1D wealth: ', self.reward + self.initial_wealth)
            print('ONS1D G: ', self.G)
            print('ONS1D beta: ', self.beta)
            print('ONS1D betamax: ', np.sqrt(self.steps)/(np.sqrt(self.steps)+1))
            print('ONS1D resets: ', self.resets)
            print('ONS1D since_last_reset: ',self.since_last_reset)
            print('ONS1D num pos: ', self.numpos)
        self.since_last_reset+=1
        self.steps += 1

        if(gradient > 0):
            self.numpos += 1

        if(abs(gradient) > self.G):
            self.resets += 1
            self.since_last_reset = 0
            ratio = self.G/abs(gradient)
            self.sum_beta_gradients *= ratio
            self.sum_beta_strong_convex_params *= ratio**2
        self.G = max(abs(gradient), self.G)

        beta_strong_convex_param = clipped_gradient**2/4
        beta_gradient = clipped_gradient/(1.0-self.beta*clipped_gradient) \
                        - self.beta*beta_strong_convex_param

        self.sum_beta_gradients += beta_gradient
        self.sum_beta_strong_convex_params += beta_strong_convex_param


        self.initial_beta_regularizer = 0.25
        self.initial_wealth = self.epsilon

        self.sumgradsquares += clipped_gradient*clipped_gradient

        self.beta = -(self.sum_beta_gradients)/(self.sum_beta_strong_convex_params + self.initial_beta_regularizer)

        self.beta = np.clip(self.beta, -np.sqrt(self.steps)/(np.sqrt(self.steps)+1) * self.positive_only, np.sqrt(self.steps)/(np.sqrt(self.steps)+1))

        self.prediction = (self.reward + self.initial_wealth) * self.beta
        if self.domain != None:
            self.prediction = np.clip(self.prediction, self.domain[0], self.domain[1])


class NDreduction(OLOalgorithm):
    def __init__(self, bounded_olo, unbounded_olo, origin=0, verbose=False):
        super(NDreduction, self).__init__(origin, verbose)
        self.bounded_olo = bounded_olo
        self.unbounded_olo = unbounded_olo
        self.origin = origin

    def hint(self, hint):
        self.unbounded_olo.hint(dot(hint, (self.bounded_olo.get_prediction() - self.origin)))
        self.bounded_olo.hint(hint)

    def update(self, gradient):
        super(NDreduction, self).update(gradient)
        if(self.verbose):
            print('NDreduction grad: ', gradient)
            print('Bounded prediction: ', self.bounded_olo.get_prediction())
            print('Unbounded prediction: ', self.unbounded_olo.get_prediction())
            print('Combo prediction: ',self.prediction)

        s = dot(gradient, (self.bounded_olo.get_prediction() - self.origin))
        self.unbounded_olo.update(s)
        self.bounded_olo.update(gradient)
        z = self.unbounded_olo.get_prediction()
        y = self.bounded_olo.get_prediction() - self.origin
        self.prediction = z*y + self.origin

class AdaGradFTRL(OLOalgorithm):
    def __init__(self, D, origin=0, normfn = lpnorm, dualnormfn = lpnorm):
        super(AdaGradFTRL, self).__init__(origin)

        self.D = D
        self.sum_gradients = 0
        self.sum_gradient_norm_square = 0
        self.epsilon = EPSILON
        self.normfn = normfn
        self.dualnormfn = dualnormfn

    def update(self, gradient):
        super(AdaGradFTRL, self).update(gradient)

        gradient_norm_square = self.dualnormfn(gradient)**2

        self.sum_gradients += gradient

        self.sum_gradient_norm_square += gradient_norm_square

        self.prediction = -self.D * self.sum_gradients / np.sqrt(2 * self.sum_gradient_norm_square + self.epsilon)

        prediction_norm = self.normfn(self.prediction)
        if(prediction_norm > self.D):
            self.prediction = self.D*self.prediction/prediction_norm



def AdaGradONSBetting(G=1.0, epsilon=1.0, origin = 0, normfn = lpnorm, dualnormfn = lpnorm):
    adagrad = AdaGradFTRL(1, origin*0, normfn, dualnormfn)

    onsbetting = ONSCoinBetting1D(G, epsilon)

    return NDreduction(adagrad, onsbetting, origin)

def AdaGradFreeRex(origin=0):
    adagrad = AdaGradFTRL(1, origin*0, lpnorm, lpnorm)
    freerex = FreeRex(0,1)
    ons_betting = ONSCoinBetting1D()
    return NDreduction(adagrad, ons_betting, origin)


def AdaGradONSBettingLp(G=1.0, epsilon=1.0, p=2):

    q = dual_lp(p)
    normfn = lambda x:lpnorm(x, p)

    dualnormfn = lambda x: lpnorm(x, q)


    return AdaGradONSBetting(G, epsilon, 0, normfn, dualnormfn)

class BoundedOptimizer(OLOalgorithm):
    """
    projector(x) = Pi_K(x), \nabla S_K(x)
    """
    def __init__(self, unbounded_olo, projector=lp_projector, \
        get_dual_norm=lpnorm, verbose=False):
        super(BoundedOptimizer, self).__init__(0, verbose)
        self.projector = projector
        self.unbounded_olo = unbounded_olo
        self.prediction, self.projection_gradient = self.projector(self.unbounded_olo.get_prediction())
        self.get_dual_norm = get_dual_norm

    def hint(self, hint):
        hint_norm = self.get_dual_norm(hint)
        self.unbounded_olo.hint(hint + hint_norm*self.projection_gradient)

    def update(self, gradient):
        super(BoundedOptimizer, self).update(gradient)
        gradient_norm = self.get_dual_norm(gradient)
        gradient_correction = gradient + gradient_norm*self.projection_gradient
        self.unbounded_olo.update(gradient_correction)
        if(self.verbose):
            print('unbounded eta: ', self.unbounded_olo.get_prediction())
            print('original gradient: ',gradient)
            print('projection_gardeitn: ', self.projection_gradient)
            print('corrected gradient: ', gradient_correction)
        self.prediction, self.projection_gradient = self.projector(self.unbounded_olo.get_prediction())


def LpBoundedOptimizer(G=1.0, d=1.0, p=2):
    unbounded_olo = AdaGradONSBettingLp(G, p)
    projector = lambda x: lp_projector(x, d, p)
    q = dual_lp(p)
    get_dual_norm = lambda x: lpnorm(x, q)
    return BoundedOptimizer(unbounded_olo, projector, get_dual_norm)

def parabolic_projector(x):
    if np.linalg.norm(x)==0:
        x = np.array([0,0])

    if(x[0]<0):
        projection = np.array([0.0, max(x[1], 0.0)])
        displacement = x - projection
        gradient = displacement/lpnorm(displacement)
        return projection, displacement
    # if(x[0]>100):
    #     projection = np.array([1, max(x[1], 1)])
    #     displacement = x - projection
    #     gradient = displacement/lpnorm(displacement)
    #     print('projection: ', projection, gradient)
    #     return projection, displacement        

    if(x[1]>x[0]**2):
        return x, np.zeros(shape=x.shape)


    # minimize
    # (a-x[0])^2 + (a^2-x[1])^2

    # differentiate and simplify:
    # 2(a-x[0]) + 4 a (a^2 - x[1]) = 0

    # -2x + a(2-4x[1]) + 4a^3 = 0

    # a^3 + (0.5- x[1]) a = 0.5 x[0]


    # This is a cubic in 'standard form'. We solve following http://mathworld.wolfram.com/CubicFormula.html 
    # Defined p and q appropriately so that
    # a^3 + p a = q
    # Make a magic substitution:
    # a = w - p/3w

    # The cubic becomes
    # w^3 - p^3/(27 * w^3) - q =0

    # Multiply through by w^3 and this becomes a quadratic in w^3. Using quadratic formula:

    # w^3 = 0.5 ( q +/- np.sqrt(q^2+ 4.0/27.0 * p^3) )

    p = 0.5 - x[1]
    q = 0.5 * x[0]

    wcubed = 0.5 * (q + np.lib.scimath.sqrt(q**2 + 4.0/27.0 * p**3))
    w = np.power(wcubed,1.0/3.0)

    a = np.real(w - p/(3 * w))

    projection = np.array([a, a**2])

    displacement = x - projection

    gradient = displacement/lpnorm(displacement)
    return projection, gradient

def parabolic_bounded_optimizer(G=1.0, epsilon=1.0, p=2):
    # unbounded_olo = AdaGradONSBettingLp(G, epsilon, p)
    unbounded_olo = AdaGradFreeRex()
    if(p==2):
        projector = parabolic_projector
    elif(p==1):
        projector = parabolic_projector_l1
    elif(p==float('inf')):
        projector = parabolic_projector_linf
    else:
        raise SyntaxError('Incorrect p!')
    return BoundedOptimizer(unbounded_olo, projector)



def parabolic_projector_l1(x):
    '''THE GRADIENT COMPUTATION IS BROKEN HERE'''
    if np.linalg.norm(x)==0:
        x = np.array([0,0])

    if(x[1]>x[0]**2):
        return x, np.zeros(shape=x.shape)

    if(x[0]<0):
        projection = np.array([0, max(x[1], 0)])
        displacement = x - projection
        gradient = np.zeros(2)
        maxcoord = np.argmax(displacement)
        gradient[maxcoord] = np.sign(displacement[maxcoord])
        gradient = np.sign(displacement)#displacement/(0.0000001+ lpnorm(displacement, p=1))

        return projection, gradient



    # minimize
    # |a-x[0]| + |a^2-x[1]|

    # differentiate and simplify:
    # sign(a-x[0]) + 2 a sign(a^2 - x[1]) = 0

    # case 1: x[0] < 0.5
    # set a = x[0]. Since x[1] < x[0]^2, sign(a^2- x[1]) = 1
    # sign(a-x[0]) + 2 x[0] = 0
    
    # Case 2: x[0] > 0.5, x[1] > 0.25
    # set a = sqrt(x[1]). Since sqrt(x[1]) < x[0], sign(a - x[0]) = -1
    # also, 2 sqrt(x[1]) >= 1
    # sign(a- x[0]) + 2 sqrt(x[1]) sign(a^2 - x[1])

    # case 3: x[0] > 0.5, x[1] < 0.25
    # set a = 0.5
    # sign(a- x[0]) + 2a sign(a^2 - x[1]) = -1 + 1 = 0



    if(x[0] < 0.5):
        projection = np.array([x[0], x[0]**2])
    elif(x[1]> 0.25):
        projection = np.array([np.sqrt(x[1]), x[1]])
    else:
        projection = np.array([0.5, 0.25])


    displacement = x - projection
    gradient = np.zeros(2)
    maxcoord = np.argmax(displacement)
    gradient[maxcoord] = np.sign(displacement[maxcoord])
    gradient = np.sign(displacement)


    return projection, gradient

def parabolic_projector_linf(x):
    '''THE GRADIENT COMPUTATION IS BROKEN HERE'''
    if np.linalg.norm(x)==0:
        x = np.array([0,0])

    if(x[1]>x[0]**2):
        return x, np.zeros(shape=x.shape)

    if(x[0]<0):
        projection = np.array([0, max(x[1], 0)])
        displacement = x - projection
        gradient = np.zeros(2)
        maxcoord = np.argmax(displacement)
        gradient[maxcoord] = np.sign(displacement[maxcoord])

        return projection, gradient



    # minimize
    # max(|a-x[0]| + |a^2-x[1]|)

    d = 0.5 * (1 + 2 * x[0] - np.sqrt(1 + 4*x[0] + 4*x[1]))
    a = x[0] - d

    projection = np.array([a, a**2])

    displacement = x - projection
    gradient = np.zeros(2)
    maxcoord = np.argmax(displacement)
    gradient[maxcoord] = np.sign(displacement[maxcoord])


    return projection, gradient
