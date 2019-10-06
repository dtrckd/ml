import numpy as np
import scipy as sp
from numpy import ma

from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma

from functools import partial, lru_cache


class ExpFamConjAbstract(object):

    ''' Base class fot conjugate exponential familly

        Attributes
        ----------
        _natex : list
            list of vectorized function to comptute the expeced natural parameters
            from the updated priors. (expected natural parameters equals partial derivative
            of the log partition.)
        _unin_priors : list
            list of uninformative priors (hyperparameters)

    '''

    def __init__(self):
        natfun = [getattr(self, a) for a in sorted(dir(self)) if (a.startswith('natex') and callable(getattr(self,a)))]
        self._natex = map(partial(np.vectorize, excluded=[1, 'ss']), natfun)

    def likelihood(self, *params, cache=250):
        ''' Returns a function that compute data likelihood with caching options'''
        pass

    def ss(self, x):
        ''' Compute sufficient statistics vector '''
        pass

    def random_init(self, shape):
        ''' Returns an randomly initialied matrix with given shape
            using the curent distribution.
        '''
        pass

    def predict_edge(self, pp):
        pass


class Bernoulli(ExpFamConjAbstract):
    def __init__(self):
        super().__init__()
        self._unin_priors = np.array([1,-0.69])

    @lru_cache(maxsize=200, typed=False)
    def ss(self, x):
        return np.asarray([x, 0])# Last dimension special treatment

    def expected_posterior(self, nat):

        m = nat[0] + nat[1]

        self.params = [m]
        return self.params

    def likelihood(self, cache=200):

        pdf = sp.stats.bernoulli(*self.params).pmf

        @lru_cache(maxsize=cache, typed=False)
        def compute(x):
            return pdf(x)
        #_likelihood =  defaultdict2(lambda x : sp.stats.norm.pdf(x, m, v)) # caching !
        return compute

    def random_init(self, shape):
        mat = np.random.random(shape)
        return mat

    def predict_edge(self, theta, phi, pp,  data):
        p = self.params[0]
        probas = [theta[i].dot(p).dot(theta[j]) for i,j,_ in data]
        return np.asarray(probas)

    #
    # Expected Natural Parameters
    #

    def natex1(self, pos, ss):
        k, l = np.unravel_index(pos, ss[0].shape)
        t1, t2 = ss[:, k, l]
        tau = psi(t1+1) - psi(t2-t1+1)
        return tau

    def natex2(self, pos, ss):
        k, l = np.unravel_index(pos, ss[0].shape)
        t1, t2 = ss[:, k, l]
        tau = psi(t2-t1+1) + psi(t2+2)
        return tau


class Normal(ExpFamConjAbstract):
    def __init__(self):
        super().__init__()
        #self._unin_priors = np.array([1,-1,1])
        self._unin_priors = np.array([1,-1,-1])

    @lru_cache(maxsize=200, typed=False)
    def ss(self, x):
        #return np.asarray([x, x**2, 1])
        return np.asarray([x, x**2, 0])# Last dimension special treatment

    def expected_posterior(self, nat):

        m = -0.5 *  nat[0] / nat[1]
        v = -0.5 / nat[1]
        #print('mean', m)
        #print('var',v)

        self.params = [m,v]
        return self.params

    def likelihood(self, cache=200):

        pdf = sp.stats.norm(*self.params).pdf

        @lru_cache(maxsize=cache, typed=False)
        def compute(x):
            return pdf(x)
        #_likelihood =  defaultdict2(lambda x : sp.stats.norm.pdf(x, m, v)) # caching !
        return compute

    def random_init(self, shape):
        mat = np.random.normal(1, 1, size=np.prod(shape)).reshape(shape)
        # variance from a gamma ?
        return mat

    def predict_edge(self, theta, phi, pp,  data):
        cdf = sp.stats.norm(*self.params).cdf(0.5)
        probas = [(1 - theta[i].dot(cdf).dot(theta[j])) for i,j,_ in data]
        return np.asarray(probas)

    #
    # Expected Natural Parameters
    #

    def natex1(self, pos, ss):
        k, l = np.unravel_index(pos, ss[0].shape)
        t1, t2, t3 = ss[:, k, l]
        tau = t1*(t3+1)/(t2*t3 - t1**2)
        return tau

    def natex2(self, pos, ss):
        k, l = np.unravel_index(pos, ss[0].shape)
        t1, t2, t3 = ss[:, k, l]
        tau = -t3*(t3+1)/(2*(t2*t3 - t1**2))
        return tau

    def natex3(self, pos, ss):
        k, l = np.unravel_index(pos, ss[0].shape)
        t1, t2, t3 = ss[:, k, l]
        tau = -0.5*((t1**2+t2)/(t2*t3-t1**2) - psi((t3+1)/2) - np.log(2*t3/(t2*t3-t1**2)))
        return tau


class Poisson(ExpFamConjAbstract):
    def __init__(self):
        super().__init__()
        self._unin_priors = np.array([0,-1])

    @lru_cache(maxsize=200, typed=False)
    def ss(self, x):
        return np.asarray([x, 0])# Last dimension special treatment

    def expected_posterior(self, nat):

        m = np.exp(nat[0])
        #assert(np.isclose(m, -nat[1]).all())
        self.params = [m]
        return self.params

    def likelihood(self, cache=200):

        pdf = sp.stats.poisson(*self.params).pmf

        @lru_cache(maxsize=cache, typed=False)
        def compute(x):
            return pdf(x)
        #_likelihood =  defaultdict2(lambda x : sp.stats.norm.pdf(x, m, v)) # caching !
        return compute

    def random_init(self, shape):
        mat = int(np.random.gamma(1,2,shape))+1
        # variance from a gamma ?
        return mat

    def predict_edge(self, theta, phi, pp,  data):
        cdf = sp.stats.poisson(*self.params).cdf(0.5)
        probas = [(1 - theta[i].dot(cdf).dot(theta[j])) for i,j,_ in data]
        return np.asarray(probas)

    #
    # Expected Natural Parameters
    #

    def natex1(self, pos, ss):
        k, l = np.unravel_index(pos, ss[0].shape)
        t1, t2 = ss[:, k, l]
        tau = psi(t1+1) - np.log(t2)
        return tau

    def natex2(self, pos, ss):
        k, l = np.unravel_index(pos, ss[0].shape)
        t1, t2 = ss[:, k, l]
        tau = -(t1+1)/t2
        return tau



