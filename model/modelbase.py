import sys
from time import time

import numpy as np
import scipy as sp
from numpy import ma

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import mean_squared_error

import pymake.io as io
from pymake import logger
from pymake.model import ModelBase

#import sppy


class RandomGraphModel(ModelBase):

    def _init_params(self, frontend):
        frontend = self.frontend

        # Save the testdata
        if hasattr(self.frontend, 'data_test'):
            data_test = frontend.data_test_w

            N = frontend.num_nodes()
            valid_ratio = frontend.get_validset_ratio() *2 # Because links + non_links...
            n_valid = np.random.choice(len(data_test), int(np.round(N*valid_ratio / (1+valid_ratio))), replace=False)
            n_test = np.arange(len(data_test))
            n_test[n_valid] = -1
            n_test = n_test[n_test>=0]
            self.data_test = data_test[n_test]
            self.data_valid = data_test[n_valid]

            # if kernel == bernoulli
            # For fast computation of bernoulli pmf.
            self._w_a = self.data_valid[:,2].T.astype(int)
            self._w_a[self._w_a > 0] = 1
            self._w_a[self._w_a == 0] = -1
            self._w_b = np.zeros(self._w_a.shape, dtype=int)
            self._w_b[self._w_a == -1] = 1

        # Data statistics
        _len = {}
        _len['K'] = self.expe.get('K')
        _len['N'] = frontend.num_nodes()
        _len['E'] = frontend.num_edges()
        _len['dims'] = frontend.num_neighbors()
        _len['nnz_ones'] = frontend.num_edges()
        _len['nnzsum'] = frontend.num_nnzsum()
        _len['nnz'] = frontend.num_nnz()
        #_len['nnz_t'] = frontend.num_nnz_t()
        self._len = _len

        self._is_symmetric = frontend.is_symmetric()

        self.log.info("Fitting `%s' model" % (type(self)))

    # for conveniance (self._params...)
    def _reduce_latent(self):
        return self._theta, self._phi


    # the Binaray case (Bernoulli model)
    def likelihood(self, theta=None, phi=None, data='valid'):
        """ Compute data likelihood (abrev. ll) with the given estimators
            onthe given set of data.

            Parameters
            ----------
            data: str
                valid -> validation data
                test -> test data
        """
        if theta is None:
            theta = self._theta
        if phi is None:
            phi = self._phi

        # @warning: assume data == 'valid' !
        assert(data == 'valid')

        if data == 'valid':
            data = self.data_valid
        elif data == 'test':
            data = self.data_test

        qijs = np.array([ theta[i].dot(phi).dot(theta[j]) for i,j,_ in data])

        qijs = qijs * self._w_a + self._w_b

        #qijs = ma.masked_invalid(qijs)
        qijs[qijs<=1e-300] = 1e-200
        return qijs

    def posterior(self, theta=None, phi=None, data='test'):
        """ Compute the predictive posterior (abrev. pp) with the given estimators
            onthe given set of data.

            Notes
            -----
            return the expected posterior $\hat \theta = E[\theta | X_train, \mu]$?
            such that x_pred ~ E_M[x_pred | \hat \theta]
        """

        if theta is None:
            theta = self._theta
        if phi is None:
            phi = self._phi

        if data == 'valid':
            data = self.data_valid
        elif data == 'test':
            data = self.data_test

        qijs = np.array([ theta[i].dot(phi).dot(theta[j]) for i,j,_ in data])

        #qijs = ma.masked_invalid(qijs)
        return qijs


    def compute_entropy(self, theta=None, phi=None, data='valid', **kws):
        if theta is None:
            theta, phi = self._reduce_latent()
        qij = self.likelihood(theta, phi, data)
        data = getattr(self, 'data_'+data)

        ll = qij

        # Log-likelihood (Perplexity is 2**H(X).)
        ll = np.log(ll).sum()
        return ll

    def compute_roc(self, theta=None, phi=None, data='test', **kws):
        if theta is None:
            theta, phi = self._reduce_latent()
        qij = self.posterior(theta, phi, data)
        data = getattr(self, 'data_'+data)

        self._y_true = np.squeeze(data[:,2].T).astype(bool)*1
        self._probas = qij

        fpr, tpr, thresholds = roc_curve(self._y_true, self._probas)
        roc = auc(fpr, tpr)
        return roc

    def compute_roc2(self, theta=None, phi=None, data='test', **kws):
        ''' Based on the count in hard class (argmax) assignement and data count. '''
        if theta is None:
            theta, phi = self._reduce_latent()

        N,K = self._theta.shape
        # class assignement
        c = self._theta.argmax(1) # @cache

        theta_hard = np.zeros_like(self._theta)
        theta_hard[np.arange(len(self._theta)), c] = 1 # @cache
        c_len = theta_hard.sum(0)

        # number of possible edges per block
        norm = np.outer(c_len,c_len)
        if not self._is_symmetric:
            np.fill_diagonal(norm, 2*(norm.diagonal()-c_len))
        else:
            np.fill_diagonal(norm, norm.diagonal()-c_len)
        norm = ma.masked_where(norm<=0, norm)

        # Expected weight per block
        pp = np.zeros((K,K))
        weights = self.frontend.data.ep['weights']
        #edges = self.frontend.data.get_edges()
        try: # 2.29 breaks comptibility
            edges = self.frontend.data.get_edges([self.frontend.data.edge_index])
        except TypeError as e:
            edges = self.frontend.data.get_edges()
        edges[:,2] = np.array([weights[i,j] for i,j,_ in edges])
        for i,j,w in edges:
            pp[c[i], c[j]] += 1

        pp = pp / norm

        data = getattr(self, 'data_'+data)
        probas = ma.array([ pp[c[i], c[j]] for i,j,_ in data])

        if ma.is_masked(probas):
            return None

        fpr, tpr, thresholds = roc_curve(self._y_true, probas)
        roc = auc(fpr, tpr)
        return roc

    def compute_pr(self, *args, **kwargs):
        # @Warning: have to be computed after compute_roc.
        # should ba place after roc string in {_format}.
        return average_precision_score(self._y_true, self._probas)

    def compute_wsim(self, theta=None, phi=None, data='test', **kws):
        ''' Based on the baysian posterior. '''
        if theta is None:
            theta, phi = self._reduce_latent()
        qij = self.posterior(theta, phi, data)
        data = getattr(self, 'data_'+data)

        wd = data[:,2].T
        ws = qij

        idx = wd > 0
        wd = wd[idx]
        ws = ws[idx]

        ## l1 norm
        #nnz = len(wd)
        #mean_dist = np.abs(ws - wd).sum() / nnz
        ## L2 norm
        mean_dist = mean_squared_error(wd, ws)

        return mean_dist


    def compute_wsim2(self, theta=None, phi=None, data='test', **kws):
        ''' Based on the count in hard class (argmax) assignement and data count. '''
        if theta is None:
            theta, phi = self._reduce_latent()

        N,K = self._theta.shape
        # class assignement
        c = self._theta.argmax(1) # @cache

        theta_hard = np.zeros_like(self._theta)
        theta_hard[np.arange(len(self._theta)), c] = 1 # @cache
        c_len = theta_hard.sum(0)

        # number of possible edges per block
        norm = np.outer(c_len,c_len)
        if not self._is_symmetric:
            np.fill_diagonal(norm, 2*(norm.diagonal()-c_len))
        else:
            np.fill_diagonal(norm, norm.diagonal()-c_len)
        norm = ma.masked_where(norm<=0, norm)

        # Expected weight per block
        pp = np.zeros((K,K))
        weights = self.frontend.data.ep['weights']
        edges = self.frontend.data.get_edges()
        edges[:,2] = np.array([weights[i,j] for i,j,_ in edges])
        for i,j,w in edges:
            pp[c[i], c[j]] += w

        pp = pp / norm

        data = getattr(self, 'data_'+data)
        qij = ma.array([ pp[c[i], c[j]] for i,j,_ in data])

        if ma.is_masked(qij):
            return None

        wd = data[:,2].T
        ws = qij

        idx = wd > 0
        wd = wd[idx]
        ws = ws[idx]

        ## l1 norm
        #nnz = len(wd)
        #mean_dist = np.abs(ws - wd).sum() / nnz
        ## L2 norm
        mean_dist = mean_squared_error(wd, ws)

        return mean_dist

    def get_ytrue_probas(self):
        return self._y_true, self._probas


### @DEPRECATED

class SVB(ModelBase):

    '''online EM/SVB

        Note: this obsolete base class is suited for dense masked numpy array.
              Base class for dense matrix svb...
    '''

    __abstractmethods__ = 'model'

    def __init__(self, expe, frontend):
        super(SVB, self).__init__(expe, frontend)

        self.entropy_diff = 0

    def _update_chunk_nnz(self, groups):
        _nnz = []
        for g in groups:
            if self._is_symmetric:
                count = 2* len(g)
                #count =  len(g)
            else:
                count = len(g)
            _nnz.append(count)

        self._nnz_vector = _nnz

    def _get_chunk(self):
        chunk = self.expe.get('chunk', 100)

        if isinstance(chunk, (int, float)):
            return chunk
        elif isinstance(chunk, str):
            pass
        else:
            raise TypeError('Unknown chunk type: %s' % type(chunk))

        mean_nodes = np.mean(self._len['dims'])
        try:
            chunk_mode, ratio = chunk.split('_')
        except ValueError as e:
            return float(chunk)

        if chunk_mode == 'adaptative':
            if '.' in ratio:
                ratio = float(ratio)
            else:
                ratio = int(ratio)

            chunk = ratio * mean_nodes
        else:
            raise ValueError('Unknown chunk mode: %s' % chunk_mode)

        return chunk


    def fit(self, *args, **kwargs):
        ''' chunk is the number of row to threat in a minibach '''

        self._init()

        data_ma = self.frontend.data_ma
        groups = self.data_iter(data_ma, randomize=True)
        chunk_groups = np.array_split(groups, self.chunk_len)
        self._update_chunk_nnz(chunk_groups)

        self._entropy = self.compute_entropy()
        print( '__init__ Entropy: %f' % self._entropy)
        for _id_mnb, minibatch in enumerate(chunk_groups):

            self.mnb_size = self._nnz_vector[_id_mnb]

            begin_it = time()
            self.sample(minibatch)

            self.compute_measures(begin_it)
            print('.', end='')
            self.log.info('Minibatch %d/%d, %s, Entropy: %f,  diff: %f' % (_id_mnb+1, self.chunk_len, '/'.join((self.expe.model, self.expe.corpus)),
                                                                        self._entropy, self.entropy_diff))
            if self.expe.get('_write'):
                self.write_current_state(self)
                if _id_mnb > 0 and _id_mnb % self.snapshot_freq == 0:
                    self.save(silent=True)
                    sys.stdout.flush()

    def sample(self, minibatch):
        '''
        Notes
        -----
        * self.iterations is actually the burnin phase of SVB.
        '''

        # not good
        #self._timestep_a = 0

        # self.iterations is actually the burnin phase of SVB.
        for _it in range(self.iterations+1):
            self.log.debug('init timestep A ?')
            #self._timestep_a = 0 # test, not good
            self._iteration = _it
            burnin = (_it < self.iterations)
            np.random.shuffle(minibatch)
            for _id_token, iter in enumerate(minibatch):
                self._id_token = _id_token
                self._xij = self.frontend.data_ma[tuple(iter)]
                self.maximization(iter)
                self.expectation(iter, burnin=burnin)


            if self._iteration != self.iterations-1 and self.expe.get('_verbose', 20) < 20:
                self.compute_measures()
                self.log.debug('it %d,  ELBO: %f, elbo diff: %f \t K=%d' % (_it, self._entropy, self.entropy_diff, self._K))
                if self.expe.get('_write'):
                    self.write_current_state(self)


        # Update global parameters after each "minibatech of links"
        # @debug if not, is the time_step evolve differently for N_Phi and N_Theta.
        minibashed_finished = True
        if minibashed_finished:
            self._purge_minibatch()

        #if self.elbo_diff < self.limit_elbo_diff:
        #    print('ELBO get stuck during data iteration: Sampling useless, return ?!')



class GibbsSampler(ModelBase):
    ''' Implemented method, except fit (other?) concerns MMM type models:
        * LDA like
        * MMSB like

        but for other (e.g IBP based), method has to be has to be overloaded...
        -> use a decorator @mmm to get latent variable ...
    '''
    __abstractmethods__ = 'model'
    def __init__(self, expe, frontend):
        super(GibbsSampler, self).__init__(expe, frontend)

    #@mmm
    def compute_measures(self, begin_it=0):

        if self.expe.get('deactivate_measures'):
            return

        self._entropy = self.compute_entropy()

        if hasattr(self, 'compute_entropy_t'):
            self._entropy_t = self.compute_entropy_t()
        else:
            self._entropy_t = np.nan

        if '_roc' in self.expe._measures:
            self._roc = self.compute_roc()

        self._alpha = np.nan
        self._gmma= np.nan
        self.alpha_mean = np.nan
        self.delta_mean = np.nan
        self.alpha_var = np.nan
        self.delta_var= np.nan

        self.time_it = time() - begin_it

    def fit(self, *args, **kwargs):

        self._init()

        self._entropy = self.compute_entropy()
        print( '__init__ Entropy: %f' % self._entropy)

        for _it in range(self.iterations):
            self._iteration = _it

            ### Sampling
            begin_it = time()
            self.sample()

            if _it >= self.iterations - self.burnin:
                if _it % self.thinning == 0:
                    self.samples.append([self._theta, self._phi])

            self.compute_measures(begin_it)
            print('.', end='')
            self.log.info('iteration %d, %s, Entropy: %f \t\t K=%d  alpha: %f gamma: %f' % (_it, '/'.join((self.expe.model, self.expe.corpus)),
                                                                                    self._entropy, self._K,self._alpha, self._gmma))
            if self.expe.get('_write'):
                self.write_current_state(self)
                if _it > 0 and _it % self.snapshot_freq == 0:
                    self.save(silent=True)
                    sys.stdout.flush()

        ### Clean Things
        if not self.samples:
            self.samples.append([self._theta, self._phi])
        self._reduce_latent()
        self.samples = None # free space
        return


    #@mmm
    def update_hyper(self, hyper):
        if hyper is None:
            return
        elif isinstance(type(hyper), (tuple, list)):
            alpha = hyper[0]
            gmma = hyper[1]
            delta = hyper[2]
        else:
            delta = hyper.get('delta')
            alpha = hyper.get('alpha')
            gmma = hyper.get('gmma')

        if delta:
            self._delta = delta
        if alpha:
            self._alpha = alpha
        if gmma:
            self._gmma = gmma


    # keep only the most representative dimension (number of topics) in the samples
    #@mmm
    def _reduce_latent(self):
        theta, phi = list(map(list, zip(*self.samples)))
        ks = [ mat.shape[1] for mat in theta]
        bn = np.bincount(ks)
        k_win = np.argmax(bn)
        self.log.debug('K selected: %d' % k_win)

        ind_rm = []
        [ind_rm.append(i) for i, v in enumerate(theta) if v.shape[1] != k_win]
        for i in sorted(ind_rm, reverse=True):
            theta.pop(i)
            phi.pop(i)

        self.log.debug('Samples Selected: %d over %s' % (len(theta), len(theta)+len(ind_rm) ))

        self._theta = np.mean(theta, 0)
        self._phi = np.mean(phi, 0)
        self.K = self._theta.shape[1]
        return self._theta, self._phi



