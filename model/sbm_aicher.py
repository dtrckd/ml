import numpy as np
import scipy as sp

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import mean_squared_error

from pymake.util.math import expnormalize, lognormalize
from ml.model import ExpFamConj
from ml.model import RandomGraphModel

import pysnooper
#np.seterr(all='raise')

class sbm_aicher(RandomGraphModel):

    ''' Implement a SBM with Variational Inferenced based on the paper:
            Aicher, Christopher, Abigail Z. Jacobs, and Aaron Clauset. "Adapting the stochastic block model to edge-weighted networks." (2013).

        It implements various kernel:
        * Bernoulli: Original SBM
        * Poisson: Weighted SBM (WSBM)
        * Gaussian: Weighted SBM (WSBM)
    '''

    def _reduce_latent(self):
        p = self.prior_model.expected_posterior(self._tau)
        self._phi = p[0]
        return self._theta, self._phi

    def likelihood(self, theta=None, phi=None, data='valid'):
        """ Compute data likelihood (abrev. ll) with the given estimators
            onthe given set of data.
            :data: str
                valid -> validation data
                test -> test data
        """
        if theta is None:
            theta = self._theta
        if phi is None:
            phi = self._phi

        if data == 'valid':
            data = self.data_valid
        elif data == 'test':
            data = self.data_test

        _likelihood = self.prior_model.likelihood()

        qijs = np.array([ theta[i].dot(_likelihood(xij)).dot(theta[j]) for i,j,xij in data])

        qijs[qijs<=1e-300] = 1e-200
        return qijs

    def compute_roc(self, theta=None, phi=None, data='test', **kws):
        if theta is None:
            theta, phi = self._reduce_latent()
        qij = self.posterior(theta, phi, data)
        data = getattr(self, 'data_'+data)

        self._y_true = np.squeeze(data[:,2].T).astype(bool)*1
        self._probas = self.prior_model.predict_edge(theta, phi, qij, data)

        fpr, tpr, thresholds = roc_curve(self._y_true, self._probas)
        roc = auc(fpr, tpr)
        return roc

    def compute_wsim(self, theta=None, phi=None, data='test', **kws):
        if self.expe.kernel == 'bernoulli':
            if theta is None:
                theta, phi = self._reduce_latent()

            N,K = self._theta.shape
            # class assignement
            c = np.argmax(self._theta, 1)

            # number of possible edges per block
            c_len = np.sum(self._theta, 0)
            norm = np.outer(c_len,c_len)
            if not self._is_symmetric:
                np.fill_diagonal(norm, 2*(norm.diagonal()-N))
            else:
                np.fill_diagonal(norm, norm.diagonal()-N)

            # Expected weight per block
            pp = np.zeros((K,K))
            weights = self.frontend.data.ep['weights']
            edges = self.frontend.data.get_edges()
            edges[:,2] = np.array([weights[i,j] for i,j,_ in edges])
            for i,j,w in edges:
                pp[c[i], c[j]] += w

            pp /= norm

            data = getattr(self, 'data_'+data)
            qij = np.array([ pp[c[i], c[j]] for i,j,_ in data])

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
        else:
            return super().compute_wsim(theta=None, phi=None, data='test', **kws)

    #@pysnooper.snoop()
    def fit(self, frontend):
        self._init(frontend)

        #Y = frontend.adj()
        K = self._len['K']
        N = self._len['N']
        kernel = self.expe.kernel

        ### Init Param
        alpha0 = 1/K
        self._theta = np.random.dirichlet([alpha0]*K, N)
        pm = self.prior_model = ExpFamConj[kernel]()

        nat_dim = len(self.prior_model._unin_priors)
        phi_shape = (nat_dim, K, K)
        phi_dim = (nat_dim, 1, 1)

        tau_sensibility = 1

        old_tau = np.zeros(phi_shape)
        old_mu = np.zeros(self._theta.shape)

        weights = frontend.data.ep['weights']
        edges = frontend.data.get_edges()
        edges[:,2] = np.array([weights[i,j] for i,j,_ in edges])
        neigs = []
        for v in range(N):
            _out = np.asarray([(int(_v),weights[v, _v]) for _v in frontend.data.vertex(v).out_neighbors()])
            _in  = np.asarray([(int(_v),weights[_v, v]) for _v in frontend.data.vertex(v).in_neighbors()])
            neigs.append([_out, _in])

        ### Loop
        max_iter = self.expe.get('max_iter', 100)
        it_loop1 = 0
        while tau_sensibility > self.expe.tau_tol and it_loop1 < max_iter:

            # block-block loop (phi updates)
            phi_sink = np.zeros(phi_shape)
            for i, j, w in edges:
                kk_outer = np.outer(self._theta[i], self._theta[j])
                kk_outer = np.tile(kk_outer, phi_dim)
                phi_sink += pm.ss(w).reshape(phi_dim) * kk_outer

            # if last dimension of T is 1 update manually
            for k1, k2 in np.ndindex(K,K):
                phi_sink[-1, k1,k2] = self._theta[:,k1].sum() * self._theta[:,k2].sum()

            phi_sink += pm._unin_priors.reshape(phi_dim)
            self._tau = self.compute_natural_expectations(phi_sink)

            tau_sensibility = np.absolute(self._tau - old_tau).sum()
            old_tau = self._tau
            print('tau: %.4f' % tau_sensibility)
            it_loop1 +=1

            mu_sensibility = 1
            it_loop2 = 0
            while mu_sensibility > self.expe.mu_tol and it_loop2 < max_iter:
                for i in np.random.choice(N, size=N, replace=False):
                    theta_sink = np.zeros(phi_shape)
                    for j, w in neigs[i][0]:
                        kk_outer = np.tile(self._theta[j], (nat_dim, K, 1))
                        theta_sink += pm.ss(w).reshape(phi_dim) * kk_outer

                    # if last dimension of T is 1 update manually
                    for k in range(K):
                        theta_sink[-1,k] += np.ones(K) * (self._theta[:i,k].sum(0)+self._theta[i+1:,k].sum(0))

                    if not self._is_symmetric:
                        for j, w in neigs[i][1]:
                            kk_outer = np.tile(self._theta[j][np.newaxis].T, (nat_dim, 1, K))
                            theta_sink += pm.ss(w).reshape(phi_dim) * kk_outer

                        # @debug symmetric, not sure
                        # if last dimension of T is 1 update manually
                        for k in range(K):
                            theta_sink[-1, :, k] += np.ones(K) * (self._theta[:i,k].sum(0)+self._theta[i+1:,k].sum(0))

                    theta_i = np.empty_like(self._theta[0])
                    for k in range(K):
                        theta_sink_cross = np.zeros(theta_sink.shape)
                        theta_sink_cross[:,k,:] = theta_sink[:,k,:]
                        theta_sink_cross[:,:,k] = theta_sink[:,:,k]
                        theta_i[k] = np.sum(theta_sink_cross * self._tau)

                    # Normalize _theta
                    self._theta[i] = expnormalize(theta_i)

                mu  = self._theta
                mu_sensibility = np.absolute(mu - old_mu).sum()
                old_mu = mu.copy()
                print('mu: %.4f' % mu_sensibility)
                it_loop2 +=1

                self.compute_measures()
                if self.expe.get('_write'):
                    self.write_current_state(self)
                    if self._measure_cpt % self.snapshot_freq == 0:
                        self.save(silent=True)

            ### DEBUG
            tau = self._tau
            mean = (-0.5*tau[0] / tau[1])
            var = (-0.5/tau[1])
            print('mean: %.3f, var: %.3f' % (mean.mean(), var.var()))
            print('mean: %.3f, var: %.3f' % (mean.max(), var.max()))




    def compute_natural_expectations(self, ss):
        ''' Compute the log partition gradient.
            (i.e. the parameter moments (mean, var, etc))

            :ss: vector of sufficient statistics
            :priors: vector of hyperpriors
        '''

        tau = np.empty_like(ss)

        for i, func in enumerate(self.prior_model._natex):
            tau[i] = func(np.arange(ss[i].size, dtype=int).reshape(ss[i].shape), ss)

        return tau




