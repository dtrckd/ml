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

        return qijs

    def posterior(self, theta=None, phi=None, data='test'):
        """ Compute the predictive posterior with the given estimators
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

        print('theta:', theta)
        print('phi:', phi)

        qijs = np.array([ theta[i].dot(phi).dot(theta[j]) for i,j,_ in data])

        print(qijs)

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

    #@pysnooper.snoop()
    def fit(self, frontend):
        self._init(frontend)

        #Y = frontend.adj()
        K = self._len['K']
        N = self._len['N']
        kernel = self.expe.kernel

        ### Init Param
        self.alpha0 = np.array([1/K]*K)
        self._theta = np.random.dirichlet([0.5]*K, N)
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
        while tau_sensibility > self.expe.tau_tol:

            # block-block loop (phi updates)
            phi_sink = np.zeros(phi_shape)
            for i, j, w in edges:
                kk_outer = np.outer(self._theta[i], self._theta[j])
                kk_outer = np.tile(kk_outer, phi_dim)
                phi_sink += pm.ss(w).reshape(phi_dim) * kk_outer

            phi_sink += pm._unin_priors.reshape(phi_dim)
            self.compute_natural_expectations(phi_sink)
            self._tau = phi_sink

            mu_sensibility = 1
            while mu_sensibility > self.expe.mu_tol:
                for i in np.random.choice(N, size=N, replace=False):
                    theta_sink = np.zeros(phi_shape)
                    for j, w in neigs[i][0]:
                        kk_outer = np.tile(self._theta[j], (nat_dim, K, 1))
                        theta_sink += pm.ss(w).reshape(phi_dim) * kk_outer

                    if not self._is_symmetric:
                        for j, w in neigs[i][1]:
                            kk_outer = np.tile(self._theta[j][np.newaxis].T, (nat_dim, 1, K))
                            theta_sink += pm.ss(w).reshape(phi_dim) * kk_outer

                    theta_i = np.empty_like(self._theta[0])
                    for k in range(K):
                        theta_sink_cross = np.zeros(theta_sink.shape)
                        theta_sink_cross[:,k,:] = theta_sink[:,k,:]
                        theta_sink_cross[:,:,k] = theta_sink[:,:,k]
                        theta_i[k] = np.sum(theta_sink_cross * phi_sink)

                    # Normalize _theta
                    self._theta[i] = expnormalize(theta_i)
                    #print('h ',self._theta[i])

                mu  = self._theta
                mu_sensibility = np.absolute(mu - old_mu).sum()
                old_mu = mu.copy()
                print('mu: %.4f' % mu_sensibility)

            self.compute_measures()
            if self.expe.get('_write'):
                self.write_current_state(self)
                if self._measure_cpt % self.snapshot_freq == 0:
                    self.save(silent=True)

            tau = phi_sink
            tau_sensibility = np.absolute(tau - old_tau).sum()
            old_tau = tau
            print('tau: %.4f' % tau_sensibility)
            self._tau = tau

            ### DEBUG
            mean = (tau[0] / tau[2]).mean()
            var = ((tau[2]+1)/2 * 2* tau[2] / (tau[1]*tau[2] - tau[0]**2)).mean()
            print('mean: %.3f, var: %.3f' % (mean, var))

            #self.compute_measures()
            #if self.expe.get('_write'):
            #    self.write_current_state(self)
            #    if self._measure_cpt % self.snapshot_freq == 0:
            #        self.save(silent=True)


    def compute_natural_expectations(self, ss):
        ''' Compute the log partition gradient.
            (i.e. the parameter moments (mean, var, etc))

            :ss: vector of sufficient statistics
            :priors: vector of hyperpriors
        '''

        for i, func in enumerate(self.prior_model._natex):
            ss[i] = func(np.arange(ss[i].size, dtype=int).reshape(ss[i].shape), ss)




