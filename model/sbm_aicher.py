import numpy as np
import scipy as sp

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import mean_squared_error

from pymake.util.math import expnormalize, lognormalize
from ml.model import ExpFamConj
from ml.model import RandomGraphModel

#import pysnooper
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

        # @cache
        theta = np.zeros_like(self._theta)
        theta[np.arange(len(self._theta)), self._theta.argmax(1)] = 1

        # @Warning: let self._theta uncertaint
        return theta, self._phi

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

        qijs = np.array([ theta[i].dot(_likelihood(int(xij>0))).dot(theta[j]) for i,j,xij in data])

        with np.errstate(all='raise'):
            try:
                qijs[qijs<=1e-300] = 1e-200
            except Exception:
                for i,j,xij in data:
                    print(theta[i], theta[j], _likelihood(xij))
                    break
                exit()

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
        alpha0 = 1/K
        self._theta = np.random.dirichlet([alpha0]*K, N)
        pm = self.prior_model = ExpFamConj[kernel]()

        nat_dim = len(self.prior_model._unin_priors)
        phi_shape = (nat_dim, K, K)
        phi_dim = (nat_dim, 1, 1)

        tau_sensibility = 1

        old_tau = np.zeros(phi_shape)
        old_mu = np.zeros(self._theta.shape)
        old_tau_sens = np.inf
        old_mus = [np.inf]
        old_taus = [np.inf]

        edges = self._edges_data
        neigs = frontend.get_neigs()

        ### Loop
        max_iter = self.expe.get('max_iter', 100)
        it_loop1 = 0
        while tau_sensibility > self.expe.tau_tol and it_loop1 < max_iter:

            # block-block loop (phi updates)
            phi_sink = np.zeros(phi_shape)
            theta_argm = self._theta.argmax(1)
            for i, j, w in edges:
                ki = theta_argm[i]
                kj = theta_argm[j]
                phi_sink[:,ki,kj] += pm.ss(w) * self._theta[i, ki]*self._theta[j, kj]
                #kk_outer = np.outer(self._theta[i], self._theta[j])
                #kk_outer = np.tile(kk_outer, phi_dim)
                #phi_sink += pm.ss(w).reshape(phi_dim) * kk_outer

            # if last dimension of T is 1 update manually
            for k1, k2 in np.ndindex(K,K):
                phi_sink[-1, k1,k2] = self._theta[:,k1].sum() * self._theta[:,k2].sum()

            self._tau = phi_sink + pm._unin_priors.reshape(phi_dim)
            nat = self.compute_natural_expectations(self._tau)

            new_tau_sens = np.absolute(self._tau - old_tau).sum()
            tau_sensibility = np.absolute(new_tau_sens-old_taus[-1])
            old_tau = self._tau
            old_taus.append(new_tau_sens)
            print('tau: %.4f' % tau_sensibility)
            it_loop1 +=1

            if len(old_taus) > 10:
                eta = np.mean(old_taus[::-1][:5]) - np.mean(old_taus[::-1][5:10])
                if eta < 0.1:
                    break

            mu_sensibility = 1
            it_loop2 = 0
            while mu_sensibility > self.expe.mu_tol and it_loop2 < max_iter:
                theta_argmax = self._theta.argmax(1)
                theta_m = np.zeros_like(self._theta)
                theta_m[np.arange(len(self._theta)), theta_argmax] = self._theta.max(1)
                for i in np.random.choice(N, size=N, replace=False):
                    theta_sink = np.zeros(phi_shape)
                    ki = theta_argmax[i]
                    for j, w in neigs[i][0]:
                        kj = theta_argmax[j]
                        theta_sink[:, ki, kj] += pm.ss(w) * theta_m[j].max()
                        #kk_outer = np.tile(self._theta[j], (nat_dim, K, 1))
                        #theta_sink += pm.ss(w).reshape(phi_dim) * kk_outer

                    # if last dimension of T is 1 update manually
                    theta_sink[-1, ki] += theta_m.sum(0)
                    theta_sink[-1, ki] -= theta_m[i]
                    #for k in range(K):
                    #    theta_sink[-1,k] += np.ones(K) * (self._theta[:i,k].sum(0)+self._theta[i+1:,k].sum(0))

                    if not self._is_symmetric:
                        for j, w in neigs[i][1]:
                            kj = theta_argmax[j]
                            theta_sink[:, kj, ki] += pm.ss(w) * theta_m[j].max()
                            #kk_outer = np.tile(self._theta[j][np.newaxis].T, (nat_dim, 1, K))
                            #theta_sink += pm.ss(w).reshape(phi_dim) * kk_outer
                    else:
                        theta_sink[:, :, ki] = theta_sink[:,ki]

                    theta_i = np.empty_like(self._theta[0])
                    for k in range(K):
                        theta_sink_cross = np.zeros(theta_sink.shape)
                        theta_sink_cross[:,k,:] = theta_sink[:,k,:]
                        theta_sink_cross[:,:,k] = theta_sink[:,:,k]
                        theta_i[k] = np.sum(theta_sink_cross * nat)

                    # Normalize _theta
                    self._theta[i] = expnormalize(theta_i)

                    theta_argmax[i] = self._theta[i].argmax()
                    theta_m[i,:] = 0
                    theta_m[i, theta_argmax[i]] = self._theta[i].max()

                mu  = self._theta
                new_mu_sens = np.absolute(mu - old_mu).sum()
                mu_sensibility = np.absolute(new_mu_sens - old_mus[-1])
                print('mu: %.4f' % mu_sensibility)
                it_loop2 +=1
                old_mu = mu.copy()
                old_mus.append(new_mu_sens)

                if len(old_mus) > 10:
                    eta = np.mean(old_mus[::-1][:5]) - np.mean(old_mus[::-1][5:10])
                    if eta < 0.1:
                        break

                self.compute_measures()
                if self.expe.get('_write'):
                    self.write_current_state(self)
                    if self._measure_cpt % self.snapshot_freq == 0:
                        self.save(silent=True)

            old_mus = old_mus[-1:]




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




