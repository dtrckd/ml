import sys
import numpy as np
import scipy as sp
from numpy import ma
import scipy.stats

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import mean_squared_error

from pymake.util.math import lognormalize, categorical, sorted_perm, adj_to_degree, gem
from ml.model import RandomGraphModel

#import warnings
#warnings.filterwarnings('error')
#warnings.catch_warnings()
##np.seterr(all='print')



class immsb_scvb3(RandomGraphModel):


    def _init_params(self, frontend):
        self.frontend = frontend

        self._edges_data =  frontend.get_edges()

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

        # Eta init
        self._eta = []
        self._eta_limit = self.expe.tol
        self._eta_control = np.nan
        self._eta_count_init = 25
        self._eta_count = self._eta_count_init

        self._init_gradient()

        # Hyperparams
        self.hyper_theta = np.asarray([1.0 / (i + np.sqrt(self._len['K'])) for i in range(self._len['K'])])
        self.hyper_theta /= self.hyper_theta.sum()

        hyper_phi = self.expe['delta']
        if hyper_phi == 'auto':
            self.hyper_phi = np.array([0.1,0.1])
            self._hyper_phi = 'auto'
        elif len(hyper_phi) == 2:
            self.hyper_phi = np.array([0.1,0.1])
            self._hyper_phi = 'fix'
        else:
            raise ValueError('hyper parmeter hyper_phi dont understood: %s' % hyper_phi)

        self.hyper_phi_sum = self.hyper_phi.sum()

        self._random_ss_init()

    def _init_gradient(self):
        N = self._len['N']
        self._timestep_a = np.zeros(N)
        self.gstep_theta = np.zeros(N)
        self._timestep_b = 0
        self._timestep_c = 0

        self._chi_a = self.expe.get('chi_a', 5)
        self._tau_a = self.expe.get('tau_a', 10)
        self._kappa_a = self.expe.get('kappa_a', 0.9)
        self._chi_b = self.expe.get('chi_b', 1)
        self._tau_b = self.expe.get('tau_b', 100)
        self._kappa_b = self.expe.get('kappa_b', 0.9)

        self._update_gstep_theta(np.arange(len(self._timestep_a)))
        self._update_gstep_phi()
        self._update_gstep_y()


    def _random_ss_init(self):
        ''' Sufficient Statistics Initialization '''
        K = self._len['K']
        N = self._len['N']
        E = self._len['E']
        nnz = self._len['nnz']
        nnzsum = self._len['nnzsum']
        dims = self._len['dims']

        alpha0 = 1/K

        self.N_theta_left = (dims[:, None] * np.random.dirichlet([alpha0]*K, N))
        self.N_theta_right = (dims[:, None] * np.random.dirichlet([alpha0]*K, N))

        self.N_phi = np.zeros((2,K,K))
        nnz0 = nnz-E
        if self.expe.get('homo') == 'assortative':
            N_phi_d = np.diag(np.random.dirichlet([0.5]*K)) *nnz0*3/4
            N_phi_d1 = np.diag(np.random.dirichlet([0.5]*(K-1)), 1) *nnz0*1/8
            if self._is_symmetric:
                N_phi_d1 += np.diag(np.random.dirichlet([0.5]*(K-1)), -1) *nnz0*1/8
                du = np.diag(np.ones(K-1), 1)==1
                dl = np.diag(np.ones(K-1), -1)==1
                N_phi_d1[dl] = N_phi_d1[du]
            else:
                N_phi_d1 += np.diag(np.random.dirichlet([0.5]*(K-1)), -1) *nnz0*1/8
            N_phi = N_phi_d + N_phi_d1
            self.N_phi[0] = ma.masked_where(N_phi==0, N_phi)

            N_phi_d = np.diag(np.random.dirichlet([0.5]*K)) *E*3/4
            N_phi_d1 = np.diag(np.random.dirichlet([0.5]*(K-1)), 1) *E*1/8
            if self._is_symmetric:
                N_phi_d1 += np.diag(np.random.dirichlet([0.5]*(K-1)), -1) *E*1/8
                du = np.diag(np.ones(K-1), 1)==1
                dl = np.diag(np.ones(K-1), -1)==1
                N_phi_d1[dl] = N_phi[du]
            else:
                N_phi_d1 += np.diag(np.random.dirichlet([0.5]*(K-1)), -1) *E*1/8
            N_phi = N_phi_d + N_phi_d1
            self.N_phi[1] = ma.masked_where(N_phi==0, N_phi)

        else:
            self.N_phi[0] = np.random.dirichlet([0.5]*K**2).reshape(K,K) * nnz0
            self.N_phi[1] = np.random.dirichlet([0.5]*K**2).reshape(K,K) * E

            if self._is_symmetric:
                self.N_theta_left = self.N_theta_right
                self.N_phi[0] = np.triu(self.N_phi[0]) + np.triu(self.N_phi[0], 1).T
                self.N_phi[1] = np.triu(self.N_phi[1]) + np.triu(self.N_phi[1], 1).T


    def _update_gstep_theta(self, idxs):
        ''' Gradient converge for kappa _in (0.5,1] '''
        chi = self._chi_a
        tau = self._tau_a
        kappa = self._kappa_a

        self.gstep_theta[idxs] = chi / ((tau + self._timestep_a[idxs])**kappa)

    def _update_gstep_phi(self):
        chi = self._chi_b
        tau = self._tau_b
        kappa = self._kappa_b

        self.gstep_phi =  chi / ((tau + self._timestep_b)**kappa)

    def _update_gstep_y(self):
        chi = self._chi_b
        tau = self._tau_b
        kappa = self._kappa_b

        self.gstep_y =  chi / ((tau + self._timestep_c)**kappa)


    def _reduce_latent(self):
        theta = self.N_theta_right + self.N_theta_left + np.tile(self.hyper_theta, (self.N_theta_left.shape[0],1))
        self._theta = (theta.T / theta.sum(axis=1)).T

        phi = self.N_phi + np.tile(self.hyper_phi, (self.N_phi.shape[1], self.N_phi.shape[2], 1)).T
        #phi = (phi / np.linalg.norm(phi, axis=0))[1]
        self._phi = (phi / phi.sum(0))[1]

        return self._theta, self._phi

    def _reduce_one(self, i, j, xij, update_local=True, update_kernel=True):

        if update_local:
            if self._is_symmetric:
                self.pik = self.pjk = self.N_theta_left[i] + self.hyper_theta
                self.pjk = self.pik
            else:
                self.pik = self.N_theta_left[i] + self.hyper_theta
                self.pjk = self.N_theta_right[j] + self.hyper_theta

        if update_kernel:
            #self.N_phi[self.N_phi<=1e-300] = 1e-300
            pxk = self.N_phi[xij] + self.hyper_phi[xij]
            # debug: Underflow
            self._kern = np.log(pxk)- np.log(self.N_phi.sum(0) + self.hyper_phi_sum)

        out = np.outer(self.pik, self.pjk)
        #out = ma.masked_invalid(out)
        #out[out<=1e-300] = 1e-300
        outer_kk = np.log(out) + self._kern

        return lognormalize(outer_kk.ravel())

    def compute_entropy(self, *args, **kwargs):
        ll = super().compute_entropy(*args, **kwargs)
        self._eta.append(ll)
        return ll

    def compute_wsim(self, theta=None, phi=None, data='test', **kws):
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
        edges = self._edges_data
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






    def fit(self, frontend):
        ''' chunk is the number of row to process in a minibach '''

        self._init(frontend)

        # Init sampling variables
        observed_pt = 0
        mnb_num = 0
        vertex = None

        self.BURNIN = 150
        qij_samples = []
        node_idxs = []
        weights = []
        _qijs_sum = 0
        _norm = 0

        self.measures['entropy'] = (self.compute_entropy(), np.inf)
        print('__init__ Entropy: %f' % self.measures['entropy'][0])
        for _it, obj in enumerate(frontend):

            source, target, weight = obj
            if type(source) is str:
                #print(_it, source, target, weight)
                _set_pos = source
                _vertex = target['vertex']
                _direction = target['direction']
                _scaler = weight
                _qij_samples = []
                _node_idxs = []
                _weights = []
                new_mnb = True

                update_kernel = True
                update_local = True
                burnin = 0
            else:
                i = source
                j = target
                weight = int(weight>0)
                weights.append(weight)
                if direction == 0:
                    node_idxs.append(j)
                else:
                    node_idxs.append(i)

                # Maximization
                qij_samples.append( self._reduce_one(i,j, weight, update_local, update_kernel).reshape(self._len['K'], self._len['K']) )

                observed_pt += 1
                burnin += 1
                update_kernel = False
                update_local = False

            if (new_mnb or burnin % self.BURNIN == 0) and qij_samples:
                qijs = np.asarray(qij_samples)
                ## Update global gradient / Expectation
                ##norm=1
                norm = qijs.shape[0]
                qijs_sum = qijs.sum(0)

                _qijs_sum += qijs_sum
                _norm += norm

                gstep_v = self.gstep_theta[vertex]
                gstep_nodes = self.gstep_theta[node_idxs][None].T

                if direction == 0:
                    self.N_theta_left[i] = (1-gstep_v)*self.N_theta_left[i] + gstep_v*scaler*qijs_sum.sum(0) /norm
                    self.N_theta_right[node_idxs] = (1-gstep_nodes)*self.N_theta_right[node_idxs] + gstep_nodes*scaler*qijs.sum(2)
                else:
                    self.N_theta_left[node_idxs] = (1-gstep_nodes)*self.N_theta_left[node_idxs] + gstep_nodes*scaler*qijs.sum(1)
                    self.N_theta_right[j] = (1-gstep_v)*self.N_theta_right[j] + gstep_v*scaler*qijs_sum.sum(1) /norm

                self._timestep_a[vertex] += norm
                self._timestep_a[node_idxs] += 1
                self._update_gstep_theta([vertex]+node_idxs)

                qij_samples.clear()
                node_idxs.clear()
                weights.clear()

                update_local = True

            if new_mnb:
                if vertex is None:
                    # Enter here only once !%!
                    mnb_total = frontend.num_mnb()

                    set_pos = _set_pos
                    vertex = _vertex
                    direction = _direction
                    scaler = _scaler
                    new_mnb = False
                    continue


                x = 1 if set_pos != '0' else 0
                self.N_phi[x] = (1 - self.gstep_phi)*self.N_phi[x] + self.gstep_phi * scaler * _qijs_sum /_norm

                self._timestep_b += _norm
                self._update_gstep_phi()


                # Allocate current state variable
                set_pos = _set_pos
                vertex = _vertex
                direction = _direction
                scaler = _scaler

                _qijs_sum = 0
                _norm = 0

                new_mnb = False
                mnb_num += 1

                if mnb_num % (self.expe['zeros_set_len']*5) == 0:
                    prop_edge = observed_pt / self._len['N']**2
                    self._observed_pt = observed_pt
                    self.compute_measures()

                    self.log.info('it %d | prop edge: %.2f | mnb %d/%d, %s, Entropy: %f,  diff: %f' % (_it, prop_edge,
                                                                                                       mnb_num, mnb_total,
                                                                                                       '/'.join((self.expe.model, self.expe.corpus)),
                                                                                                       self.measures['entropy'][0],
                                                                                                       self.measures['entropy'][1]))

                    if self.expe.get('_write'):
                        self.write_current_state(self)
                        if mnb_num % 4000 == 0:
                            self.save(silent=True)

                    if self._check_eta():
                        break



    def _check_eta(self):

        if self._eta_control is not np.nan:
            self._eta_count -=1
            if self._eta_count == 0:
                if self._eta[-1] - self._eta_control < self._eta_limit:
                    self.log.warning('Exit iteration cause eta criteria met.')
                    return True
                else:
                    self._eta_count = self._eta_count_init
                    self._eta_control = np.nan
                    self._eta = [self._eta[-1]]
        elif len(self._eta) > 10:
            if self._eta[-1] - self._eta[0] < self._eta_limit:
                self._eta_control = self._eta[-1]
                print('-', end='')
                sys.stdout.flush()
            else:
                self._eta = [self._eta[-1]]


