import os
import pickle
import numpy as np
from numpy import ma
from pymake import GramExp, ExpeFormat
from pymake.frontend.manager import ModelManager, FrontendManager
from pymake.plot import _markers, _colors, _linestyle

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import mean_squared_error


class PostCompute(ExpeFormat):

    # Documentation !
    _default_expe = dict(
        _label=lambda expe: '%s %s' % (expe._alias[expe.model], expe.get(
            'delta')) if expe.model in expe._alias else False,
        legend_size=10,
        _csv_sample=2,
        fig_burnin=0,
        _expe_silent=True,
    )

    def _preprocess(self):
        pass

    def _to_masked(self, lst, dtype=float):
        themask = lambda x: np.nan if x in ('--', 'None') else x
        if isinstance(lst, list):
            return ma.masked_invalid(np.array(list(map(themask, lst)), dtype=dtype))
        else:
            return themask(lst)

    def compute(self, meas):
        ''' from model. '''

        model = self.load_model(load=True)

        y_true, probas = model.get_ytrue_probas()

        custom_keys = ['data', 'treshold']
        # conf from spec ?
        kws = dict(treshold=self.s.get('treshold'))

        fun = getattr(model, 'compute_'+meas)
        res = fun(**kws)

        print(res)

    def compute_wsim3(self, data='test'):

        #frontend = self.load_frontend(skip_init=True)

        if "epm" in self.s.model:
            data = self.load_some()
            wd = data['Wreal']
            ws = data['Wpred']
        else:
            model = self.load_model(load=True)
            theta, phi = model._reduce_latent()

            y_true, probas = model.get_ytrue_probas()

            data = getattr(model, 'data_'+data)
            edges_data = model._edges_data
            is_symmetric = model._is_symmetric

            if 'mmsb' in self.s.model or 'wmmsb' in self.s.model:
                qij = model.posterior(theta, phi, data)
            else:
                qij = self._mean_weights(theta, phi, data, is_symmetric, edges_data)

            wd = data[:, 2].T
            ws = qij

        mse = mean_squared_error(wd, ws)

        data = {"wsim3": mse}
        self.pickle_update(data)
        print(mse, self.output_path)

    def compute_wsim4(self, data='test'):

        #frontend = self.load_frontend(skip_init=True)

        model = self.load_model(load=True)
        theta, phi = model._reduce_latent()

        y_true, probas = model.get_ytrue_probas()

        data = getattr(model, 'data_'+data)
        edges_data = model._edges_data
        is_symmetric = model._is_symmetric

        if 'mmsb' in self.s.model or 'wmmsb' in self.s.model:
            qij = model.posterior(theta, phi, data)
        else:
            qij = self._mean_weights_v2(theta, phi, data, is_symmetric, edges_data)

        if qij is None:
            return

        wd = data[:, 2].T
        ws = qij

        idx = wd > 0
        wd = wd[idx]
        ws = ws[idx]

        mse = mean_squared_error(wd, ws)

        data = {"wsim4": mse}
        self.pickle_update(data)
        print(mse, self.output_path)

    def compute_zcp(self, data='test'):

        #frontend = self.load_frontend(skip_init=True)

        model = self.load_model(load=True)
        theta, phi = model._reduce_latent()

        y_true, probas = model.get_ytrue_probas()

        data = getattr(model, 'data_'+data)
        edges_data = model._edges_data
        is_symmetric = model._is_symmetric

        if 'wsbm_ai' in self.s.model or 'wmmsb' in self.s.model:
            qij = model.posterior(theta, phi, data)
            qijs = np.array([theta[i].dot(phi/(1-np.exp(-phi))).dot(theta[j]) for i, j, _ in data])
            a = phi
            x = phi/(1-np.exp(-phi))
        else:
            print("mean weight")
            qij = self._mean_weights(theta, phi, data, is_symmetric, edges_data)

        wd = data[:, 2].T
        ws = qij

        idx = wd > 0
        wd = wd[idx]
        ws = ws[idx]

        mse = mean_squared_error(wd, ws)

        data = {"zcp": mse}
        self.pickle_update(data)
        print(mse, self.output_path)

    def get_wsim3(self):
        return self.pickle_get().get('wsim3')

    def get_wsim4(self):
        return self.pickle_get().get('wsim4')

    def get_zcp(self):
        return self.pickle_get().get('zcp')

    def pickle_get(self, fn=None):
        if not fn:
            fn = self.output_path + '.pk'

        if not os.path.isfile(fn):
            return {}

        with open(fn, 'rb') as _f:
            data = pickle.load(_f)

        return data

    def pickle_update(self, data, fn=None):
        if not fn:
            fn = self.output_path + '.pk'

        if os.path.exists(fn):
            with open(fn, 'rb') as _f:
                d = pickle.load(_f)
            d.update(data)
            data = d

        with open(fn, 'wb') as _f:
            pickle.dump(data, _f)

    def load_some(self, *args, **kwargs):
        from scipy.io import loadmat
        # Load Some hook
        expe = self.expe
        s = expe

        if expe.model == "ml.epm":
            filename = self.output_path + '.inf'

            s.iterations = 300

            outp = '/home/dtrckd/Desktop/tt/EPM2/results'
            format_id = "it%straining%sK%srep%s" % (s.iterations, s.training_ratio, s.K, s._repeat)
            ratio_id = ''.join(('_', str(s.training_ratio), '-', str(s.testset_ratio), '-', str(s.validset_ratio)))
            fnin = os.path.join(outp, s.corpus, 'wsim_all_'+format_id+ratio_id+'.mat')
            print(fnin)
            if not os.path.exists(fnin):
                self.log.warning("file not found: %s" % fnin)
                return

            trad = dict(wsim='WSIM',
                        wsim2='WSIM2',
                        roc='AUCroc',
                        time_it='timing',
                        Wpred='Wpred',
                        Wreal='Wreal',
                       )

            data = {}
            _data = loadmat(fnin)
            for k, v in trad.items():
                try:
                    data[k] = [_data[v].item()]
                except ValueError as e:
                    data[k] = _data[v][0]

            return data

        else:
            return super().load_some(*args, **kwargs)

    def _mean_weights(self, theta, phi, data, is_symmetric, edges_data):
        N, K = theta.shape
        # class assignement
        c = theta.argmax(1) # @cache

        theta_hard = np.zeros_like(theta)
        theta_hard[np.arange(len(theta)), c] = 1 # @cache
        c_len = theta_hard.sum(0)

        # number of possible edges per block
        norm = np.outer(c_len, c_len)
        if not is_symmetric:
            np.fill_diagonal(norm, 2*(norm.diagonal()-c_len))
        else:
            np.fill_diagonal(norm, norm.diagonal()-c_len)
        norm = ma.masked_where(norm <= 0, norm)

        # Expected weight per block
        pp = np.zeros((K, K))
        edges = edges_data
        for i, j, w in edges:
            pp[c[i], c[j]] += w

        pp = pp / norm

        qij = ma.array([pp[c[i], c[j]] for i, j, _ in data])

        if ma.is_masked(qij):
            return None
        else:
            return qij

    def _mean_weights_v2(self, theta, phi, data, is_symmetric, edges_data):
        # normalized by links (edge) only
        N, K = theta.shape
        # class assignement
        c = theta.argmax(1) # @cache

        theta_hard = np.zeros_like(theta)
        theta_hard[np.arange(len(theta)), c] = 1 # @cache
        c_len = theta_hard.sum(0)

        # Expected weight per block
        #
        pp = np.zeros((K, K))
        norm = np.zeros((K, K))
        edges = edges_data
        for i, j, w in edges:
            pp[c[i], c[j]] += w
            norm[c[i], c[j]] += 1

        norm = ma.masked_where(norm <= 0, norm)

        print(pp)
        print(norm)
        pp = pp / norm

        qij = ma.array([pp[c[i], c[j]] for i, j, _ in data])

        if ma.is_masked(qij):
            return None
        else:
            return qij

    def gen_data_matlab(self):
        from scipy.io import savemat
        from pymake.util.utils import hash_objects
        expe = self.expe

        expe['driver'] = 'gt'
        training_ratio = 100
        testset_ratio = 20
        validset_ratio = 10

        corpus_name = expe.corpus
        training_ratio = str(int(expe.get('training_ratio', training_ratio)))
        testset_ratio = str(int(expe.get('testset_ratio', testset_ratio)))
        validset_ratio = str(int(expe.get('validset_ratio', validset_ratio)))
        repeat = expe.get('_repeat', '')

        expe['training_ratio'] = training_ratio
        expe['testset_ratio'] = testset_ratio
        expe['validset_ratio'] = validset_ratio

        # Debug how validset is computed
        #expe['testset_ratio'] -= 0.1/1.1

        frontend = self.load_frontend()
        is_symmetric = frontend.is_symmetric()

        ### sparse matrix with test indexes at 1
        Ytest = frontend.data_test

        ### Adjacency matrix
        g = frontend.data
        g.clear_filters()
        y = frontend.adj()

        # set the weight
        for i, j, w in frontend.get_edges():
            y[i, j] = w
            if is_symmetric:
                y[j, i] = w

        # state
        seed = []
        for c in list(corpus_name):
            seed.append(str(ord(c)))
        seed.append(repeat)

        seed = ''.join([chr(int(i)) for i in list(''.join(seed))])
        seed = int((hash_objects(seed)), 32) % 2**32

        out = os.path.join(self.get_data_path(), 'mat') + '/'
        if repeat:
            out += repeat + '/'
        os.makedirs(out, exist_ok=True)

        fnout = out + corpus_name + '_' + '-'.join([training_ratio, testset_ratio, validset_ratio]) + '.mat'
        print('saving: %s' % fnout)
        savemat(fnout, {'Y': y.astype(float),
                        'Ytest': Ytest.astype(float),
                        'is_symmetric': is_symmetric,
                        'state': seed
        })

    @ExpeFormat.expe_repeat
    @ExpeFormat.table()
    def tab_mat(self, array, floc, x, y, z, *args):
        from scipy.io import loadmat
        expe = self.expe
        s = expe

        expe['driver'] = 'gt'
        training_ratio = 100
        testset_ratio = 20
        validset_ratio = 10
        K = 10
        iterations = 300

        expe['training_ratio'] = training_ratio
        expe['testset_ratio'] = testset_ratio
        expe['validset_ratio'] = validset_ratio
        expe['K'] = K
        expe['iterations'] = iterations

        outp = '/home/dtrckd/Desktop/tt/EPM2/results'
        format_id = "it%straining%sK%srep%s" % (s.iterations, s.training_ratio, s.K, s._repeat)
        ratio_id = ''.join(('_', str(s.training_ratio), '-', str(s.testset_ratio), '-', str(s.validset_ratio)))
        fnin = os.path.join(outp, s.corpus, 'wsim_all_'+format_id+ratio_id+'.mat')
        data = loadmat(fnin)

        trad = dict(wsim='WSIM',
                    wsim2='WSIM2',
                    roc='AUCroc',
                    time_it='timing',
                   )

        value = data[trad[z]].item()

        if z == "time_it":
            value = "%.2f" % (value / 3600)

        if value:
            loc = floc(expe[x], expe[y], z)
            array[loc] = value

        return

    def give_stats(self, meas):
        ''' from inf file. '''

        if self.is_first_expe():
            self.D.vals = []

        data = self.load_some()
        if not data:
            self.log.debug('No data for expe : %s' % self.output_path)
        else:
            val = float(data[meas][-1])
            self.D.vals += [val]

        if self.is_last_expe():
            vals = np.array(self.D.vals)
            res = '''%s:
                mean: %.2f \u00B1 %.2f
                max/min: %.2f/%.2f
                sum: %.2f''' % (meas,
                                vals.mean(), vals.std(),
                                vals.max(), vals.min(),
                                vals.sum())
            print(res)
