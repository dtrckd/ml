import os
import numpy as np
from numpy import ma
from pymake import GramExp, ExpeFormat
from pymake.frontend.manager import ModelManager, FrontendManager
from pymake.plot import _markers, _colors, _linestyle

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from loguru import logger
lgg = logger



class PostCompute(ExpeFormat):

    # Documentation !
    _default_expe = dict(
        _label = lambda expe: '%s %s' % (expe._alias[expe.model], expe.get('delta')) if expe.model in expe._alias else False,
        legend_size=10,
        _csv_sample = 2,
        fig_burnin = 0
    )

    def _preprocess(self):
        pass

    def _to_masked(self, lst, dtype=float):
        themask = lambda x:np.nan if x in ('--', 'None') else x
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

        frontend = self.load_frontend()

        model = self.load_model(load=True)
        theta, phi = model._reduce_latent()

        y_true, probas = model.get_ytrue_probas()

        data = getattr(model, 'data_'+data)
        edges_data = model._edges_data
        is_symmetric = model._is_symmetric

        if not ('mmsb' in self.s.model or 'wmmsb' in self.s.model):
            qij = self.posterior(theta, phi, data)
        else:
            qij = self._mean_weights(theta, phi, data, is_symmetric, edges_data)

        wd = data[:,2].T
        ws = qij

        mse = mean_squared_error(wd, ws)

        print(mse)


    def _mean_weights(self, theta, phi, data, is_symmetric, edges_data):
        N,K = theta.shape
        # class assignement
        c = theta.argmax(1) # @cache

        theta_hard = np.zeros_like(theta)
        theta_hard[np.arange(len(theta)), c] = 1 # @cache
        c_len = theta_hard.sum(0)

        # number of possible edges per block
        norm = np.outer(c_len,c_len)
        if not is_symmetric:
            np.fill_diagonal(norm, 2*(norm.diagonal()-c_len))
        else:
            np.fill_diagonal(norm, norm.diagonal()-c_len)
        norm = ma.masked_where(norm<=0, norm)

        # Expected weight per block
        pp = np.zeros((K,K))
        edges = edges_data
        for i,j,w in edges:
            pp[c[i], c[j]] += w

        pp = pp / norm

        qij = ma.array([ pp[c[i], c[j]] for i,j,_ in data])

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

        ### sparse matrix with test indexes at 1
        Ytest = frontend.data_test

        ### Adjacency matrix
        g = frontend.data
        g.clear_filters()
        y = frontend.adj()

        # set the weight
        for i,j,w in frontend.get_edges():
            y[i,j] = w

        # state
        seed = []
        for c in list(corpus_name):
            seed.append(str(ord(c)))
        seed.append(repeat)

        seed = ''.join( [chr(int(i)) for i in list(''.join(seed))])
        seed = int((hash_objects(seed)), 32) % 2**32

        out = os.path.join(self.get_data_path(), 'mat') + '/'
        if repeat:
            out += repeat + '/'
        os.makedirs(out, exist_ok=True)

        savemat(out + corpus_name + '_'+ '-'.join([training_ratio, testset_ratio, validset_ratio]) + '.mat', {'Y':y.astype(float),
                                             'Ytest':Ytest.astype(float),
                                             'is_symmetric': frontend.is_symmetric(),
                                             'state': seed
                                            })

    @ExpeFormat.expe_repeat
    @ExpeFormat.table()
    def tab_mat(self, array, floc, x, y, z, *args):
        from scipy.io import loadmat
        expe = self.expe

        expe['driver'] = 'gt'
        training_ratio = 100
        testset_ratio = 20
        validset_ratio = 10
        K = 10

        expe['training_ratio'] = training_ratio
        expe['testset_ratio'] = testset_ratio
        expe['validset_ratio'] = validset_ratio
        expe['K'] = K

        corpus = expe.corpus
        repeat = expe._repeat
        training_ratio = expe.training_ratio
        K = expe.K
        _it = '15'

        outp = '/home/dtrckd/Desktop/tt/EPM2/results'
        format_id = "it%straining%sK%srep%s" % (_it,training_ratio,K,repeat)
        ratio_id = ''.join(('_',str(training_ratio),'-',str(testset_ratio),'-',str(validset_ratio)))
        fnin = os.path.join(outp, corpus, 'wsim_all_'+format_id+ratio_id+'.mat')
        data = loadmat(fnin)

        trad = dict(wsim='WSIM',
                    wsim2='WSIM2',
                    roc='AUCroc',
                    time_it='timing',
                   )

        value = data[trad[z]].item()

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

