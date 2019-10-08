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

