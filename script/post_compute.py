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

    def compute(self, meas):

        model = self.load_model(load=True)

        y_true, probas = model.get_ytrue_probas()

        custom_keys = ['data', 'treshold']
        # conf from spec ?
        kws = dict(treshold=self.s.get('treshold'))

        fun = getattr(model, 'compute_'+meas)
        res = fun(**kws)

        print(res)

