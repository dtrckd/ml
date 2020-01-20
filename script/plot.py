import os
import numpy as np
from numpy import ma
from pymake import GramExp, ExpeFormat, ExpSpace
from pymake.frontend.manager import ModelManager, FrontendManager
from pymake.plot import _markers, _colors, _linestyle

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from loguru import logger
lgg = logger


USAGE = """\
----------------
Plot utility :
----------------
"""

class Plot(ExpeFormat):

    # Documentation !
    _default_expe = dict(
        _label = lambda expe: '%s %s' % (expe._alias[expe.model], expe.get('delta')) if expe.model in expe._alias else False,
        legend_size=10,
        #_csv_sample = 2,
        _csv_sample = None,
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

    def _extract_data(self, z, data, *args):

        # Hook
        if self.expe.get('_refdir') == 'ai19_1' and False:
            if not ('mmsb' in self.s.model or 'wmmsb' in self.s.model):
                if not 'roc2' in z:
                    z = z.replace('roc', 'roc2')
                if not 'wsim2' in z:
                    z = z.replace('wsim', 'wsim2')

        value = None

        if z in data:
            # Extract from saved measure (.inf file).
            if 'min' in args:
                value = self._to_masked(data[z]).min()
            elif 'max' in args:
                value = self._to_masked(data[z]).max()
            else:
                value = self._to_masked(data[z][-1])

        elif '@' in z:
            # Extract a value from max/min fo the second (@)
            ag, vl = z.split('@')

            if 'min' in args:
                value = self._to_masked(data[vl]).argmin()
            else:
                value = self._to_masked(data[vl]).argmax()

            value = data[ag][value]

        else:
            # Compute it directly from the model.
            self.model = ModelManager.from_expe(self.expe, load=True)
            if not self.model:
                return
            else:
                model = self.model

            if hasattr(model, 'compute_'+z):
                value = getattr(model, 'compute_'+z)(**self.expe)
            elif hasattr(self, 'get_'+z):
                _val = getattr(self, 'get_'+z)()
                if isinstance(_val, (list, np.ndarray)):
                    value = _val[-1]
                else:
                    value = _val

            else:
                self.log.error('attribute unknown: %s' % z)
                return

        return value

    def __call__(self, *args, **kwargs):
        return self.fig(*rags, **kwargs)

    @ExpeFormat.raw_plot('corpus')
    def plot_old(self, frame,  attribute='_entropy'):
        ''' Plot figure group by :corpus:.
            Notes: likelihood/perplexity convergence report
        '''
        expe = self.expe

        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            return

        values = data[attribute]
        values = self._to_masked(values)

        description = '/'.join((expe._refdir, os.path.basename(self.output_path)))

        ax = frame.ax()
        ax.plot(values, label=description, marker=frame.markers.next())
        ax.legend(loc='upper right',prop={'size':5})

    @ExpeFormat.raw_plot
    def plot_unique(self, attribute='_entropy'):
        ''' Plot all figure in the same window.
            Notes: likelihood/perplexity convergence report '''
        expe = self.expe

        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            return

        values = data[attribute]
        values = self._to_masked(values)

        description = '/'.join((expe._refdir, os.path.basename(self.output_path)))

        plt.plot(values, label=description, marker=_markers.next())
        plt.legend(loc='upper right',prop={'size':1})


    @ExpeFormat.plot()
    def fig(self, frame, attribute):
        ''' Plot all figure args is  `a:b..:c' (plot c by grouping by a, b...),
            if args is given, use for filename discrimination `key1[/key2]...'.
        '''
        expe = self.expe

        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            return

        try:
            values = data[attribute]
        except KeyError as e:
            func_name = 'get_'+attribute
            if hasattr(self, func_name):
                values = getattr(self, func_name)(data)
            else:
                self.log.error('attribute unknown: %s' % attribute)
                return

        values = self._to_masked(values)

        description = self.get_description()

        ax = frame.ax()

        if expe.get('fig_xaxis'):
            xaxis = expe['fig_xaxis']
            if isinstance(xaxis, (tuple, list)):
                xaxis_name = xaxis[0]
                xaxis_surname = xaxis[1]
            else:
                xaxis_name = xaxis_surname = xaxis

            try:
                x = np.array(data[xaxis_name], dtype=int)
            except ValueError as e:
                x = np.array(data[xaxis_name], dtype=float)

            xmax = frame.get('xmax',[])
            xmin = frame.get('xmin',[])
            xmax.append(max(x))
            xmax.append(min(x))
            frame.xmax = xmax
            frame.xmin = xmin
            frame.xaxis = xaxis_surname
        else:
            x = range(len(values))

        _sample = expe.get('_csv_sample', self._csv_sample(attribute))

        if _sample:
            s = int(_sample)
            x = x[::s]
            values = values[::s]
            self.log.warning('Subsampling data: _csv_sample=%s' % s)

        if expe.get('_label'):
            label = expe['_label'](expe)
            description = label if label else description

        #fr = self.load_frontend()
        #E = fr.num_edges()
        #N = fr.num_nodes()
        #m = self._zeros_set_len
        #pop =

        if 'cumsum' in self.expe:
            values = np.cumsum(values)

        if 'fig_burnin' in self.expe:
            burnin = self.expe.fig_burnin
            x = x[burnin:]
            values = values[burnin:]


        if frame.is_errorbar:
            if self.is_first_expe() or not hasattr(self.D, 'cont'):
                self.D.cont = {}
            cont = self.D.cont

            if description in cont:
                cont[description].x.append(x)
                cont[description].y.append(values)
            else:
                cont[description] = ExpSpace(label=label, ax=ax)
                cont[description].x = [x]
                cont[description].y = [values]

            if self.is_last_expe():
                for description, d in cont.items():
                    ax = d.ax
                    arg_max_len = np.argmax([len(e) for e in d.y])
                    x = d.x[arg_max_len]
                    max_len = len(x)
                    y = self._to_masked([y.tolist()+[np.nan]*(max_len-len(y)) for y in d.y])
                    ax.errorbar(x, y.mean(0), yerr=y.std(0), label=description, fmt=frame.markers.next(), ls=self.linestyles.next())
                    ax.legend(loc=expe.get('fig_legend',1), prop={'size':expe.get('legend_size',5)})
        else:
            ax.plot(x, values, label=description, marker=frame.markers.next())
            ax.legend(loc=expe.get('fig_legend',1), prop={'size':expe.get('legend_size',5)})

        #if self.is_last_expe() and expe.get('fig_xaxis'):
        #    for frame in self.get_figs():
        #        xmax = max(frame.xmax)
        #        xmin = min(frame.xmin)
        #        xx = np.linspace(0,xmax, 10).astype(int)
        #        ax = frame.ax()
        #        ax.set_xticks(x)

        return


    @ExpeFormat.expe_repeat
    @ExpeFormat.table()
    def tab(self, array, floc, x, y, z, *args):
        ''' Plot table according to the syntax:
            * `x:y:Z [args1/args2]'

            Z value syntax
            -------------
            The third value of the triplet is the value to print in the table. There is sepcial syntac options:
            * If several value are seperate by a '-', then for each one a table will be print out.
            * is Z is of the forme Z= a@b. Then the value of the tab is of the form data[a][data[b].argmax()].
                It takes the value b according to the max value of a.

            Special Axis
            ------------
            is x or y can have the special keywords reserved values
            * _spec: each column of the table will be attached to each different expSpace in the grobal ExpTensorsV2.

            args* syntax
            ------------
            If args is given, it'll be used for filename discrimination `key1[/key2]...'
            Args can contain special keyworkd:
            * tex: will format the table in latex
            * max/min: change the defualt value to take in the .inf file (default is max)
            * rmax/rmin: is repeat is given, it will take the min/max value of the repeats (default is mean)
        '''
        expe = self.expe
        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            data = {}

        value = self._extract_data(z, data, *args)

        if value:
            loc = floc(expe[x], expe[y], z)
            array[loc] = value

        return


    #
    # ml/model specific measure.
    #


    def get_perplexity(self, data):
        entropy = self._to_masked(data['_entropy'])
        nnz = self.model._data_test.shape[0]
        return 2 ** (-entropy / nnz)

    # @ temp specific
    def get_wsim3(self):
        ''' Based on the count in mixed membership class and data count. '''
        model = self.load_model(load=True)

        y_true, probas = model.get_ytrue_probas()

        theta, phi = model._reduce_latent()

        N,K = theta.shape
        # class assignement
        c = theta.argmax(1) # @cache

        theta_hard = np.zeros_like(theta)
        theta_hard[np.arange(len(theta)), c] = 1 # @cache
        c_len = theta_hard.sum(0)

        # number of possible edges per block
        norm = np.outer(c_len,c_len)
        if not model._is_symmetric:
            np.fill_diagonal(norm, 2*(norm.diagonal()-c_len))
        else:
            np.fill_diagonal(norm, norm.diagonal()-c_len)
        norm = ma.masked_where(norm<=0, norm)

        # Expected weight per block
        pp = np.zeros((K,K))
        edges = model._edges_data
        for i,j,w in edges:
            pp[c[i], c[j]] += w

        pp = pp / norm

        data = getattr(model, 'data_test')
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

    def roc(self):
        ''' Receiver Operating Curve. '''
        model = self.load_model(load=True)

        y_true, probas = model.get_ytrue_probas()
        fpr, tpr, thresholds = roc_curve(y_true, probas)

        roc_auc = auc(fpr, tpr)
        description = self.get_description()

        plt.plot(fpr, tpr, label='%s | auc=%0.2f' % (description, roc_auc))

        if self._it == self.expe_size -1:
            plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Luck')
            plt.legend(loc="lower right", prop={'size':self.s.get('legend_size',5)})

    def prc(self):
        ''' Precision/Recall Curve. '''
        model = self.load_model(load=True)

        y_true, probas = model.get_ytrue_probas()
        precision, recall, _ = precision_recall_curve(y_true, probas)

        avg_precision = average_precision_score(y_true, probas)
        description = self.get_description()

        plt.plot(recall, precision, label='%s | precision=%0.2f' % (description, avg_precision))


    #
    # Specific
    #

    def roc_evolution2(self, *args, _type='errorbar'):
        ''' AUC difference between two models against testset_ratio
            * _type : learnset/testset
            * _type2 : max/min/mean
            * _ratio : ration of the traning set to predict. If 100 _predictall will be true

        '''
        expe = self.expe
        if self.is_first_expe():
            D = self.D
            axis = ['_repeat', 'training_ratio', 'corpus', 'model']
            #z = ['_entropy@_roc']
            z = ['roc']
            #z = ['_entropy@_wsim']
            D.array, D.floc = self.gramexp.get_array_loc_n(axis, z)
            D.z = z
            D.axis = axis

        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            return

        array = self.D.array
        floc = self.D.floc
        z = self.D.z

        # Building tensor
        for _z in z:

            pos = floc(expe, _z)
            value = self._extract_data(_z, data, *args)
            value = float(self._to_masked(value))

            if value:

                #if 'wsbm_gt' in self.expe.model and '_wsim' in _z:
                #    value /= 1000

                array[pos] = value

        if self.is_last_expe():
            # Plotting

            #Meas = self.specname(self.get_expset('training_ratio'))
            #corpuses = self.specname(self.get_expset('corpus'))
            #models = self.specname(self.get_expset('model'))
            Meas = self.get_expset('training_ratio')
            corpuses = self.get_expset('corpus')
            models = self.get_expset('model')

            axe1 = self.D.axis.index('corpus')
            axe2 = self.D.axis.index('model')

            figs = {}

            for corpus in corpuses:

                self.markers.reset()
                self.colors.reset()
                self.linestyles.reset()
                fig = plt.figure()
                ax = fig.gca()

                jitter = np.linspace(-1, 1, len(Meas))

                for ii, model in enumerate(models):
                    idx1 = corpuses.index(corpus)
                    idx2 = models.index(model)
                    table = array[:, :, idx1, idx2]
                    _mean = table.mean(0)
                    _std = table.std(0)
                    xaxis = np.array(list(map(int, Meas))) #+ jitter[ii]
                    if _type == 'errorbar':
                        ls = self.linestyles.next()
                        _std[_std> 0.15] = 0.15
                        _std[_std< -0.15] = -0.15
                        eb = ax.errorbar(xaxis , _mean, yerr=_std,
                                         fmt=self.markers.next(), ls=ls,
                                         #errorevery=3,
                                         #c=self.colors.next(),
                                         label=self.specname(model))
                        eb[-1][0].set_linestyle(ls)
                    elif _type == 'boxplot':
                        for meu, meas in enumerate(Meas):
                            bplot = table[:, meu,]
                            w = 0.2
                            eps = 0.01
                            ax.boxplot(bplot,  widths=w,
                                       positions=[meu],
                                       #positions=[meas],
                                       #positions=[int(meas)+(meu+eps)*w],
                                       whis='range' )

                if _type == 'errorbar':
                    ax.legend(loc='lower right',prop={'size':8})
                    ymin = array.min()
                    ymin = 0.45
                    ax.set_ylim(ymin)
                else:
                    ax.set_xticklabels(Meas)
                    #ticks = list(map(int, Meas))
                    ticks = list(range(1, len(Meas)))
                    ax.set_xticks(ticks)

                ax.set_title(self.specname(corpus), fontsize=20)
                ax.set_xlabel('percentage of the training edges')
                ax.set_ylabel('AUC-ROC')
                figs[corpus] = {'fig':fig, 'base': self.D.z[0]+'_evo'}

            if expe._write:
                self.write_frames(figs)


if __name__ == '__main__':
    GramExp.generate().pymake(Plot)

