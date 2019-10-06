from .modelbase import SVB, GibbsSampler, RandomGraphModel
from .expfam import Bernoulli, Normal, Poisson

ExpFamConj = {'bernoulli': Bernoulli,
              'normal': Normal,
              'poisson': Poisson,
             }


