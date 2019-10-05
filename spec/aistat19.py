from pymake import ExpSpace, ExpTensor, Corpus, ExpDesign, ExpGroup

class Aistats19(ExpDesign):

    _alias = {'ml.iwmmsb_scvb3' : 'WMMSB',
              'ml.iwmmsb_scvb3_auto' : 'WMMSB-bg',
              'ml.immsb_scvb3' : 'MMSB',
              'ml.sbm_gt' : 'SBM',
              'ml.wsbm_gt' : 'WSBM',
              'link-dynamic-simplewiki': 'wiki-link',
              'munmun_digg_reply': 'digg-reply',
              'slashdot-threads': 'slashdot', }

    net_final = Corpus(['fb_uc',
                        'manufacturing',
                        'hep-th',
                        'link-dynamic-simplewiki',
                        'enron',
                        'slashdot-threads',
                        'prosper-loans',
                        'munmun_digg_reply',
                        'moreno_names',
                        'astro-ph'])

    base_graph = dict(
        corpus = 'manufacturing',
        testset_ratio = 20,
        validset_ratio = 10,
        training_ratio = 100,

        # Model global param
        N = 'all',
        K = 10,
        kernel = 'none',

        # plotting
        fig_legend = 4,
        legend_size = 7,
        #ticks_size = 20,
        title_size = 18,
        fig_xaxis = ('time_it', 'time'),

        driver = 'gt', # graph-tool driver
        _write = True,
        _data_type    = 'networks',
        _refdir       = 'aistat_wmmsb',
        _format       = "{model}-{kernel}-{K}_{corpus}-{training_ratio}",
        _measures     = ['time_it',
                         'entropy@data=valid',
                         'roc@data=test&measure_freq=5',
                         'pr@data=test&measure_freq=5',
                         'wsim@data=test&measure_freq=5'],
    )


    sbm_peixoto = ExpTensor(base_graph, model='sbm_gt')
    rescal_als = ExpTensor(base_graph, model='rescal_als')

    wsbm = ExpTensor(base_graph, model = 'sbm_aicher',
                    #kernel = ['bernoulli', 'normal', 'poisson'],
                    kernel = 'normal',

                     mu_tol = 0.01,
                     tau_tol = 0.01,
                     max_iter = 100,
                    )

    wmmsb = ExpTensor(base_graph, model="iwmmsb_scvb3",
                     chunk = 'stratify',
                     delta='auto',
                     sampling_coverage = 0.5, zeros_set_prob=1/2, zeros_set_len=50,
                     chi_a=1, tau_a=1024, kappa_a=0.5,
                     chi_b=1, tau_b=1024, kappa_b=0.5,
                     tol = 0.001,
                     #fig_xaxis = ('_observed_pt', 'visited edges'),
                    )
    mmsb = ExpTensor(wmmsb, model="immsb_scvb3")

    aistats_design_wmmsb = ExpGroup([wmmsb],
                                    corpus=net_final,
                                    training_ratio=[1, 5,10,20,30, 50, 100],  # subsample the edges
                                    _refdir="ai19_0",
                             )
    aistats_design_all = ExpGroup([wmmsb, mmsb, wsbm, sbm_peixoto],
                                  corpus=net_final,
                                  training_ratio=[1, 5,10,20,30, 50, 100],  # subsample the edges
                                  _refdir="ai19_0",
                             )

    # note of dev to remove:
    #
    # ow: only weighted edges (third  dimension=1)
    # now: with weighted edges (third T dimension=0)

