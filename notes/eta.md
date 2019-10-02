
best combinaison and stable on roc metric seems to be m=50, p=0.25, delta=[0,5,10].
Although, the best choice seem to be dependant of the corpus.
    pmk eta -x tab delta:zeros_set_len:_roc zeros_set_prob/corpus -m iwmmsb_scvb3 --repeat 0


beats  mmsb on roc here ! 
    pmk eta_visu -x tab corpus:model:_roc delta   --repeat 0
