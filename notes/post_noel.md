# fit

all 
    pymake noel_mmsb -x fit

mmsb relative
    pymake noel_mmsb -x fit

scvb relative
    pymake noel_scvb -x fit
    pymake noel_ada -x fit     # adaptative chunk variant


# plot

Plt entropy and roc table for corpus/model table,  separate mask:
    pymake noel_mmsb -x plot table corpus:model:_entropy-roc mask --net 1 -w 

plot entropy convergence (groups by corpus/mask):
    pymake expe -x plot fig corpus/mask:_entropy model  --net 1 -w

Plot table by table of roc and entropy by corpus/chunk:
    pymake noel_scvb_ada -x plot table corpus:chunk:_entropy-roc mask --net 1

Same, compare by mask:
    pymake noel_scvb_ada -x plot table corpus:mask:_entropy-roc mask --net 1

The scvb seems to slighly converge, but too weak, perforamance (roc) are largely inferior of cvb.

## Impact of iteration on scvb

Too much iterations seems to degrade the performance, and a small number seems enough:
    pymake noel_scvb_ada -x plot table corpus:mask:_entropy-roc iterations -i 3 20 -c blogs --net 1

## Gradient step sensitivity

best performance is chosen to be :

For generator7
* chi_a = 10, tau_a=500, kappa_a = 0.6
* chi_b=10, tau_b=100, kappa_b=0.9
(also for one test, tau_a=100, tau_b=500 seems good.

For BA
* chi_a = 1, tau_a=42, kappa_a = 0.6
* chi_b=42, tau_b=300, kappa_b=0.7

    pmk scvb_chi_2   -x plot table kappa_b:tau_b:roc tau_a --pmk chi_a=10 chi_b=10     -c generator7  --mask unbalanced --repeat 55






