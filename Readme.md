# Machine Learning Project

This repo contains models and design for several type of data structure.


# Install

 `make install`

This command will install dependancies and fetch datasets.

# Examples


Then upgrade pmk indexes with 

    pmk update

From here you can list available model:

    pmk - l model

or available script:

    pmk -l script

Then you  can fit a WMMSB model on the manufacturing network like this

    pmk wmmsb -c  manufacturing -x fit

Then, you can observe the the log-likelihood convergence

    pmk wmmsb -c  manufacturing  -x fig corpus:model:entropy

Or plot a table of final resutls

    pmk wmmsb -c  manufacturing  -x tab corpus:model:entropy


# What's inside

## Networks

Available **models**:
* IFLM : Infinte Latent Feature Model
* MMSB/IMMSB : (Infinite) Mixed Membership Stochatisc Blockmodel
* MMSB and WMMB :Stochastic Variational Inference Scheme
* Rescal from `mnick/rescal.py`
* SBM from `graph-tool` package

Available **corpuses**:
Corpus from: http://konect.uni-koblenz.de/networks/
WIP

## Text

TBD


