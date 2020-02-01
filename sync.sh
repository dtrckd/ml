#!/bin/bash

### Remote Location
# machine
#SSH='adulac@tiger'
#IN="/home/ama/adulac/workInProgress/networkofgraphs/process/repo/ml"

# aws
SSH='admin@15.188.55.178'
IN="~/ml"
# see .ssh/config to link .pem key

### Local location
OUT="./"
### Target
BASE=".pmk/results"
# from commandline
REFDIR="ai19_1"

#EXT='-20_'

#FILTER='--include "*/" --include "*noel3***"  --exclude "*"'
#FILTER='--include "*/" --include "*pnas3/***" --exclude "*" '

### Parse Args
SIMUL="-n"
OPTS="--update"

if [ "$1" == "-f" -o "$2" == "-f" ]; then
    SIMUL=""
fi

# refdir
if [ ! -z "$1" -a "$1" != "-f"  ]; then
    REFDIR="$1"
elif [ ! -z "$2" -a "$2" != "-f"  ]; then
    REFDIR="$2"
fi

if [ ! -z "$EXT" ]; then
    INCL="--include=*/ --include=*$EXT* --exclude=*"
    #INCL="--include=*/ --include=*.$EXT --exclude=*"
fi



#rsync $SIMUL  -av -u --modify-window=2 --stats -m $OPTS \
eval rsync $SIMUL $OPTS $INCL $FILTER -vah --stats -m \
    -e ssh  $SSH:$IN/$BASE/$REFDIR/ $OUT/$BASE/$REFDIR

echo
echo "rsync $SIMUL $OPTS $INCL $FILTER -vah --stats -m -e ssh  $SSH:$IN/$BASE/$REFDIR/ $OUT/$BASE/$REFDIR"

###
#rsync --dry-run  -av -u --modify-window=2  --stats --prune-empty-dirs  -e ssh --include '*/'  --include='debug/***' --exclude='*'  ./ dulac@pitmanyor:/home/dulac/ddebug
#rsync --dry-run  -av -u --modify-window=2 --stats --prune-empty-dirs  -e ssh    adulac@racer:/home/ama/adulac/workInProgress/networkofgraphs/process/PyNPB/data/networks/ ./


