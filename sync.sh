#!/bin/bash

# from pmk_config
SSH='adulac@tiger'
IN="/home/ama/adulac/workInProgress/networkofgraphs/process/repo/ml"
OUT="./"
BASE=".pmk/results"


# from commandline
REFDIR="ai19_1"
EXT='inf'

#FILTER='--include "*/" --include "*noel3***"  --exclude "*"'
#FILTER='--include "*/" --include "*pnas3/***" --exclude "*" '

### Parse Args
SIMUL="-n"
OPTS="--update"
INCL=""

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
    INCL="--include=*/ --include=*.$EXT --exclude=*"
fi




#rsync $SIMUL  -av -u --modify-window=2 --stats -m $OPTS \
eval rsync $SIMUL $OPTS $INCL  -vah --stats -m $FILTER \
    -e ssh  $SSH:$IN/$BASE/$REFDIR/ $OUT/$BASE/$REFDIR

echo
echo "rsync $SIMUL $OPTS $INCL -vah --stats -m $FILTER -e ssh  $SSH:$IN/$BASE/$REFDIR/ $OUT/$BASE/$REFDIR"

###
#rsync --dry-run  -av -u --modify-window=2  --stats --prune-empty-dirs  -e ssh --include '*/'  --include='debug/***' --exclude='*'  ./ dulac@pitmanyor:/home/dulac/ddebug
#rsync --dry-run  -av -u --modify-window=2 --stats --prune-empty-dirs  -e ssh    adulac@racer:/home/ama/adulac/workInProgress/networkofgraphs/process/PyNPB/data/networks/ ./


