#!/bin/bash

backend=$1
mode=$2
logfile=logs/$backend-$mode.csv

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 backend mode"
    echo "  backend: pyro | numpyro"
    echo "  mode:    comprehensive | mixed | generative"
    exit 1
fi

mkdir -p logs
echo --- $backend $mode
rm -f /tmp/out-$backend-$mode $logfile /tmp/err-$backend-$mode
find ../example-models -name *.stan -exec bash ./compile.sh $backend $mode $logfile {} \;
echo
echo Success:        `cat $logfile | grep ", 0" | wc -l`
echo stanc failures: `cat $logfile | grep ", 1" | wc -l`
echo stanc-$backend-$mode failures:  `cat $logfile | grep ", 2" | wc -l`
echo Total:          `cat $logfile | wc -l`
