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
success=`cat $logfile | grep ", 0" | wc -l | xargs`
stanc_failures=`cat $logfile | grep ", 1" | wc -l | xargs`
failures=`cat $logfile | grep ", 2" | wc -l | xargs`
total=`cat $logfile | wc -l | xargs`
echo
echo Success: $success
echo stanc failures: $stanc_failures
echo stanc-$backend-$mode failures: $failures
echo Total: $total
echo "examples-models $backend-$mode: { 'success': $success, 'stanc failures': $stanc_failures, 'stanc-pyro-comprehensive failures': $failures, 'total': $total}" >> logs/summary.log
