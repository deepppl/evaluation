#!/bin/bash

function compile_all {
    backend=$1
    mode=$2
    echo --- $backend $mode
    rm -f /tmp/out-$backend-$mode logs-$backend-$mode /tmp/err-$backend-$mode
    find ../example-models -name *.stan -exec bash ./compile.sh $backend $mode {} \;
    echo
    echo Success:        `cat logs-$backend-$mode | grep ": 0" | wc -l`
    echo stanc failures: `cat logs-$backend-$mode | grep ": 1" | wc -l`
    echo stanc-$backend-$mode failures:  `cat logs-$backend-$mode | grep ": 2" | wc -l`
    echo Total:          `cat logs-$backend-$mode | wc -l`
}
compile_all pyro generative
compile_all pyro comprehensive
compile_all pyro mixed
compile_all numpyro generative
compile_all numpyro comprehensive
compile_all numpyro mixed
