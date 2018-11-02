#!/bin/bash
#
# Script that contains all the experiments to evaluate pyDML's algorithms.
# Run the experiments with the argument 'run'
# Test the experiments with the argument 'test'
#
# All the experiments in this script will be executed sequentially. You may want to execute some of them in parallel.
# To do this, you can take separately any of the experiments in the 'run' block. The commands are listed below.


error=0

if [ "$1" != "" ]; then
    if [ "$1" == "test" ]; then
        echo "Testing experiments..."
        python basic.py test
        python ncm.py test
        python ker.py test
        python dim.py test
    elif [ "$1" == "run" ]; then
        echo "Running experiments..."
        python basic.py small
        python basic.py medium
        python basic.py large1
        python basic.py large2
        python basic.py large3
        python basic.py large4

        python ncm.py small
        python ncm.py medium
        python ncm.py large1
        python ncm.py large2
        python ncm.py large3
        python ncm.py large4

        python ker.py small
        python ker.py medium
        python ker.py large1
        python ker.py large2
        python ker.py large3
        python ker.py large4

        python dim.py 0
        python dim.py 1
        python dim.py 2

    else
        error=1
    fi
else
    error=1
fi

if [ $error -gt 0 ]; then
    echo "Run the script with the argument 'run' to evaluate all the algorithms and datasets."
    echo "Run the script with the argument 'test' to test the experiments."
fi
