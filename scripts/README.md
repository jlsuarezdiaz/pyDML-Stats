## Instructions for running the experiments.

To run all the experiments, you can run the script `experiments.sh` with the command `$experiments.sh run`. This script will execute all the algorithms and obtain all the results. However, all the experiments are run sequentially. The execution of all the experiments may take several weeks. You may want to run some of the experiments in parallel. To do this, you can also run indepently the scripts `basic.py`, `ncm.py`, `ker.py`, `dim.py` and `recopilate.py` with different arguments specifying different datasets, in order to be able to run with several degrees of parallelism. Execute any of the previous `.py` scripts to see all their available options. The whole set of experiments to be run is defined in the header of the file `experiments.sh`. You can also test how the script works with the command `$experiments.sh test`.

## Requirements

An appropiate version of [pyDML](https://pydml.readthedocs.io) must be installed, from [GitHub](https://github.com/jlsuarezdiaz/pyDML) or [PyPI](https://pypi.org/project/pyDML/), corresponding with the version of the PyDML-Stats software.