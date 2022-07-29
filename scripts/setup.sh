#!/bin/bash
# include ROOT
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.24.02/x86_64-centos7-gcc48-opt/bin/thisroot.sh
source hmumumlenv/bin/activate
#conda activate hmumuml

export PATH="`pwd`:${PATH}"
export PYTHONPATH="`pwd`:${PYTHONPATH}"
export THEANO_FLAGS="gcc.cxxflags='-march=core2'"

export PATH="`pwd`/scripts:${PATH}"
export PYTHONPATH="`pwd`/scripts:${PYTHONPATH}"

export PATH="`pwd`/src:$PATH"
export PYTHONPATH="`pwd`/src:$PYTHONPATH"

ulimit -S -s unlimited
