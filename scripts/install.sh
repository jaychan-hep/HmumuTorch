# include ROOT
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.24.02/x86_64-centos7-gcc48-opt/bin/thisroot.sh
python3 -m venv hmumumlenv
source hmumumlenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# setting python path
export PATH="`pwd`/scripts:${PATH}"
export PYTHONPATH="`pwd`/scripts:${PYTHONPATH}"
export PATH="`pwd`/src:$PATH"
export PYTHONPATH="`pwd`/src:$PYTHONPATH"
