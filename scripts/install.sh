# include ROOT
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.24.02/x86_64-centos7-gcc48-opt/bin/thisroot.sh
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

