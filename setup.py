from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='HmumuTorch',
   version='1.0',
   description='Pytorch package for Hmumu categorization study',
   license="MIT",
   long_description=long_description,
   long_description_content_type="text/markdown",
   author='Jay Chan',
   author_email='jay.chan@cern.ch',
   url="https://gitlab.cern.ch/wisc_atlas/HmumuTorch",
   package_dir = {'': 'src'},
   packages=['myTorch', 'utils'],
   install_requires=["matplotlib", "atlas-mpl-style", "numpy", "pandas", "scikit-learn", "torch", "tqdm", "uproot", "pytorch_lightning", "condor_assistant"],
   scripts=["scripts/train.py", "scripts/apply.py"]
)
