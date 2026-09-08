#!/bin/bash
set -e

# Adjust to whichever pyenv/venv name the target VM uses for phantomkit.
pyenv activate phantomkit

cd /home/ubuntu/git/pydra-tasks-fsl
git pull
pip install -e .

cd /home/ubuntu/git/pydra-tasks-mrtrix3
git pull
pip install -e .

cd /home/ubuntu/git/phantomkit
# CHECK WHICH BRANCH YOU ARE ON AND PULL THE LATEST CHANGES
git pull
pip install -e ".[test]"

nohup pytest tests/test_phantom_qc_preprocess.py > tests/test_phantom_qc_preprocess_OUTPUT.txt &
