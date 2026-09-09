#!/bin/bash
set -e

# Adjust to whichever pyenv/venv name the target VM uses for phantomkit.
pyenv activate phantomkit

cd /home/ubuntu/git/pydra-tasks-fsl
git pull
pip install -e .

# pydra-tasks-mrtrix3/pydra-tasks-ants have no local editable checkout on
# this VM (unlike pydra-tasks-fsl) -- pip install -e ".[test]" below pulls
# them in normally via phantomkit's own pyproject.toml dependencies.

cd /home/ubuntu/git/phantomkit
# CHECK WHICH BRANCH YOU ARE ON AND PULL THE LATEST CHANGES
git pull
pip install -e ".[test]"

nohup pytest tests/test_phantom_qc_preprocess.py > tests/test_phantom_qc_preprocess_OUTPUT.txt &
