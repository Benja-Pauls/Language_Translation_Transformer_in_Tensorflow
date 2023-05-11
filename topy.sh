#!/bin/bash

# to use this file, run the command: bash topy.sh <your_filename.ipynb>
# where <your_filename.ipynb> is the jupyter notebook you want to convert

# If you get the error, jupytext not found, run the command: pip install jupytext

jupytext --to py $1

baseFile=${1%.*}

sed -i 's/IS_PYTHON = False/IS_PYTHON = True/' "$baseFile.py" 