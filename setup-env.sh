#!/bin/sh
mkdir -p $HOME/.pythonvenv
python3 -m venv $HOME/.pythonvenv/learning-pytorch
source $HOME/.pythonvenv/learning-pytorch/bin/activate
export PATH="$HOME/.pythonvenv/learning-pytorch/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
