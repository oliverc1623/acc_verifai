#!/bin/bash

sudo apt-get update && apt-get install -y pipx
pipx ensurepath

yes "" | head -n 3 | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
micromamba create -n poetry python=3.11 -c conda-forge -y
micromamba run -n poetry pipx ensurepath
export PATH="/root/.local/bin:$PATH"
micromamba run -n poetry pipx install poetry

eval "$(micromamba shell hook --shell=bash)"
micromamba create -n acc_verifai python=3.8 -c conda-forge -y
export PATH="/root/.local/bin:$PATH"

# Quick and dirty setup
# TODO: turn into project toml
micromamba run -n acc_verifai python -m pip install verifai==2.1.0b1
micromamba run -n acc_verifai python -m pip install scenic==2.1.0
micromamba run -n acc_verifai pip3 install carla
