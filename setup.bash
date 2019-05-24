#!/usr/bin/env bash

Rscript R_install.R
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 -m ipykernel install --user --name=off-target-igsp  # for Jupyter notebook
# ALL THIS CAN BE REMOVED AND CAUSALDAG CAN BE ADDED TO REQUIREMENTS AFTER IT IS PUBLISHED
#pip3 install matplotlib typing
#yes | pip3 uninstall causaldag
