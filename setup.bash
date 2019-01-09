#!/usr/bin/env bash

python3 -m virtualenv venv
source venv/bin/activate
pip3 install ipykernel
pip3 install -r requirements.txt
python3 -m ipykernel install --user --name=off-target-igsp
# ALL THIS CAN BE REMOVED AND CAUSALDAG CAN BE ADDED TO REQUIREMENTS AFTER IT IS PUBLISHED
yes | pip3 uninstall causaldag
pip3 install matplotlib typing
pip3 install --no-index --find-link file:/Users/chandlersquires/Documents/causaldag/dist/causaldag-0.1a40.tar.gz causaldag
