#!/usr/bin/env bash

source venv/bin/activate
pip3 freeze > requirements.txt
cd ..
rsync -r --progress utigsp/ utigsp_clean --exclude venv --exclude simulations/data --exclude simulations/figures --exclude simulations/results --exclude .idea --exclude __pycache__ --exclude scratch
zip -r utigsp.zip utigsp_clean
