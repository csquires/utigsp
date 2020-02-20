This project requires Python >=3.5.0 and R >= 3.5.

To download the necessary R packages and create a Python virtual environment, run:
```
bash setup.bash
```

Then, to generate the figures from the paper, run:
```
source venv/bin/activate
bash simulations/fig1/run_fig1.sh
bash simulations/fig1_perfect/run_fig1.sh
python3 simulations/fig1/plot.py
python3 simulations/fig1_perfect/plot.py
```

The figures will be saved in `simulations/figures`.
