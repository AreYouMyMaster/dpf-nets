#!/bin/bash -l

python3 train_ae.py ./configs/generation/airplane.yaml airplane_gen_model 800 0.000256
python3 train_ae.py ./configs/generation/airplane.yaml airplane_gen_model 1200 0.000064 --resume
python3 train_ae.py ./configs/generation/airplane.yaml airplane_gen_model 1400 0.000016 --resume
python3 train_ae.py ./configs/generation/airplane.yaml airplane_gen_model 1450 0.000004 --resume