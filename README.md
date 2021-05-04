# rl-project
CSCE 689 Reinforcement Learning Semester Project

Installing requirements with conda:
    conda env create -f requirements.yml

Running code:
    python main.py
Listing all flags:
    python main.py -h

Running code with common flags:
    python main.py -M 10000 -A deep-central-q -E 100 -e 0.2 -a 0.01 -g 0.95

Common flags:
    -M   Max steps, the maximum number of steps taken during an episode
    -A   Architecture used, accepted keys: 'procedural','central-q','joint-q','dec-q','deep-central-q' 
    -E   Epochs
    -e   epsilon
    -a   alpha
    -g   gamma
