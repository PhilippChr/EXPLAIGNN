#!/usr/bin/bash 

# create directories
mkdir -p _benchmarks
mkdir -p _data
mkdir -p _intermediate_representations
mkdir -p _results
mkdir -p _results/convmix
mkdir -p out
mkdir -p out/convmix
mkdir -p out/slurm

# download 
bash scripts/download.sh convmix
bash scripts/download.sh data
bash scripts/download.sh wikipedia
bash scripts/download.sh explaignn