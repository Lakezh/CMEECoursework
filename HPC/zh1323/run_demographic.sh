#!/bin/bash

#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=1:mem=1gb

module load anaconda3/personal

echo "R is about to run"

R --vanilla < $HOME/zh1323/zh1323_HPC_2023_demographic_cluster.R
mv dem_sim_results_* $HOME/zh1323


echo "R has finished running"
