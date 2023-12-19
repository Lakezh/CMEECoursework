#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mem=1gb
#PBS -J 1-100

module load anaconda3/personal

echo "R is about to run"

R --vanilla < $HOME/zh1323/zh1323_HPC_2023_neutral_cluster.R

echo "R has finished running"