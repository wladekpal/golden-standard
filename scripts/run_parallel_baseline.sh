#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=experiment_out.txt


SEED=$1
AGENT=$2

for BOX_NUM in 1 2 3;
do
    ./scripts/run_experiment_baseline.sh $SEED $AGENT $BOX_NUM &
done

wait