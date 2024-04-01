#!/bin/bash
#SBATCH --partition=contrib-gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
#SBATCH --output=tr_output/orgl.out
#SBATCH --error=tr_output/orgl.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10                 # up to 128;
#SBATCH --gres=gpu:A100.80gb:1              # up to 8; only request what you need
#SBATCH --mem-per-cpu=12G                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL
#SBATCH --time=2-01:00:00                   # set to 1hr; please choose carefully

# to see ID and state of GPUs assigned
nvidia-smi

./install.sh --task run_copa_test_all
