#!/bin/bash
#SBATCH --job-name=jetvlad-train
#SBATCH --output=jetvlad-train-%j.log
#SBATCH --time=168:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
cd ~/NetVLAD-tagger-pytorch
/home/plattecw/.conda/envs/netvlad-pytorch/bin/python train.py \
                --train-data ~/jet-generator-tagging/output_training_all_data.root \
                --test-data ~/jet-generator-tagging/output_5to10_test_all.root \
                --val-data ~/jet-generator-tagging/output_5to10_validation_all.root \
                --variables trkvtx \
                --clusters 33 \
                --depth 2 \
                --signal hf_vs_l \
                --jobid jetvlad_test_run
