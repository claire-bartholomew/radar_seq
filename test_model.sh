#!/bin/bash
#$ -cwd
#$ -V
#$ -l coproc_p100=1
#$ -l h_rt=0:05:00
#$ -m be

./source activate myenv

# Python script with all parameters
python ./radar_seq/unet_test_and_plot.py
