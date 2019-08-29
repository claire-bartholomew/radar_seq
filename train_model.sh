#$ -cwd
#$ -V
#$ -l coproc_p100=1
#$ -l h_rt=1:00:00
#$ -m be

source /home/home01/sccsb/miniconda3/bin/activate /home/home01/sccsb/.conda/envs/myenv
#/home/home01/sccsb/miniconda3/bin/conda /home/home01/sccsb/miniconda3/bin/activate /home/home01/sccsb/.conda/envs/myenv

# Python script with all parameters
python ./radar_seq/milesial_unet_uk.py -e 1 -l 0.01 > log4.out
