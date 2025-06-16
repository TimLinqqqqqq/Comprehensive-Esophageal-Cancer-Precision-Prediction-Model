#! /bin/bash

#SBATCH --account=MST113246       # parent project to access twcc system
#SBATCH --job-name=EC_job_temp    # jobName
#SBATCH --nodes=1                 # request node number
#SBATCH --ntasks-per-node=1       # number of tasks can execute on the node
#SBATCH --gpus-per-node=1         # gpus per node
#SBATCH --cpus-per-task=1         # cpus per gpu
#SBATCH --partition=gp1d          # how long task can run
#SBATCH --output=EC.%j.out        # specify output Directory and fileName

python preprocess_and_resnet.py

