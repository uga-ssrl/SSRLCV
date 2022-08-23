#!/bin/bash
#SBATCH --job-name=build                # Job name
#SBATCH --partition=gpu_p             # Partition (queue) name, i.e., gpu_p 
#SBATCH --gres=gpu:K40:1                  # Requests one GPU device 
#SBATCH --ntasks=1                    # Run a single task       
#SBATCH --cpus-per-task=2             # Number of CPU cores per task
#SBATCH --mem=4gb                    # Job memory request
#SBATCH --time=00:30:00                # Time limit hrs:min:sec
#SBATCH --output=log/build.%j.out       # Standard output log
#SBATCH --error=log/build.%j.err        # Standard error log

#SBATCH --mail-type=BEGIN,END,FAIL      # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=%u@uga.edu  # Where to send mail    

cd $SLURM_SUBMIT_DIR

ml CUDA/10.0.130
ml GCCcore/6.4.0

make clean
make sfm -j8 SM=35 LOG_LEVEL=4
