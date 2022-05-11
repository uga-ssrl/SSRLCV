#!/bin/bash
#SBATCH --job-name=valgrind3                # Job name
#SBATCH --partition=gpu_p             # Partition (queue) name, i.e., gpu_p 
#SBATCH --gres=gpu:K40:1                  # Requests one GPU device 
#SBATCH --ntasks=1                    # Run a single task       
#SBATCH --cpus-per-task=2             # Number of CPU cores per task
#SBATCH --mem=4gb                    # Job memory request
#SBATCH --time=00:30:00                # Time limit hrs:min:sec
#SBATCH --output=log/valgrind3.%j.out       # Standard output log
#SBATCH --error=log/valgrind3.%j.err        # Standard error log

#SBATCH --mail-type=BEGIN,END,FAIL      # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=%u@uga.edu  # Where to send mail    

cd $SLURM_SUBMIT_DIR

ml CUDA/10.0.130
ml GCCcore/6.4.0

valgrind --error-exitcode=1 --tool=memcheck --errors-for-leak-kinds=definite --leak-check=full --show-leak-kinds=all bin/SFM -d /work/demlab/sfm/SSRLCV-Sample-Data/everest1024/3view -s /work/demlab/sfm/SSRLCV-Sample-Data/seeds/seed_spongebob.png
