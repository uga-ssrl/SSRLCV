#!/bin/bash
#SBATCH --job-name=sfm                # Job name
#SBATCH --partition=gpu_p             # Partition (queue) name, i.e., gpu_p 
#SBATCH --gres=gpu:K40:1              # Requests one GPU device 
#SBATCH --ntasks=1                    # Run a single task       
#SBATCH --cpus-per-task=2             # Number of CPU cores per task
#SBATCH --mem=4gb                     # Job memory request
#SBATCH --time=00:40:00               # Time limit hrs:min:sec
#SBATCH --output=log/sfm.%j.out       # Standard output log
#SBATCH --error=log/sfm.%j.err        # Standard error log

#SBATCH --mail-type=BEGIN,END,FAIL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ejv88036@uga.edu  # Where to send mail     

cd $SLURM_SUBMIT_DIR

mkdir -p log

ml CUDA/10.0.130
ml GCCcore/6.4.0

make sfm -j8 SM=35
bin/SFM -d SSRLCV-Sample-Data/everest1024/2view -s SSRLCV-Sample-Data/seeds/seed_spongebob.png