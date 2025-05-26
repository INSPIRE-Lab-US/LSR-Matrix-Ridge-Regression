#!/bin/bash
#SBATCH --job-name=lsr34447                       # Unique job name
#SBATCH --output=Output/Sep3_Tuck44_0/Output/output_%j.txt       # Output file
#SBATCH --error=Output/Sep3_Tuck44_0/Error/error_%j.txt         # Error file
#SBATCH --ntasks=1                                      # Number of tasks (jobs)
#SBATCH --cpus-per-task=20                              # Number of CPU cores per task
#SBATCH --mem=124000                                    # Memory in MB (or use --mem=4G for GB)
#SBATCH --time=24:00:00                                 # Time limit hrs:min:sec
#SBATCH --mail-type=END,FAIL                            # Notifications for job done & fail
#SBATCH --mail-user=lsr105@scarletmail.rutgers.edu      # Change to your email
#SBATCH --partition=main 				#The partition you want to use

# Load necessary modules (if any)
module load python/3.8.5                        # Load the Python module (adjust the version)

# Activate your Conda environment (if needed)
source ~/miniconda3/etc/profile.d/conda.sh      # Adjust this if necessary
conda activate torchtensor                      # Replace with your Conda environment

# Navigate to your scratch directory
cd /projects/f_ah1491_1/Rama/LSR_Tensor_Regression_Multiple_Init

# Run your Python script
export OMP_NUM_THREADS=1

python execute.py                               # Replace with your actual Python script name

                                
