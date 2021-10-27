#!/bin/bash

#SBATCH --partition=GPU             #Partition to submit to
#SBATCH --time=0-00:60:00             #Time limit for this job
#SBATCH --nodes=1                     #Nodes to be used for this job during runtime. Use MPI jobs with multiple nodes.
#SBATCH --ntasks-per-node=1           #Number of CPUs. Cannot be greater than number of CPUs on the node.
#SBATCH --mem=10000                     #Total memory for this job
#SBATCH --job-name="HAR ML Model"     #Name of this job in work queue
#SBATCH --output=har-out.txt          #Output file name
#SBATCH --error=har-error.txt          #Error file name
#SBATCH --mail-user=teewz1076@uwec.edu  #Email to send notifications to
#SBATCH --mail-type=ALL               #Email notification type (BEGIN, END, FAIL, ALL). To have multiple use a comma separated list. i.e END,FAIL
#SBATCH --gpus=1                        # How many GPUs to run this script on

# What machine ran this job?
echo "Machine running job: $(hostname)"

# Run Commands Below
module load python-libs
python processData.py