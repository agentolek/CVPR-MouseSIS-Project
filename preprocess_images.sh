#!/bin/bash
#SBATCH --job-name=preprocess_images
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plggolemml25-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
module load Miniconda3/4.9.2
eval "$(conda shell.bash hook)"

# Activate enviroment
conda activate /net/tscratch/people/plgagentolek/conda_env

# Navigate to project directory
cd /net/tscratch/people/plgagentolek/CVPR-MouseSIS-Project

# Preprocess events
# python scripts/preprocess_events_to_e2vid_images.py --data_root data/MouseSIS

# Run full inference on entire test dataset
python3 scripts/inference.py --config configs/predict/sis_challenge_baseline.yaml