#!/usr/bin/env bash
# container/slurm_example.sh
# =============================================================================
# Example SLURM job script for running dwiforge via Apptainer on an HPC.
# Copy and modify for your cluster — partition names, account, paths will differ.
#
# Submitting a single subject:
#   sbatch --export=SUBJECT=sub-001 container/slurm_example.sh
#
# Submitting all subjects as an array:
#   sbatch container/slurm_example.sh --all
# =============================================================================

#SBATCH --job-name=dwiforge
#SBATCH --partition=gpu                  # adjust for your cluster
#SBATCH --account=your_account           # adjust for your cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1                     # remove if no GPU needed
#SBATCH --output=logs/dwiforge_%j.out
#SBATCH --error=logs/dwiforge_%j.err

# ---- Paths — edit these ----
SIF="/path/to/images/dwiforge_2.0.sif"
BIDS="/data/bids"
OUTPUT="/data/output"
SCRATCH="/scratch/${USER}/dwiforge_${SLURM_JOB_ID}"
FS_LICENSE="${HOME}/freesurfer_license.txt"
CONFIG="${BIDS}/dwiforge.toml"

# ---- Apptainer bind mounts ----
BINDS=(
    "${BIDS}:/data/bids:ro"
    "${OUTPUT}:/data/output"
    "${SCRATCH}:/scratch"
    "${FS_LICENSE}:/opt/freesurfer/license.txt:ro"
)

BIND_STRING=$(IFS=','; echo "${BINDS[*]}")

mkdir -p "${SCRATCH}" logs

# ---- Load Apptainer module if needed ----
# module load apptainer/1.3   # uncomment and adjust for your cluster

# ---- Run ----
apptainer run \
    --bind "${BIND_STRING}" \
    --nv \
    "${SIF}" \
    --config "${CONFIG}" \
    ${SUBJECT:+--subject ${SUBJECT}} \
    ${SLURM_ARRAY_TASK_ID:+--subject-index ${SLURM_ARRAY_TASK_ID}}
