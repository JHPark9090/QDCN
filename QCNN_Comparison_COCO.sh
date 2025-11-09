#!/bin/bash
#SBATCH -A m4138_g
#SBATCH -J QCNN_Comparison_COCO
#SBATCH -C gpu&hbm80g
#SBATCH --qos shared
#SBATCH -t 48:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --chdir='/pscratch/sd/j/junghoon/'
#SBATCH --output=/pscratch/sd/j/junghoon/qcnn_comparison_results/QCNN_Comparison_COCO_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/qcnn_comparison_results/QCNN_Comparison_COCO_%j.e
#SBATCH --mail-user=utopie9090@snu.ac.kr
#!/bin/bash
set +x

echo "=========================================="
echo "QCNN vs Quantum Dilated CNN - COCO"
echo "=========================================="
echo ""
echo "Comparing two quantum CNN architectures:"
echo "  1. QCNN (Cong et al. 2019) - Nearest-neighbor entanglement"
echo "  2. Quantum Dilated CNN - Non-adjacent entanglement"
echo ""

cd /pscratch/sd/j/junghoon
module load python
conda activate ./conda-envs/qml_eeg

# Create output directory
mkdir -p /pscratch/sd/j/junghoon/qcnn_comparison_results

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo ""

# Configuration:
# - Dataset: COCO (80 classes, 224x224x3 images)
# - Qubits: 8
# - Layers: 2
# - Optimizer: Adam
# - Epochs: 50 (< 100 as requested)
# - Batch size: 16 (smaller due to larger images)

python QCNN_Comparison.py \
    --dataset=coco \
    --n-qubits=8 \
    --n-layers=2 \
    --models qcnn dilated \
    --n-epochs=50 \
    --batch-size=16 \
    --lr=1E-3 \
    --wd=1E-4 \
    --seed=2025 \
    --output-dir='/pscratch/sd/j/junghoon/qcnn_comparison_results/coco' \
    --job-id='coco_comparison' \
    --wandb \
    --wandb-project='QCNN_Comparison' \
    --wandb-entity='QML_Research'

echo ""
echo "=========================================="
echo "COCO Comparison Complete"
echo "End Time: $(date)"
echo "=========================================="
