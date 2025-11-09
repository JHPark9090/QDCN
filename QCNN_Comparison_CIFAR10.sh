#!/bin/bash
#SBATCH -A m4138_g
#SBATCH -J QCNN_Comparison_CIFAR10
#SBATCH -C gpu&hbm80g
#SBATCH --qos shared
#SBATCH -t 48:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --chdir='/pscratch/sd/j/junghoon/'
#SBATCH --output=/pscratch/sd/j/junghoon/qcnn_comparison_results/QCNN_Comparison_CIFAR10_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/qcnn_comparison_results/QCNN_Comparison_CIFAR10_%j.e
#SBATCH --mail-user=utopie9090@snu.ac.kr
#!/bin/bash
set +x

echo "=========================================="
echo "QCNN vs Quantum Dilated CNN - CIFAR-10"
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
# - Dataset: CIFAR-10 (10 classes, 32x32x3 images)
# - Qubits: 8
# - Layers: 2
# - Optimizer: Adam
# - Epochs: 50 (< 100 as requested)
# - Batch size: 32

python QCNN_Comparison.py \
    --dataset=cifar10 \
    --n-qubits=8 \
    --n-layers=2 \
    --models qcnn dilated \
    --n-epochs=50 \
    --batch-size=32 \
    --lr=1E-3 \
    --wd=1E-4 \
    --seed=2025 \
    --output-dir='/pscratch/sd/j/junghoon/qcnn_comparison_results/cifar10' \
    --job-id='cifar10_comparison' \
    --wandb \
    --wandb-project='QCNN_Comparison' \
    --wandb-entity='QML_Research'

echo ""
echo "=========================================="
echo "CIFAR-10 Comparison Complete"
echo "End Time: $(date)"
echo "=========================================="
