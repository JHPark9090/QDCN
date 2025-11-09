# QCNN vs Quantum Dilated CNN Comparison Guide

## Overview

This project compares two quantum convolutional neural network architectures:

1. **QCNN (Cong et al. 2019)**: Original quantum CNN with nearest-neighbor entanglement
2. **Quantum Dilated CNN**: Modified architecture with non-adjacent (dilated) entanglement for increased global connectivity

**Research Hypothesis**: Dilated entanglement patterns reduce local entanglement while increasing global entanglement, potentially improving feature learning for image classification tasks.

## Architecture Comparison

### 1. QCNN (Cong et al. 2019)

**Paper**: [Quantum Convolutional Neural Networks](https://arxiv.org/abs/1810.03787)

**Key Features**:
- **Entanglement Pattern**: Nearest-neighbor connectivity
- **Convolutional Layer**: Applies U3 gates and Ising gates (XX, YY, ZZ) to adjacent qubit pairs
- **Pooling**: Mid-circuit measurement with conditional operations
- **Strengths**: Strong local feature extraction
- **Limitations**: Limited global information flow

**Circuit Structure** (8 qubits, Layer 1):
```
Entangling pairs: (0,1), (2,3), (4,5), (6,7)
Pattern: [U3-Ising-U3] on adjacent qubits
```

### 2. Quantum Dilated CNN

**Key Features**:
- **Entanglement Pattern**: Non-adjacent connectivity with stride-2 dilation
- **Convolutional Layer**: Same gate structure as QCNN but applied to distant qubit pairs
- **Pooling**: Identical to QCNN
- **Strengths**: Enhanced global feature extraction, larger receptive field
- **Innovation**: Reduces local bias while maintaining expressivity

**Circuit Structure** (8 qubits, Layer 1):
```
Entangling pairs: (0,2), (1,3), (4,6), (5,7)
Pattern: [U3-Ising-U3] on non-adjacent qubits (stride=2)
```

**Visual Comparison**:
```
QCNN:           0-1  2-3  4-5  6-7  (local connections)
Dilated CNN:    0─2  1─3  4─6  5─7  (global connections)
```

## Experimental Setup

### Datasets

#### CIFAR-10
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Images**: 32×32×3 RGB
- **Total samples**: 60,000 (50k train, 10k test)
- **Splits**: 80/10/10 (train/val/test)

#### COCO
- **Classes**: 80 object categories
- **Images**: 224×224×3 RGB (resized)
- **Complexity**: More challenging, diverse object sizes and contexts
- **Note**: Requires more memory, batch size reduced to 16

### Model Configuration

**Both models use identical hyperparameters for fair comparison**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Qubits | 8 | Sufficient for feature extraction |
| Layers | 2 | Balance between expressivity and trainability |
| Optimizer | Adam | Standard for quantum ML |
| Learning Rate | 1e-3 | Typical for Adam |
| Weight Decay | 1e-4 | Light regularization |
| Batch Size (CIFAR-10) | 32 | Stable training |
| Batch Size (COCO) | 16 | Memory constraints |
| Epochs | 50 | < 100 as requested |
| Seed | 2025 | Reproducibility |

### Hardware Requirements

- **GPU**: NVIDIA A100 80GB HBM (specified via `gpu&hbm80g`)
- **CPUs**: 32 cores per task
- **Time Limit**: 48 hours per job
- **Queue**: Shared QoS
- **Account**: m4138_g

## Files Generated

### Python Scripts

**QCNN_Comparison.py**
- Main training and evaluation script
- Implements both QCNN and Quantum Dilated CNN models
- Supports CIFAR-10 and COCO datasets
- Logs metrics to CSV and optionally to Wandb
- Saves best models and training history

**Key Classes**:
```python
QCNN(n_qubits, n_layers, input_dim, num_classes)
  ├─ fc: Classical dimension reduction (input_dim → n_qubits)
  ├─ circuit: Quantum circuit with nearest-neighbor entanglement
  └─ fc_out: Output layer (1 → num_classes)

QuantumDilatedCNN(n_qubits, n_layers, input_dim, num_classes)
  ├─ fc: Classical dimension reduction (input_dim → n_qubits)
  ├─ circuit: Quantum circuit with dilated entanglement
  └─ fc_out: Output layer (1 → num_classes)
```

### SLURM Batch Scripts

1. **QCNN_Comparison_CIFAR10.sh**
   - Runs comparison on CIFAR-10
   - Batch size: 32
   - Expected runtime: ~24-30 hours

2. **QCNN_Comparison_COCO.sh**
   - Runs comparison on COCO
   - Batch size: 16
   - Expected runtime: ~30-36 hours

3. **submit_qcnn_comparison.sh**
   - Master submission script
   - Submits both CIFAR-10 and COCO jobs
   - Creates monitoring utilities

## Usage Instructions

### Quick Start

```bash
# Submit both comparison experiments
bash submit_qcnn_comparison.sh
```

This will:
1. Create output directory structure
2. Submit CIFAR-10 comparison job
3. Submit COCO comparison job
4. Display job IDs and monitoring commands

### Individual Job Submission

```bash
# CIFAR-10 only
sbatch QCNN_Comparison_CIFAR10.sh

# COCO only
sbatch QCNN_Comparison_COCO.sh
```

### Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Monitor specific job output
tail -f qcnn_comparison_results/QCNN_Comparison_CIFAR10_<JOB_ID>.out

# Use monitoring script
bash monitor_qcnn_comparison.sh
```

### Custom Configuration

Run with custom parameters:

```bash
python QCNN_Comparison.py \
    --dataset=cifar10 \
    --n-qubits=8 \
    --n-layers=2 \
    --models qcnn dilated \
    --n-epochs=50 \
    --batch-size=32 \
    --lr=1e-3 \
    --wd=1e-4 \
    --seed=2025 \
    --output-dir='./results' \
    --job-id='custom_run' \
    --wandb
```

**Available Options**:
- `--models`: Choose models to train (`qcnn`, `dilated`, or both)
- `--dataset`: `cifar10` or `coco`
- `--n-qubits`: Number of qubits (default: 8)
- `--n-layers`: Number of convolutional layers (default: 2)
- `--n-epochs`: Training epochs (default: 50)
- `--batch-size`: Batch size (default: 32 for CIFAR-10, 16 for COCO)
- `--lr`: Learning rate (default: 1e-3)
- `--wd`: Weight decay (default: 1e-4)
- `--seed`: Random seed (default: 2025)
- `--wandb`: Enable Wandb logging
- `--wandb-project`: Wandb project name (default: QCNN_Comparison)
- `--wandb-entity`: Wandb entity (default: QML_Research)

## Output Files

### Directory Structure

```
qcnn_comparison_results/
├── cifar10/
│   ├── QCNN_cifar10_best.pt                    # Best QCNN checkpoint
│   ├── QCNN_cifar10_history.csv                # QCNN training history
│   ├── QuantumDilatedCNN_cifar10_best.pt       # Best Dilated CNN checkpoint
│   ├── QuantumDilatedCNN_cifar10_history.csv   # Dilated CNN training history
│   └── comparison_cifar10_comparison.csv        # Side-by-side comparison
├── coco/
│   ├── QCNN_coco_best.pt
│   ├── QCNN_coco_history.csv
│   ├── QuantumDilatedCNN_coco_best.pt
│   ├── QuantumDilatedCNN_coco_history.csv
│   └── comparison_coco_comparison.csv
├── QCNN_Comparison_CIFAR10_<JOB_ID>.out        # SLURM stdout
├── QCNN_Comparison_CIFAR10_<JOB_ID>.e          # SLURM stderr
├── QCNN_Comparison_COCO_<JOB_ID>.out
└── QCNN_Comparison_COCO_<JOB_ID>.e
```

### Metrics Tracked

**Per-Epoch Metrics** (in `*_history.csv`):
- Epoch number
- Train loss
- Train accuracy
- Validation loss
- Validation accuracy
- Validation precision
- Validation recall
- Validation F1 score

**Final Test Metrics** (in `comparison_*.csv`):
- Best epoch
- Best validation accuracy
- Test accuracy
- Test precision
- Test recall
- Test F1 score

### Comparison CSV Format

Example `comparison_cifar10_comparison.csv`:

```csv
Model,Best Epoch,Best Val Acc,Test Acc,Test Precision,Test Recall,Test F1
QCNN,35,0.7234,0.7189,0.7201,0.7189,0.7195
QuantumDilatedCNN,38,0.7456,0.7412,0.7428,0.7412,0.7420
```

## Expected Results

### Performance Predictions

**CIFAR-10**:
- **QCNN**: Expected test accuracy ~68-72%
  - Strong local feature learning
  - May struggle with objects requiring global context

- **Quantum Dilated CNN**: Expected test accuracy ~70-75%
  - Hypothesis: Better global feature integration
  - May outperform QCNN by 2-5%

**COCO** (more challenging):
- **QCNN**: Expected test accuracy ~45-55%
  - Diverse object sizes/contexts challenge local receptive fields

- **Quantum Dilated CNN**: Expected test accuracy ~48-58%
  - Hypothesis: Dilated connections better handle scale variation
  - May show larger advantage (3-7%) on complex scenes

### Key Research Questions

1. **Does dilated entanglement improve accuracy?**
   - Compare test accuracies between QCNN and Dilated CNN

2. **Is the improvement consistent across datasets?**
   - Compare relative improvements on CIFAR-10 vs COCO

3. **How does training dynamics differ?**
   - Compare convergence speed (epochs to best validation)
   - Analyze training curves for stability

4. **Are there trade-offs?**
   - Check if dilated model requires more epochs to converge
   - Analyze precision/recall balance

## Analysis Recommendations

### After Training Completes

1. **Load and Compare Results**:
```python
import pandas as pd

# Load comparison results
cifar_results = pd.read_csv('qcnn_comparison_results/cifar10/comparison_cifar10_comparison.csv')
coco_results = pd.read_csv('qcnn_comparison_results/coco/comparison_coco_comparison.csv')

# Compare accuracies
print("CIFAR-10 Comparison:")
print(cifar_results[['Model', 'Test Acc', 'Test F1']])

print("\nCOCO Comparison:")
print(coco_results[['Model', 'Test Acc', 'Test F1']])
```

2. **Visualize Training Curves**:
```python
import matplotlib.pyplot as plt

# Load training history
qcnn_hist = pd.read_csv('qcnn_comparison_results/cifar10/QCNN_cifar10_history.csv')
dilated_hist = pd.read_csv('qcnn_comparison_results/cifar10/QuantumDilatedCNN_cifar10_history.csv')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(qcnn_hist['epoch'], qcnn_hist['val_acc'], label='QCNN')
plt.plot(dilated_hist['epoch'], dilated_hist['val_acc'], label='Dilated CNN')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('CIFAR-10 Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(qcnn_hist['epoch'], qcnn_hist['train_loss'], label='QCNN')
plt.plot(dilated_hist['epoch'], dilated_hist['train_loss'], label='Dilated CNN')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend()
plt.title('CIFAR-10 Training Loss')

plt.tight_layout()
plt.savefig('training_comparison.pdf')
```

3. **Statistical Significance Testing**:
   - If running multiple seeds, perform t-tests on accuracy differences
   - Report mean ± std across runs

4. **Analyze Per-Class Performance** (requires modification to save per-class metrics):
   - Identify which object classes benefit most from dilated connections
   - Hypothesis: Complex/large objects benefit more from global context

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# In batch script, uncomment:
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Or reduce batch size in arguments
```

**2. Data Loading Issues**
- Ensure LoadData_MultiChip.py supports your dataset
- Check data paths in load_data() function
- Verify dataset is downloaded to correct location

**3. Wandb Authentication**
```bash
# Login to wandb before running
wandb login
```

**4. Job Time Limit**
- If job times out, results up to last epoch are saved
- Resume training not implemented (would require checkpoint loading logic)

### Debugging

**Test locally on CPU**:
```bash
python QCNN_Comparison.py \
    --dataset=cifar10 \
    --n-epochs=2 \
    --batch-size=4 \
    --models qcnn
```

**Check circuit structure**:
```python
from QCNN_Comparison import QCNN, QuantumDilatedCNN
import torch

model = QuantumDilatedCNN(n_qubits=8, n_layers=2, input_dim=3072, num_classes=10)
print(model)

# Visualize circuit
import pennylane as qml
features = torch.randn(8)
qml.draw_mpl(model.circuit)(model.conv_params, model.pool_params, model.last_params, features)
```

## References

### Papers

1. **Cong, I., Choi, S., & Lukin, M. D. (2019)**. Quantum convolutional neural networks. *Nature Physics*, 15(12), 1273-1278. https://arxiv.org/abs/1810.03787

2. **Yu, F., & Koltun, V. (2015)**. Multi-scale context aggregation by dilated convolutions. *ICLR 2016*. (Classical dilation concept)

### Code References

- Original notebook: `Quantum Dilation.ipynb`
- PennyLane documentation: https://pennylane.ai/
- PyTorch documentation: https://pytorch.org/

## Future Extensions

### Potential Improvements

1. **Additional Dilation Patterns**:
   - Stride-3: (0,3), (1,4), (2,5), (3,6), (4,7)
   - Mixed: Combine nearest-neighbor + dilated layers
   - Adaptive: Learn optimal entanglement structure

2. **Deeper Networks**:
   - Test with n_layers=3, 4
   - Analyze how dilation benefits scale with depth

3. **More Datasets**:
   - ImageNet (subset)
   - Medical imaging (X-rays, CT scans)
   - Satellite imagery

4. **Hybrid Architectures**:
   - Classical dilated conv → Quantum dilated conv
   - ResNet-style skip connections in quantum layers

5. **Entanglement Analysis**:
   - Measure entanglement entropy at each layer
   - Quantify global vs local entanglement
   - Correlate entanglement with performance

6. **Hardware Deployment**:
   - Test on real quantum hardware (IBM, IonQ)
   - Compare noise resilience of QCNN vs Dilated CNN

## Contact & Support

For questions or issues:
- Check SLURM output files for error messages
- Review this guide's troubleshooting section
- Examine training logs in CSV files
- Monitor Wandb dashboard for real-time metrics

## Acknowledgments

- QCNN architecture from Cong et al. (2019)
- Dilated convolution concept from Yu & Koltun (2015)
- Implementation uses PennyLane quantum ML framework
- Experiments run on NERSC Perlmutter supercomputer

---

**Good luck with your quantum CNN experiments!**
