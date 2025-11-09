#!/bin/bash

# Master submission script for QCNN comparison experiments
# Compares QCNN vs Quantum Dilated CNN on CIFAR-10 and COCO datasets

echo "=========================================="
echo "Submitting QCNN Comparison Experiments"
echo "=========================================="
echo ""

# Create output directory if it doesn't exist
mkdir -p /pscratch/sd/j/junghoon/qcnn_comparison_results

# Submit CIFAR-10 comparison
echo "Submitting CIFAR-10 comparison..."
JOB_CIFAR10=$(sbatch QCNN_Comparison_CIFAR10.sh | awk '{print $4}')
echo "  Job ID: $JOB_CIFAR10"

# Submit COCO comparison
echo "Submitting COCO comparison..."
JOB_COCO=$(sbatch QCNN_Comparison_COCO.sh | awk '{print $4}')
echo "  Job ID: $JOB_COCO"

echo ""
echo "=========================================="
echo "All Jobs Submitted Successfully"
echo "=========================================="
echo ""
echo "Job IDs:"
echo "  CIFAR-10: $JOB_CIFAR10"
echo "  COCO: $JOB_COCO"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check output files:"
echo "  tail -f /pscratch/sd/j/junghoon/qcnn_comparison_results/QCNN_Comparison_CIFAR10_${JOB_CIFAR10}.out"
echo "  tail -f /pscratch/sd/j/junghoon/qcnn_comparison_results/QCNN_Comparison_COCO_${JOB_COCO}.out"
echo ""
echo "Estimated completion time: ~24-36 hours per job"
echo ""

# Create monitoring script
cat > monitor_qcnn_comparison.sh << 'EOF'
#!/bin/bash
echo "=========================================="
echo "QCNN Comparison Jobs Status"
echo "=========================================="
echo ""

squeue -u $USER --format="%.18i %.9P %.30j %.8T %.10M %.6D %R" | grep "QCNN_Comparison"

echo ""
echo "To cancel all comparison jobs:"
echo "  scancel <JOB_ID>"
echo ""
EOF

chmod +x monitor_qcnn_comparison.sh

echo "Monitoring script created: monitor_qcnn_comparison.sh"
echo ""
