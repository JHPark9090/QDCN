#!/usr/bin/env python3
"""
QCNN vs Quantum Dilated CNN Comparison
========================================
Compares performance of:
1. QCNN (Cong et al. 2019) - nearest-neighbor entanglement
2. Quantum Dilated CNN - non-adjacent entanglement for global connectivity

Supports CIFAR-10 and COCO datasets with configurable parameters.
"""

import os, random, copy, time
import argparse
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import pennylane as qml

# Import data loaders
from LoadData_MultiChip import load_data

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging disabled.")


def set_all_seeds(seed: int = 42) -> None:
    """Seed every RNG we rely on (Python, NumPy, Torch, PennyLane, CUDNN)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    qml.numpy.random.seed(seed)


class QCNN(nn.Module):
    """
    Original QCNN from Cong et al. (2019) with nearest-neighbor entanglement.
    https://arxiv.org/abs/1810.03787
    """
    def __init__(self, n_qubits=8, n_layers=2, input_dim=3072, num_classes=10):
        super(QCNN, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_classes = num_classes

        # Classical dimension reduction
        self.fc = nn.Linear(input_dim, n_qubits)

        # Quantum parameters
        self.conv_params = nn.Parameter(torch.randn(n_layers, n_qubits, 15))
        self.pool_params = nn.Parameter(torch.randn(n_layers, n_qubits // 2, 3))
        self.last_params = nn.Parameter(torch.randn(15))

        # Output layer for multi-class
        self.fc_out = nn.Linear(1, num_classes)

        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(self, conv_weights, pool_weights, last_weights, features):
        wires = list(range(self.n_qubits))

        # Variational Embedding
        qml.AngleEmbedding(features, wires=wires, rotation='Y')

        for layer in range(self.n_layers):
            # Convolutional Layer
            self._apply_convolution(conv_weights[layer], wires)
            # Pooling Layer
            self._apply_pooling(pool_weights[layer], wires)
            wires = wires[::2]

        # Final unitary
        qml.ArbitraryUnitary(last_weights, wires)

        return qml.expval(qml.PauliZ(0))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten

        # Classical reduction
        reduced_x = torch.tanh(self.fc(x))

        # Quantum circuit execution
        qnode = qml.qnode(self.dev, interface="torch")(self.circuit)
        quantum_out = []
        for i in range(batch_size):
            out = qnode(self.conv_params, self.pool_params, self.last_params, reduced_x[i])
            quantum_out.append(out)

        quantum_out = torch.stack(quantum_out).unsqueeze(1)

        # Output layer
        logits = self.fc_out(quantum_out)

        return logits

    def _apply_convolution(self, weights, wires):
        """Nearest-neighbor convolutional layer (original QCNN)."""
        n_wires = len(wires)
        for p in [0, 1]:
            for indx, w in enumerate(wires):
                if indx % 2 == p and indx < n_wires - 1:
                    qml.U3(*weights[indx, :3], wires=w)
                    qml.U3(*weights[indx + 1, 3:6], wires=wires[indx + 1])
                    qml.IsingZZ(weights[indx, 6], wires=[w, wires[indx + 1]])
                    qml.IsingYY(weights[indx, 7], wires=[w, wires[indx + 1]])
                    qml.IsingXX(weights[indx, 8], wires=[w, wires[indx + 1]])
                    qml.U3(*weights[indx, 9:12], wires=w)
                    qml.U3(*weights[indx + 1, 12:], wires=wires[indx + 1])

    def _apply_pooling(self, pool_weights, wires):
        """Pooling using mid-circuit measurement."""
        n_wires = len(wires)
        assert n_wires >= 2, "Need at least two wires for pooling."
        for indx, w in enumerate(wires):
            if indx % 2 == 1 and indx < n_wires:
                measurement = qml.measure(w)
                qml.cond(measurement, qml.U3)(*pool_weights[indx // 2], wires=wires[indx - 1])


class QuantumDilatedCNN(nn.Module):
    """
    Quantum Dilated CNN with non-adjacent entanglement for global connectivity.
    Reduces local entanglement, increases global entanglement.
    """
    def __init__(self, n_qubits=8, n_layers=2, input_dim=3072, num_classes=10):
        super(QuantumDilatedCNN, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_classes = num_classes

        # Classical dimension reduction
        self.fc = nn.Linear(input_dim, n_qubits)

        # Quantum parameters (same structure as QCNN for fair comparison)
        self.conv_params = nn.Parameter(torch.randn(n_layers, n_qubits, 15))
        self.pool_params = nn.Parameter(torch.randn(n_layers, n_qubits // 2, 3))
        self.last_params = nn.Parameter(torch.randn(15))

        # Output layer
        self.fc_out = nn.Linear(1, num_classes)

        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(self, conv_weights, pool_weights, last_weights, features):
        wires = list(range(self.n_qubits))

        # Variational Embedding
        qml.AngleEmbedding(features, wires=wires, rotation='Y')

        for layer in range(self.n_layers):
            # Dilated Convolutional Layer
            self._apply_dilated_convolution(conv_weights[layer], wires)
            # Pooling Layer (same as QCNN)
            self._apply_pooling(pool_weights[layer], wires)
            wires = wires[::2]

        # Final unitary
        qml.ArbitraryUnitary(last_weights, wires)

        return qml.expval(qml.PauliZ(0))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten

        # Classical reduction
        reduced_x = torch.tanh(self.fc(x))

        # Quantum circuit execution
        qnode = qml.qnode(self.dev, interface="torch")(self.circuit)
        quantum_out = []
        for i in range(batch_size):
            out = qnode(self.conv_params, self.pool_params, self.last_params, reduced_x[i])
            quantum_out.append(out)

        quantum_out = torch.stack(quantum_out).unsqueeze(1)

        # Output layer
        logits = self.fc_out(quantum_out)

        return logits

    def _apply_dilated_convolution(self, weights, wires):
        """
        Non-adjacent entanglement pattern for global connectivity.
        Uses dilation to skip neighbors and connect distant qubits.
        """
        n_wires = len(wires)

        # Define dilated entanglement pairs based on number of wires
        if n_wires == 8:
            entanglement_pairs = [(0, 2), (1, 3), (4, 6), (5, 7)]
        elif n_wires == 4:
            entanglement_pairs = [(0, 2), (1, 3)]
        elif n_wires == 2:
            entanglement_pairs = [(0, 1)]  # Fall back to nearest neighbor
        else:
            # General case: stride of 2
            entanglement_pairs = [(i, i+2) for i in range(n_wires-2)]

        processed_qubits = set()

        # Apply entangling blocks to dilated pairs
        for q1, q2 in entanglement_pairs:
            if q1 in wires and q2 in wires:
                qml.U3(*weights[q1, :3], wires=q1)
                qml.U3(*weights[q2, 3:6], wires=q2)
                qml.IsingZZ(weights[q1, 6], wires=[q1, q2])
                qml.IsingYY(weights[q1, 7], wires=[q1, q2])
                qml.IsingXX(weights[q1, 8], wires=[q1, q2])
                qml.U3(*weights[q1, 9:12], wires=q1)
                qml.U3(*weights[q2, 12:], wires=q2)
                processed_qubits.add(q1)
                processed_qubits.add(q2)

        # Apply single-qubit gates to remaining qubits
        for w in wires:
            if w not in processed_qubits:
                for i in range(5):  # 5 U3 gates = 15 parameters
                    qml.U3(*weights[w, i*3:(i+1)*3], wires=w)

    def _apply_pooling(self, pool_weights, wires):
        """Pooling using mid-circuit measurement (same as QCNN)."""
        n_wires = len(wires)
        assert n_wires >= 2, "Need at least two wires for pooling."
        for indx, w in enumerate(wires):
            if indx % 2 == 1 and indx < n_wires:
                measurement = qml.measure(w)
                qml.cond(measurement, qml.U3)(*pool_weights[indx // 2], wires=wires[indx - 1])


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, (data, target) in enumerate(tqdm(loader, desc="Training")):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in tqdm(loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    # Additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    return avg_loss, accuracy, precision, recall, f1


def main(args):
    # Set seeds
    set_all_seeds(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset.lower() == 'cifar10':
        input_dim = 3072  # 32x32x3
        num_classes = 10
    elif args.dataset.lower() == 'coco':
        input_dim = 224 * 224 * 3  # Typical COCO image size
        num_classes = 80  # COCO has 80 classes
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader, val_loader, test_loader = load_data(
        dataset_name=args.dataset.lower(),
        batch_size=args.batch_size,
        num_workers=4
    )

    # Initialize models
    print(f"\nInitializing models with {args.n_qubits} qubits, {args.n_layers} layers...")

    models = {}
    if 'qcnn' in args.models:
        models['QCNN'] = QCNN(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            input_dim=input_dim,
            num_classes=num_classes
        ).to(device)

    if 'dilated' in args.models:
        models['QuantumDilatedCNN'] = QuantumDilatedCNN(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            input_dim=input_dim,
            num_classes=num_classes
        ).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    results = {}

    # Train each model
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        # Initialize wandb
        if args.wandb and WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{model_name}_{args.dataset}_{args.job_id}",
                config={
                    "model": model_name,
                    "dataset": args.dataset,
                    "n_qubits": args.n_qubits,
                    "n_layers": args.n_layers,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "wd": args.wd,
                    "epochs": args.n_epochs,
                    "seed": args.seed
                }
            )

        best_val_acc = 0.0
        best_epoch = 0
        history = []

        for epoch in range(args.n_epochs):
            print(f"\nEpoch {epoch+1}/{args.n_epochs}")

            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
                model, val_loader, criterion, device
            )

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

            # Log to wandb
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1
                })

            # Save history
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1
            })

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc
                }, output_dir / f"{model_name}_{args.dataset}_best.pt")

        # Test best model
        print(f"\nLoading best model from epoch {best_epoch}...")
        checkpoint = torch.load(output_dir / f"{model_name}_{args.dataset}_best.pt")
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
            model, test_loader, criterion, device
        )

        print(f"\nTest Results for {model_name}:")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1 Score: {test_f1:.4f}")

        # Save results
        results[model_name] = {
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'history': history
        }

        # Save training history
        pd.DataFrame(history).to_csv(
            output_dir / f"{model_name}_{args.dataset}_history.csv",
            index=False
        )

        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

    # Save comparison results
    comparison_df = pd.DataFrame({
        model_name: {
            'Best Epoch': info['best_epoch'],
            'Best Val Acc': info['best_val_acc'],
            'Test Acc': info['test_acc'],
            'Test Precision': info['test_precision'],
            'Test Recall': info['test_recall'],
            'Test F1': info['test_f1']
        }
        for model_name, info in results.items()
    }).T

    comparison_df.to_csv(output_dir / f"comparison_{args.dataset}_{args.job_id}.csv")

    print(f"\n{'='*60}")
    print("Comparison Results:")
    print(f"{'='*60}")
    print(comparison_df)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QCNN vs Quantum Dilated CNN Comparison")

    # Model parameters
    parser.add_argument('--n-qubits', type=int, default=8, help='Number of qubits')
    parser.add_argument('--n-layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--models', nargs='+', default=['qcnn', 'dilated'],
                       choices=['qcnn', 'dilated'], help='Models to compare')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'coco'], help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')

    # Training parameters
    parser.add_argument('--n-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')

    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./qcnn_comparison_results',
                       help='Output directory')
    parser.add_argument('--job-id', type=str, default='comparison',
                       help='Job ID for naming')

    # Wandb parameters
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--wandb-project', type=str, default='QCNN_Comparison',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default='QML_Research',
                       help='Wandb entity name')

    args = parser.parse_args()

    main(args)
