#!/usr/bin/env python3
"""
Visualization script for entanglement measurement results
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Set style
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

def load_results(n_qubits=6, n_layers=2, results_dir='entanglement_results'):
    """Load all entanglement measurement results."""
    results_dir = Path(results_dir)

    data = {
        'QCNN': {
            'meyer_wallach': np.load(results_dir / f'QCNN_meyer_wallach_{n_qubits}q_{n_layers}l.npy'),
            'concentratable': np.load(results_dir / f'QCNN_concentratable_{n_qubits}q_{n_layers}l.npy')
        },
        'QuantumDilatedCNN': {
            'meyer_wallach': np.load(results_dir / f'QuantumDilatedCNN_meyer_wallach_{n_qubits}q_{n_layers}l.npy'),
            'concentratable': np.load(results_dir / f'QuantumDilatedCNN_concentratable_{n_qubits}q_{n_layers}l.npy')
        }
    }

    return data


def plot_distributions(data, output_file='entanglement_distributions.png'):
    """Create comprehensive visualization of entanglement distributions."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Color scheme
    colors = {'QCNN': '#3498db', 'QuantumDilatedCNN': '#e74c3c'}

    # Plot 1: Meyer-Wallach Histograms
    ax = axes[0, 0]
    for model_name, model_data in data.items():
        ax.hist(model_data['meyer_wallach'], bins=20, alpha=0.6,
                label=model_name, color=colors[model_name], edgecolor='black')
    ax.set_xlabel('Meyer-Wallach Q (Entangling Capability)')
    ax.set_ylabel('Frequency')
    ax.set_title('Meyer-Wallach Measure Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Concentratable Entanglement Histograms
    ax = axes[0, 1]
    for model_name, model_data in data.items():
        ax.hist(model_data['concentratable'], bins=20, alpha=0.6,
                label=model_name, color=colors[model_name], edgecolor='black')
    ax.set_xlabel('Concentratable Entanglement C')
    ax.set_ylabel('Frequency')
    ax.set_title('Concentratable Entanglement Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Box plots comparison
    ax = axes[1, 0]
    mw_data = [data['QCNN']['meyer_wallach'], data['QuantumDilatedCNN']['meyer_wallach']]
    bp = ax.boxplot(mw_data, labels=['QCNN', 'Dilated QCNN'],
                    patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Meyer-Wallach Q')
    ax.set_title('Meyer-Wallach Comparison (Box Plot)')
    ax.grid(alpha=0.3, axis='y')

    # Plot 4: Scatter plot
    ax = axes[1, 1]
    for model_name, model_data in data.items():
        ax.scatter(model_data['meyer_wallach'], model_data['concentratable'],
                  alpha=0.6, s=50, label=model_name, color=colors[model_name],
                  edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Meyer-Wallach Q (Entangling Capability)')
    ax.set_ylabel('Concentratable Entanglement C')
    ax.set_title('Correlation: Meyer-Wallach vs Concentratable Entanglement')
    ax.legend()
    ax.grid(alpha=0.3)

    # Add correlation coefficients
    for model_name, model_data in data.items():
        corr = np.corrcoef(model_data['meyer_wallach'], model_data['concentratable'])[0, 1]
        print(f"{model_name} correlation: {corr:.3f}")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def create_summary_table(data):
    """Create detailed summary statistics table."""

    stats = {}

    for model_name, model_data in data.items():
        stats[model_name] = {}
        for measure_name, values in model_data.items():
            stats[model_name][measure_name] = {
                'Mean': np.mean(values),
                'Std': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values),
                'Median': np.median(values),
                'Q1': np.percentile(values, 25),
                'Q3': np.percentile(values, 75)
            }

    # Print summary
    print("\n" + "="*80)
    print("DETAILED STATISTICS SUMMARY")
    print("="*80)

    for model_name in data.keys():
        print(f"\n{model_name}:")
        print("-" * 80)

        for measure_name in ['meyer_wallach', 'concentratable']:
            display_name = "Meyer-Wallach Q" if measure_name == 'meyer_wallach' else "Concentratable Ent"
            print(f"\n  {display_name}:")
            for stat_name, value in stats[model_name][measure_name].items():
                print(f"    {stat_name:>10}: {value:.6f}")

    # Comparison
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)

    mw_diff = stats['QuantumDilatedCNN']['meyer_wallach']['Mean'] - stats['QCNN']['meyer_wallach']['Mean']
    ce_diff = stats['QuantumDilatedCNN']['concentratable']['Mean'] - stats['QCNN']['concentratable']['Mean']

    print(f"\nMeyer-Wallach Difference (Dilated - QCNN): {mw_diff:+.6f}")
    print(f"  Relative difference: {100*mw_diff/stats['QCNN']['meyer_wallach']['Mean']:+.2f}%")

    print(f"\nConcentratable Entanglement Difference (Dilated - QCNN): {ce_diff:+.6f}")
    print(f"  Relative difference: {100*ce_diff/stats['QCNN']['concentratable']['Mean']:+.2f}%")

    return stats


def main():
    print("Loading entanglement measurement results...")

    try:
        data = load_results(n_qubits=6, n_layers=2)

        print("\nCreating visualizations...")
        plot_distributions(data)

        print("\nComputing summary statistics...")
        stats = create_summary_table(data)

        print("\n" + "="*80)
        print("Analysis complete!")
        print("="*80)

    except FileNotFoundError as e:
        print(f"\nError: Could not find results files.")
        print(f"Make sure you've run 'measure_entanglement.py' first.")
        print(f"\nMissing file: {e.filename}")


if __name__ == "__main__":
    main()
