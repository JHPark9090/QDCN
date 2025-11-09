#!/usr/bin/env python3
"""
Visualization for local vs global entanglement analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

def load_results(results_dir='local_global_results'):
    """Load analysis results."""
    results_dir = Path(results_dir)

    qcnn_data = np.load(results_dir / 'QCNN_local_global_results.npy', allow_pickle=True).item()
    dilated_data = np.load(results_dir / 'QuantumDilatedCNN_local_global_results.npy', allow_pickle=True).item()

    return {'QCNN': qcnn_data, 'QuantumDilatedCNN': dilated_data}


def plot_analysis(data, output_file='local_global_entanglement_analysis.png'):
    """Create comprehensive visualization."""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    colors = {'QCNN': '#3498db', 'QuantumDilatedCNN': '#e74c3c'}

    # Plot 1: Distance-dependent entanglement
    ax1 = fig.add_subplot(gs[0, 0])

    for model_name, model_data in data.items():
        distances = []
        means = []
        stds = []

        for dist in sorted(model_data['distance_entanglement'].keys()):
            distances.append(dist)
            means.append(model_data['distance_entanglement'][dist]['mean'])
            stds.append(model_data['distance_entanglement'][dist]['std'])

        ax1.errorbar(distances, means, yerr=stds, marker='o', markersize=8,
                    linewidth=2, capsize=5, label=model_name, color=colors[model_name])

    ax1.set_xlabel('Qubit Distance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mutual Information (bits)', fontsize=12, fontweight='bold')
    ax1.set_title('Distance-Dependent Entanglement\n(Local vs Global)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Add annotations
    ax1.annotate('LOCAL\n(nearest neighbor)', xy=(1, 0.25), xytext=(1, 0.4),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax1.annotate('GLOBAL\n(long-range)', xy=(4, 0.5), xytext=(3.5, 0.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # Plot 2: Bar chart of local vs global
    ax2 = fig.add_subplot(gs[0, 1])

    qcnn_local = data['QCNN']['distance_entanglement'][1]['mean']
    qcnn_global = np.mean([data['QCNN']['distance_entanglement'][d]['mean']
                          for d in range(2, 6)])

    dilated_local = data['QuantumDilatedCNN']['distance_entanglement'][1]['mean']
    dilated_global = np.mean([data['QuantumDilatedCNN']['distance_entanglement'][d]['mean']
                             for d in range(2, 6)])

    x = np.arange(2)
    width = 0.35

    bars1 = ax2.bar(x - width/2, [qcnn_local, qcnn_global], width,
                   label='QCNN', color=colors['QCNN'], alpha=0.7)
    bars2 = ax2.bar(x + width/2, [dilated_local, dilated_global], width,
                   label='QuantumDilatedCNN', color=colors['QuantumDilatedCNN'], alpha=0.7)

    ax2.set_ylabel('Mutual Information (bits)', fontsize=12, fontweight='bold')
    ax2.set_title('Local vs Global Entanglement\n(Summary)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['LOCAL\n(distance=1)', 'GLOBAL\n(distance>1)'], fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 3: Bipartite entanglement entropy
    ax3 = fig.add_subplot(gs[1, 0])

    for model_name, model_data in data.items():
        cuts = []
        means = []
        stds = []

        for cut in sorted(model_data['bipartite_entropy'].keys()):
            cuts.append(cut)
            means.append(model_data['bipartite_entropy'][cut]['mean'])
            stds.append(model_data['bipartite_entropy'][cut]['std'])

        ax3.errorbar(cuts, means, yerr=stds, marker='s', markersize=8,
                    linewidth=2, capsize=5, label=model_name, color=colors[model_name])

    ax3.set_xlabel('Bipartite Cut Position', fontsize=12, fontweight='bold')
    ax3.set_ylabel('von Neumann Entropy (bits)', fontsize=12, fontweight='bold')
    ax3.set_title('Bipartite Entanglement Entropy\n(Entanglement Across Cuts)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3)

    # Plot 4: Concentratable entanglement by subset size
    ax4 = fig.add_subplot(gs[1, 1])

    for model_name, model_data in data.items():
        sizes = []
        means = []
        stds = []

        for k in sorted(model_data['concentratable_by_size'].keys()):
            sizes.append(k)
            means.append(model_data['concentratable_by_size'][k]['mean'])
            stds.append(model_data['concentratable_by_size'][k]['std'])

        ax4.errorbar(sizes, means, yerr=stds, marker='D', markersize=8,
                    linewidth=2, capsize=5, label=model_name, color=colors[model_name])

    ax4.set_xlabel('Subset Size k', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Concentratable Entanglement\nContribution', fontsize=12, fontweight='bold')
    ax4.set_title('k-Body Concentratable Entanglement\n(Local to Global)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(alpha=0.3)
    ax4.set_xticks(sizes)

    # Add shaded regions
    ax4.axvspan(1, 2.5, alpha=0.1, color='blue', label='_nolegend_')
    ax4.text(1.75, ax4.get_ylim()[1]*0.95, 'Local\n(Few-body)', ha='center', va='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    ax4.axvspan(4.5, 6, alpha=0.1, color='red', label='_nolegend_')
    ax4.text(5.25, ax4.get_ylim()[1]*0.95, 'Global\n(Many-body)', ha='center', va='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    # Plot 5: Total concentratable entanglement comparison
    ax5 = fig.add_subplot(gs[2, 0])

    qcnn_total = sum([data['QCNN']['concentratable_by_size'][k]['mean']
                     for k in data['QCNN']['concentratable_by_size'].keys()])
    dilated_total = sum([data['QuantumDilatedCNN']['concentratable_by_size'][k]['mean']
                        for k in data['QuantumDilatedCNN']['concentratable_by_size'].keys()])

    models = ['QCNN', 'QuantumDilatedCNN']
    totals = [qcnn_total, dilated_total]

    bars = ax5.bar(models, totals, color=[colors[m] for m in models], alpha=0.7, width=0.5)
    ax5.set_ylabel('Total Concentratable\nEntanglement', fontsize=12, fontweight='bold')
    ax5.set_title('Total Concentratable Entanglement\n(Sum Over All k)', fontsize=14, fontweight='bold')
    ax5.grid(alpha=0.3, axis='y')

    for bar, total in zip(bars, totals):
        ax5.text(bar.get_x() + bar.get_width()/2., total,
                f'{total:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 6: Ratio analysis
    ax6 = fig.add_subplot(gs[2, 1])

    sizes = sorted(data['QCNN']['concentratable_by_size'].keys())
    ratios = []

    for k in sizes:
        qcnn_val = data['QCNN']['concentratable_by_size'][k]['mean']
        dilated_val = data['QuantumDilatedCNN']['concentratable_by_size'][k]['mean']
        ratio = dilated_val / qcnn_val if qcnn_val > 0 else 0
        ratios.append(ratio)

    colors_gradient = ['#2ecc71' if r > 1 else '#e74c3c' for r in ratios]
    bars = ax6.bar(sizes, ratios, color=colors_gradient, alpha=0.7)
    ax6.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Equal')
    ax6.set_xlabel('Subset Size k', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Ratio: Dilated / QCNN', fontsize=12, fontweight='bold')
    ax6.set_title('k-Body Entanglement Ratio\n(Dilated/QCNN)', fontsize=14, fontweight='bold')
    ax6.set_xticks(sizes)
    ax6.grid(alpha=0.3, axis='y')
    ax6.legend(fontsize=10)

    # Add value labels
    for bar, ratio in zip(bars, ratios):
        ax6.text(bar.get_x() + bar.get_width()/2., ratio,
                f'{ratio:.2f}×', ha='center', va='bottom', fontsize=9)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def print_summary(data):
    """Print detailed summary."""
    print("\n" + "="*80)
    print("LOCAL VS GLOBAL ENTANGLEMENT SUMMARY")
    print("="*80)

    # Local vs global
    qcnn_local = data['QCNN']['distance_entanglement'][1]['mean']
    qcnn_global = np.mean([data['QCNN']['distance_entanglement'][d]['mean']
                          for d in range(2, 6)])

    dilated_local = data['QuantumDilatedCNN']['distance_entanglement'][1]['mean']
    dilated_global = np.mean([data['QuantumDilatedCNN']['distance_entanglement'][d]['mean']
                             for d in range(2, 6)])

    print("\n1. MUTUAL INFORMATION ANALYSIS")
    print("-" * 80)
    print(f"\nLOCAL Entanglement (distance=1):")
    print(f"  QCNN:              {qcnn_local:.4f}")
    print(f"  QuantumDilatedCNN: {dilated_local:.4f}")
    if dilated_local > 0:
        print(f"  Ratio (QCNN/Dilated): {qcnn_local/dilated_local:.2f}×")
    else:
        print(f"  → QCNN has infinitely more local entanglement!")

    print(f"\nGLOBAL Entanglement (distance>1, average):")
    print(f"  QCNN:              {qcnn_global:.4f}")
    print(f"  QuantumDilatedCNN: {dilated_global:.4f}")
    print(f"  Ratio (Dilated/QCNN): {dilated_global/qcnn_global:.2f}×")

    # Concentratable entanglement
    print("\n2. CONCENTRATABLE ENTANGLEMENT BY SIZE")
    print("-" * 80)

    for k in sorted(data['QCNN']['concentratable_by_size'].keys()):
        qcnn_val = data['QCNN']['concentratable_by_size'][k]['mean']
        dilated_val = data['QuantumDilatedCNN']['concentratable_by_size'][k]['mean']
        ratio = dilated_val / qcnn_val

        print(f"\nk={k} ({['', 'single', 'pair', 'triplet', 'quad', 'quint', 'all'][k]}):")
        print(f"  QCNN:              {qcnn_val:.4f}")
        print(f"  QuantumDilatedCNN: {dilated_val:.4f}")
        print(f"  Ratio (Dilated/QCNN): {ratio:.2f}×", end="")
        if ratio > 1.1:
            print(" ✓ (Dilated wins)")
        elif ratio < 0.9:
            print(" ✗ (QCNN wins)")
        else:
            print(" ≈ (Similar)")

    # Total
    qcnn_total = sum([data['QCNN']['concentratable_by_size'][k]['mean']
                     for k in data['QCNN']['concentratable_by_size'].keys()])
    dilated_total = sum([data['QuantumDilatedCNN']['concentratable_by_size'][k]['mean']
                        for k in data['QuantumDilatedCNN']['concentratable_by_size'].keys()])

    print(f"\nTOTAL Concentratable Entanglement:")
    print(f"  QCNN:              {qcnn_total:.4f}")
    print(f"  QuantumDilatedCNN: {dilated_total:.4f}")
    print(f"  Ratio (Dilated/QCNN): {dilated_total/qcnn_total:.2f}×")


def main():
    print("Loading local vs global entanglement results...")

    try:
        data = load_results()

        print("\nCreating visualizations...")
        plot_analysis(data)

        print_summary(data)

        print("\n" + "="*80)
        print("Analysis complete!")
        print("="*80)

    except FileNotFoundError as e:
        print(f"\nError: Could not find results files.")
        print(f"Make sure you've run 'measure_local_global_entanglement.py' first.")


if __name__ == "__main__":
    main()
