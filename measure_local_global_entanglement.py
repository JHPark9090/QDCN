#!/usr/bin/env python3
"""
Local vs Global Entanglement Analysis
======================================
Measures spatial structure of entanglement in quantum circuits:
1. Distance-dependent pairwise entanglement
2. Bipartite entanglement entropy (von Neumann entropy)
3. Concentratable entanglement decomposed by subset size
4. Mutual information between qubit pairs

Distinguishes:
- LOCAL entanglement: nearest-neighbor, small subsets
- GLOBAL entanglement: long-range, large subsets
"""

import os
import random
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import torch
import pennylane as qml
from tqdm import tqdm
from itertools import combinations
from scipy.linalg import sqrtm


def set_all_seeds(seed: int = 42) -> None:
    """Seed every RNG we rely on."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    qml.numpy.random.seed(seed)


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Compute von Neumann entropy: S = -Tr(ρ log ρ)"""
    eigenvalues = np.linalg.eigvalsh(rho)
    # Remove numerical zeros
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log2(eigenvalues))


def partial_trace(state_vector: np.ndarray, n_qubits: int, keep_qubits: List[int]) -> np.ndarray:
    """
    Compute partial trace, keeping only specified qubits.

    Args:
        state_vector: State vector of shape (2^n_qubits,)
        n_qubits: Total number of qubits
        keep_qubits: List of qubit indices to keep

    Returns:
        Reduced density matrix for kept qubits
    """
    # Reshape to tensor
    psi = state_vector.reshape([2] * n_qubits)

    # Determine qubits to trace out
    trace_qubits = [i for i in range(n_qubits) if i not in keep_qubits]

    # Move kept qubits to front
    axes_order = list(keep_qubits) + trace_qubits
    psi_reordered = np.moveaxis(psi, axes_order, range(n_qubits))

    # Reshape for partial trace
    keep_dim = 2 ** len(keep_qubits)
    trace_dim = 2 ** len(trace_qubits)
    psi_matrix = psi_reordered.reshape(keep_dim, trace_dim)

    # Compute reduced density matrix
    rho = psi_matrix @ psi_matrix.conj().T

    return rho


def mutual_information(state_vector: np.ndarray, n_qubits: int,
                       qubit_i: int, qubit_j: int) -> float:
    """
    Compute mutual information I(i:j) = S(ρ_i) + S(ρ_j) - S(ρ_{ij})

    Measures correlations (classical + quantum) between two qubits.
    """
    # Single qubit reduced density matrices
    rho_i = partial_trace(state_vector, n_qubits, [qubit_i])
    rho_j = partial_trace(state_vector, n_qubits, [qubit_j])

    # Two-qubit reduced density matrix
    rho_ij = partial_trace(state_vector, n_qubits, [qubit_i, qubit_j])

    # Compute entropies
    S_i = von_neumann_entropy(rho_i)
    S_j = von_neumann_entropy(rho_j)
    S_ij = von_neumann_entropy(rho_ij)

    return S_i + S_j - S_ij


def concurrence(rho: np.ndarray) -> float:
    """
    Compute concurrence for a two-qubit density matrix.
    Measures entanglement between two qubits (0 = separable, 1 = maximally entangled).
    """
    # Pauli Y matrix
    sigma_y = np.array([[0, -1j], [1j, 0]])

    # Two-qubit Pauli Y
    sigma_y_kron = np.kron(sigma_y, sigma_y)

    # Spin-flipped density matrix
    rho_tilde = sigma_y_kron @ rho.conj() @ sigma_y_kron

    # Compute eigenvalues of rho * rho_tilde
    R = rho @ rho_tilde
    eigenvalues = np.linalg.eigvalsh(R)
    eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))  # Ensure non-negative
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    C = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])

    return C


def bipartite_entanglement_entropy(state_vector: np.ndarray, n_qubits: int,
                                   cut_position: int) -> float:
    """
    Compute von Neumann entropy for bipartite cut at position k.
    Partitions qubits as {0,...,k-1} | {k,...,n-1}
    """
    subsystem_A = list(range(cut_position))
    rho_A = partial_trace(state_vector, n_qubits, subsystem_A)
    return von_neumann_entropy(rho_A)


def concentratable_entanglement_by_size(state_vector: np.ndarray, n_qubits: int) -> Dict[int, float]:
    """
    Compute concentratable entanglement decomposed by subset size.

    Returns:
        Dictionary mapping subset size k -> concentratable entanglement contribution
    """
    psi = state_vector.reshape([2] * n_qubits)

    contributions = {}

    for k in range(1, n_qubits + 1):
        sum_purities = 0.0
        n_subsets = 0

        for subset in combinations(range(n_qubits), k):
            purity = compute_subset_purity(psi, n_qubits, subset)
            sum_purities += purity
            n_subsets += 1

        # Contribution from subsets of size k
        contributions[k] = sum_purities / (2 ** n_qubits)

    return contributions


def compute_subset_purity(psi: np.ndarray, n_qubits: int, subset: Tuple[int]) -> float:
    """Compute purity Tr(rho^2) for a subset of qubits."""
    qubits_to_trace = [i for i in range(n_qubits) if i not in subset]
    axes_order = list(subset) + qubits_to_trace
    psi_reordered = np.moveaxis(psi, axes_order, range(n_qubits))

    subset_dim = 2 ** len(subset)
    complement_dim = 2 ** len(qubits_to_trace)
    psi_matrix = psi_reordered.reshape(subset_dim, complement_dim)

    rho = psi_matrix @ psi_matrix.conj().T
    purity = np.trace(rho @ rho).real

    return purity


def distance_dependent_entanglement(state_vector: np.ndarray, n_qubits: int) -> Dict[int, List[float]]:
    """
    Compute pairwise entanglement as function of qubit distance.

    Returns:
        Dictionary mapping distance -> list of mutual information values
    """
    distances = {}

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            dist = j - i
            mi = mutual_information(state_vector, n_qubits, i, j)

            if dist not in distances:
                distances[dist] = []
            distances[dist].append(mi)

    return distances


# Circuit building functions (same as before)
def apply_qcnn_convolution(weights, wires):
    """Nearest-neighbor convolutional layer."""
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


def apply_dilated_convolution(weights, wires):
    """Non-adjacent entanglement pattern."""
    n_wires = len(wires)

    if n_wires == 8:
        entanglement_pairs = [(0, 2), (1, 3), (4, 6), (5, 7)]
    elif n_wires == 6:
        entanglement_pairs = [(0, 2), (1, 3), (2, 4), (3, 5)]
    elif n_wires == 4:
        entanglement_pairs = [(0, 2), (1, 3)]
    elif n_wires == 2:
        entanglement_pairs = [(0, 1)]
    else:
        entanglement_pairs = [(i, i+2) for i in range(n_wires-2)]

    processed_qubits = set()

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

    for w in wires:
        if w not in processed_qubits:
            for i in range(5):
                qml.U3(*weights[w, i*3:(i+1)*3], wires=w)


def build_qcnn_circuit(n_qubits: int, n_layers: int):
    """Build QCNN circuit (nearest-neighbor)."""
    def circuit(params):
        conv_params = params['conv']
        features = params['features']
        wires = list(range(n_qubits))

        qml.AngleEmbedding(features, wires=wires, rotation='Y')

        for layer in range(n_layers):
            apply_qcnn_convolution(conv_params[layer], wires)
            wires = wires[::2]

    return circuit


def build_dilated_qcnn_circuit(n_qubits: int, n_layers: int):
    """Build QuantumDilatedCNN circuit (non-adjacent)."""
    def circuit(params):
        conv_params = params['conv']
        features = params['features']
        wires = list(range(n_qubits))

        qml.AngleEmbedding(features, wires=wires, rotation='Y')

        for layer in range(n_layers):
            apply_dilated_convolution(conv_params[layer], wires)
            wires = wires[::2]

    return circuit


def generate_random_params(n_qubits: int, n_layers: int, n_samples: int) -> List[Dict]:
    """Generate random parameters."""
    params_list = []
    for _ in range(n_samples):
        features = np.random.uniform(-np.pi, np.pi, n_qubits)
        conv_params = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 15))

        params_list.append({
            'features': features,
            'conv': conv_params
        })

    return params_list


def analyze_local_global_entanglement(circuit_function, n_qubits: int,
                                     params_list: List[Dict]) -> Dict:
    """
    Comprehensive local vs global entanglement analysis.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        circuit_function(params)
        return qml.state()

    # Storage for results
    distance_ent_all = {d: [] for d in range(1, n_qubits)}
    bipartite_entropy_all = {k: [] for k in range(1, n_qubits)}
    concentratable_by_size_all = {k: [] for k in range(1, n_qubits + 1)}

    for params in tqdm(params_list, desc="Analyzing entanglement structure"):
        state = circuit(params)

        # 1. Distance-dependent entanglement
        dist_ent = distance_dependent_entanglement(state, n_qubits)
        for dist, values in dist_ent.items():
            distance_ent_all[dist].extend(values)

        # 2. Bipartite entanglement entropy
        for cut_pos in range(1, n_qubits):
            entropy = bipartite_entanglement_entropy(state, n_qubits, cut_pos)
            bipartite_entropy_all[cut_pos].append(entropy)

        # 3. Concentratable entanglement by subset size
        conc_by_size = concentratable_entanglement_by_size(state, n_qubits)
        for k, value in conc_by_size.items():
            concentratable_by_size_all[k].append(value)

    # Compute statistics
    results = {
        'distance_entanglement': {
            dist: {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
            for dist, values in distance_ent_all.items()
        },
        'bipartite_entropy': {
            cut: {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
            for cut, values in bipartite_entropy_all.items()
        },
        'concentratable_by_size': {
            k: {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
            for k, values in concentratable_by_size_all.items()
        }
    }

    return results


def main(args):
    set_all_seeds(args.seed)

    print("="*70)
    print("LOCAL vs GLOBAL ENTANGLEMENT ANALYSIS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Number of qubits: {args.n_qubits}")
    print(f"  Number of layers: {args.n_layers}")
    print(f"  Number of samples: {args.n_samples}")
    print(f"  Random seed: {args.seed}")

    # Generate parameters
    print(f"\nGenerating {args.n_samples} random parameter samples...")
    params_samples = generate_random_params(args.n_qubits, args.n_layers, args.n_samples)

    results = {}

    # Analyze QCNN
    print("\n" + "="*70)
    print("Analyzing QCNN (Nearest-Neighbor Entanglement)")
    print("="*70)
    qcnn_circuit = build_qcnn_circuit(args.n_qubits, args.n_layers)
    results['QCNN'] = analyze_local_global_entanglement(qcnn_circuit, args.n_qubits, params_samples)

    # Analyze QuantumDilatedCNN
    print("\n" + "="*70)
    print("Analyzing QuantumDilatedCNN (Dilated Entanglement)")
    print("="*70)
    dilated_circuit = build_dilated_qcnn_circuit(args.n_qubits, args.n_layers)
    results['QuantumDilatedCNN'] = analyze_local_global_entanglement(dilated_circuit, args.n_qubits, params_samples)

    # Display results
    print("\n" + "="*70)
    print("DISTANCE-DEPENDENT ENTANGLEMENT (Mutual Information)")
    print("="*70)
    print("\nQCNN:")
    for dist in sorted(results['QCNN']['distance_entanglement'].keys()):
        stats = results['QCNN']['distance_entanglement'][dist]
        print(f"  Distance {dist}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print("\nQuantumDilatedCNN:")
    for dist in sorted(results['QuantumDilatedCNN']['distance_entanglement'].keys()):
        stats = results['QuantumDilatedCNN']['distance_entanglement'][dist]
        print(f"  Distance {dist}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print("\n" + "="*70)
    print("BIPARTITE ENTANGLEMENT ENTROPY")
    print("="*70)
    print("\nQCNN:")
    for cut in sorted(results['QCNN']['bipartite_entropy'].keys()):
        stats = results['QCNN']['bipartite_entropy'][cut]
        print(f"  Cut at position {cut}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print("\nQuantumDilatedCNN:")
    for cut in sorted(results['QuantumDilatedCNN']['bipartite_entropy'].keys()):
        stats = results['QuantumDilatedCNN']['bipartite_entropy'][cut]
        print(f"  Cut at position {cut}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print("\n" + "="*70)
    print("CONCENTRATABLE ENTANGLEMENT BY SUBSET SIZE")
    print("="*70)
    print("\nQCNN:")
    for k in sorted(results['QCNN']['concentratable_by_size'].keys()):
        stats = results['QCNN']['concentratable_by_size'][k]
        print(f"  k={k} ({k}-body): {stats['mean']:.4f} ± {stats['std']:.4f}")

    print("\nQuantumDilatedCNN:")
    for k in sorted(results['QuantumDilatedCNN']['concentratable_by_size'].keys()):
        stats = results['QuantumDilatedCNN']['concentratable_by_size'][k]
        print(f"  k={k} ({k}-body): {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as numpy files
    for model_name, model_results in results.items():
        np.save(output_dir / f'{model_name}_local_global_results.npy', model_results)

    print(f"\n\nResults saved to: {output_dir}")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    # Compare local (distance=1) vs global (distance>1) entanglement
    qcnn_local = results['QCNN']['distance_entanglement'][1]['mean']
    qcnn_global = np.mean([results['QCNN']['distance_entanglement'][d]['mean']
                           for d in range(2, args.n_qubits)])

    dilated_local = results['QuantumDilatedCNN']['distance_entanglement'][1]['mean']
    dilated_global = np.mean([results['QuantumDilatedCNN']['distance_entanglement'][d]['mean']
                              for d in range(2, args.n_qubits)])

    print(f"\nLOCAL Entanglement (distance=1):")
    print(f"  QCNN: {qcnn_local:.4f}")
    print(f"  QuantumDilatedCNN: {dilated_local:.4f}")
    print(f"  → QCNN has {(qcnn_local/dilated_local - 1)*100:+.1f}% {'MORE' if qcnn_local > dilated_local else 'LESS'} local entanglement")

    print(f"\nGLOBAL Entanglement (distance>1):")
    print(f"  QCNN: {qcnn_global:.4f}")
    print(f"  QuantumDilatedCNN: {dilated_global:.4f}")
    print(f"  → QuantumDilatedCNN has {(dilated_global/qcnn_global - 1)*100:+.1f}% {'MORE' if dilated_global > qcnn_global else 'LESS'} global entanglement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure local vs global entanglement in quantum circuits"
    )

    parser.add_argument('--n-qubits', type=int, default=6,
                       help='Number of qubits (default: 6)')
    parser.add_argument('--n-layers', type=int, default=2,
                       help='Number of layers (default: 2)')
    parser.add_argument('--n-samples', type=int, default=50,
                       help='Number of random parameter samples (default: 50)')
    parser.add_argument('--seed', type=int, default=2025,
                       help='Random seed (default: 2025)')
    parser.add_argument('--output-dir', type=str, default='./local_global_results',
                       help='Output directory for results')

    args = parser.parse_args()

    main(args)
