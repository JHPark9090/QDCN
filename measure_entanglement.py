#!/usr/bin/env python3
"""
Entanglement Measurement for QCNN and QuantumDilatedCNN
=========================================================
Measures quantum entanglement using:
1. Meyer-Wallach measure (Entangling Capability) - Sim et al. 2019
2. Concentratable Entanglement - Beckey et al. 2021

References:
- Sim et al. (2019): https://arxiv.org/abs/1905.10876
- Beckey et al. (2021): https://arxiv.org/pdf/2104.06923
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

# Note: We implement the circuits directly in this file for independence


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


def meyer_wallach_measure(state_vector: np.ndarray, n_qubits: int) -> float:
    """
    Compute Meyer-Wallach entanglement measure Q.

    Q = 2 * (1 - (1/n) * sum_of_purities)

    where purity of qubit i is Tr(rho_i^2) for reduced density matrix rho_i.

    Args:
        state_vector: Complex state vector of shape (2^n_qubits,)
        n_qubits: Number of qubits

    Returns:
        Q: Meyer-Wallach measure in [0, 1]
           Q=0 for product states, Q=1 for maximally entangled states
    """
    # Reshape state vector to tensor of shape (2, 2, ..., 2)
    psi = state_vector.reshape([2] * n_qubits)

    sum_purities = 0.0

    # Compute purity for each qubit
    for qubit_idx in range(n_qubits):
        # Move target qubit to first position
        psi_reordered = np.moveaxis(psi, qubit_idx, 0)
        # Reshape to (2, 2^(n-1)) for reduced density matrix
        psi_matrix = psi_reordered.reshape(2, -1)

        # Compute reduced density matrix: rho = |psi><psi|
        rho = psi_matrix @ psi_matrix.conj().T

        # Compute purity: Tr(rho^2)
        purity = np.trace(rho @ rho).real
        sum_purities += purity

    # Meyer-Wallach formula
    Q = 2 * (1 - sum_purities / n_qubits)

    return Q


def concentratable_entanglement(state_vector: np.ndarray, n_qubits: int) -> float:
    """
    Compute Concentratable Entanglement (Beckey et al. 2021).

    C(|psi>) = 1 - (1/2^n) * sum_{alpha in P(n)} Tr(rho_alpha^2)

    where P(n) is the power set of n qubits (all subsets).

    Args:
        state_vector: Complex state vector of shape (2^n_qubits,)
        n_qubits: Number of qubits

    Returns:
        C: Concentratable entanglement measure in [0, 1]
    """
    # Reshape state vector to tensor of shape (2, 2, ..., 2)
    psi = state_vector.reshape([2] * n_qubits)

    sum_purities = 0.0

    # Iterate over all non-empty subsets of qubits (power set minus empty set)
    for subset_size in range(1, n_qubits + 1):
        for subset in combinations(range(n_qubits), subset_size):
            # Compute reduced density matrix for this subset
            purity = compute_subset_purity(psi, n_qubits, subset)
            sum_purities += purity

    # Concentratable entanglement formula
    C = 1 - sum_purities / (2 ** n_qubits)

    return C


def compute_subset_purity(psi: np.ndarray, n_qubits: int, subset: Tuple[int]) -> float:
    """
    Compute purity Tr(rho^2) for a subset of qubits.

    Args:
        psi: State tensor of shape (2, 2, ..., 2) with n_qubits dimensions
        n_qubits: Total number of qubits
        subset: Tuple of qubit indices to keep

    Returns:
        purity: Tr(rho_subset^2)
    """
    # Get qubits to trace out (complement of subset)
    qubits_to_trace = [i for i in range(n_qubits) if i not in subset]

    # Reshape to separate kept and traced qubits
    # Move subset qubits to the front
    axes_order = list(subset) + qubits_to_trace
    psi_reordered = np.moveaxis(psi, axes_order, range(n_qubits))

    # Reshape: (2^|subset|, 2^|complement|)
    subset_dim = 2 ** len(subset)
    complement_dim = 2 ** len(qubits_to_trace)
    psi_matrix = psi_reordered.reshape(subset_dim, complement_dim)

    # Compute reduced density matrix: rho = Tr_complement(|psi><psi|)
    rho = psi_matrix @ psi_matrix.conj().T

    # Compute purity
    purity = np.trace(rho @ rho).real

    return purity


def entangling_capability(circuit_function, n_qubits: int, params_list: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute entangling capability using Meyer-Wallach measure.

    Averages over multiple random parameter samples following Sim et al. (2019).

    Args:
        circuit_function: Function that builds the quantum circuit
        n_qubits: Number of qubits
        params_list: List of parameter arrays to average over

    Returns:
        dict with 'mean', 'std', 'min', 'max' of Meyer-Wallach Q values
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        circuit_function(params)
        return qml.state()

    q_values = []

    for params in tqdm(params_list, desc="Computing entangling capability"):
        state = circuit(params)
        Q = meyer_wallach_measure(state, n_qubits)
        q_values.append(Q)

    return {
        'mean': np.mean(q_values),
        'std': np.std(q_values),
        'min': np.min(q_values),
        'max': np.max(q_values),
        'all_values': q_values
    }


def concentratable_ent_measure(circuit_function, n_qubits: int, params_list: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute concentratable entanglement measure.

    Averages over multiple random parameter samples.

    Args:
        circuit_function: Function that builds the quantum circuit
        n_qubits: Number of qubits
        params_list: List of parameter arrays to average over

    Returns:
        dict with 'mean', 'std', 'min', 'max' of concentratable entanglement values
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        circuit_function(params)
        return qml.state()

    c_values = []

    for params in tqdm(params_list, desc="Computing concentratable entanglement"):
        state = circuit(params)
        C = concentratable_entanglement(state, n_qubits)
        c_values.append(C)

    return {
        'mean': np.mean(c_values),
        'std': np.std(c_values),
        'min': np.min(c_values),
        'max': np.max(c_values),
        'all_values': c_values
    }


def apply_qcnn_convolution(weights, wires):
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


def apply_dilated_convolution(weights, wires):
    """Non-adjacent entanglement pattern for global connectivity."""
    n_wires = len(wires)

    # Define dilated entanglement pairs based on number of wires
    if n_wires == 8:
        entanglement_pairs = [(0, 2), (1, 3), (4, 6), (5, 7)]
    elif n_wires == 6:
        entanglement_pairs = [(0, 2), (1, 3), (2, 4), (3, 5)]
    elif n_wires == 4:
        entanglement_pairs = [(0, 2), (1, 3)]
    elif n_wires == 2:
        entanglement_pairs = [(0, 1)]
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


def build_qcnn_circuit(n_qubits: int, n_layers: int):
    """Build circuit function for QCNN model."""
    def circuit(params):
        # Unpack parameters
        conv_params = params['conv']
        pool_params = params['pool']
        last_params = params['last']
        features = params['features']

        wires = list(range(n_qubits))

        # Variational Embedding
        qml.AngleEmbedding(features, wires=wires, rotation='Y')

        for layer in range(n_layers):
            # Convolutional Layer
            apply_qcnn_convolution(conv_params[layer], wires)
            # Pooling Layer (simplified - just apply gates, no measurement)
            # Note: For entanglement measurement, we skip mid-circuit measurements
            wires = wires[::2]

        # Final unitary on remaining wires
        if len(wires) > 0:
            # Only apply on remaining active wires
            qml.ArbitraryUnitary(last_params[:2**(2*len(wires))-1], wires=wires[:min(len(wires), len(last_params))])

    return circuit


def build_dilated_qcnn_circuit(n_qubits: int, n_layers: int):
    """Build circuit function for QuantumDilatedCNN model."""
    def circuit(params):
        # Unpack parameters
        conv_params = params['conv']
        pool_params = params['pool']
        last_params = params['last']
        features = params['features']

        wires = list(range(n_qubits))

        # Variational Embedding
        qml.AngleEmbedding(features, wires=wires, rotation='Y')

        for layer in range(n_layers):
            # Dilated Convolutional Layer
            apply_dilated_convolution(conv_params[layer], wires)
            # Pooling Layer (simplified - skip mid-circuit measurements)
            wires = wires[::2]

        # Final unitary on remaining wires
        if len(wires) > 0:
            qml.ArbitraryUnitary(last_params[:2**(2*len(wires))-1], wires=wires[:min(len(wires), len(last_params))])

    return circuit


def generate_random_params(n_qubits: int, n_layers: int, n_samples: int) -> List[Dict]:
    """Generate random parameters for sampling circuit configurations."""
    params_list = []

    for _ in range(n_samples):
        # Random features (input data)
        features = np.random.uniform(-np.pi, np.pi, n_qubits)

        # Random circuit parameters
        conv_params = np.random.uniform(0, 2 * np.pi,
                                       (n_layers, n_qubits, 15))
        pool_params = np.random.uniform(0, 2 * np.pi,
                                       (n_layers, n_qubits // 2, 3))
        last_params = np.random.uniform(0, 2 * np.pi, 15)

        params_list.append({
            'features': features,
            'conv': conv_params,
            'pool': pool_params,
            'last': last_params
        })

    return params_list


def main(args):
    set_all_seeds(args.seed)

    print("="*70)
    print("Quantum Entanglement Measurement")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Number of qubits: {args.n_qubits}")
    print(f"  Number of layers: {args.n_layers}")
    print(f"  Number of samples: {args.n_samples}")
    print(f"  Random seed: {args.seed}")
    print()

    # Generate random parameter samples
    print(f"\nGenerating {args.n_samples} random parameter samples...")
    params_samples = generate_random_params(args.n_qubits, args.n_layers, args.n_samples)

    # Results dictionary
    results = {}

    # ========== QCNN Measurements ==========
    print("\n" + "="*70)
    print("Measuring QCNN Entanglement")
    print("="*70)

    qcnn_circuit = build_qcnn_circuit(args.n_qubits, args.n_layers)

    print("\n1. Computing Meyer-Wallach Measure (Entangling Capability)...")
    qcnn_mw = entangling_capability(qcnn_circuit, args.n_qubits, params_samples)
    print(f"   QCNN Meyer-Wallach Q: {qcnn_mw['mean']:.6f} ± {qcnn_mw['std']:.6f}")
    print(f"   Range: [{qcnn_mw['min']:.6f}, {qcnn_mw['max']:.6f}]")

    results['QCNN'] = {'meyer_wallach': qcnn_mw}

    # Concentratable entanglement (computationally intensive for large n_qubits)
    if args.n_qubits <= 6:
        print("\n2. Computing Concentratable Entanglement...")
        qcnn_ce = concentratable_ent_measure(qcnn_circuit, args.n_qubits, params_samples)
        print(f"   QCNN Concentratable Entanglement: {qcnn_ce['mean']:.6f} ± {qcnn_ce['std']:.6f}")
        print(f"   Range: [{qcnn_ce['min']:.6f}, {qcnn_ce['max']:.6f}]")
        results['QCNN']['concentratable'] = qcnn_ce
    else:
        print("\n2. Skipping Concentratable Entanglement (too many qubits, computationally expensive)")
        print(f"   Note: Concentratable entanglement requires 2^{args.n_qubits} = {2**args.n_qubits} subset calculations")

    # ========== QuantumDilatedCNN Measurements ==========
    print("\n" + "="*70)
    print("Measuring QuantumDilatedCNN Entanglement")
    print("="*70)

    dilated_circuit = build_dilated_qcnn_circuit(args.n_qubits, args.n_layers)

    print("\n1. Computing Meyer-Wallach Measure (Entangling Capability)...")
    dilated_mw = entangling_capability(dilated_circuit, args.n_qubits, params_samples)
    print(f"   QuantumDilatedCNN Meyer-Wallach Q: {dilated_mw['mean']:.6f} ± {dilated_mw['std']:.6f}")
    print(f"   Range: [{dilated_mw['min']:.6f}, {dilated_mw['max']:.6f}]")

    results['QuantumDilatedCNN'] = {'meyer_wallach': dilated_mw}

    if args.n_qubits <= 6:
        print("\n2. Computing Concentratable Entanglement...")
        dilated_ce = concentratable_ent_measure(dilated_circuit, args.n_qubits, params_samples)
        print(f"   QuantumDilatedCNN Concentratable Entanglement: {dilated_ce['mean']:.6f} ± {dilated_ce['std']:.6f}")
        print(f"   Range: [{dilated_ce['min']:.6f}, {dilated_ce['max']:.6f}]")
        results['QuantumDilatedCNN']['concentratable'] = dilated_ce
    else:
        print("\n2. Skipping Concentratable Entanglement (too many qubits, computationally expensive)")

    # ========== Summary ==========
    print("\n" + "="*70)
    print("SUMMARY OF ENTANGLEMENT MEASUREMENTS")
    print("="*70)

    summary_data = {
        'Model': ['QCNN', 'QuantumDilatedCNN'],
        'Meyer-Wallach (Mean)': [
            results['QCNN']['meyer_wallach']['mean'],
            results['QuantumDilatedCNN']['meyer_wallach']['mean']
        ],
        'Meyer-Wallach (Std)': [
            results['QCNN']['meyer_wallach']['std'],
            results['QuantumDilatedCNN']['meyer_wallach']['std']
        ]
    }

    if args.n_qubits <= 6:
        summary_data['Concentratable (Mean)'] = [
            results['QCNN']['concentratable']['mean'],
            results['QuantumDilatedCNN']['concentratable']['mean']
        ]
        summary_data['Concentratable (Std)'] = [
            results['QCNN']['concentratable']['std'],
            results['QuantumDilatedCNN']['concentratable']['std']
        ]

    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary_df.to_csv(output_dir / f"entanglement_summary_{args.n_qubits}q_{args.n_layers}l.csv", index=False)

    # Save detailed results
    for model_name, model_results in results.items():
        for measure_name, measure_data in model_results.items():
            values = np.array(measure_data['all_values'])
            np.save(
                output_dir / f"{model_name}_{measure_name}_{args.n_qubits}q_{args.n_layers}l.npy",
                values
            )

    print(f"\nResults saved to: {output_dir}")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("\nMeyer-Wallach Measure (Entangling Capability):")
    print("  - Range: [0, 1]")
    print("  - 0 = product state (no entanglement)")
    print("  - 1 = maximally entangled state")
    print("  - Higher values indicate better entangling capability")

    if args.n_qubits <= 6:
        print("\nConcentratable Entanglement:")
        print("  - Range: [0, 1]")
        print("  - Measures entanglement that can be concentrated into Bell pairs")
        print("  - Higher values indicate more concentratable entanglement")

    # Compare models
    mw_diff = results['QuantumDilatedCNN']['meyer_wallach']['mean'] - results['QCNN']['meyer_wallach']['mean']
    print(f"\nDifference (QuantumDilatedCNN - QCNN):")
    print(f"  Meyer-Wallach: {mw_diff:+.6f}")

    if mw_diff > 0:
        print("  → QuantumDilatedCNN has HIGHER entangling capability")
    elif mw_diff < 0:
        print("  → QCNN has HIGHER entangling capability")
    else:
        print("  → Both models have SIMILAR entangling capability")

    if args.n_qubits <= 6:
        ce_diff = results['QuantumDilatedCNN']['concentratable']['mean'] - results['QCNN']['concentratable']['mean']
        print(f"  Concentratable Ent: {ce_diff:+.6f}")

        if ce_diff > 0:
            print("  → QuantumDilatedCNN has MORE concentratable entanglement")
        elif ce_diff < 0:
            print("  → QCNN has MORE concentratable entanglement")
        else:
            print("  → Both models have SIMILAR concentratable entanglement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure entanglement in QCNN and QuantumDilatedCNN circuits"
    )

    # Circuit parameters
    parser.add_argument('--n-qubits', type=int, default=6,
                       help='Number of qubits (default: 6, max 8 for concentratable entanglement)')
    parser.add_argument('--n-layers', type=int, default=2,
                       help='Number of layers (default: 2)')

    # Sampling parameters
    parser.add_argument('--n-samples', type=int, default=100,
                       help='Number of random parameter samples (default: 100)')
    parser.add_argument('--seed', type=int, default=2025,
                       help='Random seed (default: 2025)')

    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./entanglement_results',
                       help='Output directory for results')

    args = parser.parse_args()

    main(args)
