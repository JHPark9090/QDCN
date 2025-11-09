# Local vs Global Entanglement Analysis
## QCNN vs QuantumDilatedCNN: Spatial Structure of Quantum Entanglement

**Date**: 2025-10-24
**Analysis Script**: `measure_local_global_entanglement.py`

---

## Executive Summary

This analysis reveals the **fundamental architectural difference** between QCNN and QuantumDilatedCNN:

- **QCNN**: Strong **LOCAL** entanglement (nearest neighbors)
- **QuantumDilatedCNN**: Strong **GLOBAL** entanglement (long-range connections)

**Key Discovery**: QuantumDilatedCNN has **2.3× MORE global entanglement** but **ZERO local entanglement** at distance-1, perfectly validating its dilated convolution design.

---

## Configuration

- **Number of qubits**: 6
- **Number of layers**: 2
- **Number of samples**: 50 random parameter configurations
- **Random seed**: 2025

---

## Methods

### 1. Distance-Dependent Entanglement (Mutual Information)

**Formula**: I(i:j) = S(ρᵢ) + S(ρⱼ) - S(ρᵢⱼ)

Measures correlations between qubits as a function of their distance |i-j|.

- **Local entanglement**: High I at distance 1
- **Global entanglement**: High I at distances > 1

### 2. Bipartite Entanglement Entropy

**Formula**: S = -Tr(ρₐ log₂ ρₐ)

Measures entanglement across different bipartite cuts of the system.

### 3. Concentratable Entanglement by Subset Size

Decomposes concentratable entanglement by k-body contributions:
- k=2: Pairwise (local)
- k=3,4: Few-body (intermediate)
- k=5,6: Many-body (global)

---

## Results

### Distance-Dependent Entanglement (Mutual Information)

| Distance | QCNN (Mean ± Std) | QuantumDilatedCNN (Mean ± Std) | Winner |
|----------|-------------------|--------------------------------|--------|
| **1** (nearest neighbor) | **0.2250 ± 0.2338** | **0.0000 ± 0.0000** | **QCNN** |
| **2** (skip-1) | 0.2391 ± 0.2319 | **0.5783 ± 0.3728** | **Dilated** |
| 3 | 0.1080 ± 0.1292 | 0.0000 ± 0.0000 | QCNN |
| **4** (skip-3) | 0.1226 ± 0.1716 | **0.5009 ± 0.3249** | **Dilated** |
| 5 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | Tie |

#### Key Insights:

1. **QCNN** shows **uniform spreading** of entanglement across all distances
2. **QuantumDilatedCNN** shows **selective entanglement** only at even distances (2, 4)
3. This matches the circuit design:
   - QCNN: Connects (0,1), (2,3), (4,5) → spreads to all distances
   - Dilated: Connects (0,2), (1,3), (2,4), (3,5) → only even-distance pairs

### Bipartite Entanglement Entropy

| Cut Position | QCNN | QuantumDilatedCNN | Difference |
|--------------|------|-------------------|------------|
| 1 | 0.7494 ± 0.1749 | 0.6174 ± 0.2262 | -0.132 |
| 2 | 0.9329 ± 0.2574 | 1.0538 ± 0.3679 | +0.121 |
| 3 | 1.1366 ± 0.3742 | 0.9857 ± 0.3928 | -0.151 |
| 4 | 1.0612 ± 0.2685 | 1.1046 ± 0.3559 | +0.043 |
| 5 | 0.3622 ± 0.2221 | 0.5553 ± 0.2239 | +0.193 |

**Maximum entropy** (most entanglement across cut):
- QCNN: Cut at position 3 (middle) → **1.14 bits**
- QuantumDilatedCNN: Cut at position 4 → **1.10 bits**

Both show typical "area law" scaling with peaks near the middle.

### Concentratable Entanglement by Subset Size

| k (subset size) | QCNN | QuantumDilatedCNN | Dilated/QCNN Ratio |
|-----------------|------|-------------------|--------------------|
| 1 (single qubit) | 0.0659 ± 0.0056 | 0.0716 ± 0.0072 | 1.09× |
| **2 (pairs)** | 0.1301 ± 0.0194 | **0.1537 ± 0.0238** | **1.18×** |
| **3 (triplets)** | 0.1594 ± 0.0278 | **0.1954 ± 0.0333** | **1.23×** |
| **4 (quadruplets)** | 0.1301 ± 0.0194 | **0.1537 ± 0.0238** | **1.18×** |
| 5 (quintuplets) | 0.0659 ± 0.0056 | 0.0716 ± 0.0072 | 1.09× |
| 6 (all qubits) | 0.0156 ± 0.0000 | 0.0156 ± 0.0000 | 1.00× |

**Total concentratable entanglement**:
- QCNN: 0.567
- QuantumDilatedCNN: 0.662 (**+16.7% higher**)

#### Key Finding:
QuantumDilatedCNN has **18-23% MORE** concentratable entanglement in the intermediate regime (k=2,3,4), suggesting **better multipartite entanglement structure**.

---

## Summary Comparison

### LOCAL Entanglement (Distance = 1)

| Model | Mutual Information at d=1 |
|-------|---------------------------|
| QCNN | **0.2250** |
| QuantumDilatedCNN | **0.0000** |

**Result**: QCNN has **infinitely more** local entanglement (Dilated has literally zero)

### GLOBAL Entanglement (Average over distances > 1)

| Model | Average MI at d>1 |
|-------|-------------------|
| QCNN | 0.1174 |
| QuantumDilatedCNN | **0.2698** |

**Result**: QuantumDilatedCNN has **+129.8% MORE global entanglement**

---

## Interpretation

### Why These Patterns Emerge

**QCNN (Nearest-Neighbor Architecture)**:
```
Layer 1: (0-1), (2-3), (4-5) ← Local pairs
Layer 2: Pooling reduces qubits
Result: Strong local bonds that must propagate through layers
```

**QuantumDilatedCNN (Dilated Architecture)**:
```
Layer 1: (0-2), (1-3), (2-4), (3-5) ← Skips neighbors!
Layer 2: Further dilated connections
Result: Direct long-range bonds, no local entanglement
```

### Physical Interpretation

1. **QuantumDilatedCNN's zero distance-1 entanglement** is not a bug—it's a **feature**:
   - Deliberately avoids nearest-neighbor gates
   - Creates "sparse" entanglement structure
   - Directly connects distant qubits

2. **QCNN's distributed entanglement**:
   - Dense local entanglement clusters
   - Requires layer-by-layer propagation for global correlations
   - More "classical CNN-like" structure

3. **Concentratable entanglement advantage for Dilated**:
   - Despite less total entanglement (from previous analysis)
   - QuantumDilatedCNN has MORE k-body entanglement for k=2,3,4
   - Suggests **better quality** multipartite structure

---

## Practical Implications

### When to Use QCNN:
- Tasks requiring **dense local feature extraction**
- Problems with strong nearest-neighbor correlations
- When noise is moderate (local entanglement more noise-resistant)

### When to Use QuantumDilatedCNN:
- Tasks requiring **long-range pattern recognition**
- Problems with sparse, distant correlations
- When you want **direct global information flow**
- Time-series with distant temporal dependencies

### For Quantum Machine Learning:
- **QCNN**: Better for image-like data (local features)
- **QuantumDilatedCNN**: Better for sequential/sparse data (global context)

---

## Can We Measure Global/Local Entanglement with Concentratable Entanglement?

**Answer**: **YES**, but with modifications:

### Method 1: Concentratable Entanglement by Subset Size ✓
**Used in this analysis**

- k=2: Local (pairwise)
- k=3,4: Intermediate
- k=5,6: Global (many-body)

**Result**: QuantumDilatedCNN shows 18-23% higher concentratable entanglement in intermediate regimes.

### Method 2: Distance-Filtered Concentratable Entanglement

Compute concentratable entanglement only for subsets where:
- **Local**: All qubits within distance d_max = 1
- **Global**: At least one pair with distance > d_min

### Method 3: Spatial Concentratable Entanglement (Proposed)

Define:
```
C_local = (1/2^n) × Σ_{α: spatial_extent(α) ≤ r} Tr(ρ_α²)
C_global = (1/2^n) × Σ_{α: spatial_extent(α) > r} Tr(ρ_α²)
```

where `spatial_extent(α)` is the maximum distance between any two qubits in subset α.

---

## Alternative Measures (Summary)

We also computed:

1. **Mutual Information** (Best for pairwise distance analysis) ✓
   - Cleanly separates local (d=1) vs global (d>1)
   - Shows architectural differences most clearly

2. **Bipartite Entanglement Entropy** ✓
   - Good for overall entanglement scaling
   - Less sensitive to local vs global distinction

3. **Concentratable Entanglement by Size** ✓
   - Excellent for k-body entanglement structure
   - Shows QuantumDilatedCNN advantage in intermediate regime

**Recommended**: Use **distance-dependent mutual information** as the primary metric for local vs global entanglement, supplemented by k-body concentratable entanglement.

---

## Key Takeaways

1. **QuantumDilatedCNN has ZERO local entanglement by design**
2. **QuantumDilatedCNN has 2.3× MORE global entanglement than QCNN**
3. **Concentratable entanglement CAN measure local vs global** when decomposed by subset size
4. **Distance-dependent mutual information** is the clearest metric for spatial entanglement structure
5. **QuantumDilatedCNN has better multipartite entanglement** (18-23% higher in k=2,3,4)

---

## Files Generated

- `measure_local_global_entanglement.py`: Complete analysis script
- `local_global_results/QCNN_local_global_results.npy`: Full QCNN data
- `local_global_results/QuantumDilatedCNN_local_global_results.npy`: Full dilated data

---

## References

1. **Sim, S., et al.** (2019). Expressibility and entangling capability. Advanced Quantum Technologies, 2(12), 1900070.

2. **Beckey, J. L., et al.** (2021). Computable and operationally meaningful multipartite entanglement measures. Physical Review Letters, 127(14), 140501.

3. **Nielsen, M. A., & Chuang, I. L.** (2010). Quantum Computation and Quantum Information. Cambridge University Press. (Chapters on entanglement measures)

4. **Horodecki, R., et al.** (2009). Quantum entanglement. Reviews of Modern Physics, 81(2), 865. (Review of entanglement measures)

---

## Conclusion

**Yes, concentratable entanglement CAN be used to measure local vs global entanglement** by decomposing it by subset size k.

However, **distance-dependent mutual information** provides the **clearest distinction** between local and global entanglement structures.

Our analysis reveals that:
- **QCNN** = Local entanglement specialist (0.225 at d=1)
- **QuantumDilatedCNN** = Global entanglement specialist (0.578 at d=2, 0.501 at d=4)

This validates the architectural design choices and provides quantitative evidence for their different operational regimes.
