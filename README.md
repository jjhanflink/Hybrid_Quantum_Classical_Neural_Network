# Hybrid Quantum-Classical Neural Network with Local Privacy Sanitization

A production-grade benchmark and modular implementation of a Hybrid Quantum-Classical Neural Network (QNN) built with PennyLane, PyTorch, and Mitiq. The pipeline implements local feature sanitization, parameterized variational quantum circuits (VQCs) with local observable measurements to mitigate barren plateaus, and direct head-to-head performance evaluations against classical Multi-Layer Perceptrons (MLPs).

---

## System Architecture

```text
========================= 1. LOCAL DATA PRE-PROCESSING =========================
   [ Raw Tabular Input (N features) ]
                   │
                   ▼
   [ PCA Feature Compression ] ──> Reduces dimension: ℝ^N ──> ℝ^K (K = n_qubits)
                   │
                   ▼
   [ MinMax Angle Scaling ]   ──> Normalizes continuous features to [-π, π]

========================= 2. QUANTUM VARIATIONAL LAYER =========================
   [ State Preparation ]      ──> AngleEmbedding: |0⟩^⊗K ──> |ψ(x)⟩ via R_Y(θ_i)
                   │
                   ▼
   [ Parameterized Layers ]   ──> L-layer ansatz:
                                   • Parameterized Single-Qubit Rotations: Rot(α, β, γ)
                                   • Entangling Ring: Cyclic CNOT lattice
                   │
                   ▼
   [ Observable Measurement ] ──> Local Pauli-Z expectations: ⟨Z_i⟩ for i ∈ [0, K-1]
                                  (Preserves gradient variance / mitigates barren plateaus)

========================= 3. CLASSICAL HEAD & INFERENCE =========================
   [ Classical Feedforward ]  ──> Linear(K ──> 16) ──> ReLU ──> Linear(16 ──> num_classes)
                   │
                   ▼
   [ Optimization ]           ──> CrossEntropyLoss + Adam (End-to-End Backprop)
