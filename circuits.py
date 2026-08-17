import pennylane as qml
import torch
import torch.nn as nn


def create_quantum_layer(n_qubits: int = 4, n_layers: int = 3):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # 1. Angle Embedding for input state preparation
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        # 2. Parameterized Variational Layers
        for layer in range(n_layers):
            # Parameterized Rotations
            for i in range(n_qubits):
                qml.Rot(
                    weights[layer, i, 0],
                    weights[layer, i, 1],
                    weights[layer, i, 2],
                    wires=i,
                )
            # Entangling Ring
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # 3. Local Observables Measurement (Mitigates Barren Plateaus)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class HybridQNN(nn.Module):
    """End-to-end Hybrid Quantum-Classical Neural Network."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.quantum_layer = create_quantum_layer(n_qubits, n_layers)
        self.classical_head = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = self.quantum_layer(x)
        logits = self.classical_head(q_out)
        return logits