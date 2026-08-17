from benchmarks.classical_baselines import ClassicalMLPBaseline
from core.circuits import HybridQNN
from core.sanitizer import LocalDataSanitizer
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim


def run_benchmark():
    print("=" * 60)
    print("Hybrid Quantum-Classical vs. Classical Baseline Benchmark")
    print("=" * 60)

    # 1. Generate synthetic tabular dataset (e.g., fraud / risk classification)
    X_raw, y_raw = make_classification(
        n_samples=400,
        n_features=12,
        n_informative=6,
        n_classes=2,
        random_state=42,
    )

    # 2. Local Privacy Pre-processing & Sanitization
    n_qubits = 4
    sanitizer = LocalDataSanitizer(target_qubits=n_qubits)
    X_sanitized = sanitizer.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sanitized, y_raw, test_size=0.25, random_state=42
    )

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # 3. Initialize Models
    hybrid_model = HybridQNN(n_qubits=n_qubits, n_layers=2, num_classes=2)
    classical_model = ClassicalMLPBaseline(
        input_dim=n_qubits, hidden_dim=16, num_classes=2
    )

    criterion = nn.CrossEntropyLoss()
    hybrid_opt = optim.Adam(hybrid_model.parameters(), lr=0.05)
    classical_opt = optim.Adam(classical_model.parameters(), lr=0.05)

    epochs = 15
    print(f"Training both models across {epochs} epochs...")

    for epoch in range(epochs):
        # Train Hybrid QNN
        hybrid_opt.zero_grad()
        out_q = hybrid_model(X_train_t)
        loss_q = criterion(out_q, y_train_t)
        loss_q.backward()
        hybrid_opt.step()

        # Train Classical MLP
        classical_opt.zero_grad()
        out_c = classical_model(X_train_t)
        loss_c = criterion(out_c, y_train_t)
        loss_c.backward()
        classical_opt.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] | Hybrid Loss: {loss_q.item():.4f} | Classical Loss: {loss_c.item():.4f}"
            )

    # 4. Evaluation
    hybrid_model.eval()
    classical_model.eval()

    with torch.no_grad():
        preds_q = torch.argmax(hybrid_model(X_test_t), dim=1).numpy()
        preds_c = torch.argmax(classical_model(X_test_t), dim=1).numpy()

    q_acc = accuracy_score(y_test, preds_q)
    c_acc = accuracy_score(y_test, preds_c)
    q_f1 = f1_score(y_test, preds_q)
    c_f1 = f1_score(y_test, preds_c)

    q_params = sum(p.numel() for p in hybrid_model.parameters())
    c_params = sum(p.numel() for p in classical_model.parameters())

    # 5. Output Summary
    print("\n" + "-" * 60)
    print(f"{'Metric':<25} | {'Hybrid QNN':<15} | {'Classical MLP':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<25} | {q_acc * 100:>13.2f}% | {c_acc * 100:>13.2f}%")
    print(f"{'F1 Score':<25} | {q_f1:>15.4f} | {c_f1:>15.4f}")
    print(f"{'Trainable Parameters':<25} | {q_params:>15} | {c_params:>15}")
    print("-" * 60)


if __name__ == "__main__":
    run_benchmark()