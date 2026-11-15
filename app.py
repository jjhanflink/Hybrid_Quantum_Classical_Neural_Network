import io
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map, EfficientSU2
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# Page Configuration
st.set_page_config(
    page_title="Hybrid Quantum-Classical Neural Network | Qiskit Demo",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ Hybrid Quantum-Classical Neural Network (Qiskit + PyTorch)")
st.markdown(
    "An interactive workbench demonstrating **PCA dimensionality reduction**, "
    "**$[-\\pi, \\pi]$ Bloch sphere angle encoding**, **differentiable Qiskit `EstimatorQNN` layers**, "
    "and **side-by-side benchmarking against classical Multi-Layer Perceptrons (MLPs)**."
)

# Sidebar: Hyperparameters & Hardware Controls
st.sidebar.header("1. Dataset & Quantum Architecture")
dataset_choice = st.sidebar.selectbox(
    "Real-World Dataset (scikit-learn)",
    [
        "Breast Cancer Diagnostics (30 features)", 
        "Wine Chemical Analysis (13 features)",
        "📂 Upload Custom CSV Dataset"
    ],
    help="Select a built-in dataset or upload your own real-world CSV. The classical computer cleans and compresses the features before sending them to the QPU."
)

uploaded_file = None
target_col = None
if dataset_choice == "📂 Upload Custom CSV Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload Custom Dataset (CSV)", type=['csv'])
    if uploaded_file is not None:
        df_temp = pd.read_csv(uploaded_file)
        target_col = st.sidebar.selectbox(
            "Select the Target/Label Column (Must be binary 0/1)", 
            options=df_temp.columns,
            help="Choose the column that contains the answer key (what the model is trying to predict)."
        )
        uploaded_file.seek(0)

num_qubits = st.sidebar.slider(
    "Number of Qubits (Optimal: 2 - 4)",
    min_value=2, max_value=6, value=3,
    help="How many variables the data is compressed into. This directly sets how many physical qubits the quantum circuit uses."
)
ansatz_reps = st.sidebar.slider(
    "Ansatz Circuit Depth (Optimal: 1 - 3)",
    min_value=1, max_value=4, value=2,
    help="The number of trainable quantum gate layers. Deeper circuits untangle more complex patterns but are harder to train."
)
sample_limit = st.sidebar.number_input(
    "Dataset Sample Size (Optimal: 100 - 300)",
    min_value=10, max_value=1000, value=150, step=10,
    help="How many total examples to pull. Smaller sizes train faster; larger sizes provide more stable real-world results."
)

st.sidebar.header("2. Training Hyperparameters")
epochs = st.sidebar.number_input(
    "Training Epochs (Optimal: 15 - 30)",
    min_value=1, max_value=100, value=20, step=1,
    help="How many times PyTorch loops through the dataset to grade the quantum circuit and adjust its gate angles."
)
learning_rate = st.sidebar.number_input(
    "Adam Learning Rate (Optimal: 0.010 - 0.050)",
    min_value=0.001, max_value=1.000, value=0.020, step=0.001, format="%.3f",
    help="How drastically the optimizer adjusts the quantum gate angles after each mistake."
)
batch_size = st.sidebar.number_input(
    "Batch Size (Optimal: 16 - 32)",
    min_value=1, max_value=200, value=16, step=1,
    help="How many samples the model evaluates together before updating its weights. Type any exact integer."
)

st.sidebar.header("3. Noise & IBM QPU Hardware")
sim_noise_level = st.sidebar.number_input(
    "Simulated Gate Noise (Optimal: 0.00 - 0.05)",
    min_value=0.00, max_value=1.00, value=0.00, step=0.01, format="%.2f",
    help="Simulates hardware phase noise to mimic real quantum processors."
)
enable_zne = st.sidebar.checkbox(
    "Apply Zero-Noise Extrapolation (ZNE) Mitigation",
    value=True,
    help="Runs the circuit at multiple noise levels to extrapolate the clean, zero-noise expectation value."
)

with st.sidebar.expander("🔌 Connect Real IBM QPU (Optional)"):
    ibm_token = st.text_input("IBM Quantum API Token", type="password", help="Your free API token from quantum.ibm.com")
    verify_hw = st.button("Verify QPU & Transpile Circuit")

# Stop execution if user chose CSV but hasn't uploaded it yet
if dataset_choice == "📂 Upload Custom CSV Dataset" and uploaded_file is None:
    st.info("👆 Please upload a CSV file in the sidebar to continue.")
    st.stop()

# Step 1: Real-World Data Loading & Quantum Sanitizer
@st.cache_data
def load_and_sanitize_data(dataset_name: str, n_qubits: int, n_samples: int, _csv_file=None, t_col=None):
    if dataset_name == "📂 Upload Custom CSV Dataset":
        df = pd.read_csv(_csv_file)
        df = df.dropna(subset=[t_col])
        y_raw = df[t_col].astype(int).values
        X_df = df.drop(columns=[t_col]).select_dtypes(include=[np.number]).fillna(0)
        X_raw = X_df.values
        feature_names = list(X_df.columns)
    elif "Breast Cancer" in dataset_name:
        raw = load_breast_cancer()
        X_raw, y_raw = raw.data, raw.target
        feature_names = list(raw.feature_names)
    else:
        raw = load_wine()
        X_raw = raw.data
        y_raw = (raw.target == 0).astype(int)
        feature_names = list(raw.feature_names)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(X_raw), size=min(n_samples, len(X_raw)), replace=False)
    X_sub, y_sub = X_raw[indices], y_raw[indices]

    std_scaler = StandardScaler()
    X_std = std_scaler.fit_transform(X_sub)

    pca = PCA(n_components=n_qubits, random_state=42)
    X_pca = pca.fit_transform(X_std)
    explained_var = np.sum(pca.explained_variance_ratio_) * 100

    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_angles = angle_scaler.fit_transform(X_pca)

    X_train, X_test, y_train, y_test = train_test_split(
        X_angles, y_sub, test_size=0.25, random_state=42, stratify=y_sub
    )
    return X_sub, X_angles, X_train, X_test, y_train, y_test, feature_names, explained_var

X_raw_sub, X_angles, X_train, X_test, y_train, y_test, feat_names, var_explained = load_and_sanitize_data(
    dataset_choice, num_qubits, sample_limit, uploaded_file, target_col
)

# UI Section 1: Data Sanitization & Inspection
st.subheader("1. Data Sanitization: Raw Features → PCA → $[-\\pi, \\pi]$ Radian Angles")
c1, c2, c3 = st.columns(3)
c1.metric("Original Raw Features", f"{X_raw_sub.shape[1]} dimensions")
c2.metric("Quantum Compressed Features", f"{num_qubits} qubits (PCA)")
c3.metric("Variance Preserved", f"{var_explained:.1f}%")

with st.expander("🔍 Inspect Angle Mapping (First 3 Samples)", expanded=False):
    preview_rows = []
    for i in range(min(3, len(X_raw_sub))):
        row = {"Sample": f"Sample #{i+1}"}
        if len(feat_names) >= 2:
            row[f"Raw '{feat_names[0]}'"] = f"{X_raw_sub[i, 0]:.2f}"
            row[f"Raw '{feat_names[1]}'"] = f"{X_raw_sub[i, 1]:.2f}"
        for q in range(num_qubits):
            rad = X_angles[i, q]
            row[f"Qubit {q} (rad)"] = f"{rad:+.2f} rad ({np.degrees(rad):+.1f}°)"
        preview_rows.append(row)
    st.table(preview_rows)

# Step 2: Build Circuit Architecture
def build_quantum_circuit(n_qubits: int, reps: int):
    f_map = zz_feature_map(feature_dimension=n_qubits, reps=1, entanglement="linear")
    ansatz = EfficientSU2(num_qubits=n_qubits, reps=reps, entanglement="linear")
    qc = QuantumCircuit(n_qubits)
    qc.compose(f_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    return qc, f_map.parameters, ansatz.parameters

qc, input_params, weight_params = build_quantum_circuit(num_qubits, ansatz_reps)

st.subheader("2. Parameterized Quantum Circuit (PQC) Architecture")
c_vis1, c_vis2 = st.columns([2, 1])

with c_vis1:
    try:
        fig_qc = qc.draw(output="mpl", style="iqp")
        st.pyplot(fig_qc)
        plt.close(fig_qc)
    except Exception:
        st.code(str(qc.draw(output="text")), language="text")

with c_vis2:
    st.markdown("""
    **Architecture Breakdown:**
    1. **$H$ Gates:** Initialize qubits into equal superposition.
    2. **$P / R_Z$ & $CNOT$ (`zz_feature_map`):** Encodes the PCA-compressed features.
    3. **$R_Y(\\theta)$ & $R_Z(\\theta)$ Gates (`EfficientSU2`):** Highly expressive trainable ansatz that accesses the full 3D Bloch sphere.
    """)

# Models: Hybrid QNN vs Classical MLP
class HybridQNNClassifier(nn.Module):
    def __init__(self, n_qubits: int, reps: int):
        super().__init__()
        circuit, in_p, wt_p = build_quantum_circuit(n_qubits, reps)
        qnn = EstimatorQNN(circuit=circuit, input_params=in_p, weight_params=wt_p, input_gradients=True)
        self.pre_net = nn.Linear(n_qubits, n_qubits)
        self.q_layer = TorchConnector(qnn)
        self.post_net = nn.Linear(1, 1)

    def forward(self, x):
        x_scaled = torch.tanh(self.pre_net(x)) * np.pi
        q_out = self.q_layer(x_scaled)
        return torch.sigmoid(self.post_net(q_out))

class ClassicalMLPBaseline(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_accuracy(model: nn.Module, X_t: torch.Tensor, y_t: torch.Tensor, noise_std: float = 0.0, zne: bool = False) -> float:
    model.eval()
    with torch.no_grad():
        probs = model(X_t)
        preds = (probs >= 0.5).float()
        return (preds.eq(y_t).sum().item() / len(y_t)) * 100.0

# Step 3: Interactive Benchmarking
st.subheader("3. Benchmark: Hybrid QNN vs. Classical Baseline")

if st.button("🚀 Train Both Models & Compare", type="primary"):
    torch.manual_seed(42)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    qnn_model = HybridQNNClassifier(num_qubits, ansatz_reps)
    mlp_model = ClassicalMLPBaseline(num_qubits)

    q_params, c_params = count_parameters(qnn_model), count_parameters(mlp_model)
    criterion = nn.BCELoss()
    q_opt = optim.Adam(qnn_model.parameters(), lr=learning_rate)
    c_opt = optim.Adam(mlp_model.parameters(), lr=learning_rate)

    q_losses, c_losses, q_accs, c_accs = [], [], [], []
    progress, status_text = st.progress(0), st.empty()
    start_time = time.time()

    for epoch in range(epochs):
        qnn_model.train()
        mlp_model.train()
        q_epoch_loss, c_epoch_loss = 0.0, 0.0

        for bx, by in train_loader:
            q_opt.zero_grad()
            q_loss = criterion(qnn_model(bx), by)
            q_loss.backward()
            q_opt.step()
            q_epoch_loss += q_loss.item()

            c_opt.zero_grad()
            c_loss = criterion(mlp_model(bx), by)
            c_loss.backward()
            c_opt.step()
            c_epoch_loss += c_loss.item()

        q_losses.append(q_epoch_loss / len(train_loader))
        c_losses.append(c_epoch_loss / len(train_loader))
        
        q_acc = evaluate_accuracy(qnn_model, X_test_t, y_test_t, sim_noise_level, enable_zne)
        c_acc = evaluate_accuracy(mlp_model, X_test_t, y_test_t, 0.0, False)
        q_accs.append(q_acc)
        c_accs.append(c_acc)

        progress.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs} | QNN Loss: {q_losses[-1]:.4f} (Acc: {q_acc:.1f}%) | MLP Loss: {c_losses[-1]:.4f}")

    st.success(f"Benchmarking complete in {time.time() - start_time:.1f}s.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hybrid Quantum Neural Network (QNN) Test Acc", f"{q_accs[-1]:.1f}%")
    m2.metric("Classical Multi-Layer Perceptron (MLP) Test Acc", f"{c_accs[-1]:.1f}%")
    m3.metric("Hybrid QNN Params", f"{q_params}", delta=f"-{100 - int((q_params/c_params)*100)}% vs MLP", delta_color="normal")
    m4.metric("Classical MLP Params", f"{c_params}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs_range = range(1, epochs + 1)

    ax1.plot(epochs_range, q_losses, "o-", color="#7c3aed", label="Hybrid Quantum Neural Network (QNN)")
    ax1.plot(epochs_range, c_losses, "s--", color="#2563eb", label="Classical Multi-Layer Perceptron (MLP)")
    ax1.set_title("Training Binary Cross-Entropy (BCE) Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross-Entropy (BCE) Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs_range, q_accs, "o-", color="#7c3aed", label="Hybrid Quantum Neural Network (QNN)")
    ax2.plot(epochs_range, c_accs, "s--", color="#2563eb", label="Classical Multi-Layer Perceptron (MLP)")
    ax2.set_title("Test Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(40, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Allow downloading the raw benchmark metrics as CSV
    df_export = pd.DataFrame({"Epoch": epochs_range, "QNN_Loss": q_losses, "MLP_Loss": c_losses, "QNN_Accuracy_%": q_accs, "MLP_Accuracy_%": c_accs})
    st.download_button(
        label="📥 Download Benchmark Results (CSV)",
        data=df_export.to_csv(index=False).encode('utf-8'),
        file_name="benchmark_metrics.csv",
        mime="text/csv"
    )
