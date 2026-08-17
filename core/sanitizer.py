import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class LocalDataSanitizer:
    """Sanitizes raw tabular datasets locally by removing proprietary feature names

    and mapping high-dimensional data down to normalized quantum rotation
    angles.
    """

    def __init__(self, target_qubits: int = 4):
        self.target_qubits = target_qubits
        self.pca = PCA(n_components=target_qubits)
        self.scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))

    def fit_transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include=[np.number]).values

        # 1. Dimensionality reduction to match qubit register count
        if X.shape[1] > self.target_qubits:
            X_reduced = self.pca.fit_transform(X)
        else:
            X_reduced = X

        # 2. Scale features to bounded rotation angles [-pi, pi]
        sanitized_angles = self.scaler.fit_transform(X_reduced)
        return sanitized_angles.astype(np.float32)

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include=[np.number]).values
        if X.shape[1] > self.target_qubits:
            X_reduced = self.pca.transform(X)
        else:
            X_reduced = X
        return self.scaler.transform(X_reduced).astype(np.float32)
