from mitiq.zne.inference import LinearFactory, RichardsonFactory
from mitiq.zne.scaling import fold_global
import pennylane as qml


class QuantumNoiseMitigator:
    """Applies Zero-Noise Extrapolation (ZNE) over noisy quantum nodes."""

    def __init__(self, scale_factors: list[int] = None):
        if scale_factors is None:
            scale_factors = [1, 3, 5]
        self.scale_factors = scale_factors

    def wrap_noisy_qnode(self, qnode, factory_type: str = "richardson"):
        extrapolate_fn = (
            RichardsonFactory.extrapolate
            if factory_type == "richardson"
            else LinearFactory.extrapolate
        )

        mitigated_qnode = qml.transforms.mitigate_with_zne(
            scale_factors=self.scale_factors,
            folding=fold_global,
            extrapolate=extrapolate_fn,
        )(qnode)

        return mitigated_qnode