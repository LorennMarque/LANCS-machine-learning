#######################################################
#
# DISCLAIMER
# This file was made with AI to test debug the models
# during development.
#
#######################################################

"""Quick smoke test: train the MLP on a tiny synthetic dataset.

Run from project root:
    python test_mlp.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make project root importable when running: python test_mlp.py
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.loss_functions import mse_loss  # noqa: E402


def load_mlp_module():
    # File is named 00_mlp_baseline.py — not a valid package module name
    path = ROOT / "models" / "00_mlp_baseline.py"
    spec = importlib.util.spec_from_file_location("mlp_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MLP = load_mlp_module().MLP


def make_sample_data(n_samples: int = 40, seed: int = 0):
    """2 features, binary labels — linearly separable-ish blob data."""
    rng = np.random.default_rng(seed)

    # Class 0 around (-1, -1), class 1 around (1, 1)
    x0 = rng.normal(loc=-1.0, scale=0.4, size=(n_samples // 2, 2))
    x1 = rng.normal(loc=1.0, scale=0.4, size=(n_samples // 2, 2))

    x = np.vstack([x0, x1])
    y = pd.Series([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Shuffle
    idx = rng.permutation(n_samples)
    return x[idx], y.iloc[idx].reset_index(drop=True)


def main():
    x, y = make_sample_data()
    print(f"X shape: {x.shape}")
    print(f"Y values: {sorted(y.unique().tolist())}")
    print(f"First 5 rows:\n{x[:5]}")
    print(f"First 5 labels: {y.head().tolist()}\n")

    # Single hidden layer with 4 neurons (matches current MLP note)
    model = MLP(loss_function=mse_loss, hidden_layer_nodes=[3])

    print("Starting training...\n")
    model.train(x, y, learning_rate=0.5, max_iter=5)
    print("\nTraining call finished.")


if __name__ == "__main__":
    main()
