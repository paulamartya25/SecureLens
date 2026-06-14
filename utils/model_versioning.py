"""
utils/model_versioning.py
SecureLens — Model Version Manager
Tracks and manages different model versions.
"""

import os, json
from datetime import datetime

MODELS_DIR   = os.path.join(
    os.path.dirname(__file__), "..", "cloud_server", "models")
VERSION_FILE = os.path.join(MODELS_DIR, "model_versions.json")


def register_model(
    version: str,
    accuracy: float,
    notes: str = "",
):
    """Registers a trained model version."""
    versions = _load_versions()
    versions[version] = {
        "version"     : version,
        "accuracy"    : accuracy,
        "registered"  : datetime.utcnow().isoformat() + "Z",
        "notes"       : notes,
        "files"       : [
            "best_model.pth",
            "feature_weights.json",
            "linear_weights.json",
            "first_weights.json",
        ],
        "ckks_params" : {
            "poly_modulus_degree": 8192,
            "global_scale"       : "2^40",
            "security_bits"      : 128,
        }
    }
    _save_versions(versions)
    print(f"[Versioning] Registered model v{version} "
          f"(accuracy={accuracy}%)")
    return versions[version]


def get_current_version():
    """Returns the current active model version."""
    versions = _load_versions()
    if not versions:
        return None
    latest = sorted(versions.keys())[-1]
    return versions[latest]


def list_versions():
    """Lists all registered model versions."""
    return _load_versions()


def _load_versions():
    if not os.path.exists(VERSION_FILE):
        return {}
    with open(VERSION_FILE) as f:
        return json.load(f)


def _save_versions(versions):
    with open(VERSION_FILE, "w") as f:
        json.dump(versions, f, indent=2)


if __name__ == "__main__":
    register_model(
        version="1.0.0",
        accuracy=89.42,
        notes="ResNet-18 transfer learning, 20 epochs, "
              "128x128 input, CKKS 128-bit"
    )
    v = get_current_version()
    print(f"\nCurrent model: v{v['version']} "
          f"({v['accuracy']}% accuracy)")
    print("✅ Model versioning working.")