#!/usr/bin/env python
"""Test script: import torchgeo from local path and try to import OlmoEarth model."""

import sys
from pathlib import Path

# Add local torchgeo package to path (parent of script = repo root)
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def main() -> None:
    print("Testing OlmoEarth import from local torchgeo...")
    print(f"Python: {sys.executable}")
    print(f"sys.path[0]: {sys.path[0]}\n")

    try:
        from torchgeo.models.olmoearth_pretrain_v1 import (
            OlmoEarthPretrain_v1,
            OlmoEarthPretrainV1_Weights,
            olmoearth_pretrain_v1,
            Normalizer,
            Modality,
        )
        print("Import OK: OlmoEarthPretrain_v1, Weights, builder, Normalizer, Modality")
    except Exception as e:
        print(f"Import failed: {e}")
        raise

    try:
        model = OlmoEarthPretrain_v1(model_size="nano")
        print("Build OK: OlmoEarthPretrain_v1(model_size='nano')")
    except Exception as e:
        print(f"Build failed: {e}")
        raise

    try:
        normalizer = Normalizer()
        print("Normalizer OK")
    except Exception as e:
        print(f"Normalizer failed: {e}")
        raise

    try:
        from torchgeo.models import get_model, get_model_weights, list_models
        if "olmoearth_pretrain_v1" in list_models():
            print("list_models() OK: olmoearth_pretrain_v1 is registered")
        weights_cls = get_model_weights(olmoearth_pretrain_v1)
        print(f"get_model_weights OK: {weights_cls.__name__}")
    except Exception as e:
        print(f"API check failed: {e}")
        raise

    print("\nAll OlmoEarth checks passed.")

if __name__ == "__main__":
    main()
