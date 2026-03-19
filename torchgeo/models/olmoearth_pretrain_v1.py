# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""OlmoEarth Pretrain v1 model and data preprocessing.

Uses the ``olmoearth-pretrain-minimal`` optional dependency. Install with
``pip install torchgeo[models]`` to load pre-trained weights from Hugging Face
(allenai/OlmoEarth-v1-*).
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision.models._api import Weights, WeightsEnum

from olmoearth_pretrain_minimal import Normalizer, OlmoEarthPretrain_v1
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import (
    Modality,
    ModalitySpec,
)
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
)

# No-op transforms for OlmoEarth; use Normalizer for modality-specific preprocessing.
_olmoearth_transforms = nn.Identity()


class OlmoEarthPretrainV1_Weights(WeightsEnum):  # type: ignore[misc]
    """OlmoEarth v1 pre-trained weights from Hugging Face (allenai/OlmoEarth-v1-*)."""

    NANO = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Nano/resolve/main/weights.pth',
        transforms=_olmoearth_transforms,
        meta={'model_size': 'nano', 'repo': 'allenai/OlmoEarth-v1-Nano'},
    )
    TINY = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Tiny/resolve/main/weights.pth',
        transforms=_olmoearth_transforms,
        meta={'model_size': 'tiny', 'repo': 'allenai/OlmoEarth-v1-Tiny'},
    )
    BASE = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Base/resolve/main/weights.pth',
        transforms=_olmoearth_transforms,
        meta={'model_size': 'base', 'repo': 'allenai/OlmoEarth-v1-Base'},
    )
    LARGE = Weights(
        url='https://huggingface.co/allenai/OlmoEarth-v1-Large/resolve/main/weights.pth',
        transforms=_olmoearth_transforms,
        meta={'model_size': 'large', 'repo': 'allenai/OlmoEarth-v1-Large'},
    )


def olmoearth_pretrain_v1(
    weights: OlmoEarthPretrainV1_Weights | None = None, **kwargs: Any
) -> OlmoEarthPretrain_v1:
    """OlmoEarth Pretrain v1 model.

    Args:
        weights: Pre-trained weights. If None, model is randomly initialized.
        **kwargs: Passed to OlmoEarthPretrain_v1 (e.g. model_size, max_patch_size).

    Returns:
        OlmoEarthPretrain_v1 instance.
    """
    model_size = kwargs.pop('model_size', 'nano')
    if weights is not None:
        model_size = weights.meta.get('model_size', model_size)
        kwargs['model_size'] = model_size
    model = OlmoEarthPretrain_v1(model_size=model_size, **kwargs)
    if weights is not None:
        state_dict = weights.get_state_dict(progress=True)
        if not any(k.startswith('model.') for k in state_dict):
            state_dict = {f'model.{k}': v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    return model


__all__ = [
    'MaskValue',
    'MaskedOlmoEarthSample',
    'Modality',
    'ModalitySpec',
    'Normalizer',
    'OlmoEarthPretrainV1_Weights',
    'OlmoEarthPretrain_v1',
    'olmoearth_pretrain_v1',
]
