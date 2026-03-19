# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import OlmoEarthPretrainV1_Weights, olmoearth_pretrain_v1

pytest.importorskip('olmoearth_pretrain_minimal')


class TestOlmoEarthPretrainV1:
    @pytest.fixture(params=[*OlmoEarthPretrainV1_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthPretrainV1_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> OlmoEarthPretrainV1_Weights:
        weights = OlmoEarthPretrainV1_Weights.NANO
        path = tmp_path / 'weights.pth'
        model = olmoearth_pretrain_v1(model_size='nano')
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_olmoearth_pretrain_v1(self) -> None:
        olmoearth_pretrain_v1()

    def test_olmoearth_pretrain_v1_weights(
        self, mocked_weights: OlmoEarthPretrainV1_Weights
    ) -> None:
        olmoearth_pretrain_v1(weights=mocked_weights)

    @pytest.mark.slow
    def test_olmoearth_pretrain_v1_download(
        self, weights: OlmoEarthPretrainV1_Weights
    ) -> None:
        olmoearth_pretrain_v1(weights=weights)
