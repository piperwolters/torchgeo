# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import Any

import pytest
import torch
import torchvision
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import OlmoEarthV1_Weights, olmoearth_v1

pytest.importorskip('olmoearth_pretrain_minimal')


class TestOlmoEarthV1:
    @pytest.fixture(params=[*OlmoEarthV1_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthV1_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> OlmoEarthV1_Weights:
        weights = OlmoEarthV1_Weights.NANO
        path = tmp_path / 'weights.pth'
        model = olmoearth_v1(model_size='nano')
        # Save the *inner* state dict (encoder.*/decoder.*), which is how the published
        # checkpoints are keyed. Saving the wrapper's dict here would already carry the
        # model.* prefix and so would not exercise the re-keying that loading requires.
        torch.save(model.model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_olmoearth_v1(self) -> None:
        olmoearth_v1()

    def test_olmoearth_v1_weights(self, mocked_weights: OlmoEarthV1_Weights) -> None:
        olmoearth_v1(weights=mocked_weights)

    def test_olmoearth_v1_weights_are_applied(
        self, mocked_weights: OlmoEarthV1_Weights
    ) -> None:
        """Two models built from the same weights must hold identical tensors.

        Regression test for the ``model.*`` prefix mismatch: the checkpoint keys and the
        parameter names were disjoint, so ``strict=False`` dropped every tensor and each call
        returned a freshly randomized model.
        """
        one = olmoearth_v1(weights=mocked_weights).state_dict()
        two = olmoearth_v1(weights=mocked_weights).state_dict()
        assert one.keys() == two.keys()
        for key, value in one.items():
            assert torch.equal(value, two[key]), key

    def test_olmoearth_v1_cache_file_names_are_distinct(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Every entry must cache under its own file name.

        All four URLs end in ``weights.pth`` and :func:`torch.hub.load_state_dict_from_url`
        caches by file name, so entries sharing one would silently reuse whichever size was
        downloaded first, then fail to load it into a different size.
        """
        seen: list[str | None] = []

        def capture(
            url: str, *args: Any, file_name: str | None = None, **kwargs: Any
        ) -> dict[str, Any]:
            seen.append(file_name)
            return {}

        monkeypatch.setattr(
            torchvision.models._api, 'load_state_dict_from_url', capture
        )
        for weights in OlmoEarthV1_Weights:
            weights.get_state_dict()
        assert len(seen) == len(OlmoEarthV1_Weights)
        assert len(set(seen)) == len(seen)

    @pytest.mark.slow
    def test_olmoearth_v1_download(self, weights: OlmoEarthV1_Weights) -> None:
        olmoearth_v1(weights=weights)
