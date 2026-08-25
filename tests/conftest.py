# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import Any

import matplotlib
import pytest
import torch
import torchvision
from pytest import MonkeyPatch


def load(
    *args: Any,
    progress: bool = False,
    check_hash: bool = False,
    file_name: str | None = None,
    **kwargs: Any,
) -> Any:
    # check_hash and file_name only concern the torch.hub download and its cache entry, which
    # this stub bypasses by loading the path directly, so accept and ignore them.
    return torch.load(*args, **kwargs)


@pytest.fixture
def load_state_dict_from_url(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(torchvision.models._api, 'load_state_dict_from_url', load)


@pytest.fixture(autouse=True, scope='session')
def matplotlib_backend() -> None:
    matplotlib.use('agg')


@pytest.fixture(autouse=True)
def torch_hub(tmp_path: Path) -> None:
    torch.hub.set_dir(tmp_path)
