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
    *args: Any, progress: bool = False, file_name: str | None = None, **kwargs: Any
) -> Any:
    # file_name only names the torch.hub cache entry, which this stub bypasses by loading the
    # path directly, so accept and ignore it the way the real signature allows.
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
