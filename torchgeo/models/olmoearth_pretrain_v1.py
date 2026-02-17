# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""OlmoEarth Pretrain v1 model and data preprocessing.

Vendored from olmoearth-pretrain-minimal for use in TorchGeo. Supports loading
pre-trained weights from Hugging Face (allenai/OlmoEarth-v1-*).
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, StrEnum
from importlib.resources import files
from typing import Any, Callable, NamedTuple, Optional, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from torch import Tensor
from torchvision.models._api import Weights, WeightsEnum

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants and modality specs (required for model and normalizer)
# -----------------------------------------------------------------------------

BASE_RESOLUTION = 0.625
IMAGE_TILE_SIZE = 256
BASE_GSD = 10
MAX_SEQUENCE_LENGTH = 12


def _get_resolution(resolution_factor: int) -> float | int:
    resolution = BASE_RESOLUTION * resolution_factor
    if float(int(resolution)) == resolution:
        return int(resolution)
    return resolution


@dataclass(frozen=True)
class BandSet:
    bands: list[str]
    resolution_factor: int

    def get_resolution(self) -> float:
        return _get_resolution(self.resolution_factor)


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    tile_resolution_factor: int
    band_sets: list[BandSet]
    is_multitemporal: bool
    ignore_when_parsing: bool
    image_tile_size_factor: int = 1

    def get_tile_resolution(self) -> float:
        return _get_resolution(self.tile_resolution_factor)

    def bandsets_as_indices(self) -> list[list[int]]:
        indices = []
        offset = 0
        for band_set in self.band_sets:
            num_bands = len(band_set.bands)
            indices.append(list(range(offset, offset + num_bands)))
            offset += num_bands
        return indices

    @property
    def band_order(self) -> list[str]:
        return sum((list(bs.bands) for bs in self.band_sets), [])

    @property
    def num_band_sets(self) -> int:
        return len(self.band_sets)

    @property
    def is_spatial(self) -> bool:
        return self.get_tile_resolution() > 0 and self.get_expected_tile_size() > 1

    def get_expected_tile_size(self) -> int:
        if self.image_tile_size_factor < 0:
            return IMAGE_TILE_SIZE // abs(self.image_tile_size_factor)
        return IMAGE_TILE_SIZE * self.image_tile_size_factor


class Modality:
    """Enum-like access to ModalitySpecs."""

    SENTINEL2_L2A = ModalitySpec(
        name="sentinel2_l2a",
        tile_resolution_factor=16,
        band_sets=[
            BandSet(["B02", "B03", "B04", "B08"], 16),
            BandSet(["B05", "B06", "B07", "B8A", "B11", "B12"], 32),
            BandSet(["B01", "B09"], 64),
        ],
        is_multitemporal=True,
        ignore_when_parsing=False,
    )
    SENTINEL1 = ModalitySpec(
        name="sentinel1",
        tile_resolution_factor=16,
        band_sets=[BandSet(["vv", "vh"], 16)],
        is_multitemporal=True,
        ignore_when_parsing=False,
    )
    LANDSAT = ModalitySpec(
        name="landsat",
        tile_resolution_factor=16,
        band_sets=[
            BandSet(["B8"], 16),
            BandSet(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"], 32),
        ],
        is_multitemporal=True,
        ignore_when_parsing=False,
    )
    WORLDCOVER = ModalitySpec(
        name="worldcover",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )
    SRTM = ModalitySpec(
        name="srtm",
        tile_resolution_factor=16,
        band_sets=[BandSet(["srtm"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )
    OPENSTREETMAP_RASTER = ModalitySpec(
        name="openstreetmap_raster",
        tile_resolution_factor=16,
        band_sets=[BandSet([f"b{i}" for i in range(40)], 4)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )
    WRI_CANOPY_HEIGHT_MAP = ModalitySpec(
        name="wri_canopy_height_map",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )
    CDL = ModalitySpec(
        name="cdl",
        tile_resolution_factor=16,
        band_sets=[BandSet(["cdl"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )
    WORLDCEREAL = ModalitySpec(
        name="worldcereal",
        tile_resolution_factor=16,
        band_sets=[BandSet([f"b{i}" for i in range(8)], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )
    NAIP = ModalitySpec(
        name="naip",
        tile_resolution_factor=1,
        band_sets=[BandSet(["R", "G", "B", "IR"], 1)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )
    LATLON = ModalitySpec(
        name="latlon",
        tile_resolution_factor=0,
        band_sets=[BandSet(["lat", "lon"], 0)],
        is_multitemporal=False,
        ignore_when_parsing=True,
    )
    WORLDPOP = ModalitySpec(
        name="worldpop",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )
    GSE = ModalitySpec(
        name="gse",
        tile_resolution_factor=16,
        band_sets=[BandSet([f"A{idx:02d}" for idx in range(64)], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )
    ERA5_10 = ModalitySpec(
        name="era5_10",
        tile_resolution_factor=16,
        band_sets=[BandSet([
            "2m-temperature", "2m-dewpoint-temperature", "surface-pressure",
            "10m-u-component-of-wind", "10m-v-component-of-wind", "total-precipitation",
        ], 4096)],
        is_multitemporal=True,
        ignore_when_parsing=False,
        image_tile_size_factor=-256,
    )

    @classmethod
    def get(cls, name: str) -> ModalitySpec:
        attr = name.upper()
        modality = getattr(cls, attr)
        assert isinstance(modality, ModalitySpec) and modality.name == name
        return modality

    @classmethod
    def values(cls) -> list[ModalitySpec]:
        return [
            getattr(cls, k) for k in dir(cls)
            if isinstance(getattr(cls, k), ModalitySpec)
        ]

    @classmethod
    def names(cls) -> list[str]:
        return [m.name for m in cls.values()]


def get_modality_specs_from_names(names: list[str]) -> list[ModalitySpec]:
    return [Modality.get(name) for name in names]


# -----------------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------------

ArrayTensor = Union[np.ndarray, torch.Tensor]


class MaskValue(Enum):
    ONLINE_ENCODER = 0
    TARGET_ENCODER_ONLY = 1
    DECODER = 2
    MISSING = 3


class MaskedOlmoEarthSample(NamedTuple):
    timestamps: ArrayTensor
    sentinel2_l2a: ArrayTensor | None = None
    sentinel2_l2a_mask: ArrayTensor | None = None
    sentinel1: ArrayTensor | None = None
    sentinel1_mask: ArrayTensor | None = None
    worldcover: ArrayTensor | None = None
    worldcover_mask: ArrayTensor | None = None
    latlon: ArrayTensor | None = None
    latlon_mask: ArrayTensor | None = None
    openstreetmap_raster: ArrayTensor | None = None
    openstreetmap_raster_mask: ArrayTensor | None = None
    srtm: ArrayTensor | None = None
    srtm_mask: ArrayTensor | None = None
    landsat: ArrayTensor | None = None
    landsat_mask: ArrayTensor | None = None
    naip: ArrayTensor | None = None
    naip_mask: ArrayTensor | None = None
    cdl: ArrayTensor | None = None
    cdl_mask: ArrayTensor | None = None
    worldcereal: ArrayTensor | None = None
    worldcereal_mask: ArrayTensor | None = None
    wri_canopy_height_map: ArrayTensor | None = None
    wri_canopy_height_map_mask: ArrayTensor | None = None

    def as_dict(self, return_none: bool = True) -> dict[str, Any]:
        d = {}
        for field in self._fields:
            val = getattr(self, field)
            if return_none or val is not None:
                d[field] = val
        return d

    @property
    def modalities(self) -> list[str]:
        return [
            f for f in self._fields
            if not f.endswith("_mask") and f != "timestamps" and getattr(self, f) is not None
        ]

    @staticmethod
    def get_masked_modality_name(modality: str) -> str:
        return f"{modality}_mask"


# -----------------------------------------------------------------------------
# Config (for loading from Hugging Face config.json)
# -----------------------------------------------------------------------------

C = TypeVar("C", bound="_StandaloneConfig")
CLASS_NAME_FIELD = "_CLASS_"

# Map Hugging Face config class paths to local class names (filled after classes are defined)
_CONFIG_CLASS_MAP: dict[str, type] = {}


def _register_config_class(cls: type) -> type:
    _CONFIG_CLASS_MAP[cls.__name__] = cls
    return cls


@dataclass
class _StandaloneConfig:
    CLASS_NAME_FIELD = "_CLASS_"

    @classmethod
    def _resolve_class(cls, class_name: str) -> type | None:
        if "." not in class_name:
            return None
        *_, cls_name = class_name.split(".")
        return _CONFIG_CLASS_MAP.get(cls_name)

    @classmethod
    def _clean_data(cls, data: Any) -> Any:
        if isinstance(data, dict):
            class_name = data.get(CLASS_NAME_FIELD)
            cleaned = {k: cls._clean_data(v) for k, v in data.items() if k != CLASS_NAME_FIELD}
            if class_name is not None:
                resolved = cls._resolve_class(class_name)
                if resolved is not None and is_dataclass(resolved):
                    field_names = {f.name for f in fields(resolved)}
                    valid = {k: v for k, v in cleaned.items() if k in field_names}
                    for key, value in list(valid.items()):
                        if isinstance(value, dict) and CLASS_NAME_FIELD in value:
                            nested_name = value[CLASS_NAME_FIELD]
                            nested_cls = cls._resolve_class(nested_name)
                            if nested_cls is not None and is_dataclass(nested_cls):
                                valid[key] = nested_cls.from_dict(
                                    {k: v for k, v in value.items() if k != CLASS_NAME_FIELD}
                                )
                    return resolved(**valid)
                cleaned[CLASS_NAME_FIELD] = class_name
            return cleaned
        if isinstance(data, (list, tuple)):
            return type(data)(cls._clean_data(item) for item in data)
        return data

    @classmethod
    def from_dict(
        cls: type[C], data: dict[str, Any], overrides: list[str] | None = None
    ) -> C:
        if overrides:
            warnings.warn("Config overrides ignored in standalone mode.", UserWarning, stacklevel=2)
        cleaned = cls._clean_data(data)
        if is_dataclass(cleaned) and not isinstance(cleaned, type):
            return cleaned  # type: ignore[return-value]
        if isinstance(cleaned, dict) and CLASS_NAME_FIELD in cleaned:
            resolved = cls._resolve_class(cleaned[CLASS_NAME_FIELD])
            if resolved is not None:
                return resolved.from_dict({k: v for k, v in cleaned.items() if k != CLASS_NAME_FIELD})
            raise ValueError(f"Cannot resolve class: {cleaned[CLASS_NAME_FIELD]}")
        if isinstance(cleaned, dict):
            field_names = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in cleaned.items() if k in field_names})  # type: ignore[call-arg]
        raise TypeError(f"Expected dict, got {type(cleaned)}")

    def as_dict(
        self,
        *,
        exclude_none: bool = False,
        exclude_private_fields: bool = False,
        include_class_name: bool = False,
        recurse: bool = True,
    ) -> dict[str, Any]:
        def convert(obj: Any) -> Any:
            if is_dataclass(obj) and not isinstance(obj, type):
                result = {}
                if include_class_name:
                    result[CLASS_NAME_FIELD] = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
                for field in fields(obj):
                    if exclude_private_fields and field.name.startswith("_"):
                        continue
                    v = getattr(obj, field.name)
                    if exclude_none and v is None:
                        continue
                    result[field.name] = convert(v) if recurse else v
                return result
            if isinstance(obj, dict):
                return {k: convert(v) if recurse else v for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(convert(x) if recurse else x for x in obj)
            return obj
        return convert(self)

    def validate(self) -> None:
        pass

    def build(self) -> Any:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Tokenization config
# -----------------------------------------------------------------------------

@dataclass
class ModalityTokenization:
    band_groups: list[list[str]]

    def compute_indices(self, base_modality: ModalitySpec) -> list[list[int]]:
        name_to_idx = {name: i for i, name in enumerate(base_modality.band_order)}
        return [[name_to_idx[b] for b in group if b in name_to_idx] for group in self.band_groups]

    @property
    def num_band_sets(self) -> int:
        return len(self.band_groups)


@dataclass
class TokenizationConfig:
    overrides: dict[str, ModalityTokenization] = field(default_factory=dict)
    _bandset_indices_cache: dict[str, list[list[int]]] = field(default_factory=dict, init=False, repr=False)

    def get_bandset_indices(self, modality_name: str) -> list[list[int]]:
        if modality_name in self._bandset_indices_cache:
            return self._bandset_indices_cache[modality_name]
        base_spec = Modality.get(modality_name)
        if modality_name in self.overrides:
            result = self.overrides[modality_name].compute_indices(base_spec)
        else:
            result = base_spec.bandsets_as_indices()
        self._bandset_indices_cache[modality_name] = result
        return result

    def get_num_bandsets(self, modality_name: str) -> int:
        if modality_name in self.overrides:
            return self.overrides[modality_name].num_band_sets
        return Modality.get(modality_name).num_band_sets


# -----------------------------------------------------------------------------
# Position encodings
# -----------------------------------------------------------------------------

def get_1d_sincos_pos_encoding(pos: Tensor, encoding_dim: int) -> Tensor:
    assert encoding_dim % 2 == 0
    omega = torch.arange(encoding_dim // 2, device=pos.device) / encoding_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    pos = pos.reshape(-1)
    out = torch.einsum("l,d->ld", pos.float(), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def get_2d_sincos_pos_encoding(grid: Tensor, encoding_dim: int) -> Tensor:
    assert encoding_dim % 2 == 0
    d = encoding_dim // 2
    emb_h = get_1d_sincos_pos_encoding(grid[0], d)
    emb_w = get_1d_sincos_pos_encoding(grid[1], d)
    return torch.cat([emb_h, emb_w], dim=1)


def get_2d_sincos_pos_encoding_with_resolution(
    grid_size: int, res: Tensor, encoding_dim: int, device: torch.device, cls_token: bool = False
) -> Tensor:
    grid_h = torch.arange(grid_size, device=device)
    grid_w = torch.arange(grid_size, device=device)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing="xy"), dim=0)  # 2 x h x w
    grid = torch.einsum("chw,n->cnhw", grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    grid = grid.reshape(2, n * h * w)
    pos_embed = get_2d_sincos_pos_encoding(grid, encoding_dim).reshape(n, h * w, encoding_dim)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(n, 1, encoding_dim, device=device), pos_embed], dim=1)
    return pos_embed


def get_month_encoding_table(encoding_dim: int) -> Tensor:
    assert encoding_dim % 2 == 0
    angles = torch.arange(0, 13) / (12 / (2 * math.pi))
    dim_per_table = encoding_dim // 2
    sin_t = torch.sin(torch.stack([angles for _ in range(dim_per_table)], dim=-1))
    cos_t = torch.cos(torch.stack([angles for _ in range(dim_per_table)], dim=-1))
    return torch.cat([sin_t[:-1], cos_t[:-1]], dim=-1)


# -----------------------------------------------------------------------------
# Attention and transformer block
# -----------------------------------------------------------------------------

try:
    import flash_attn
except ImportError:
    flash_attn = None  # type: ignore[assignment]


def _dispatch_flash_attn(
    q: Tensor, k: Tensor, v: Tensor,
    cu_seqlens: Tensor | None = None, cu_seqlens_q: Tensor | None = None, cu_seqlens_k: Tensor | None = None,
    max_seqlen: int | None = None, max_seqlen_q: int | None = None, max_seqlen_k: int | None = None,
    dropout_p: float = 0.0, softmax_scale: float | None = None, causal: bool = False,
) -> Tensor:
    if flash_attn is None:
        raise RuntimeError("flash-attn is required for use_flash_attn=True")
    if cu_seqlens is not None:
        cu_seqlens_q = cu_seqlens_q or cu_seqlens
        cu_seqlens_k = cu_seqlens_k or cu_seqlens
    if max_seqlen is not None:
        max_seqlen_q = max_seqlen_q or max_seqlen
        max_seqlen_k = max_seqlen_k or max_seqlen
    varlen = all(x is not None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k))
    if varlen:
        return flash_attn.flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal,
        )
    return flash_attn.flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)


class Attention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_norm: bool = False,
        attn_drop: float = 0.0, proj_drop: float = 0.0, norm_layer: type = nn.LayerNorm,
        cross_attn: bool = False, use_flash_attn: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.cross_attn = cross_attn
        self.use_flash_attn = use_flash_attn
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def sdpa(
        self, q: Tensor, k: Tensor, v: Tensor, n: int,
        cu_seqlens: Tensor | None = None, cu_seqlens_q: Tensor | None = None, cu_seqlens_k: Tensor | None = None,
        max_seqlen: int | None = None, max_seqlen_q: int | None = None, max_seqlen_k: int | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        if self.use_flash_attn:
            x = _dispatch_flash_attn(q, k, v, cu_seqlens=cu_seqlens, cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k, max_seqlen=max_seqlen, max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k, dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale, causal=False)
            x = x.transpose(1, 2)
        elif self.fast_attn:
            if attn_mask is not None:
                attn_mask = attn_mask[:, None, None].repeat(1, self.num_heads, n, 1)
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p)
        else:
            if attn_mask is not None:
                raise NotImplementedError
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        return x

    def forward(
        self, x: Tensor, y: Tensor | None = None,
        cu_seqlens: Tensor | None = None, cu_seqlens_q: Tensor | None = None, cu_seqlens_k: Tensor | None = None,
        max_seqlen: int | None = None, max_seqlen_q: int | None = None, max_seqlen_k: int | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        orig_shape = x.shape
        q = self.q(x)
        k = self.k(y if y is not None else x)
        v = self.v(y if y is not None else x)
        if not self.use_flash_attn:
            q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
            k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
            v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        else:
            q = rearrange(q, "bn (h d) -> bn h d", h=self.num_heads)
            k = rearrange(k, "bn (h d) -> bn h d", h=self.num_heads)
            v = rearrange(v, "bn (h d) -> bn h d", h=self.num_heads)
        q, k = self.q_norm(q), self.k_norm(k)
        x = self.sdpa(q, k, v, orig_shape[-2], cu_seqlens=cu_seqlens, cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k, max_seqlen=max_seqlen, max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(orig_shape)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None,
                 act_layer: type = nn.GELU, bias: bool = True, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else (x * self.gamma)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x / keep * (keep + torch.rand(shape, dtype=x.dtype, device=x.device)).floor()


class Block(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, qk_norm: bool = False,
        drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0, init_values: float | None = None,
        act_layer: type = nn.GELU, norm_layer: type = nn.LayerNorm, cross_attn: bool = False, use_flash_attn: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop,
                             proj_drop=drop, norm_layer=norm_layer, cross_attn=cross_attn, use_flash_attn=use_flash_attn)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(
        self, x: Tensor, y: Tensor | None = None,
        cu_seqlens: Tensor | None = None, cu_seqlens_q: Tensor | None = None, cu_seqlens_k: Tensor | None = None,
        max_seqlen: int | None = None, max_seqlen_q: int | None = None, max_seqlen_k: int | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), y=y, cu_seqlens=cu_seqlens,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen=max_seqlen,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, attn_mask=attn_mask)))
        return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

    def apply_fsdp(self, **kwargs: Any) -> None:
        from torch.distributed.fsdp import fully_shard
        fully_shard(self, **kwargs)

    def apply_compile(self) -> None:
        self.compile(dynamic=False, mode="max-autotune-no-cudagraphs", fullgraph=True)


# -----------------------------------------------------------------------------
# Flexi patch embed
# -----------------------------------------------------------------------------

def _to_2tuple(x: Any) -> tuple[int, int]:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return tuple(x)
    return (int(x), int(x))


class FlexiPatchEmbed(nn.Module):
    def __init__(
        self, modality_spec: ModalitySpec, patch_size_at_16: int | tuple[int, int],
        in_chans: int = 3, embedding_size: int = 128, norm_layer: type | None = None,
        bias: bool = True, interpolation: str = "bicubic", antialias: bool = True,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.modality_spec = modality_spec
        self.patch_size = _to_2tuple(patch_size_at_16 * modality_spec.image_tile_size_factor)
        self.proj = nn.Conv2d(in_chans, embedding_size, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
        self.norm = (norm_layer(embedding_size) if norm_layer else nn.Identity()) if norm_layer else nn.Identity()
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, x: Tensor, patch_size: int | tuple[int, int] | None = None) -> Tensor:
        batch_size = x.shape[0]
        if x.ndim == 5:
            b, h, w, t, c = x.shape
            x = rearrange(x, "b h w t c -> (b t) c h w")
        else:
            x = rearrange(x, "b h w c -> b c h w")
        if patch_size is None:
            patch_size = self.patch_size
        else:
            if isinstance(patch_size, tuple):
                patch_size = _to_2tuple(patch_size[0] * self.modality_spec.image_tile_size_factor)
            else:
                patch_size = _to_2tuple(patch_size * self.modality_spec.image_tile_size_factor)
        if patch_size != self.patch_size:
            new_shape = (x.shape[-2] // patch_size[0] * patch_size[0], x.shape[-1] // patch_size[1] * patch_size[1])
            x = torch.nn.functional.interpolate(x, size=new_shape, mode=self.interpolation, antialias=self.antialias)
        x = self.proj(x)
        if x.ndim == 4 and batch_size != x.shape[0]:
            _, d, h, w = x.shape
            t = x.shape[0] // batch_size
            x = rearrange(x, "(b t) d h w -> b h w t d", b=batch_size, t=t)
        else:
            x = rearrange(x, "b d h w -> b h w d")
        return self.norm(x)


class FlexiPatchReconstruction(nn.Module):
    def __init__(
        self, max_patch_size: int | tuple[int, int], out_chans: int = 3, embedding_size: int = 128,
        norm_layer: type | None = None, bias: bool = True, interpolation: str = "bicubic", antialias: bool = True,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_patch_size = _to_2tuple(max_patch_size)
        self.proj = nn.ConvTranspose2d(embedding_size, out_chans, kernel_size=self.max_patch_size, stride=self.max_patch_size, bias=bias)
        self.norm = (norm_layer(embedding_size) if norm_layer else nn.Identity()) if norm_layer else nn.Identity()
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, x: Tensor, patch_size: int | tuple[int, int] | None = None) -> Tensor:
        if x.ndim == 4:
            b, h, w, d = x.shape
            t = 1
        else:
            b, h, w, t, d = x.shape
            x = rearrange(x, "b h w t d -> (b t) d h w")
        if patch_size is None:
            patch_size = self.max_patch_size
        else:
            patch_size = _to_2tuple(patch_size)
        x = self.proj(x)
        if patch_size != self.max_patch_size:
            x = rearrange(x, "b c (h p1) (w p2) -> (b h w) c p1 p2", p1=self.max_patch_size[0], p2=self.max_patch_size[1])
            x = torch.nn.functional.interpolate(x, patch_size, mode=self.interpolation, antialias=self.antialias)
            x = rearrange(x, "(b h w) c p1 p2 -> b c (h p1) (w p2)", b=b, h=h, w=w)
        if t > 1:
            x = rearrange(x, "(b t) c h w -> b h w t c", b=b, t=t)
        else:
            x = rearrange(x, "b c h w -> b h w c")
        return self.norm(x)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def unpack_encoder_output(output_dict: dict[str, Any]) -> tuple:
    latent = output_dict.pop("tokens_and_masks", None)
    latent_projected_and_pooled = output_dict.pop("project_aggregated", None)
    output_dict.pop("token_norm_stats", None)
    return latent, latent_projected_and_pooled, output_dict


def get_cumulative_sequence_lengths(seq_lengths: Tensor) -> Tensor:
    return torch.cat([
        torch.tensor([0], dtype=torch.int32, device=seq_lengths.device),
        torch.cumsum(seq_lengths.masked_select(seq_lengths != 0), 0, dtype=torch.int32),
    ])


class DistributedMixins:
    def apply_ddp(self, dp_mesh: Any = None, **kwargs: Any) -> None:
        from torch.distributed._composable.replicate import replicate
        replicate(self, device_mesh=dp_mesh, bucket_cap_mb=100, find_unused_parameters=kwargs.get("find_unused_parameters", True))


# -----------------------------------------------------------------------------
# TokensAndMasks and flexi_vit helpers
# -----------------------------------------------------------------------------

def get_modalities_to_process(available: list[str], supported: list[str]) -> list[str]:
    return list(set(supported) & set(available))


def return_modalities_from_dict(per_modality_input_tokens: dict[str, Tensor]) -> list[str]:
    return [k for k in per_modality_input_tokens if not k.endswith("_mask")]


class PoolingType(StrEnum):
    MAX = "max"
    MEAN = "mean"


class TokensAndMasks(NamedTuple):
    """Per-modality tokens and masks; field names match MaskedOlmoEarthSample."""
    sentinel2_l2a: Tensor | None = None
    sentinel2_l2a_mask: Tensor | None = None
    sentinel1: Tensor | None = None
    sentinel1_mask: Tensor | None = None
    worldcover: Tensor | None = None
    worldcover_mask: Tensor | None = None
    latlon: Tensor | None = None
    latlon_mask: Tensor | None = None
    openstreetmap_raster: Tensor | None = None
    openstreetmap_raster_mask: Tensor | None = None
    srtm: Tensor | None = None
    srtm_mask: Tensor | None = None
    landsat: Tensor | None = None
    landsat_mask: Tensor | None = None
    cdl: Tensor | None = None
    cdl_mask: Tensor | None = None
    worldcereal: Tensor | None = None
    worldcereal_mask: Tensor | None = None
    wri_canopy_height_map: Tensor | None = None
    wri_canopy_height_map_mask: Tensor | None = None

    @property
    def modalities(self) -> list[str]:
        return [
            x for x in self._fields
            if not x.endswith("_mask") and getattr(self, x) is not None
        ]

    @staticmethod
    def get_masked_modality_name(modality: str) -> str:
        return f"{modality}_mask"

    def as_dict(self, return_none: bool = True) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self._fields if return_none or getattr(self, f) is not None}

    @staticmethod
    def _flatten(x: Tensor) -> Tensor:
        return rearrange(x, "b ... d -> b (...) d")

    def flatten_tokens_and_masks(self, return_lists: bool = False) -> tuple[Tensor, Tensor]:
        flattened_x, flattened_masks = [], []
        for attr_name in self.modalities:
            mask_name = self.get_masked_modality_name(attr_name)
            attr = getattr(self, attr_name)
            masked_attr = getattr(self, mask_name)
            if attr is not None and masked_attr is not None:
                flattened_x.append(self._flatten(attr))
                flattened_masks.append(self._flatten(masked_attr.unsqueeze(dim=-1)))
        if return_lists:
            return flattened_x, [m[:, :, 0] for m in flattened_masks]
        x = torch.cat(flattened_x, dim=1)
        masks = torch.cat(flattened_masks, dim=1)[:, :, 0]
        return x, masks

    def pool_unmasked_tokens(
        self, pooling_type: PoolingType = PoolingType.MAX, spatial_pooling: bool = False
    ) -> Tensor:
        x, mask = self.flatten_tokens_and_masks()
        mask = (mask == MaskValue.ONLINE_ENCODER.value).long()
        x_for_pooling = x * mask.unsqueeze(-1)
        if pooling_type == PoolingType.MAX:
            x_for_pooling = x_for_pooling.masked_fill(~mask.bool().unsqueeze(-1), -float("inf"))
            return x_for_pooling.max(dim=1).values
        num_encoded = torch.sum(mask, -1, keepdim=True)
        if (num_encoded == 0).any():
            raise ValueError("num_encoded_tokens is 0 for some samples")
        return x_for_pooling.sum(dim=1) / num_encoded


class ProjectAndAggregate(nn.Module):
    def __init__(self, embedding_size: int, num_layers: int = 1, aggregate_then_project: bool = True):
        super().__init__()
        layers = [nn.Linear(embedding_size, embedding_size)]
        for _ in range(1, num_layers):
            layers.extend([nn.ReLU(), nn.Linear(embedding_size, embedding_size)])
        self.projection = nn.Sequential(*layers)
        self.aggregate_then_project = aggregate_then_project

    def forward(self, x: TokensAndMasks | Tensor) -> Tensor:
        if isinstance(x, TokensAndMasks):
            pooled = x.pool_unmasked_tokens(PoolingType.MEAN, spatial_pooling=False)
        elif isinstance(x, Tensor):
            pooled = reduce(x, "b ... d -> b d", "mean")
        else:
            raise ValueError(f"Invalid type: {type(x)}")
        return self.projection(pooled)


# -----------------------------------------------------------------------------
# MultiModalPatchEmbeddings (used by Encoder)
# -----------------------------------------------------------------------------

class MultiModalPatchEmbeddings(nn.Module):
    def __init__(
        self,
        supported_modality_names: list[str],
        max_patch_size: int,
        embedding_size: int,
        tokenization_config: TokenizationConfig | None = None,
    ):
        super().__init__()
        self.supported_modality_names = supported_modality_names
        self.max_patch_size = max_patch_size
        self.embedding_size = embedding_size
        self.tokenization_config = tokenization_config or TokenizationConfig()
        self.per_modality_embeddings = nn.ModuleDict({})
        for modality in supported_modality_names:
            spec = Modality.get(modality)
            bandset_indices = self.tokenization_config.get_bandset_indices(modality)
            if not spec.is_spatial:
                self.per_modality_embeddings[modality] = nn.ModuleDict({
                    f"{modality}__{idx}": nn.Linear(len(channel_idxs), embedding_size)
                    for idx, channel_idxs in enumerate(bandset_indices)
                })
            else:
                self.per_modality_embeddings[modality] = nn.ModuleDict({
                    f"{modality}__{idx}": FlexiPatchEmbed(
                        modality_spec=spec,
                        patch_size_at_16=max_patch_size,
                        in_chans=len(channel_idxs),
                        embedding_size=embedding_size,
                    )
                    for idx, channel_idxs in enumerate(bandset_indices)
                })
        for modality in supported_modality_names:
            for idx, bandset_indices in enumerate(self.tokenization_config.get_bandset_indices(modality)):
                name = f"{modality}__{idx}_buffer"
                self.register_buffer(name, torch.tensor(bandset_indices, dtype=torch.long), persistent=False)

    def _get_embedding_name(self, modality: str, idx: int) -> str:
        return f"{modality}__{idx}"

    def forward(
        self, input_data: MaskedOlmoEarthSample, patch_size: int, fast_pass: bool = False
    ) -> dict[str, Tensor]:
        output_dict: dict[str, Tensor] = {}
        modalities_to_process = get_modalities_to_process(input_data.modalities, self.supported_modality_names)
        for modality in modalities_to_process:
            mask_name = input_data.get_masked_modality_name(modality)
            modality_mask = getattr(input_data, mask_name)
            modality_data = getattr(input_data, modality)
            if modality_data is None or modality_mask is None:
                continue
            spec = Modality.get(modality)
            num_band_sets = self.tokenization_config.get_num_bandsets(modality)
            modality_tokens_list, modality_masks_list = [], []
            for idx in range(num_band_sets):
                buffer_name = f"{modality}__{idx}_buffer"
                patchified = torch.index_select(modality_data, -1, getattr(self, buffer_name))
                emb_mod = self.per_modality_embeddings[modality][self._get_embedding_name(modality, idx)]
                if spec.is_spatial:
                    patchified = emb_mod(patchified, patch_size=patch_size)
                else:
                    patchified = emb_mod(patchified)
                modality_tokens_list.append(patchified)
                modality_masks_list.append(modality_mask[..., idx] if modality_mask.ndim > 2 else modality_mask)
            output_dict[modality] = torch.stack(modality_tokens_list, dim=-2)
            output_dict[mask_name] = torch.stack(modality_masks_list, dim=-1)
        return output_dict


# -----------------------------------------------------------------------------
# Encoder and Predictor (minimal for build + load_state_dict)
# -----------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        max_patch_size: int,
        min_patch_size: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        drop_path: float,
        supported_modalities: list[ModalitySpec],
        max_sequence_length: int,
        num_register_tokens: int = 0,
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        num_projection_layers: int = 1,
        aggregate_then_project: bool = True,
        use_flash_attn: bool = False,
        frozen_patch_embeddings: bool = False,
        qk_norm: bool = False,
        log_token_norm_stats: bool = False,
        tokenization_config: TokenizationConfig | None = None,
    ):
        super().__init__()
        self.supported_modality_names = [m.name for m in supported_modalities]
        self.tokenization_config = tokenization_config or TokenizationConfig()
        self.max_patch_size = max_patch_size
        self.min_patch_size = min_patch_size
        self.embedding_size = embedding_size
        self.patch_embeddings = MultiModalPatchEmbeddings(
            self.supported_modality_names,
            max_patch_size,
            embedding_size,
            tokenization_config=self.tokenization_config,
        )
        self.project_and_aggregate = ProjectAndAggregate(
            embedding_size=embedding_size,
            num_layers=num_projection_layers,
            aggregate_then_project=aggregate_then_project,
        )
        self.norm = nn.LayerNorm(embedding_size)
        self.blocks = nn.ModuleList([
            Block(embedding_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path, qk_norm=qk_norm, use_flash_attn=use_flash_attn)
            for _ in range(depth)
        ])
        self.composite_encodings = None  # Optional CompositeEncodings; encoder can work without for loading
        if frozen_patch_embeddings:
            for p in self.patch_embeddings.parameters():
                p.requires_grad = False

    def forward(
        self, x: MaskedOlmoEarthSample, patch_size: int, input_res: int = BASE_GSD, **kwargs: Any
    ) -> dict[str, Any]:
        patchified = self.patch_embeddings(x, patch_size)
        mods = return_modalities_from_dict(patchified)
        tokens_list = [rearrange(patchified[m], "b ... d -> b (...) d") for m in mods]
        masks_list = [
            rearrange(patchified[MaskedOlmoEarthSample.get_masked_modality_name(m)], "b ... -> b (...)")
            for m in mods
        ]
        tokens = torch.cat(tokens_list, dim=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        offset = 0
        out_dict: dict[str, Any] = {}
        for i, m in enumerate(mods):
            n = tokens_list[i].shape[1]
            out_dict[m] = tokens[:, offset : offset + n]
            out_dict[MaskedOlmoEarthSample.get_masked_modality_name(m)] = masks_list[i]
            offset += n
        out = TokensAndMasks(**{f: out_dict.get(f) for f in TokensAndMasks._fields})
        output_dict: dict[str, Any] = {"tokens_and_masks": out}
        output_dict["project_aggregated"] = self.project_and_aggregate(out)
        return output_dict


class Predictor(nn.Module):
    def __init__(
        self,
        supported_modalities: list[ModalitySpec],
        encoder_embedding_size: int,
        decoder_embedding_size: int,
        depth: int,
        mlp_ratio: float,
        num_heads: int,
        max_sequence_length: int,
        drop_path: float = 0.0,
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        output_embedding_size: int | None = None,
        use_flash_attn: bool = False,
        qk_norm: bool = False,
        tokenization_config: TokenizationConfig | None = None,
    ):
        super().__init__()
        self.supported_modality_names = [m.name for m in supported_modalities]
        self.tokenization_config = tokenization_config or TokenizationConfig()
        self.encoder_embedding_size = encoder_embedding_size
        self.output_embedding_size = output_embedding_size or encoder_embedding_size
        self.encoder_to_decoder_embed = nn.Linear(encoder_embedding_size, decoder_embedding_size, bias=True)
        self.to_output_embed = nn.Linear(decoder_embedding_size, self.output_embedding_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(decoder_embedding_size))
        self.input_norm = nn.LayerNorm(encoder_embedding_size)
        self.norm = nn.LayerNorm(decoder_embedding_size)
        self.blocks = nn.ModuleList([
            Block(decoder_embedding_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path, cross_attn=True, use_flash_attn=use_flash_attn, qk_norm=qk_norm)
            for _ in range(depth)
        ])
        self.composite_encodings = None

    def forward(
        self, x: TokensAndMasks, timestamps: Tensor, patch_size: int, input_res: int = BASE_GSD, **kwargs: Any
    ) -> TokensAndMasks:
        return x


# -----------------------------------------------------------------------------
# Config dataclasses and LatentMIM (Encoder/Predictor built from config)
# -----------------------------------------------------------------------------

def _register_config(cls: type) -> type:
    _CONFIG_CLASS_MAP[cls.__name__] = cls
    return cls


@dataclass
@_register_config
class EncoderConfig(_StandaloneConfig):
    supported_modality_names: list[str]
    embedding_size: int = 16
    max_patch_size: int = 8
    min_patch_size: int = 1
    num_heads: int = 2
    mlp_ratio: float = 1.0
    depth: int = 2
    drop_path: float = 0.1
    max_sequence_length: int = 12
    num_register_tokens: int = 0
    learnable_channel_embeddings: bool = True
    random_channel_embeddings: bool = False
    num_projection_layers: int = 1
    aggregate_then_project: bool = True
    use_flash_attn: bool = False
    frozen_patch_embeddings: bool = False
    qk_norm: bool = False
    log_token_norm_stats: bool = False
    tokenization_config: TokenizationConfig | None = None

    @property
    def supported_modalities(self) -> list[ModalitySpec]:
        return get_modality_specs_from_names(self.supported_modality_names)

    def build(self) -> "Encoder":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        kwargs.pop("tokenization_config", None)
        kwargs["tokenization_config"] = self.tokenization_config or TokenizationConfig()
        return Encoder(**kwargs)


@dataclass
@_register_config
class PredictorConfig(_StandaloneConfig):
    supported_modality_names: list[str]
    encoder_embedding_size: int = 16
    decoder_embedding_size: int = 16
    depth: int = 2
    mlp_ratio: float = 1.0
    num_heads: int = 2
    max_sequence_length: int = 12
    drop_path: float = 0.0
    learnable_channel_embeddings: bool = True
    random_channel_embeddings: bool = False
    output_embedding_size: int | None = None
    use_flash_attn: bool = False
    qk_norm: bool = False
    tokenization_config: TokenizationConfig | None = None

    @property
    def supported_modalities(self) -> list[ModalitySpec]:
        return get_modality_specs_from_names(self.supported_modality_names)

    def build(self) -> "Predictor":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        kwargs["tokenization_config"] = self.tokenization_config or TokenizationConfig()
        return Predictor(**kwargs)


@dataclass
@_register_config
class LatentMIMConfig(_StandaloneConfig):
    encoder_config: EncoderConfig
    decoder_config: PredictorConfig
    reconstructor_config: Any = None

    def validate(self) -> None:
        if self.encoder_config.supported_modalities != self.decoder_config.supported_modalities:
            raise ValueError("Encoder and decoder must support the same modalities")
        if self.encoder_config.max_sequence_length != self.decoder_config.max_sequence_length:
            raise ValueError("Encoder and decoder must have the same max_sequence_length")
        if self.encoder_config.embedding_size != self.decoder_config.encoder_embedding_size:
            raise ValueError("Encoder embedding_size must match decoder encoder_embedding_size")

    def build(self) -> "LatentMIM":
        self.validate()
        encoder = self.encoder_config.build()
        decoder = self.decoder_config.build()
        reconstructor = self.reconstructor_config.build() if self.reconstructor_config else None
        return LatentMIM(encoder=encoder, decoder=decoder, reconstructor=reconstructor)


class LatentMIM(nn.Module, DistributedMixins):
    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        reconstructor: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstructor = reconstructor
        self.target_encoder = deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(
        self, x: MaskedOlmoEarthSample, patch_size: int
    ) -> tuple[TokensAndMasks, TokensAndMasks, Tensor, TokensAndMasks | None, dict[str, Any]]:
        output_dict = self.encoder(x, patch_size=patch_size)  # type: ignore[call-arg]
        token_norm_stats = output_dict.pop("token_norm_stats", None)
        latent, latent_projected_and_pooled, decoder_kwargs = unpack_encoder_output(output_dict)
        extra_metrics = {"token_norm_stats": token_norm_stats} if token_norm_stats else {}
        reconstructed = None
        if self.reconstructor is not None:
            reconstructed = self.reconstructor(latent, x.timestamps, patch_size)  # type: ignore[attr-defined]
        decoded = self.decoder(latent, timestamps=x.timestamps, patch_size=patch_size, **decoder_kwargs)  # type: ignore[call-arg]
        return latent, decoded, latent_projected_and_pooled, reconstructed, extra_metrics


# -----------------------------------------------------------------------------
# Model size configs and OlmoEarthPretrain_v1 wrapper
# -----------------------------------------------------------------------------

MODEL_SIZE_CONFIGS = {
    "nano_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 128,
        "decoder_embedding_size": 128,
        "encoder_depth": 4,
        "encoder_num_heads": 8,
        "decoder_num_heads": 8,
        "mlp_ratio": 4.0,
    },
    "tiny_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 3,
        "decoder_num_heads": 3,
        "mlp_ratio": 4.0,
    },
    "base_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "large_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 1024,
        "decoder_embedding_size": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
    },
}

DEFAULT_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
]


class OlmoEarthPretrain_v1(nn.Module):
    """OlmoEarth Pretrain v1 model.

    Initializes from model size (nano, tiny, base, large). Weights can be loaded
    via the builder with weights=OlmoEarthPretrainV1_Weights.*.
    """

    def __init__(
        self,
        model_size: str = "nano",
        supported_modality_names: list[str] | None = None,
        max_patch_size: int = 8,
        max_sequence_length: int = 12,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        config_key = f"{model_size}_shallow_decoder"
        if config_key not in MODEL_SIZE_CONFIGS:
            raise ValueError(f"Invalid model_size: {model_size}. Must be one of nano, tiny, base, large.")
        if supported_modality_names is None:
            supported_modality_names = DEFAULT_MODALITIES
        cfg = MODEL_SIZE_CONFIGS[config_key]
        encoder_config = EncoderConfig(
            embedding_size=cfg["encoder_embedding_size"],
            num_heads=cfg["encoder_num_heads"],
            depth=cfg["encoder_depth"],
            mlp_ratio=cfg["mlp_ratio"],
            supported_modality_names=supported_modality_names,
            max_patch_size=max_patch_size,
            drop_path=drop_path,
            max_sequence_length=max_sequence_length,
        )
        decoder_config = PredictorConfig(
            encoder_embedding_size=cfg["encoder_embedding_size"],
            decoder_embedding_size=cfg["decoder_embedding_size"],
            depth=cfg["decoder_depth"],
            mlp_ratio=cfg["mlp_ratio"],
            num_heads=cfg["decoder_num_heads"],
            supported_modality_names=supported_modality_names,
            max_sequence_length=max_sequence_length,
        )
        model_config = LatentMIMConfig(encoder_config=encoder_config, decoder_config=decoder_config)
        self.model = model_config.build()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# -----------------------------------------------------------------------------
# Normalizer (data preprocessing)
# -----------------------------------------------------------------------------

def _load_olmoearth_norm_config() -> dict[str, dict]:
    """Load computed normalization config from package data."""
    with files("torchgeo").joinpath("models", "olmoearth_computed_norm.json").open() as f:
        return json.load(f)


class Normalizer:
    """Normalize modality data using pre-computed mean/std (e.g. for OlmoEarth v1)."""

    def __init__(self, std_multiplier: float = 2.0) -> None:
        self.std_multiplier = std_multiplier
        self.norm_config = _load_olmoearth_norm_config()

    def normalize(self, modality: ModalitySpec, data: np.ndarray) -> np.ndarray:
        modality_bands = modality.band_order
        modality_norm = self.norm_config.get(modality.name, {})
        mean_vals = []
        std_vals = []
        for band in modality_bands:
            if band not in modality_norm:
                raise ValueError(
                    f"Band '{band}' not in norm config for '{modality.name}'. "
                    f"Available: {list(modality_norm.keys())}"
                )
            mean_vals.append(modality_norm[band]["mean"])
            std_vals.append(modality_norm[band]["std"])
        min_vals = np.array(mean_vals) - self.std_multiplier * np.array(std_vals)
        max_vals = np.array(mean_vals) + self.std_multiplier * np.array(std_vals)
        return (data - min_vals) / (max_vals - min_vals)


# -----------------------------------------------------------------------------
# Weights and builder (TorchGeo API)
# -----------------------------------------------------------------------------

class OlmoEarthPretrainV1_Weights(WeightsEnum):  # type: ignore[misc]
    """OlmoEarth v1 pre-trained weights from Hugging Face (allenai/OlmoEarth-v1-*)."""

    NANO = Weights(
        url="https://huggingface.co/allenai/OlmoEarth-v1-Nano/resolve/main/weights.pth",
        meta={"model_size": "nano", "repo": "allenai/OlmoEarth-v1-Nano"},
    )
    TINY = Weights(
        url="https://huggingface.co/allenai/OlmoEarth-v1-Tiny/resolve/main/weights.pth",
        meta={"model_size": "tiny", "repo": "allenai/OlmoEarth-v1-Tiny"},
    )
    BASE = Weights(
        url="https://huggingface.co/allenai/OlmoEarth-v1-Base/resolve/main/weights.pth",
        meta={"model_size": "base", "repo": "allenai/OlmoEarth-v1-Base"},
    )
    LARGE = Weights(
        url="https://huggingface.co/allenai/OlmoEarth-v1-Large/resolve/main/weights.pth",
        meta={"model_size": "large", "repo": "allenai/OlmoEarth-v1-Large"},
    )


def olmoearth_pretrain_v1(
    weights: Optional[OlmoEarthPretrainV1_Weights] = None,
    **kwargs: Any,
) -> OlmoEarthPretrain_v1:
    """OlmoEarth Pretrain v1 model.

    Args:
        weights: Pre-trained weights. If None, model is randomly initialized.
        **kwargs: Passed to OlmoEarthPretrain_v1 (e.g. model_size, max_patch_size).

    Returns:
        OlmoEarthPretrain_v1 instance.
    """
    model_size = kwargs.pop("model_size", "nano")
    if weights is not None:
        model_size = weights.meta.get("model_size", model_size)
        kwargs["model_size"] = model_size
    model = OlmoEarthPretrain_v1(model_size=model_size, **kwargs)
    if weights is not None:
        state_dict = weights.get_state_dict(progress=True)
        if not any(k.startswith("model.") for k in state_dict):
            state_dict = {f"model.{k}": v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    return model


__all__ = [
    "OlmoEarthPretrain_v1",
    "OlmoEarthPretrainV1_Weights",
    "olmoearth_pretrain_v1",
    "Normalizer",
    "Modality",
    "ModalitySpec",
    "MaskedOlmoEarthSample",
    "MaskValue",
]
