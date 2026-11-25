# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from iree.runtime import ParameterIndex, ParameterIndexEntry

from iree.importers.torch.support.tensor_traits import (
    ExternalTensorTrait,
)

__all__ = [
    "externalize_module_parameters",
    "save_module_parameters",
    "ParameterArchiveEntry",
    "ParameterArchive",
    "ParameterArchiveBuilder",
]


def _new_parameter_index() -> ParameterIndex:
    from iree.runtime import ParameterIndex

    return ParameterIndex()


################################################################################
# Parameter externalization
################################################################################


def externalize_module_parameters(
    module: nn.Module, *, external_scope: str = "", prefix: str = ""
):
    """Externalizes parameters and persistent buffers in a module by name."""

    for tensor_name, tensor in _yield_saveable_tensors(module, prefix=prefix):
        trait = ExternalTensorTrait(
            external_scope=external_scope, external_name=tensor_name
        )
        trait.set(tensor)


################################################################################
# Metadata
################################################################################

_dtype_to_name: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


# Deal with datatypes not yet added in all versions of Torch.
def _add_optional_dtype(name: str):
    try:
        dtype = getattr(torch, name)
    except AttributeError:
        return
    _dtype_to_name[dtype] = name


_add_optional_dtype("float8_e4m3fn")
_add_optional_dtype("float8_e4m3fnuz")
_add_optional_dtype("float8_e5m2")
_add_optional_dtype("float8_e5m2fnuz")
_add_optional_dtype("uint16")
_add_optional_dtype("uint32")
_add_optional_dtype("uint64")


_name_to_dtype: dict[str, torch.dtype] = {v: k for k, v in _dtype_to_name.items()}

_metadata_prefix = "PYTORCH:"


def _make_tensor_metadata(t: torch.Tensor) -> str:
    """Makes a tensor metadata blob that can be used to reconstruct the tensor."""
    dtype = t.dtype
    try:
        dtype_name = _dtype_to_name[dtype]
    except KeyError:
        dtype_name = "unknown"
        warnings.warn(
            f"Unknown dtype saving params: {dtype} (missing entry in params._dtype_to_name)"
        )
    dtype_desc = {
        "class_name": type(dtype).__name__,
        "is_complex": dtype.is_complex,
        "is_floating_point": dtype.is_floating_point,
        "is_signed": dtype.is_signed,
        "itemsize": dtype.itemsize,
    }
    d = {
        "type": "Tensor",
        "dtype": dtype_name,
        "shape": list(t.shape),
        "dtype_desc": dtype_desc,
    }
    encoded = f"{_metadata_prefix}{json.dumps(d)}"
    return encoded


################################################################################
# Parameter archives save/load
################################################################################


def save_module_parameters(
    file_path: str | Path, module: nn.Module, *, prefix: str = ""
):
    """One shot save of parameters and persistent buffers on a module.

    More options are available by using a ParameterArchiveBuilder.
    """
    builder = ParameterArchiveBuilder()
    builder.add_module(module, prefix=prefix)
    builder.save(file_path)


class ParameterArchiveEntry:
    """Wraps a raw ParameterIndexEntry with additional helpers."""

    def __init__(self, raw: ParameterIndexEntry):
        self.raw = raw

    @property
    def key(self) -> str:
        return self.raw.key

    def as_flat_tensor(self) -> torch.Tensor:
        """Accesses the contents as a uint8 flat tensor.

        If it is a splat, then the tensor will be a view of the splat pattern.

        Raises a ValueError on unsupported entries.
        """
        if self.raw.is_file:
            wrapper = np.asarray(self.raw.file_view)
        elif self.raw.is_splat:
            wrapper = np.array(self.raw.splat_pattern)
        else:
            raise ValueError(f"Unsupported ParameterIndexEntry: {self.raw}")

        return torch.from_numpy(wrapper)

    def as_tensor(self) -> torch.Tensor:
        """Returns a tensor viewed with appropriate shape/dtype from metadata.

        Raises a ValueError if unsupported.
        """
        # Decode metadata.
        metadata = self.raw.metadata.decode()
        if not metadata.startswith(_metadata_prefix):
            raise ValueError(
                f"No metadata for parameter entry {self.key}: Cannot convert to tensor"
            )
        metadata = metadata[len(_metadata_prefix) :]
        d = json.loads(metadata)
        try:
            type_name = d["type"]
            if d["type"] != "Tensor":
                raise ValueError(
                    f"Metadata for parameter entry {self.key} is not a Tensor ('{type_name}')"
                )
            dtype_name = d["dtype"]
            shape = d["shape"]
        except KeyError as e:
            raise ValueError(f"Bad metadata for parameter entry {self.key}") from e

        # Unpack/validate.
        try:
            dtype = _name_to_dtype[dtype_name]
        except KeyError:
            raise ValueError(f"Unknown dtype name '{dtype_name}'")
        try:
            shape = [int(d) for d in shape]
        except ValueError as e:
            raise ValueError(f"Illegal shape for parameter entry {self.key}") from e

        t = self.as_flat_tensor()
        return t.view(dtype=dtype).view(shape)

    def __repr__(self):
        return f"ParameterArchiveEntry({self.raw}, metadata={self.raw.metadata})"


class ParameterArchive:
    """Allows access to a parameter archive as CPU tensors.

    TODO: Add more helpers for reading tensors once we get upstream versions that
    have that integrated.
    """

    def __init__(
        self,
        file_path: str | Path | None = None,
        *,
        mmap: bool = True,
        readable: bool = True,
        writable: bool = False,
    ):
        self._index = _new_parameter_index()
        if file_path is not None:
            self.load(file_path, mmap=mmap, readable=readable, writable=writable)

    def load(
        self,
        file_path: str | Path,
        *,
        mmap: bool = True,
        readable: bool = True,
        writable: bool = False,
    ):
        """Loads index entries from a file adding them to the in-memory archive."""
        self._index.load(
            str(file_path),
            mmap=mmap,
            readable=readable,
            writable=writable,
        )

    @property
    def index(self) -> ParameterIndex:
        return self._index

    def items(self) -> list[tuple[str, ParameterArchiveEntry]]:
        """Returns the items in the archive.

        Note that there can be duplicates if the archive was constructed that way.
        """
        return [(k, ParameterArchiveEntry(v)) for k, v in self._index.items()]

    def __repr__(self):
        return repr(self._index)


class ParameterArchiveBuilder:
    """Helper for building parameter archives from live modules."""

    def __init__(self):
        self._index = _new_parameter_index()

    @property
    def index(self) -> ParameterIndex:
        return self._index

    def save(self, file_path: str | Path):
        """Saves the archive."""
        self._index.create_archive_file(str(file_path))

    def add_tensor(self, name: str, tensor: torch.Tensor):
        """Adds an named tensor to the archive."""
        metadata = _make_tensor_metadata(tensor)
        # 0d arrays are special in both torch/numpy in different ways that makes
        # it hard to reliably get a memory view of their contents. Since we
        # know that 0d is always small, we just force a copy when in numpy
        # land and that seems to get it on the happy path.
        # See: https://github.com/iree-org/iree-turbine/issues/29
        if len(tensor.shape) == 0:
            flat_array_np = tensor.detach().cpu().numpy().copy()
            host_array = flat_array_np
        else:
            flat_array = tensor.detach().flatten().contiguous().cpu().view(torch.uint8)
            host_array = flat_array.numpy()
        self._index.add_buffer(name, host_array, metadata=metadata)

    def add_module(self, module: nn.Module, *, prefix: str = ""):
        """Adds all parameters and persistent buffers from a module hierarchy."""
        for name, t in _yield_saveable_tensors(module, prefix=prefix):
            self.add_tensor(name, t)

    def add_blob(self, key: str, blob):
        """Adds a raw blob to the index.

        The blob must be interpretable as a buffer.
        """
        self._index.add_buffer(key, blob)


def _yield_saveable_tensors(
    module: nn.Module, *, prefix: str = ""
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yields tuple of name/tensor for all saveable tensors in a module.

    This includes parameters and persistent buffers.
    """
    memo: set[str] = set()
    for sub_name, sub_module in module.named_modules(prefix=prefix):
        state_dict = sub_module.state_dict()
        for param_name, param in sub_module.named_parameters(recurse=False):
            full_param_name = f"{sub_name}.{param_name}" if sub_name else param_name
            if full_param_name in memo:
                continue
            memo.add(full_param_name)
            yield full_param_name, param
        for buffer_name, buffer in sub_module.named_buffers(recurse=False):
            full_buffer_name = f"{sub_name}.{buffer_name}" if sub_name else buffer_name
            if full_buffer_name in memo:
                continue
            memo.add(full_buffer_name)
            if buffer_name not in state_dict:
                # Non persistent
                continue
            yield full_buffer_name, buffer
