# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Sequence, Union

import array
from functools import reduce
import json
import numpy as np
from os import PathLike

from ._binding import ParameterIndex, ParameterIndexEntry
from .dtypes import map_name_to_dtype_info, map_dtype_to_dtype_info

__all__ = [
    "parameter_index_add_numpy_ndarray",
    "parameter_index_entry_as_numpy_flat_ndarray",
    "parameter_index_entry_as_numpy_ndarray",
    "SplatValue",
    "save_archive_file",
]


class SplatValue:
    def __init__(
        self,
        pattern: Union[array.array, np.ndarray, np.generic],
        count: Union[Sequence[int], int],
    ):
        if isinstance(pattern, np.ndarray):
            shape = pattern.shape
            if not shape:
                total_elements = 1
            else:
                total_elements = reduce(lambda x, y: x * y, shape)
        elif isinstance(pattern, np.generic):
            total_elements = 1
        else:
            total_elements = len(pattern)
        item_size = pattern.itemsize
        if total_elements != 1:
            raise ValueError("SplatValue requires an array of a single element")
        self.pattern = pattern
        if isinstance(count, int):
            logical_length = count
        elif len(count) == 1:
            logical_length = count[0]
        else:
            logical_length = reduce(lambda x, y: x * y, count)
        self.total_length = logical_length * item_size

    def __repr__(self):
        return f"SplatValue({self.pattern} of {self.total_length})"


def save_archive_file(entries: dict[str, Union[Any, SplatValue]], file_path: PathLike):
    """Creates an IRPA (IREE Parameter Archive) from contents.

    Similar to the safetensors.numpy.save_file function, this takes
    a dictionary of key-value pairs where the value is a buffer. It
    writes a file with the contents.
    """
    index = ParameterIndex()
    for key, value in entries.items():
        if isinstance(value, SplatValue):
            index.add_splat(key, value.pattern, value.total_length)
        else:
            index.add_buffer(key, value)
    index.create_archive_file(str(file_path))


def parameter_index_add_numpy_ndarray(
    index: ParameterIndex, name: str, array: np.ndarray
):
    """Adds an ndarray to the index."""
    metadata = _make_tensor_metadata(array)
    # 0d arrays are special in both torch/numpy in different ways that makes
    # it hard to reliably get a memory view of their contents. Since we
    # know that 0d is always small, we just force a copy when in numpy
    # land and that seems to get it on the happy path.
    # See: https://github.com/iree-org/iree-turbine/issues/29
    if len(array.shape) == 0:
        flat_array = array.copy()
    else:
        flat_array = np.ascontiguousarray(array).view(np.uint8)
    index.add_buffer(name, flat_array, metadata=metadata)


def parameter_index_entry_as_numpy_flat_ndarray(
    index_entry: ParameterIndexEntry,
) -> np.ndarray:
    """Accesses the contents as a uint8 flat tensor.

    If it is a splat, then the tensor will be a view of the splat pattern.

    Raises a ValueError on unsupported entries.
    """
    if index_entry.is_file:
        wrapper = np.array(index_entry.file_view, copy=False)
    elif index_entry.is_splat:
        wrapper = np.array(index_entry.splat_pattern, copy=True)
    else:
        raise ValueError(f"Unsupported ParameterIndexEntry: {index_entry}")

    return wrapper


def parameter_index_entry_as_numpy_ndarray(
    index_entry: ParameterIndexEntry,
) -> np.ndarray:
    """Returns a tensor viewed with appropriate shape/dtype from metadata.

    Raises a ValueError if unsupported.
    """

    # Decode metadata.
    versioned_metadata = index_entry.metadata.decode()
    metadata_parts = versioned_metadata.split(_metadata_version_separator, maxsplit=1)
    if len(metadata_parts) == 1:
        raise ValueError(
            (
                f'Invalid metadata for parameter index entry "{index_entry.key}".'
                f' Expected format version prefix not found in "{metadata_parts[0][:100]}".'
            )
        )
    format_version = metadata_parts[0]
    metadata = metadata_parts[1]
    if (
        format_version != _metadata_version
        and format_version != _metadata_iree_turbine_version
    ):
        raise ValueError(
            (
                f'Unsupported metadata format version "{format_version}" for parameter '
                'index entry "{index_entry.key}": Cannot convert to tensor'
            )
        )
    d = json.loads(metadata)
    try:
        type_name = d["type"]
        if d["type"] != "Tensor":
            raise ValueError(
                f"Metadata for parameter entry {index_entry.key} is not a Tensor ('{type_name}')"
            )
        dtype_name = d["dtype"]
        shape = d["shape"]
    except KeyError as e:
        raise ValueError(f"Bad metadata for parameter entry {index_entry.key}") from e

    # Unpack/validate.
    try:
        dtype = map_name_to_dtype_info(dtype_name).dtype
    except KeyError as e:
        raise ValueError(
            f"Unsupported dtype for parameter entry {index_entry.key}"
        ) from e
    try:
        shape = [int(d) for d in shape]
    except ValueError as e:
        raise ValueError(f"Illegal shape for parameter entry {index_entry.key}") from e

    t = parameter_index_entry_as_numpy_flat_ndarray(index_entry)
    return t.view(dtype=dtype).reshape(shape)


_metadata_version = "TENSORv0"
"""Magic number to identify the format version.
The current version that will be used when adding tensors to a parameter index."""

_metadata_iree_turbine_version = "PYTORCH"
"""There are files created with IREE Turbine that use this prefix.
This is here to maintain the ability to load such files."""

_metadata_version_separator = ":"
"""The separator between the format version and the actual metadata.
The metadata has the following format <format-version><separator><metadata>"""


def _make_tensor_metadata(t: np.ndarray) -> str:
    """Makes a tensor metadata blob that can be used to reconstruct the tensor."""
    dtype = t.dtype
    dtype_info = map_dtype_to_dtype_info(dtype)
    dtype_desc = {
        "class_name": type(dtype).__name__,
        "is_complex": dtype_info.is_complex,
        "is_floating_point": dtype_info.is_floating_point,
        "is_signed": dtype_info.is_signed,
        "itemsize": dtype.itemsize,
    }
    d = {
        "type": "Tensor",
        "dtype": dtype_info.name,
        "shape": list(t.shape),
        "dtype_desc": dtype_desc,
    }
    encoded = f"{_metadata_version}{_metadata_version_separator}{json.dumps(d)}"
    return encoded
