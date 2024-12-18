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
        pattern: Union[array.array, np.ndarray],
        count: Union[Sequence[int], int],
    ):
        if hasattr(pattern, "shape"):
            shape = pattern.shape
            if not shape:
                total_elements = 1
            else:
                total_elements = reduce(lambda x, y: x * y, pattern.shape)
        else:
            total_elements = len(pattern)
        item_size = pattern.itemsize
        if total_elements != 1:
            raise ValueError(f"SplatValue requires an array of a single element")
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
        dtype = _NAME_TO_DTYPE[dtype_name]
    except KeyError:
        raise ValueError(f"Unknown dtype name '{dtype_name}'")
    try:
        shape = [int(d) for d in shape]
    except ValueError as e:
        raise ValueError(f"Illegal shape for parameter entry {index_entry.key}") from e

    t = parameter_index_entry_as_numpy_flat_ndarray(index_entry)
    return t.view(dtype=dtype).reshape(shape)


_DTYPE_TO_NAME = (
    (np.float16, "float16"),
    (np.float32, "float32"),
    (np.float64, "float64"),
    (np.int32, "int32"),
    (np.int64, "int64"),
    (np.int16, "int16"),
    (np.int8, "int8"),
    (np.uint32, "uint32"),
    (np.uint64, "uint64"),
    (np.uint16, "uint16"),
    (np.uint8, "uint8"),
    (np.bool_, "bool"),
    (np.complex64, "complex64"),
    (np.complex128, "complex128"),
)

_NAME_TO_DTYPE: dict[str, np.dtype] = {
    name: np_dtype for np_dtype, name in _DTYPE_TO_NAME
}


def _map_dtype_to_name(dtype) -> str:
    for match_dtype, dtype_name in _DTYPE_TO_NAME:
        if match_dtype == dtype:
            return dtype_name

    raise KeyError(f"Numpy dtype {dtype} not found.")


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
    dtype_name = _map_dtype_to_name(dtype)
    is_complex = np.issubdtype(dtype, np.complexfloating)
    is_floating_point = np.issubdtype(dtype, np.floating)
    is_signed = np.issubdtype(dtype, np.signedinteger)
    dtype_desc = {
        "class_name": type(dtype).__name__,
        "is_complex": is_complex,
        "is_floating_point": is_floating_point,
        "is_signed": is_signed,
        "itemsize": dtype.itemsize,
    }
    d = {
        "type": "Tensor",
        "dtype": dtype_name,
        "shape": list(t.shape),
        "dtype_desc": dtype_desc,
    }
    encoded = f"{_metadata_version}{_metadata_version_separator}{json.dumps(d)}"
    return encoded
