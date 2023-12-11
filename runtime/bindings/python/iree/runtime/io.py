# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Sequence, Union

import array
from functools import reduce
import numpy
from os import PathLike
from pathlib import Path

from ._binding import ParameterIndex

__all__ = [
    "SplatValue",
    "save_archive_file",
]


class SplatValue:
    def __init__(
        self, pattern: Union[array.array, numpy.ndarray], count: Union[Sequence[int], int]
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
