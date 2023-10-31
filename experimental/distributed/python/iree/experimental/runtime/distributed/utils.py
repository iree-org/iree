# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from numpy.typing import ArrayLike
from typing import List
import numpy as np


def read_numpy_arrays_from_file(filepath: str) -> List[ArrayLike]:
    res = []
    with open(filepath, "rb") as f:
        while True:
            try:
                res.append(np.load(f))
            except EOFError:
                break
    return res


def write_numpy_arrays_to_file(filepath: str, arrays: List[ArrayLike]):
    with open(filepath, "wb") as f:
        for arr in arrays:
            np.save(f, np.asarray(arr), allow_pickle=False)
