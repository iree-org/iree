# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np

# zero bytes
with open("empty.npy", "wb") as f:
    f.flush()

# single array
with open("single.npy", "wb") as f:
    np.save(f, np.array([1.1, 2.2, 3.3], dtype=np.float32))

# multiple arrays
with open("multiple.npy", "wb") as f:
    np.save(f, np.array([1.1, 2.2, 3.3], dtype=np.float32))
    np.save(f, np.array([[0, 1], [2, 3]], dtype=np.int32))
    np.save(f, np.array(42, dtype=np.int32))

# arrays of various shapes
with open("array_shapes.npy", "wb") as f:
    np.save(f, np.array(1, dtype=np.int8))
    np.save(f, np.array([], dtype=np.int8))
    np.save(f, np.array([1], dtype=np.int8))
    np.save(f, np.array([[1], [2]], dtype=np.int8))
    np.save(f, np.array([[0], [1], [2], [3], [4], [5], [6], [7]], dtype=np.int8))
    np.save(f, np.array([[1, 2], [3, 4]], dtype=np.int8))
    np.save(f, np.array([[[1], [2]], [[3], [4]]], dtype=np.int8))

# arrays of various types
with open("array_types.npy", "wb") as f:
    np.save(f, np.array([True, False], dtype=np.bool_))
    np.save(f, np.array([-1, 1], dtype=np.int8))
    np.save(f, np.array([-20000, 20000], dtype=np.int16))
    np.save(f, np.array([-2000000, 2000000], dtype=np.int32))
    np.save(f, np.array([-20000000000, 20000000000], dtype=np.int64))
    np.save(f, np.array([1, 255], dtype=np.uint8))
    np.save(f, np.array([1, 65535], dtype=np.uint16))
    np.save(f, np.array([1, 4294967295], dtype=np.uint32))
    np.save(f, np.array([1, 18446744073709551615], dtype=np.uint64))
    np.save(f, np.array([-1.1, 1.1], dtype=np.float16))
    np.save(f, np.array([-1.1, 1.1], dtype=np.float32))
    np.save(f, np.array([-1.1, 1.1], dtype=np.float64))
    np.save(f, np.array([1 + 5j, 2 + 6j], dtype=np.complex64))
    np.save(f, np.array([1 + 5j, 2 + 6j], dtype=np.complex128))
