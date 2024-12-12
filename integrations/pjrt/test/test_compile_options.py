# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import partial
import jax.numpy as jnp
from jax import jit

a = jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9])


@partial(jit, compiler_options={"iree-scheduling-dump-statistics-format": "csv"})
def f(a, b):
    return a + b


print(f(a, a))
