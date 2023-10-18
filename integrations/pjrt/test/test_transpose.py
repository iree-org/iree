# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import jax.numpy as jnp
import numpy as np

a = np.arange(15, dtype=np.float32).reshape(3, 5)
a = a.T
transpose = jnp.array(a)
transpose_copy = jnp.array(a.copy())

assert (transpose == transpose_copy).all()
