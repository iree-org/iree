# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import jax

a = jax.numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9])

print(a + a)
