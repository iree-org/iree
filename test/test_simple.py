# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import jax

# Do once and print.
a = jax.numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9])
b = a
for i in range(100):
  b = jax.numpy.asarray([i]) * a + b
print(b)

# Do once and print.
a = jax.numpy.asarray([10, 20, 30, 40, 50, 60, 70, 80, 90])
b = a
for i in range(100):
  b = jax.numpy.asarray([i]) * a + b
print(b)
