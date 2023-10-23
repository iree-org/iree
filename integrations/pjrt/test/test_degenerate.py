# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import jax

ones_splat =  jax.numpy.ones((3, 4))
print(ones_splat)

twos = ones_splat + ones_splat
print(twos)

ones_degenerate = jax.numpy.ones((4, 0))
twos_degenerate = ones_degenerate + ones_degenerate
print(twos_degenerate)
