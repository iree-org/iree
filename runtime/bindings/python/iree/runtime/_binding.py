# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Imports the native runtime library.

All code in the runtime should use runtime imports via this module, which
locates the actual _runtime module based on environment configuration.

TODO: We could rename this to _runtime since it a trampoline for
the _runtime native extension module we load from elsehwhere.
"""

import sys

# "_runtime" is the native extension within the _runtime_libs package
# selected.
from .._runtime.libs import _runtime

sys.modules[__name__] = _runtime
