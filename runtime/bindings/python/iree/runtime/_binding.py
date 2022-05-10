# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Imports the native runtime library.

All code in the runtime should use runtime imports via this module, which
locates the actual _runtime module based on environment configuration.
Currently, we only bundle a single runtime library, but in the future, this
will let us dynamically switch between instrumented, debug, etc by changing
the way this trampoline functions.
"""

import sys

from iree import _runtime
sys.modules[__name__] = _runtime
