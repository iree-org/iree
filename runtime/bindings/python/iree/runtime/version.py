# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Trampoline to the generated version.py.

The generated version.py comes from the selected _runtime_libs
package.
"""

from .._runtime_libs.version import *
