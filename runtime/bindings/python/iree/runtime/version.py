# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Trampoline to the generated version.py.

The generated version.py comes from the selected _runtime_libs
package.
"""

from . import _binding

PACKAGE_SUFFIX = _binding.version.PACKAGE_SUFFIX if _binding.version else ""
VERSION = _binding.version.VERSION if _binding.version else ""
REVISIONS = _binding.version.REVISIONS if _binding.version else {}
