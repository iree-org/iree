# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The logical namespace the compiler's llvm/mlir symbols are renamed into."""

# Any identifier no real namespace uses will do; the 18 is arbitrary and tracks
# nothing. A plugin resolves against these names, so changing it invalidates
# every plugin already built: bump it only to force that. CMake takes the same
# value from IREE_COMPILER_ABI_PREFIX.
IREE_COMPILER_ABI_PREFIX = "IREE18"
