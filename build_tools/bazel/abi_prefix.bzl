# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The logical namespace the compiler's llvm/mlir symbols are renamed into."""

# This is a symbol namespace, not a compatibility version; 18 has no version
# meaning. Keep it stable across plugin API changes. CMake uses the same default
# for IREE_COMPILER_ABI_PREFIX and exports the host's value to plugin builds.
IREE_COMPILER_ABI_PREFIX = "IREE18"
