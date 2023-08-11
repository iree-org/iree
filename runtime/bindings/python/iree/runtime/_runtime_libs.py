# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Trampoline to the right iree._runtime_libs* module."""

import os
import sys
import warnings

variant = os.getenv("IREE_PY_RUNTIME", "default")
if variant == "tracy":
    try:
        import iree._runtime_libs_tracy as _libs
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "IREE Tracy runtime requested via IREE_PY_RUNTIME but it is not "
            "enabled in this build"
        ) from e
    print("-- Using Tracy runtime (IREE_PY_RUNTIME=tracy)", file=sys.stderr)
else:
    if variant != "default":
        warnings.warn(
            f"Unknown value for IREE_PY_RUNTIME env var ({variant}): " f"Using default"
        )
        variant = "default"
    import iree._runtime_libs as _libs

_libs.name = variant
_libs.library_path = _libs.__path__[0]
sys.modules[__name__] = _libs
