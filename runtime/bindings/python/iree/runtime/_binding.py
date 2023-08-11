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

import os
import sys
import warnings

# Detect which runtime variant we are using.
variant = os.getenv("IREE_PY_RUNTIME", "default")
if variant == "default":
    import iree._runtime_libs as _libs
    try:
        import iree._runtime_libs.version as _version
    except ModuleNotFoundError:
        _version = None
    from iree._runtime_libs import _runtime
elif variant == "tracy":
    try:
        import iree._runtime_libs_tracy as _libs
        from iree._runtime_libs_tracy import _runtime
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "IREE Tracy runtime requested via IREE_PY_RUNTIME but it is not "
            "enabled in this build"
        ) from e
    try:
        import iree._runtime_libs.version as _version
    except ModuleNotFoundError:
        _version = None
    print("-- Using Tracy runtime (IREE_PY_RUNTIME=tracy)", file=sys.stderr)
else:
    warnings.warn(
        f"Unknown value for IREE_PY_RUNTIME env var ({variant}): " f"Using default"
    )


_runtime.name = variant
_runtime.library_path = _libs.__path__[0]
_runtime.version = _version
sys.modules[__name__] = _runtime
