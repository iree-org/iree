# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from pathlib import Path
import platform
import sys

import jax._src.xla_bridge as xb

logger = logging.getLogger(__name__)


def probe_iree_compiler_dylib() -> str:
    """Probes an installed iree.compiler for the compiler dylib."""
    # TODO: Move this out of the ctypes API initialization.
    from iree.compiler.api import ctypes_dl

    return ctypes_dl._probe_iree_compiler_dylib()


def initialize():
    import iree._pjrt_libs.cpu as lib_package

    path = Path(lib_package.__file__).resolve().parent / "pjrt_plugin_iree_cpu.so"
    if not path.exists():
        logger.warning(
            f"WARNING: Native library {path} does not exist. "
            f"This most likely indicates an issue with how {__package__} "
            f"was built or installed."
        )
    xb.register_plugin(
        "iree_cpu",
        priority=500,
        library_path=str(path),
        options={
            "COMPILER_LIB_PATH": str(probe_iree_compiler_dylib()),
        },
    )
