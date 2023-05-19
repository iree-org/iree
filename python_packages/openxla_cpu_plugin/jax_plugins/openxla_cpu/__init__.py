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
  # TODO: Make this an API on iree.compiler itself.
  from iree.compiler import _mlir_libs
  from iree.compiler import version
  logger.debug(f"Found installed iree-compiler package {version.VERSION}")
  dylib_basename = "libIREECompiler.so"
  system = platform.system()
  if system == "Darwin":
    dylib_basename = "libIREECompiler.dylib"
  elif system == "Windows":
    dylib_basename = "IREECompiler.dll"

  paths = _mlir_libs.__path__
  for p in paths:
    dylib_path = Path(p) / dylib_basename
    if dylib_path.exists():
      logger.debug(f"Found --iree-compiler-dylib={dylib_path}")
      return dylib_path
  raise ValueError(f"Could not find {dylib_basename} in {paths}")


def initialize():
  path = Path(__file__).resolve().parent / "pjrt_plugin_iree_cpu.so"
  if not path.exists():
    logger.warning(
        f"WARNING: Native library {path} does not exist. "
        f"This most likely indicates an issue with how {__package__} "
        f"was built or installed.")
  xb.register_plugin("iree_cpu",
                     priority=500,
                     library_path=str(path),
                     options={
                       "COMPILER_LIB_PATH": str(probe_iree_compiler_dylib()),
                     })
