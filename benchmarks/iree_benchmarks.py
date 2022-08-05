## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List

from definitions.common import ArchitectureInfo, DeviceArchitecture, DevicePlatform
from definitions.iree import IreeBenchmarkCompileSpec, IreeBenchmarkRunSpec, IreeCompileTarget, IreeCompileConfig, IreeRuntimeConfig, IreeRuntimeDriver, IreeRuntimeLoader, IreeTargetBackend

X86_64_CASCADELAKE_CPU_TARGET = IreeCompileTarget(
    target_architecture=DeviceArchitecture.X86_64_CASCADELAKE,
    target_backend=IreeTargetBackend.LLVM_CPU)


def get_iree_compile_target_flags(target: IreeCompileTarget):
  arch_info: ArchitectureInfo = target.arch.value
  flags = []
  if arch_info.architecture == "x86_64":
    flags.append(
        f"--iree-llvm-target-triple=x86_64-unknown-{target.platform.value}",
        f"--iree-llvm-target-cpu={arch_info.microarchitecture.lower()}")
  else:
    raise ValueError("Unsupported architecture.")
  return flags


def build_iree_compile_config(id: str, input_type: str,
                              compile_target: IreeCompileTarget,
                              tags: List[str]):
  compile_flags = [
      "--iree-mlir-to-vm-bytecode-module", f"--iree-input-type={input_type}",
      f"--iree-hal-target-backends={compile_target.target_backend.value}"
  ]
  compile_flags.extend(get_iree_compile_target_flags(compile_target))
  return IreeCompileConfig(id=id,
                           compile_targets=[compile_target],
                           tags=tags,
                           compile_flags=compile_flags)
