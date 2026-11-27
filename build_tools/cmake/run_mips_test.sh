#!/bin/bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Wrapper to run a cross-compiled MIPS artifact, under QEMU when available.

set -x
set -e

# Export QEMU_LD_PREFIX from the -L <sysroot> argument. The -L flag only sets the
# sysroot for this top-level QEMU; child processes a test spawns are re-executed
# by binfmt_misc with a bare QEMU that reads the sysroot only from this env var.
# Deriving it from -L keeps them in sync and overrides any stale value in the env.
_args=("$@")
for ((_i = 0; _i < ${#_args[@]}; ++_i)); do
  if [[ "${_args[_i]}" == "-L" ]]; then
    export QEMU_LD_PREFIX="${_args[_i + 1]}"
    break
  fi
done

if [[ ! -z "${QEMU_BIN}" ]] && [[ ! -z "${QEMU_CPU_FLAGS}" ]]; then
  "${QEMU_BIN}" "-cpu" "${QEMU_CPU_FLAGS}" "$@"
else
  "$@"
fi
