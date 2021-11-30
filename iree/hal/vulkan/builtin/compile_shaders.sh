#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Compiles input .glsl files into output .spv binary files. As these files are
# updated infrequently and their binary sizes are small, we check in both files
# and don't take a hard dependency on the shader compiler tool.
#
# To use, ensure `glslc` is on your PATH (such as by installing the Vulkan SDK
# or builting it from its source at https://github.com/google/shaderc) and run
# the script.

set -e
set -x

BUILTIN_DIR="$(dirname $0)"

glslc \
  -Os -fshader-stage=compute -mfmt=bin \
  ${BUILTIN_DIR}/fill_unaligned.glsl \
  -o ${BUILTIN_DIR}/fill_unaligned.spv
