# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Helper to, given an MLIR file from a "frontend", produce two variants:
#  1. An .mlirbc with unaltered contents.
#  2. An _stripped.mlir with stripped constants.

set -e
set -x

input_file="$1"
mlirbc_file="${input_file%%.mlir*}.mlirbc"
stripped_file="${input_file%%.mlir*}_stripped.mlir"

echo "Copying to bytecode $input_file -> $mlirbc_file"
iree-ir-tool copy --emit-bytecode -o "$mlirbc_file" "$input_file"

echo "Stripping $mlirbc_file -> $stripped_file"
iree-ir-tool strip-data -o "$stripped_file" "$mlirbc_file"
