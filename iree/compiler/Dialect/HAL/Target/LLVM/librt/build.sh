# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
OUT="${SCRIPT_DIR}/bin"
SRC="${SCRIPT_DIR}/src"
LL_FILE="${OUT}/librt.ll"
BC_FILE="${OUT}/librt.bc"

# Generate an LLVM IR assembly listing so we can easily read the file.
# This is not checked in or used by the compiler.
clang \
    -target wasm32 \
    -std=c17 \
    -O2 \
    -Xclang -disable-llvm-passes \
    -fno-ident \
    -fvisibility=hidden \
    -nostdinc \
    -g0 \
    -S \
    -emit-llvm \
    -fno-verbose-asm \
    -fdiscard-value-names \
    -o "${LL_FILE}" \
    -c \
    "${SRC}/libm.c"

# Clang adds a bunch of bad attributes and host-specific information that we
# don't want (so we get at least somewhat deterministic builds).
sed -i 's/^;.*$//' ${LL_FILE}
sed -i 's/^source_filename.*$//' ${LL_FILE}
sed -i 's/^target datalayout.*$//' ${LL_FILE}
sed -i 's/^target triple.*$//' ${LL_FILE}
sed -i 's/^\(attributes #[0-9]* = {\).*$/\1 inlinehint }/' ${LL_FILE}

# Generate a binary bitcode file embedded into the compiler binary.
# NOTE: we do this from stdin so that the filename on the user's system is not
# embedded in the bitcode file (making it non-deterministic).
cat ${LL_FILE} | llvm-as -o=${BC_FILE}
