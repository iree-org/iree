# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Example command line:
#   LLVM_AS=/usr/bin/llvm-as \
#   CLANG=/usr/bin/clang-13 \
#   ./iree/builtins/device/bin/build.sh

set -x
set -e

CLANG="${CLANG:-clang}"
# TODO(benvanik): figure out how to get this path from clang itself.
CLANG_INCLUDE="${CLANG_INCLUDE:-/usr/lib/llvm-13/lib/clang/13.0.0/include/}"
IREE_SRC_DIR="$(git rev-parse --show-toplevel)"
IREE_BUILD_DIR="${IREE_BUILD_DIR:-${IREE_SRC_DIR?}/../build}"
LLVM_AS="${LLVM_AS:-${IREE_BUILD_DIR}/llvm-project/bin/llvm-as}"

SCRIPT_DIR="$(realpath `dirname $0`)"
OUT="${SCRIPT_DIR?}/"
SRC="${SCRIPT_DIR?}/.."

function make_arch_bc {
  local ARCH=$1
  local FEATURES=$2
  local SOURCE_FILE=$3
  local FILE_BASENAME="${OUT}/libdevice_${ARCH}_${FEATURES}"

  # Generate an LLVM IR assembly listing so we can easily read the file.
  # This is not checked in or used by the compiler.
  ${CLANG?} \
      "${@:4}" \
      -isystem "${CLANG_INCLUDE?}" \
      -std=c17 \
      -O3 \
      -fno-ident \
      -fvisibility=hidden \
      -nostdinc \
      -S \
      -emit-llvm \
      -fdiscard-value-names \
      -DIREE_DEVICE_STANDALONE \
      -o "${FILE_BASENAME}.ll" \
      -c \
      "${SRC}/${SOURCE_FILE}"

  # Clang adds a bunch of bad attributes and host-specific information that we
  # don't want (so we get at least somewhat deterministic builds).
  sed -i 's/^;.*$//' "${FILE_BASENAME}.ll"
  sed -i 's/^source_filename.*$//' "${FILE_BASENAME}.ll"
  sed -i 's/^target datalayout.*$//' "${FILE_BASENAME}.ll"
  sed -i 's/^target triple.*$//' "${FILE_BASENAME}.ll"
  sed -i 's/^\(attributes #[0-9]* = {\).*$/\1 inlinehint }/' "${FILE_BASENAME}.ll"

  # Generate a binary bitcode file embedded into the compiler binary.
  # NOTE: we do this from stdin so that the filename on the user's system is not
  # embedded in the bitcode file (making it non-deterministic).
  cat "${FILE_BASENAME}.ll" | ${LLVM_AS} -opaque-pointers=0 -o="${FILE_BASENAME}.bc"
}

make_arch_bc "wasm32" "generic" "device_generic.c" \
    --target=wasm32
make_arch_bc "wasm64" "generic" "device_generic.c" \
    --target=wasm64
