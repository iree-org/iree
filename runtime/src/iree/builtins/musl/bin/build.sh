# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Example command line:
#   LLVM_AS=/usr/bin/llvm-as \
#   LLVM_LINK=/usr/bin/llvm-link \
#   CLANG=/usr/bin/clang-13 \
#   ./iree/builtins/musl/bin/build.sh

set -x
set -e

CLANG="${CLANG:-clang}"
CLANGXX="${CLANGXX:-$(which clang++)}"
LLVM_AS="${LLVM_AS:-${IREE_BUILD_DIR}/llvm-project/bin/llvm-as}"
LLVM_LINK="${LLVM_LINK:-${IREE_BUILD_DIR}/llvm-project/bin/llvm-link}"
LLVM_OPT="${LLVM_OPT:-${IREE_BUILD_DIR}/llvm-project/bin/opt}"

IREE_SRC_DIR="$(git rev-parse --show-toplevel)"
IREE_BUILD_DIR="${IREE_BUILD_DIR:-${IREE_SRC_DIR?}/../build}"

SCRIPT_DIR="$(realpath `dirname $0`)"
OUT="${SCRIPT_DIR?}/"
SRC="${SCRIPT_DIR?}/.."

function make_arch_bc {
  local ARCH=$1
  local FEATURES=$2
  local FILE_BASENAME="${OUT}/libmusl_${ARCH}_${FEATURES}"
  local MUSL_MAKEFILE="${SCRIPT_DIR?}/../Makefile_${ARCH}.iree"

  # Generate IR with 32-bit target.
  MUSL_DIR=${IREE_SRC_DIR?}/third_party/musl
  cd ${MUSL_DIR}
  rm -rf obj/
  CC=${CLANG?} CXX=${CLANGXX?} ./configure
  MUSL_DIR=${MUSL_DIR} make -f ${MUSL_MAKEFILE} iree
  MUSL_LL_FILES=`find obj/ -name "*.ll"`
  cp ${MUSL_LL_FILES?} ${OUT}
  rm ${MUSL_LL_FILES?}
  cd ${SCRIPT_DIR?}

  ALL_LL_FILES=`find ${OUT} -name "*.ll"`

  cd ${OUT}
  # git restore ${FILE_BASENAME}.bc
  for file in ${ALL_LL_FILES}
  do
    # Run full LLVM optimizations.
    # TODO(benvanik): defer this? Some of these opts may not be portable/safe.
    ${LLVM_OPT?} ${file} -O3 -S -opaque-pointers=0 -o ${file}.opt.ll

    # Clang adds a bunch of bad attributes and host-specific information that we
    # don't want (so we get at least somewhat deterministic builds).
    sed -i 's/^;.*$//' "${file}.opt.ll"
    sed -i 's/^source_filename.*$//' "${file}.opt.ll"
    sed -i 's/^target datalayout.*$//' "${file}.opt.ll"
    sed -i 's/^target triple.*$//' "${file}.opt.ll"
    sed -i 's/^\(attributes #[0-9]* = {\).*$/\1 inlinehint }/' "${file}.opt.ll"

    # Generate a binary bitcode file embedded into the compiler binary.
    # NOTE: we do this from stdin so that the filename on the user's system is not
    # embedded in the bitcode file (making it non-deterministic).
    cat ${file}.opt.ll | ${LLVM_AS?} -opaque-pointers=0 -o=${file}.opt.ll.bc
    rm ${file}.opt.ll
  done
  rm ${ALL_LL_FILES}

  ALL_BC_FILES=`ls *.ll.bc`
  ${LLVM_LINK?} -opaque-pointers=0 ${ALL_BC_FILES} -o ${FILE_BASENAME}.bc
  rm ${ALL_BC_FILES}
}

make_arch_bc "wasm32" "generic" \
    --target=wasm32
make_arch_bc "wasm64" "generic" \
    --target=wasm64
