# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -x
set -e

SCRIPT_DIR="$(realpath `dirname $0`)"
OUT="${SCRIPT_DIR?}/bin"
SRC="${SCRIPT_DIR?}/src"
LL_FILE="${OUT}/librt.ll"
BC_FILE="${OUT}/librt.bc"
BC64_FILE="${OUT}/librt64.bc"
IREE_SRC_DIR="$(git rev-parse --show-toplevel)"
IREE_BUILD_DIR="${IREE_BUILD_DIR:-${IREE_SRC_DIR?}/../build}"
CLANG="${CLANG:-$(which clang)}"
CLANGXX="${CLANGXX:-$(which clang++)}"
LLVM_AS="${LLVM_AS:-${IREE_BUILD_DIR}/third_party/llvm-project/llvm/bin/llvm-as}"
LLVM_LINK="${LLVM_DIS:-${IREE_BUILD_DIR}/third_party/llvm-project/llvm/bin/llvm-link}"
OPT="${OPT:-${IREE_BUILD_DIR}/third_party/llvm-project/llvm/bin/opt}"

function generate_librt_from_musl ()
{
  ## Generate the LLVM IR assembly for the required math files from muls
  MUSL_DIR=${IREE_SRC_DIR?}/third_party/musl

  ## Generate IR with 32-bit target. This is linked by default.
  cd ${MUSL_DIR}
  rm -rf obj/
  CC=${CLANG?} CXX=${CLANGXX?} ./configure
  MUSL_DIR=${MUSL_DIR} make -f $1 iree
  MUSL_LL_FILES=`find obj/ -name *.ll`
  cp ${MUSL_LL_FILES?} ${OUT}
  rm ${MUSL_LL_FILES?}
  cd ${SCRIPT_DIR?}

  ALL_LL_FILES=`find ${OUT} -name *.ll`

  cd ${OUT}
  git restore $2
  for file in ${ALL_LL_FILES}
  do
    ${OPT?} ${file} -O3 -S -o ${file}.opt.ll
    # Clang adds a bunch of bad attributes and host-specific information that we
    # don't want (so we get at least somewhat deterministic builds).
    sed -i 's/^;.*$//' ${file}.opt.ll
    sed -i 's/^source_filename.*$//' ${file}.opt.ll
    sed -i 's/^target datalayout.*$//' ${file}.opt.ll
    sed -i 's/^target triple.*$//' ${file}.opt.ll
    sed -i 's/^\(attributes #[0-9]* = {\).*$/\1 inlinehint }/' ${file}.opt.ll

    # Generate a binary bitcode file embedded into the compiler binary.
    # NOTE: we do this from stdin so that the filename on the user's system is not
    # embedded in the bitcode file (making it non-deterministic).
    cat ${file}.opt.ll | ${LLVM_AS?} -o=${file}.opt.ll.bc
    rm ${file}.opt.ll
  done
  rm ${ALL_LL_FILES}

  ALL_BC_FILES=`ls *.ll.bc`
  ${LLVM_LINK?} ${ALL_BC_FILES} -o $2
  rm ${ALL_BC_FILES}
}

# Generate an LLVM IR assembly listing so we can easily read the file.
# This is not checked in or used by the compiler.
${CLANG?} \
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

### Generate the LLVM IR assembly for the required math files from musl

## Generate the librt functions with `wasm32` target to be used on all
## backends
generate_librt_from_musl ${SCRIPT_DIR?}/Makefile_musl.iree ${BC_FILE}

## Generate the librty functions with `wasm64` target to be used on 64-bit
## backends
generate_librt_from_musl ${SCRIPT_DIR?}/Makefile_musl_64.iree ${BC64_FILE}
