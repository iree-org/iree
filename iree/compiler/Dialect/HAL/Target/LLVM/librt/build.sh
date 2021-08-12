# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -x
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

## Generate the LLVM IR assembly for the required math files from muls
IREE_SRC_DIR="$( cd ${SCRIPT_DIR} && cd ../../../../../../../ && pwd)"
MUSL_DIR=${IREE_SRC_DIR}/third_party/musl
cd ${MUSL_DIR}
CC=clang CXX=clang++ ./configure
MUSL_DIR=${MUSL_DIR} make -f ${SCRIPT_DIR}/Makefile_musl.iree iree
MUSL_LL_FILES=`find obj/ -name *.ll`
cp ${MUSL_LL_FILES} ${OUT}
rm ${MUSL_LL_FILES}
cd ${SCRIPT_DIR}

ALL_LL_FILES=`find ${OUT} -name *.ll`

cd ${OUT}
git restore ${BC_FILE}
for file in ${ALL_LL_FILES}
do
  opt ${file} -O3 -S -o ${file}.opt.ll
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
  cat ${file}.opt.ll | llvm-as -o=${file}.opt.ll.bc
done

ALL_BC_FILES=`ls *.ll.bc`
llvm-link ${ALL_BC_FILES} -o ${BC_FILE}
rm ${ALL_BC_FILES}
ALL_LL_FILES=`ls *.ll`
rm ${ALL_LL_FILES}
