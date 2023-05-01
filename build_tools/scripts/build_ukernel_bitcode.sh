#!/bin/bash

set -e
set -x

IREE_BUILD_DIR="${IREE_BUILD_DIR:-$(pwd)}"
LLVM_BIN_DIR="${LLVM_BIN_DIR:-$IREE_BUILD_DIR/llvm-project/bin}"
CLANG="${CLANG:-$LLVM_BIN_DIR/clang}"
LLVM_LINK="${LLVM_LINK:-$LLVM_BIN_DIR/llvm-link}"
LLVM_DIS="${LLVM_DIS:-$LLVM_BIN_DIR/llvm-dis}"
IREE_SRC_DIR="${IREE_SRC_DIR:-$(grep '^IREE_SOURCE_DIR[:=]' CMakeCache.txt | cut -d '=' -f 2)}"
RUNTIME_BIN_DIR="${RUNTIME_BIN_DIR:-${IREE_BUILD_DIR}/runtime/src}"
IMMINTRIN_INCLUDE_DIR="$(echo "$(dirname "$(find "${IREE_BUILD_DIR}" -name immintrin.h | grep -F 'include/immintrin.h')")")"
LIBDEVICE_FLAGS="-std=c17 -nostdinc -ffreestanding -O3 -fno-ident -fdiscard-value-names -c -emit-llvm -DIREE_DEVICE_STANDALONE=1"
CLANG_FLAGS="-I${IREE_SRC_DIR}/runtime/src -I${IMMINTRIN_INCLUDE_DIR} ${LIBDEVICE_FLAGS}"

files_default=(
  runtime/src/iree/builtins/ukernel/mmt4d.c
  runtime/src/iree/builtins/ukernel/mmt4d_tile.c
  runtime/src/iree/builtins/ukernel/unpack_tile.c
  runtime/src/iree/builtins/ukernel/pack.c
  runtime/src/iree/builtins/ukernel/query_tile_sizes.c
  runtime/src/iree/builtins/ukernel/arch/x86_64/query_tile_sizes_x86_64.c
  runtime/src/iree/builtins/ukernel/arch/x86_64/unpack_x86_64.c
  runtime/src/iree/builtins/ukernel/arch/x86_64/pack_x86_64.c
  runtime/src/iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64.c
  runtime/src/iree/builtins/ukernel/unpack.c
  runtime/src/iree/builtins/ukernel/pack_tile.c
)

files_avx2_fma=(
  runtime/src/iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_avx2_fma.c
  runtime/src/iree/builtins/ukernel/arch/x86_64/pack_x86_64_avx2_fma.c
  runtime/src/iree/builtins/ukernel/arch/x86_64/unpack_x86_64_avx2_fma.c
)

files_avx512_base=(
  runtime/src/iree/builtins/ukernel/arch/x86_64/unpack_x86_64_avx512_base.c
  runtime/src/iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_avx512_base.c
  runtime/src/iree/builtins/ukernel/arch/x86_64/pack_x86_64_avx512_base.c
)

files_avx512_vnni=(
  runtime/src/iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_avx512_vnni.c
)

bc_files=()

for file in "${files_default[@]}"
do
  bc_file="$(basename "${file}").bc"
  "${CLANG}" ${CLANG_FLAGS} "${IREE_SRC_DIR}/${file}" -o ${bc_file}
  bc_files+=("${bc_file}")
done

for file in "${files_avx2_fma[@]}"
do
  bc_file="$(basename "${file}").bc"
  "${CLANG}" -mavx2 -mfma ${CLANG_FLAGS} "${IREE_SRC_DIR}/${file}" -o ${bc_file}
  bc_files+=("${bc_file}")
done

for file in "${files_avx512_base[@]}"
do
  bc_file="$(basename "${file}").bc"
  "${CLANG}" -mavx512f -mavx512vl -mavx512cd -mavx512bw -mavx512dq ${CLANG_FLAGS} "${IREE_SRC_DIR}/${file}" -o ${bc_file}
  bc_files+=("${bc_file}")
done

for file in "${files_avx512_vnni[@]}"
do
  bc_file="$(basename "${file}").bc"
  "${CLANG}" -mavx512f -mavx512vl -mavx512cd -mavx512bw -mavx512dq -mavx512vnni ${CLANG_FLAGS} "${IREE_SRC_DIR}/${file}" -o ${bc_file}
  bc_files+=("${bc_file}")
done

${LLVM_LINK} "${bc_files[@]}" -o ukernel.bc

cmake --build . --target llvm-dis
${LLVM_DIS} ukernel.bc -o ukernel.ll
