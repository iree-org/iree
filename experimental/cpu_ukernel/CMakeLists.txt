# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_BUILD_EXPERIMENTAL_CPU_UKERNEL_PLUGINS)
  return()
endif()

add_subdirectory(test)

include(iree_experimental_system_plugin.cmake)
include(iree_experimental_standalone_plugin.cmake)

iree_experimental_system_plugin(
  NAME
    builtin_ukernel_system_plugin
  SRCS
    plugin.c
  DEPS
    iree::builtins::ukernel
)

set(IREE_UK_X86_64_AVX2_FMA_COPTS
 "-mavx2"
 "-mfma"
)

set(IREE_UK_X86_64_AVX512_BASE_COPTS
  ${IREE_UK_X86_64_AVX2_FMA_COPTS}
  "-mavx512f"
  "-mavx512vl"
  "-mavx512cd"
  "-mavx512bw"
  "-mavx512dq"
)

set(IREE_UK_X86_64_AVX512_VNNI_COPTS
  ${IREE_UK_X86_64_AVX512_BASE_COPTS}
  "-mavx512vnni"
)

set(IREE_UK_ARM_64_DOTPROD_COPTS
  "-march=armv8.2-a+dotprod"
)

set(IREE_UK_ARM_64_I8MM_COPTS
  "-march=armv8.2-a+i8mm"
)

iree_experimental_standalone_plugin(
  NAME
    builtin_ukernel_standalone_plugin
  ARCHS
    "x86_64"
    "arm_64"
  SRCS
    plugin.c
    # ukernel/arch/ file come before ukernel/ (non-arch) files because they
    # contains symbols that should override weak symbols in ukernel/weak.c.
    "x86_64:runtime/src/iree/builtins/ukernel/arch/x86_64/query_tile_sizes_x86_64.c"
    "x86_64:runtime/src/iree/builtins/ukernel/arch/x86_64/unpack_x86_64.c"
    "x86_64:runtime/src/iree/builtins/ukernel/arch/x86_64/pack_x86_64.c"
    "x86_64:runtime/src/iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64.c"
    "x86_64:runtime/src/iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_avx2_fma.c:IREE_UK_X86_64_AVX2_FMA_COPTS"
    "x86_64:runtime/src/iree/builtins/ukernel/arch/x86_64/pack_x86_64_avx2_fma.c:IREE_UK_X86_64_AVX2_FMA_COPTS"
    "x86_64:runtime/src/iree/builtins/ukernel/arch/x86_64/unpack_x86_64_avx2_fma.c:IREE_UK_X86_64_AVX2_FMA_COPTS"
    "x86_64:runtime/src/iree/builtins/ukernel/arch/x86_64/unpack_x86_64_avx512_base.c:IREE_UK_X86_64_AVX512_BASE_COPTS"
    "x86_64:runtime/src/iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_avx512_base.c:IREE_UK_X86_64_AVX512_BASE_COPTS"
    "x86_64:runtime/src/iree/builtins/ukernel/arch/x86_64/pack_x86_64_avx512_base.c:IREE_UK_X86_64_AVX512_BASE_COPTS"
    "x86_64:runtime/src/iree/builtins/ukernel/arch/x86_64/mmt4d_x86_64_avx512_vnni.c:IREE_UK_X86_64_AVX512_VNNI_COPTS"
    "arm_64:runtime/src/iree/builtins/ukernel/arch/arm_64/query_tile_sizes_arm_64.c"
    "arm_64:runtime/src/iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64.c"
    "arm_64:runtime/src/iree/builtins/ukernel/arch/arm_64/pack_arm_64.c"
    "arm_64:runtime/src/iree/builtins/ukernel/arch/arm_64/unpack_arm_64.c"
    "arm_64:runtime/src/iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_dotprod.c:IREE_UK_ARM_64_DOTPROD_COPTS"
    "arm_64:runtime/src/iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_i8mm.c:IREE_UK_ARM_64_I8MM_COPTS"
    runtime/src/iree/builtins/ukernel/mmt4d.c
    runtime/src/iree/builtins/ukernel/mmt4d_tile.c
    runtime/src/iree/builtins/ukernel/unpack_tile.c
    runtime/src/iree/builtins/ukernel/pack.c
    runtime/src/iree/builtins/ukernel/query_tile_sizes.c
    runtime/src/iree/builtins/ukernel/unpack.c
    runtime/src/iree/builtins/ukernel/pack_tile.c
    runtime/src/iree/builtins/ukernel/weak.c
)
