// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/mmt4d.h"

#if defined(IREE_UKERNEL_ARCH_ARM_64)
#include "iree/builtins/ukernel/mmt4d_arm_64.h"
#endif

#if defined(IREE_UKERNEL_ARCH_GENERIC_32) || \
    defined(IREE_UKERNEL_ARCH_GENERIC_64)
#include "iree/builtins/ukernel/mmt4d_generic.h"
#endif

IREE_UKERNEL_EXPORT int iree_ukernel_mmt4d_f32f32f32(
    const iree_ukernel_mmt4d_f32f32f32_params_t* params) {
  if (params->flags & ~IREE_VMVX_MATMUL_FLAG_ACCUMULATE) {
    return IREE_UKERNEL_MMT4D_ERROR_BAD_FLAGS;
  }

#if defined(IREE_UKERNEL_ARCH_ARM_64)
  return iree_ukernel_mmt4d_f32f32f32_arm_64(params);
#endif

#if defined(IREE_UKERNEL_ARCH_GENERIC_32) || \
    defined(IREE_UKERNEL_ARCH_GENERIC_64)
  return iree_ukernel_mmt4d_f32f32f32_generic(params);
#endif

  return IREE_UKERNEL_MMT4D_ERROR_UNIMPLEMENTED;
}

IREE_UKERNEL_EXPORT int iree_ukernel_mmt4d_i8i8i32(
    const iree_ukernel_mmt4d_i8i8i32_params_t* params) {
  if (params->flags & ~IREE_VMVX_MATMUL_FLAG_ACCUMULATE) {
    return IREE_UKERNEL_MMT4D_ERROR_BAD_FLAGS;
  }

#if defined(IREE_UKERNEL_ARCH_ARM_64)
  return iree_ukernel_mmt4d_i8i8i32_arm_64(params);
#endif

#if defined(IREE_UKERNEL_ARCH_GENERIC_32) || \
    defined(IREE_UKERNEL_ARCH_GENERIC_64)
  return iree_ukernel_mmt4d_i8i8i32_generic(params);
#endif

  return IREE_UKERNEL_MMT4D_ERROR_UNIMPLEMENTED;
}

const char* iree_ukernel_mmt4d_error_message(int retcode) {
  switch (retcode) {
    case IREE_UKERNEL_MMT4D_ERROR_UNIMPLEMENTED:
      return "hit unimplemented code path in mmt4d";
    case IREE_UKERNEL_MMT4D_ERROR_BAD_FLAGS:
      return "bad mmt4d flags";
    default:
      return "unknown";
  }
}
