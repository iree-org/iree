// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/mmt4d_arm_64.h"

// TODO: once actual ARM64 code is implemented, we shouldn't need this anymore
#include "iree/builtins/ukernel/mmt4d_generic.h"

#if defined(IREE_UKERNEL_ARCH_ARM_64)

int iree_ukernel_mmt4d_f32f32f32_arm_64(
    const iree_ukernel_mmt4d_f32f32f32_params_t* params) {
  // TODO: implement actual arm assembly kernels instead of calling _generic.
  return iree_ukernel_mmt4d_f32f32f32_generic(params);
}

int iree_ukernel_mmt4d_i8i8i32_arm_64(
    const iree_ukernel_mmt4d_i8i8i32_params_t* params) {
  // TODO: implement actual arm assembly kernels instead of calling _generic.
  return iree_ukernel_mmt4d_i8i8i32_generic(params);
}

#endif  // IREE_UKERNEL_ARCH_ARM_64
