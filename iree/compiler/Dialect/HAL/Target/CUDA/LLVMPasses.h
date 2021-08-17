// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_CUDA_PASS_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_CUDA_PASS_H_

namespace llvm {

class Pass;

/// Pass to mark all loops with llvm metadata to disable unrolling.
Pass *createSetNoUnrollPass();

}  // namespace llvm

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_CUDA_PASS_H_
