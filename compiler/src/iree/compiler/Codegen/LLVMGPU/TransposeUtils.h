// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_TRANSPOSEUTILS_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_TRANSPOSEUTILS_H_

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {

/// Returns true if the index map represents a transpose that benefits from
/// shared mem.
static bool sharedMemTransposeFilter(AffineMap indexMap) {
  if (!indexMap.isEmpty() && indexMap.isPermutation()) {
    // Ensure that the fasted moving dimension (the last one) is permuted,
    // Otherwise shared memory promotion will not benefit the operation.
    if (indexMap.getDimPosition(indexMap.getNumDims() - 1) !=
        indexMap.getNumDims() - 1) {
      return true;
    }
  }
  return false;
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_LLVMGPU_TRANSPOSEUTILS_H_
