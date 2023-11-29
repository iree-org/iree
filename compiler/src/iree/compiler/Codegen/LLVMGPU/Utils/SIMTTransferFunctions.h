// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_SIMTTRANSFERFUNCTIONS_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_SIMTTRANSFERFUNCTIONS_H_

#include "iree/compiler/Codegen/LLVMGPU/Utils/SIMTLayoutAnalysis.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"

namespace mlir {
namespace iree_compiler {

void propagationTransferFunction(
    Operation *op, ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update);

void enforcementTransferFunction(
    Operation *op, ArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update);

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_SIMTTRANSFERFUNCTIONS_H_
