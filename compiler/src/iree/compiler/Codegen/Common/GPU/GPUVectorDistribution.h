// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_

#include "iree/compiler/Codegen/Common/GPU/VectorLayoutProvider.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::iree_compiler {

/// Lookup the distributed value for the given SIMD value. If the value was not
/// distributed yet, wrap it in a ToSIMTOp.
TypedValue<VectorType> getDistributed(RewriterBase &rewriter,
                                      TypedValue<VectorType> value,
                                      LayoutProvider &provider);

/// Replace an op with it's distributed replacement values.
void replaceOpWithDistributedValues(RewriterBase &rewriter, Operation *op,
                                    LayoutProvider &provider,
                                    ValueRange values);

/// Replace an op with a new distributed op. The results of the distributed op
/// must be distributed vector values.
template <typename OpTy, typename... Args>
OpTy replaceOpWithNewDistributedOp(LayoutProvider &provider,
                                   RewriterBase &rewriter, Operation *op,
                                   Args &&...args) {
  auto newOp = rewriter.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
  replaceOpWithDistributedValues(rewriter, op, provider, newOp->getResults());
  return newOp;
}

class VectorDistribution {
public:
  VectorDistribution(func::FuncOp root, VectorLayoutAnalysis &analysis,
                     LayoutProvider &provider);

  void distribute();

private:
  func::FuncOp root;
  VectorLayoutAnalysis &analysis;
  LayoutProvider &provider;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_
