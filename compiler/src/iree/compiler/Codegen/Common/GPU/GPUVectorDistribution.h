// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::iree_compiler {

/// Forward declarations.
class VectorLayoutOptions;

/// Lookup the distributed value for the given SIMD value. If the value was not
/// distributed yet, wrap it in a ToSIMTOp.
TypedValue<VectorType> getDistributed(RewriterBase &rewriter,
                                      TypedValue<VectorType> value,
                                      VectorLayoutOptions &options);

/// Replace an op with its distributed replacement values.
void replaceOpWithDistributedValues(RewriterBase &rewriter, Operation *op,
                                    VectorLayoutOptions &options,
                                    ValueRange values);

/// Replace an op with a new distributed op. The results of the distributed op
/// must be distributed vector values.
template <typename OpTy, typename... Args>
OpTy replaceOpWithNewDistributedOp(VectorLayoutOptions &options,
                                   RewriterBase &rewriter, Operation *op,
                                   Args &&...args) {
  auto newOp = rewriter.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
  replaceOpWithDistributedValues(rewriter, op, options, newOp->getResults());
  return newOp;
}

class VectorLayoutOptions {
public:
  VectorLayoutOptions(VectorLayoutAnalysis &analysis, Operation *root)
      : analysis(analysis), root(root) {
    assert(root && "root operation must be non-null");
  }

  virtual ~VectorLayoutOptions() = default;

  VectorLayoutAnalysis &getAnalysis() { return analysis; }

  /// Set the anchor ops in the analysis rooted on the root operation.
  virtual void setAnchorOps() = 0;

  /// Given a Value of type VectorType, return the distributed shape of the
  /// value, based on its layout in the analysis.
  virtual SmallVector<int64_t>
  getDistributedShape(TypedValue<VectorType> val) = 0;

protected:
  VectorLayoutAnalysis &analysis;
  Operation *root;
}; // namespace iree_compiler

void distributeVectorOps(Operation *root, VectorLayoutOptions &options);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_
