// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_VECTOR_LAYOUT_PROVIDER_H
#define IREE_COMPILER_CODEGEN_VECTOR_LAYOUT_PROVIDER_H

#include <array>
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"

namespace mlir::iree_compiler {

class LayoutProvider {
public:
  LayoutProvider(VectorLayoutAnalysis &analysis, Operation *root)
      : analysis(analysis), root(root) {}

  virtual ~LayoutProvider() = default;

  VectorLayoutAnalysis &getAnalysis() { return analysis; }

  /// Set the anchor ops in the analysis rooted on the root operation.
  virtual void setAnchorOps() = 0;

  /// Given a Value of type VectorType, return the distributed shape of the
  /// value, based on it's layout in the analysis.
  virtual SmallVector<int64_t>
  getDistributedShape(TypedValue<VectorType> val) = 0;

  /// Given an operation, do specialized distribution for it. Return true if
  /// the operation if a specialized distribution is done.
  /// Return false if the operation is not specialized.
  virtual LogicalResult specializedDistribution(RewriterBase &rewriter,
                                                Operation *op) {
    // No specialization by default.
    return failure();
  }

  virtual SmallVector<Value> getThreadGrid(RewriterBase &rewriter) = 0;

protected:
  VectorLayoutAnalysis &analysis;
  Operation *root;
}; // namespace iree_compiler

class HigherDimLayoutProvider : public LayoutProvider {
public:
  HigherDimLayoutProvider(VectorLayoutAnalysis &analysis, Operation *root)
      : LayoutProvider(analysis, root) {}

  virtual SmallVector<int64_t>
  getDistributedShape(TypedValue<VectorType> value) override;

protected:
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_VECTOR_LAYOUT_PROVIDER_H
