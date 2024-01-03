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

/// Options that control how vector values are distributed.
///
/// The way to use these options is to derive from this class and implement
/// methods to control how VectorLayoutAnalysis is initialized by passing it
/// initial anchors and how to get the distributed shape of a given vector
/// value.
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

/// Distribute vector operations in the IR rooted at `root`.
///
/// The flow of distribution looks like:
///   - Make `options` set some initial information about how to distribute
///     some vector values. This is usually done on operations like
///     vector.contract, vector.transfer_read/vector.transfer_write,
///     vector.multi_reduction, where we are trying to target a specific
///     hardware instructions. This information is provided in the form of a
///     layout for the value.
///   - Run a global analysis to determine how to distribute rest of the vector
///     values keeping the initial anchors in mind.
///   - Use the analysis information to distribute each operation.
void distributeVectorOps(Operation *root, VectorLayoutOptions &options);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_
