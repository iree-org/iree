// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

/// A signature describing the layout for each operand of vector type for
/// an operation.
struct DistributionSignature {
  SmallVector<VectorLayoutInterface> operands;
  SmallVector<VectorLayoutInterface> results;
};

struct DistributionPattern : RewritePattern {
  using RewritePattern::RewritePattern;

  /// Lookup the distributed value for the given SIMD value. If the value
  /// was not distributed yet, wrap it in a ToSIMTOp.
  TypedValue<VectorType> getDistributed(RewriterBase &rewriter,
                                        TypedValue<VectorType> value,
                                        VectorLayoutInterface layout) const;

  /// Replace an op with its distributed replacement values.
  void replaceOpWithDistributedValues(RewriterBase &rewriter, Operation *op,
                                      ValueRange values) const;

  /// Get the signature for the given operation.
  std::optional<DistributionSignature> getOpSignature(Operation *op) const;
};

template <typename SourceOp>
struct OpDistributionPattern : DistributionPattern {
  OpDistributionPattern<SourceOp>(MLIRContext *context,
                                  PatternBenefit benefit = 1)
      : DistributionPattern(SourceOp::getOperationName(), benefit, context) {}

  virtual LogicalResult matchAndRewrite(SourceOp op,
                                        DistributionSignature &opSignature,
                                        PatternRewriter &rewriter) const = 0;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    std::optional<DistributionSignature> opSignature = getOpSignature(op);
    if (!opSignature) {
      return failure();
    }
    return matchAndRewrite(cast<SourceOp>(op), *opSignature, rewriter);
  }
};

/// Options to control how the layout analysis is initialised for vector
/// distribution.
class VectorLayoutOptions {
public:
  VectorLayoutOptions(Operation *root) : root(root) {
    assert(root && "root operation must be non-null");
  }

  virtual ~VectorLayoutOptions() = default;

  /// Set the anchor ops in the analysis rooted on the root operation.
  virtual void setAnchorOps(VectorLayoutAnalysis &analysis) = 0;

protected:
  Operation *root;
}; // namespace iree_compiler

/// Distribute vector operations in the IR rooted at `root`.
///
/// The flow of distribution looks like:
///   - Make `options` set some initial information about how to distribute
///     some vector values. This is usually done on operations like
///     vector.contract, vector.transfer_read/vector.transfer_write,
///     vector.multi_reduction, where we are trying to target specific
///     hardware instructions. This information is provided in the form of a
///     layout for the value.
///   - Run a global analysis to determine how to distribute rest of the vector
///     values keeping the initial anchors in mind.
///   - Use the analysis information to distribute each operation.
void distributeVectorOps(Operation *root,
                         RewritePatternSet &distributionPatterns,
                         VectorLayoutOptions &options);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_
