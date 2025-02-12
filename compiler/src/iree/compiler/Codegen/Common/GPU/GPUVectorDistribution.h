// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_

#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

/// A signature describing the layout for each value of vector type which is
/// an operand or result of this operation.
///
/// Two operands may be the same value, but since each value can only have
/// one layout, we only need to keep track of the value, not the two operands
/// separately.
using DistributionSignature =
    DenseMap<TypedValue<VectorType>, VectorLayoutInterface>;

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

protected:
  // Sets new layout/signature for op, and mark it for redistribution.
  // When "vector layout storage" and "vector layout redistribution"
  // is defined, VectorDistributionRewriter would add it to worklist
  // of operations to be distributed.
  void setSignatureForRedistribution(PatternRewriter &rewriter, Operation *op,
                                     Attribute inputLayoutsAttr,
                                     Attribute outputLayoutsAttr) const;
};

template <typename SourceOp>
struct OpDistributionPattern : DistributionPattern {
  OpDistributionPattern(MLIRContext *context, PatternBenefit benefit = 1)
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

template <template <typename> class TraitType>
class OpTraitDistributionPattern : public DistributionPattern {
public:
  OpTraitDistributionPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : DistributionPattern(Pattern::MatchTraitOpTypeTag(),
                            TypeID::get<TraitType>(), benefit, context) {}

  virtual LogicalResult matchAndRewrite(Operation *op,
                                        DistributionSignature &opSignature,
                                        PatternRewriter &rewriter) const = 0;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    std::optional<DistributionSignature> opSignature = getOpSignature(op);
    if (!opSignature) {
      return failure();
    }
    return matchAndRewrite(op, *opSignature, rewriter);
  }
};

/// Options to control how the layout analysis is initialised for vector
/// distribution.
class VectorLayoutOptions {
public:
  VectorLayoutOptions(Operation *root) : root(root), fullConversion(true) {
    assert(root && "root operation must be non-null");
  }
  VectorLayoutOptions(Operation *root, bool fullConversion)
      : root(root), fullConversion(fullConversion) {
    assert(root && "root operation must be non-null");
  }

  virtual ~VectorLayoutOptions() = default;

  bool verifyConversion() const { return fullConversion; }

  virtual VectorLayoutInterface getDefaultLayout(VectorType type) const = 0;

protected:
  Operation *root;
  bool fullConversion = true;
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
LogicalResult distributeVectorOps(Operation *root,
                                  RewritePatternSet &distributionPatterns,
                                  VectorLayoutOptions &options);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_VECTOR_DISTRIBUTION_H_
