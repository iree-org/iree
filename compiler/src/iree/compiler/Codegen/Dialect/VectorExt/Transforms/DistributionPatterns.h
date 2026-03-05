// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_DISTRIBUTION_PATTERNS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_DISTRIBUTION_PATTERNS_H_

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::VectorExt {

class VectorLayoutOptions;

/// A signature describing the layout for each value of vector type which is
/// an operand or result of this operation.
///
/// Two operands may be the same value, but since each value can only have
/// one layout, we only need to keep track of the value, not the two operands
/// separately.
using DistributionSignature =
    DenseMap<TypedValue<VectorType>, VectorLayoutInterface>;

/// Set signature for the operation based on the analysis. Returns failure if
/// an operation contains vectors that cannot be distributed i.e. they have no
/// layout.
LogicalResult
setOpSignature(Operation *op,
               const llvm::MapVector<Value, VectorLayoutInterface> &layouts,
               const VectorLayoutOptions &options);

/// Check if an operation has a distribution signature attribute.
bool hasOpSignature(Operation *op);

/// Remove the distribution signature attribute from an operation.
void removeOpSignature(Operation *op);

/// Check if an operation is marked for redistribution.
bool isMarkedForRedistribution(Operation *op);

/// Clear the redistribution mark from an operation.
void clearRedistributionMark(Operation *op);

/// Get the distribution signature from an operation's attributes.
DistributionSignature getOpSignature(Operation *op);

struct DistributionPattern : RewritePattern {
  using RewritePattern::RewritePattern;

  /// Lookup the distributed value for the given SIMD value. If the value
  /// was not distributed yet, wrap it in a ToSIMTOp.
  TypedValue<VectorType> getDistributed(RewriterBase &rewriter,
                                        TypedValue<VectorType> value,
                                        VectorLayoutInterface layout) const;

  /// Get the distributed values that could replace the op
  SmallVector<Value> getOpDistributedReplacements(RewriterBase &rewriter,
                                                  Operation *op,
                                                  ValueRange values) const;

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
  void setSignatureForRedistribution(
      RewriterBase &rewriter, Operation *op,
      ArrayRef<VectorLayoutInterface> inputLayouts,
      ArrayRef<VectorLayoutInterface> outputLayouts) const;

  LogicalResult replaceParentMask(PatternRewriter &rewriter,
                                  vector::MaskOp) const;
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

template <typename SourceOp>
struct MaskedOpDistributionPattern : DistributionPattern {
  MaskedOpDistributionPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : DistributionPattern(SourceOp::getOperationName(), benefit, context) {}

  virtual LogicalResult
  matchAndRewrite(SourceOp op, DistributionSignature &opSignature,
                  vector::MaskOp maskOp,
                  std::optional<DistributionSignature> &maskSignature,
                  PatternRewriter &rewriter) const = 0;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    std::optional<DistributionSignature> opSignature = getOpSignature(op);
    if (!opSignature) {
      return failure();
    }
    auto maskOp = op->getParentOfType<vector::MaskOp>();
    std::optional<DistributionSignature> maskSignature;
    if (maskOp) {
      maskSignature = getOpSignature(maskOp);
      if (!maskSignature) {
        return failure();
      }
    }
    LogicalResult result = matchAndRewrite(cast<SourceOp>(op), *opSignature,
                                           maskOp, maskSignature, rewriter);
    if (failed(result)) {
      return failure();
    }
    if (maskOp) {
      return replaceParentMask(rewriter, maskOp);
    }
    return success();
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
};

/// Populate patterns for distributing vector operations with NestedLayoutAttr.
void populateNestedLayoutDistributionPatterns(RewritePatternSet &patterns,
                                              Value threadId,
                                              int64_t subgroupSize,
                                              ArrayRef<int64_t> workgroupSize,
                                              int64_t maxBitsPerShuffle = 32);

} // namespace mlir::iree_compiler::IREE::VectorExt

#endif // IREE_COMPILER_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_DISTRIBUTION_PATTERNS_H_
