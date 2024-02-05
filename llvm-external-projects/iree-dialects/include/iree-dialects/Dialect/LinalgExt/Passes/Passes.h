// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

class ConversionTarget;
class TypeConverter;

namespace iree_compiler {
namespace IREE {
namespace LinalgExt {
// Marker used as attribute name in generated Linalg rewriting transformations.
struct LinalgTransforms {
  static const StringLiteral kLinalgTransformMarker;
};

/// Helper class to control application of linalg transformation patterns.
/// Control comes in 2 forms:
///   1. attribute matching and setting behavior using the attribute named
///      `kLinalgTransformMarker`. This can be used to build a state machine
///      using attributes and incrementally applying patterns to advance states.
///   2. filter function, which is a simple lambda on the Operation* that
///      returns a LogicalResult.
struct LinalgTransformationFilter {
  using FilterFunction = std::function<LogicalResult(Operation *)>;

  explicit LinalgTransformationFilter(
      ArrayRef<StringAttr> matchDisjunction = {},
      std::optional<StringAttr> replacement = std::nullopt);

  explicit LinalgTransformationFilter(
      const FilterFunction &f, ArrayRef<StringAttr> matchDisjunction = {},
      std::optional<StringAttr> replacement = std::nullopt);

  LinalgTransformationFilter(LinalgTransformationFilter &&) = default;
  LinalgTransformationFilter(const LinalgTransformationFilter &) = default;
  LogicalResult checkAndNotify(RewriterBase &rewriter, Operation *op) const;
  void replaceLinalgTransformationFilter(RewriterBase &rewriter,
                                         Operation *op) const;
  bool hasReplacementFilter(Operation *op) const;

  LinalgTransformationFilter &addFilter(const FilterFunction &f) {
    if (f)
      filters.push_back(f);
    return *this;
  }

  template <typename... OpTypes>
  LinalgTransformationFilter &addOpFilter() {
    return addFilter(
        [](Operation *op) { return success(isa<OpTypes...>(op)); });
  }

  LinalgTransformationFilter &addOpNameFilter(StringRef opName) {
    return addFilter([opName](Operation *op) {
      return success(op->getName().getStringRef() == opName);
    });
  }

  LinalgTransformationFilter &setMatchByDefault() {
    matchByDefault = true;
    return *this;
  }

private:
  SmallVector<FilterFunction> filters;
  SmallVector<StringAttr> matchDisjunction;
  std::optional<StringAttr> replacement;
  /// When set to true, if the attribute is not set, it will be treated as
  /// a match. Default is false.
  bool matchByDefault;
};

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLinalgExtToLoopsPass();

std::unique_ptr<OperationPass<>> createPadContractionToBlockSizePass();

/// Function signature to control reduction splitting. This returns the split
/// reduction ratio used to split the reduction dimension. The ratio is applied
/// to the reduction dimension of TopK. If the ratio value is less or equal to 1
/// then nothing will be done. Input is the current depth of recursive split
/// reduction, starting from 0 (first level).
using TopkSplitReductionControlFn =
    std::function<int64_t(int64_t splitReductionDepth)>;

/// Patterns to apply `topk split reduction` pass.
void populateTopkSplitReductionPattern(
    RewritePatternSet &patterns,
    const TopkSplitReductionControlFn &splitReductionFn,
    const LinalgExt::LinalgTransformationFilter &f =
        LinalgExt::LinalgTransformationFilter());

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTopkSplitReductionPass();

/// Tile and decompose the winograd transform ops into a sequence
/// of linalg ops.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTileAndDecomposeWinogradTransformPass();

// Creates a pass to convert linalg convolution ops into a sequence of
// linalg_ext.winograd.* ops and linalg.batch_matmul ops using the winograd
// tranformation.
std::unique_ptr<Pass> createConvertConv2DToWinogradPass();

// Transform dialect version of tile and decompose attention wrapper.
// The optional tile size specifies the step for the innermost for loop.
void tileAndDecomposeAttention(IREE::LinalgExt::AttentionOp attnOp,
                               SmallVectorImpl<Operation *> &ops,
                               RewriterBase &rewriter, bool onlyTile = false,
                               std::optional<uint64_t> tileSize = std::nullopt);

IREE::LinalgExt::AttentionOp
tileAttention(IREE::LinalgExt::AttentionOp attnOp,
              SmallVectorImpl<Operation *> &ops, RewriterBase &rewriter,
              std::optional<uint64_t> tileSize = std::nullopt);

void decomposeTiledAttention(IREE::LinalgExt::AttentionOp tiledAttnOp,
                             SmallVectorImpl<Operation *> &ops,
                             RewriterBase &rewriter,
                             std::optional<uint64_t> tileSize = std::nullopt);

// Creates a pass to convert the attention op into a sequence of
// linalg ops.
std::unique_ptr<Pass> createTileAndDecomposeAttentionPass();

// Marker used as attribute the depth of the split reduction transformations.
const StringLiteral kSplitReductionDepthMarker = "__split_reduction_depth__";

//===---------------------------------------------------------------------===//
// Codegen Strategy passes that are moved into IREE.
//===---------------------------------------------------------------------===//

void registerPasses();

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_
