// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
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
  LogicalResult checkAndNotify(PatternRewriter &rewriter, Operation *op) const;
  void replaceLinalgTransformationFilter(PatternRewriter &rewriter,
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

std::unique_ptr<OperationPass<func::FuncOp>> createTilingInterfaceTilingPass();

std::unique_ptr<OperationPass<func::FuncOp>> createLinalgExtToLoopsPass();

/// TypeConverter to use for materializing the encoding.
struct MaterializeEncodingTypeConverter : public TypeConverter {
  MaterializeEncodingTypeConverter(MaterializeEncodingFn fn);
  MaterializeEncodingFn &getMaterializeEncodingFn() {
    return materializeEncodingFn;
  }

private:
  MaterializeEncodingFn materializeEncodingFn;
};

/// Conversion target to use for for materializing the encoding.
struct MaterializeEncodingConversionTarget : public ConversionTarget {
  MaterializeEncodingConversionTarget(MLIRContext &context);
};

/// Base class for patterns that materialize encoding.
template <typename OpTy>
class OpMaterializeEncodingPattern : public OpConversionPattern<OpTy> {
public:
  OpMaterializeEncodingPattern(
      MLIRContext *context, MaterializeEncodingTypeConverter &typeConverter,
      MaterializeEncodingValueFn materializeEncodingValueFn = {},
      PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(typeConverter, context, benefit),
        materializeEncodingValueFn(materializeEncodingValueFn) {}

protected:
  const MaterializeEncodingValueFn materializeEncodingValueFn;
};

/// Method to populate the patterns to convert operations that have operands
/// with tensor encodings into ops that materialize the layout specified by the
/// encoding, as well as ops that perform the computation on the materialized
/// layout. For now these hard-code a fixed way the lowering is encoded, but the
/// encoding can be made backend specific. Also initializes the
/// `conversionTarget` and `typeConverter`.
void populateMaterializeEncodingPatterns(
    RewritePatternSet &patterns,
    MaterializeEncodingConversionTarget &conversionTarget,
    MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn = {});

/// Pass to apply patterns specified by `populateMaterializeEncodingPass`.
std::unique_ptr<OperationPass<func::FuncOp>> createMaterializeEncodingPass();

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

std::unique_ptr<OperationPass<func::FuncOp>> createTopkSplitReductionPass();

/// Tile and decompose the winograd transform ops into a sequence
/// of linalg ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createTileAndDecomposeWinogradTransformPass();

// Creates a pass to convert linalg convolution ops into a sequence of
// linalg_ext.winograd.* ops and linalg.batch_matmul ops using the winograd
// tranformation.
std::unique_ptr<Pass> createConvertConv2DToWinogradPass();

// Creates a pass to convert the softmax op into a sequence of
// linalg generic ops.
std::unique_ptr<Pass> createDecomposeSoftmaxPass();

// Transform dialect version of tile and decompose attention
SmallVector<Operation *>
tileAndDecomposeAttention(IREE::LinalgExt::AttentionOp attnOp,
                          IRRewriter &rewriter);

// Creates a pass to convert the attention op into a sequence of
// linalg ops.
std::unique_ptr<Pass> createTileAndDecomposeAttentionPass();

// Marker used as attribute the depth of the split reduction transformations.
const StringLiteral kSplitReductionDepthMarker = "__split_reduction_depth__";

//===---------------------------------------------------------------------===//
// Codegen Strategy passes that are moved into IREE.
//===---------------------------------------------------------------------===//
/// Options to control the application of enabling transformations.
/// Hoisting transformations are always deemed beneficial and must be disabled
/// explicitly.
struct LinalgEnablingOptions {
  /// Enable LICM.
  bool licm = true;
  LinalgEnablingOptions &enableLICM(bool val = true) {
    licm = val;
    return *this;
  }
  /// Enable hoisting of redundant vector transfer ops.
  /// TODO: MLIR does currently not model parallel contexts. It can be unsafe
  /// to hoist transfers from buffers in a multi-threaded environment, so this
  /// should not be enabled by default.
  bool hoistRedundantVectorTransfers = false;
  LinalgEnablingOptions &enableHoistRedundantVectorTransfers(bool val = true) {
    hoistRedundantVectorTransfers = val;
    return *this;
  }
  /// Enable hoisting of redundant vector transfer ops on tensor.
  bool hoistRedundantVectorTransfersOnTensor = true;
  LinalgEnablingOptions &
  enableHoistRedundantVectorTransfersOnTensor(bool val = true) {
    hoistRedundantVectorTransfersOnTensor = val;
    return *this;
  }
};

/// Create a LinalgStrategyDecomposePass.
// TODO: if/when we need finer control add an `opName` parameter.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyDecomposePass(
    const LinalgExt::LinalgTransformationFilter &filter =
        LinalgExt::LinalgTransformationFilter());

/// Create a LinalgStrategyVectorizePass.
using VectorSizeComputationFunction =
    std::function<SmallVector<int64_t>(linalg::LinalgOp, ArrayRef<int64_t>)>;

struct LinalgVectorizationOptions {
  /// Enable vector masking during vectorization.
  bool enableVectorMasking = false;

  LinalgVectorizationOptions &setEnableVectorMasking(bool val) {
    enableVectorMasking = val;
    return *this;
  }

  /// Canonical vector sizes for the vector iteration space (i.e., vectorization
  /// factors). They are optional for input code with full static shapes.
  SmallVector<int64_t> canonicalVectorSizes;

  LinalgVectorizationOptions &
  setCanonicalVectorSizes(ArrayRef<int64_t> vecSizes) {
    assert(canonicalVectorSizes.empty() &&
           "Canonical vector sizes are already set");
    canonicalVectorSizes.append(vecSizes.begin(), vecSizes.end());
    return *this;
  }

  /// Computation function that returns the vector sizes to vectorize a given
  /// Linalg operation and the canonical vector sizes of the iteration space.
  VectorSizeComputationFunction vectorSizeComputationFunction = nullptr;

  LinalgVectorizationOptions &
  setVectorSizeComputationFunction(VectorSizeComputationFunction fun) {
    vectorSizeComputationFunction = std::move(fun);
    return *this;
  }

  /// Enable vectorization of padding operations.
  bool vectorizePadding = false;

  LinalgVectorizationOptions &setVectorizePadding(bool vecPad) {
    vectorizePadding = vecPad;
    return *this;
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyVectorizePass(
    StringRef opName = "",
    const LinalgVectorizationOptions &options = LinalgVectorizationOptions(),
    const LinalgExt::LinalgTransformationFilter &filter =
        LinalgExt::LinalgTransformationFilter());

/// Create a LinalgStrategyEnablePass.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyEnablePass(
    LinalgEnablingOptions opt = LinalgEnablingOptions(),
    const LinalgExt::LinalgTransformationFilter &filter =
        LinalgExt::LinalgTransformationFilter());

/// Create a LinalgStrategyLowerVectorsPass.
/// Vector lowering options control how ops are lowered down to 1-D and scf.for
/// form.
struct LinalgVectorLoweringOptions {
  /// Enable lowering of vector.contract.
  /// In a progressive lowering of vectors, this would be the 1st step.
  bool contractionLowering = false;
  LinalgVectorLoweringOptions &enableContractionLowering(bool val = true) {
    contractionLowering = val;
    return *this;
  }
  /// Enable lowering of vector.multi_reduce.
  /// In a progressive lowering of vectors, this would be the 2nd step.
  bool multiReductionLowering = false;
  LinalgVectorLoweringOptions &enableMultiReductionLowering(bool val = true) {
    multiReductionLowering = val;
    return *this;
  }
  /// Trigger full / partial vector.transfer splits.
  /// In a progressive lowering of vectors, this would be the 3rd step.
  bool transferPartialRewrite = false;
  LinalgVectorLoweringOptions &enableTransferPartialRewrite(bool val = true) {
    transferPartialRewrite = val;
    return *this;
  }
  /// Enable lowering of vector.transfer to scf.
  /// In a progressive lowering of vectors, this would be the 4th step.
  bool transferToSCFConversion = false;
  LinalgVectorLoweringOptions &enableTransferToSCFConversion(bool val = true) {
    transferToSCFConversion = val;
    return *this;
  }
  /// Maximal transfer rank under which we do not lower further.
  int64_t maxTransferRank = 1;
  LinalgVectorLoweringOptions &setMaxTransferRank(int64_t val) {
    maxTransferRank = val;
    return *this;
  }
  /// Vector lowering operations may result in surprising behavior when
  /// composing multiple codegen strategies and must be enabled explicitly.
  /// In a progressive lowering of vectors, this would be the 5th step.
  bool transferLowering = true;
  LinalgVectorLoweringOptions &enableTransferLowering(bool val = true) {
    transferLowering = val;
    return *this;
  }
  /// Enable lowering of vector.shape_cast to insert/extract.
  /// In a progressive lowering of vectors, this would be the 6th step.
  bool shapeCastLowering = true;
  LinalgVectorLoweringOptions &enableShapeCastLowering(bool val = true) {
    shapeCastLowering = val;
    return *this;
  }
  /// Enable lowering of vector.transpose.
  /// In a progressive lowering of vectors, this would be the 7th step.
  bool transposeLowering = false;
  LinalgVectorLoweringOptions &enableVectorTransposeLowering(bool val = true) {
    transposeLowering = val;
    return *this;
  }
  /// Enable AVX2-specific lowerings.
  bool avx2Lowering = false;
  LinalgVectorLoweringOptions &enableAVX2Lowering(bool val = true) {
    avx2Lowering = val;
    return *this;
  }

  /// Configure the post staged-patterns late vector.transfer to scf
  /// conversion.
  VectorTransferToSCFOptions vectorTransferToSCFOptions;
  LinalgVectorLoweringOptions &
  setVectorTransferToSCFOptions(VectorTransferToSCFOptions options) {
    vectorTransferToSCFOptions = options;
    return *this;
  }
  /// Configure late vector transformations.
  vector::VectorTransformsOptions vectorTransformOptions;
  LinalgVectorLoweringOptions &
  setVectorTransformsOptions(vector::VectorTransformsOptions options) {
    vectorTransformOptions = options;
    return *this;
  }
  /// Configure specialized vector lowerings.
  x86vector::avx2::LoweringOptions avx2LoweringOptions;
  LinalgVectorLoweringOptions &
  setAVX2LoweringOptions(x86vector::avx2::LoweringOptions options) {
    avx2LoweringOptions = options;
    return *this;
  }
};

std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgStrategyLowerVectorsPass(
    LinalgVectorLoweringOptions opt = LinalgVectorLoweringOptions(),
    const LinalgTransformationFilter &filter = LinalgTransformationFilter());

/// Create a LinalgStrategyRemoveMarkersPass.
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgStrategyRemoveMarkersPass();

void registerPasses();

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_
