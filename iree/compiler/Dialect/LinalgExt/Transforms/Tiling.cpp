// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace linalg_ext {

//===----------------------------------------------------------------------===//
// Utility methods for tiling a linalg_ext operation that implements a
// TiledOpInterface
//===----------------------------------------------------------------------===//

/// Returns failure if the options are unsupported.
static LogicalResult verifySupportedTilingOptions(
    PatternRewriter &rewriter, Operation *op,
    const linalg::LinalgTilingOptions &options) {
  if (!options.interchangeVector.empty()) {
    return rewriter.notifyMatchFailure(op,
                                       "unsupported interchange during tiling");
  }
  if (options.paddingValueComputationFunction) {
    return rewriter.notifyMatchFailure(op, "unsupported tile + pad option");
  }
  if (options.loopType != linalg::LinalgTilingLoopType::Loops) {
    return rewriter.notifyMatchFailure(op,
                                       "only tiling with scf.for is supported");
  }
  if (options.distribution) {
    if (llvm::any_of(options.distribution->distributionMethod,
                     [](linalg::DistributionMethod method) {
                       return method != linalg::DistributionMethod::Cyclic;
                     })) {
      return rewriter.notifyMatchFailure(op,
                                         "only cyclic distibution is allowed");
    }
  }
  return success();
}

/// Converts a `Value` to an `OpFoldRedult` by extracting the constant value if
/// the value is defined by a constant op.
static OpFoldResult getOpFoldResult(Value value) {
  IntegerAttr::ValueType attr;
  if (matchPattern(value, m_ConstantInt(&attr))) {
    return IntegerAttr::get(value.getType(), attr);
  }
  return value;
}
static SmallVector<OpFoldResult, 4> getOpFoldResult(ArrayRef<Value> values) {
  return llvm::to_vector<4>(llvm::map_range(
      values, [](Value value) { return getOpFoldResult(value); }));
}

/// Converts an `OpFoldResult` to a `Value` by building a constant op if
/// if the `OpFoldResult` is an `IntegerAttr`.
static Value getValue(OpBuilder &builder, Location loc,
                      OpFoldResult valueOrAttr) {
  if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
    return builder.create<ConstantIndexOp>(loc,
                                           attr.cast<IntegerAttr>().getInt());
  }
  return valueOrAttr.get<Value>();
}

/// Returns the constant value in `valueOrAttr` if it is not a dynamic `Value`.
static Optional<int64_t> getConstantValue(OpFoldResult valueOrAttr) {
  if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
    return attr.cast<IntegerAttr>().getInt();
  }
  return {};
}

/// Checks if `valueOrAttr` represents a constant value `val`.
static bool isValue(OpFoldResult valueOrAttr, int64_t val) {
  auto attr = valueOrAttr.dyn_cast<Attribute>();
  return attr && attr.cast<IntegerAttr>().getValue() == val;
}

/// Returns true if loop is untiled. Only checks if the value is statically
/// zero. It is assumed that a `Value` defined by a constant op is already
/// converted to an `IntegerAttr` of that value. So here just return true if
/// this is an attribute with a zero value.
static bool isUntiledLoop(OpFoldResult valueOrAttr) {
  return isValue(valueOrAttr, 0);
}

/// Generates the tiled loops and the body by invoking the interface methods of
/// TiledOpInterface.
/// - `outputs` are the operands to use for outputs of the tiled operation.
/// - `tileSizes` are tile sizes specified for all loops of the operation. If a
///   loop is to be untiled it is set to 0.
/// - `iteratorType` is the type of the loop iterator returned by the
///   TiledOpInterface.
/// - `loopBounds` are the bounds of all the loops of the op returned by the
///   TiledOpInterface.
/// - `loopDepth` is the current loop depth being processed.
/// - `offsets` are the `Value`s that represent the position of the tile being
///   operated on. The offsets are computed as the tiled loops are being
///   generated.
/// - `distributionInfo` is the proc_id and nprocs `Value`s to be used for
///   distributed loops. It is a stack, and once an entry at the top of the
///   stack is used for distribution it is popped before processing the inner
///   loops.
static FailureOr<TiledOp> tileInterfaceOpImpl(
    OpBuilder &builder, TiledOpInterface tilableOp, ValueRange outputs,
    MutableArrayRef<OpFoldResult> tileSizes, ArrayRef<StringRef> iteratorTypes,
    ArrayRef<Range> loopBounds, unsigned loopDepth,
    SmallVectorImpl<OpFoldResult> &offsets,
    ArrayRef<linalg::ProcInfo> distributionInfo) {
  Location loc = tilableOp.getLoc();
  // If this is the innermost loop, then generated the tiled implementation of
  // the op by invoking the TiledOpInterface methods.
  if (loopDepth == tileSizes.size()) {
    SmallVector<SmallVector<OpFoldResult, 4>> resultOffsets;
    SmallVector<SmallVector<OpFoldResult, 4>> resultSizes;
    Operation *tiledOp = tilableOp.getTiledImplementation(
        builder, outputs, offsets, tileSizes, resultOffsets, resultSizes);
    if (!tiledOp) {
      return static_cast<LogicalResult>(
          tilableOp.emitOpError("failed to get tiled implementation"));
    }
    assert(tiledOp->getNumResults() == 0 ||
           (resultOffsets.size() == tiledOp->getNumResults()));
    TiledOp ret;
    ret.op = tiledOp;

    // If the operation has results, then the result of the tiled operation is
    // to be inserted into the `initValues` and returned.
    if (tiledOp->getNumResults()) {
      SmallVector<Value> results;
      auto oneAttr = builder.getI64IntegerAttr(1);
      results.reserve(tiledOp->getNumResults());
      for (auto en : llvm::enumerate(tiledOp->getResults())) {
        Value result = en.value();
        ArrayRef<OpFoldResult> offsets(resultOffsets[en.index()]);
        ArrayRef<OpFoldResult> sizes(resultSizes[en.index()]);
        SmallVector<OpFoldResult> strides(offsets.size(), oneAttr);
        Value insert = builder.create<tensor::InsertSliceOp>(
            loc, result, outputs[en.index()], offsets, sizes, strides);
        results.push_back(insert);
      }
      std::swap(ret.results, results);
    }
    return ret;
  }

  // If tile size at this depth is empty, do nothing.
  if (isUntiledLoop(tileSizes[loopDepth])) {
    auto zeroAttr = builder.getI64IntegerAttr(0);
    offsets.push_back(zeroAttr);
    assert(matchPattern(loopBounds[loopDepth].offset, m_Zero()) &&
           "expected loop bounds to have lower bound of zero");
    tileSizes[loopDepth] = getOpFoldResult(loopBounds[loopDepth].size);
    return tileInterfaceOpImpl(builder, tilableOp, outputs, tileSizes,
                               iteratorTypes, loopBounds, loopDepth + 1,
                               offsets, distributionInfo);
  }

  // Generate an scf.for for the current loop depth.
  Value lb = loopBounds[loopDepth].offset;
  Value ub = loopBounds[loopDepth].size;
  if (!matchPattern(loopBounds[loopDepth].stride, m_One())) {
    return static_cast<LogicalResult>(
        tilableOp.emitOpError("expected stride to be 1"));
  }
  Value step = getValue(builder, loc, tileSizes[loopDepth]);

  // Update lb, ub and step for cyclic distribution.
  if (!distributionInfo.empty() &&
      iteratorTypes[loopDepth] == getParallelIteratorTypeName()) {
    linalg::updateBoundsForCyclicDistribution(
        builder, loc, distributionInfo.front().procId,
        distributionInfo.front().nprocs, lb, ub, step);
    distributionInfo = distributionInfo.drop_front();
  }
  FailureOr<TiledOp> innerReturnValue;
  bool isBufferTiling = tilableOp->getNumResults() == 0;
  ValueRange initValues(isBufferTiling ? ValueRange{} : outputs);
  auto forOp = builder.create<scf::ForOp>(
      loc, lb, ub, step, initValues,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        offsets.push_back(iv);
        auto affineMaps = AffineMap::inferFromExprList({ArrayRef<AffineExpr>{
            b.getAffineSymbolExpr(0),
            b.getAffineSymbolExpr(1) - b.getAffineDimExpr(0)}})[0];
        // Similar to linalg tiling, the tile size is the min(tileSizes, ub -
        // iv) to account for cases where tile size does not divide (ub - lb)
        // exactly.
        Value inBoundsTileSize = b.create<AffineMinOp>(
            loc, affineMaps,
            ValueRange{iv, getValue(builder, loc, tileSizes[loopDepth]), ub});
        tileSizes[loopDepth] = getOpFoldResult(inBoundsTileSize);
        // Recursively proceed to generate the tiled loop for the next level.
        innerReturnValue =
            tileInterfaceOpImpl(b, tilableOp, (isBufferTiling ? outputs : args),
                                tileSizes, iteratorTypes, loopBounds,
                                loopDepth + 1, offsets, distributionInfo);
        if (failed(innerReturnValue)) return;
        b.create<scf::YieldOp>(loc, innerReturnValue->results);
      });
  if (failed(innerReturnValue)) {
    return innerReturnValue;
  }
  innerReturnValue->loops.insert(innerReturnValue->loops.begin(),
                                 forOp.getOperation());
  innerReturnValue->results = forOp.getResults();
  return innerReturnValue;
}

FailureOr<TiledOp> tileInterfaceOp(OpBuilder &b, Operation *op, ValueRange dest,
                                   const linalg::LinalgTilingOptions &options) {
  TiledOpInterface tilableOp = dyn_cast<TiledOpInterface>(op);
  if (!tilableOp) return TiledOp{};

  SmallVector<StringRef> iteratorTypes = tilableOp.getLoopIteratorTypes();
  SmallVector<Value, 4> tileSizesVals =
      options.tileSizeComputationFunction(b, op);
  auto zeroAttr = b.getI64IntegerAttr(0);

  // The actual tile sizes used converts `Value` defined as constant 0, to a
  // zero integer attributes. Currently if the iterator type is not "parallel",
  // the tile size is forced to zero as well.
  auto tileSizes = getOpFoldResult(tileSizesVals);
  tileSizes.resize(iteratorTypes.size(), zeroAttr);
  for (auto en : llvm::enumerate(iteratorTypes)) {
    if (en.value() == getParallelIteratorTypeName()) continue;
    if (!isUntiledLoop(tileSizes[en.index()])) {
      return static_cast<LogicalResult>(tilableOp.emitOpError(
          "unimplemented tiling of non-parallel loop iterator type"));
    }
  }

  // Trivial early exit case of tile sizes being zero for all parallel loops.
  if (llvm::all_of(tileSizes, isUntiledLoop)) {
    return TiledOp{op, {}, {}};
  }

  SmallVector<Range> loopBounds = tilableOp.getLoopBounds(b);
  SmallVector<linalg::ProcInfo> distributionInfo;
  // If the tiled loops are distributed, get the proc_id and nprocs for the
  // distributed loops. First collect the parallel loops by iterating over the
  // tileSizes and getting the loops that are distribute, i.e.,
  // - parallel, i.e. iteratorTypes is "parallel"
  // - tiled, i.e. tileSize != 0
  if (options.distribution) {
    SmallVector<Range> distributedLoopRange;
    for (auto i : llvm::seq<unsigned>(0, tileSizes.size())) {
      if (isUntiledLoop(tileSizes[i])) continue;
      if (iteratorTypes[i] != getParallelIteratorTypeName()) continue;
      distributedLoopRange.push_back(loopBounds[i]);
    }
    distributionInfo = options.distribution->procInfo(b, tilableOp.getLoc(),
                                                      distributedLoopRange);
  }

  SmallVector<OpFoldResult> offsets;
  return tileInterfaceOpImpl(b, tilableOp, dest, tileSizes, iteratorTypes,
                             loopBounds, 0, offsets, distributionInfo);
}

//===----------------------------------------------------------------------===//
// Interface implementations for external operations.
//===----------------------------------------------------------------------===//

namespace {
struct InsertSliceTiledOpInterface
    : public TiledOpInterface::ExternalModel<InsertSliceTiledOpInterface,
                                             tensor::InsertSliceOp> {
  SmallVector<StringRef> getLoopIteratorTypes(Operation *op) const {
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    return SmallVector<StringRef>(insertSliceOp.getSourceType().getRank(),
                                  getParallelIteratorTypeName());
  }

  SmallVector<Range> getLoopBounds(Operation *op, OpBuilder &b) const {
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    Value source = insertSliceOp.source();
    RankedTensorType sourceType = insertSliceOp.getSourceType();
    Location loc = op->getLoc();
    Value zero = b.create<ConstantIndexOp>(loc, 0);
    Value one = b.create<ConstantIndexOp>(loc, 1);
    SmallVector<Range> loopBounds(sourceType.getRank(),
                                  Range{zero, nullptr, one});
    for (auto dim :
         llvm::seq<int64_t>(0, insertSliceOp.getSourceType().getRank())) {
      loopBounds[dim].size = b.create<tensor::DimOp>(loc, source, dim);
    }
    return loopBounds;
  }

  Operation *getTiledImplementation(
      Operation *op, OpBuilder &b, ValueRange outputs,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<SmallVector<OpFoldResult, 4>> &resultOffsets,
      SmallVectorImpl<SmallVector<OpFoldResult, 4>> &resultSizes) const {
    // Compute a subtensor of the source based on the offsets.
    auto insertOp = cast<tensor::InsertSliceOp>(op);
    auto opOffsets = insertOp.getMixedOffsets();
    auto opSizes = insertOp.getMixedSizes();
    auto opStrides = insertOp.getMixedStrides();
    if (!llvm::all_of(opStrides, [&](OpFoldResult valueOrAttr) {
          return isValue(valueOrAttr, 1);
        })) {
      op->emitOpError("unable to tile operation with non-unit stride");
      return nullptr;
    }
    // The operation returned is just a tensor.extract_slice of the source with
    // the given offsets, sizes and strides. Setting the correct result offset
    // will make the sure the tiling algorithm will insert this slice into the
    // correct place in the destination.
    // The result offset is just the offset passed in plus the offset specified
    // in the op (since all strides are checked to be 1).
    unsigned offsetIndex = 0;
    ArrayRef<int64_t> sourceShape = insertOp.getSourceType().getShape();
    int64_t destRank = insertOp.getType().getRank();
    resultOffsets.resize(1);
    resultOffsets[0].resize(destRank);
    resultSizes.resize(1);
    resultSizes[0].resize(destRank);
    Location loc = insertOp.getLoc();
    auto zeroAttr = b.getI64IntegerAttr(0);
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(offsets.size(), oneAttr);
    for (auto opOffset : llvm::enumerate(opOffsets)) {
      // Check for rank-reducing by checking that
      // 1) The corresponding opSize value is 1
      // 2) The current rank of the source is not 1.
      // Then the opOffset is for the rank-reduced dimension. Skip.
      unsigned opOffsetIndex = opOffset.index();
      if (isValue(opSizes[opOffsetIndex], 1) && sourceShape[offsetIndex] != 1) {
        resultOffsets[0][opOffsetIndex] = zeroAttr;
        resultSizes[0][opOffsetIndex] = oneAttr;
        continue;
      }
      OpFoldResult opOffsetVal = opOffset.value();
      OpFoldResult offset = offsets[offsetIndex];
      if (opOffsetVal.is<Attribute>() && offset.is<Attribute>()) {
        resultOffsets[0][opOffsetIndex] = b.getI64IntegerAttr(
            *getConstantValue(opOffsetVal) + *getConstantValue(offset));
      } else {
        AffineMap map = AffineMap::get(
            1, 1, {b.getAffineDimExpr(0) + b.getAffineSymbolExpr(0)});
        resultOffsets[0][opOffsetIndex] =
            b.create<AffineApplyOp>(loc, map,
                                    ValueRange{getValue(b, loc, offset),
                                               getValue(b, loc, opOffsetVal)})
                .getResult();
      }
      resultSizes[0][opOffsetIndex] = sizes[offsetIndex];
      offsetIndex++;
    }
    return b.create<tensor::ExtractSliceOp>(loc, insertOp.source(), offsets,
                                            sizes, strides);
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// Patterns for tiling LinalgExtOps.
//===----------------------------------------------------------------------===//

namespace {
/// Base pattern for tiling TiledOpInterfaceOps.
struct TiledOpInterfaceBaseTilingPattern : public RewritePattern {
  TiledOpInterfaceBaseTilingPattern(StringRef opName, MLIRContext *context,
                                    linalg::LinalgTilingOptions options,
                                    linalg::LinalgTransformationFilter filter =
                                        linalg::LinalgTransformationFilter(),
                                    PatternBenefit benefit = 1)
      : RewritePattern(opName, benefit, context),
        filter(filter),
        options(options) {}

  LogicalResult matchAndRewriteBase(Operation *op, ValueRange dest,
                                    PatternRewriter &rewriter,
                                    TiledOp &result) const;

 private:
  /// LinalgTransformMarker handles special attribute manipulations.
  linalg::LinalgTransformationFilter filter;
  /// Options to control tiling;
  linalg::LinalgTilingOptions options;
};

template <typename OpTy>
struct LinalgExtTilingPattern : public TiledOpInterfaceBaseTilingPattern {
  LinalgExtTilingPattern(MLIRContext *context,
                         linalg::LinalgTilingOptions options,
                         linalg::LinalgTransformationFilter filter =
                             linalg::LinalgTransformationFilter(),
                         PatternBenefit benefit = 1)
      : TiledOpInterfaceBaseTilingPattern(OpTy::getOperationName(), context,
                                          options, filter, benefit) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto linalgExtOp = cast<LinalgExtOp>(op);
    TiledOp tiledOp;
    // Check for failure.
    if (failed(TiledOpInterfaceBaseTilingPattern::matchAndRewriteBase(
            op, linalgExtOp.outputs(), rewriter, tiledOp))) {
      return failure();
    }
    // Check for do-nothing case.
    if (!tiledOp.op) return failure();
    if (tiledOp.op != op) {
      if (tiledOp.results.empty()) {
        rewriter.eraseOp(op);
      } else {
        rewriter.replaceOp(op, tiledOp.results);
      }
    }
    return success();
  }
};

struct InsertSliceTilingPattern : public TiledOpInterfaceBaseTilingPattern {
  InsertSliceTilingPattern(MLIRContext *context,
                           linalg::LinalgTilingOptions options,
                           linalg::LinalgTransformationFilter filter =
                               linalg::LinalgTransformationFilter(),
                           PatternBenefit benefit = 1)
      : TiledOpInterfaceBaseTilingPattern(
            tensor::InsertSliceOp::getOperationName(), context, options, filter,
            benefit) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    tensor::InsertSliceOp insertSliceOp = cast<tensor::InsertSliceOp>(op);
    TiledOp tiledOp;
    // Check for failure.
    if (failed(TiledOpInterfaceBaseTilingPattern::matchAndRewriteBase(
            insertSliceOp, insertSliceOp.dest(), rewriter, tiledOp))) {
      return failure();
    }
    // Check for do-nothing case.
    if (!tiledOp.op) return failure();
    if (tiledOp.op != op) {
      if (tiledOp.results.empty()) {
        rewriter.eraseOp(op);
      } else {
        rewriter.replaceOp(op, tiledOp.results);
      }
    }
    return success();
  }
};
}  // namespace

LogicalResult TiledOpInterfaceBaseTilingPattern::matchAndRewriteBase(
    Operation *op, ValueRange dest, PatternRewriter &rewriter,
    TiledOp &result) const {
  if (failed(filter.checkAndNotify(rewriter, op))) return failure();
  if (failed(verifySupportedTilingOptions(rewriter, op, options))) {
    return failure();
  }

  FailureOr<TiledOp> res = tileInterfaceOp(rewriter, op, dest, options);
  if (failed(res)) return res;
  result = *res;
  if (result.op) {
    filter.replaceLinalgTransformationFilter(rewriter, result.op);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Test pass for tiling Linalg Ext ops
//===----------------------------------------------------------------------===//

namespace {
struct TiledOpInterfaceTilingPass
    : public TiledOpInterfaceTilingBase<TiledOpInterfaceTilingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AffineDialect, IREE::Flow::FlowDialect, linalg::LinalgDialect,
                memref::MemRefDialect, StandardOpsDialect,
                tensor::TensorDialect, scf::SCFDialect>();
  }

  LogicalResult initialize(MLIRContext *context) override;
  void runOnOperation() override;
};
}  // namespace

template <typename OpTy>
static Value buildFlowWorkgroupInfoOp(OpBuilder &b, unsigned dim) {
  return b.template create<OpTy>(b.getInsertionPoint()->getLoc(), dim);
}

LogicalResult TiledOpInterfaceTilingPass::initialize(MLIRContext *context) {
  tensor::InsertSliceOp::attachInterface<InsertSliceTiledOpInterface>(*context);
  return success();
}

void TiledOpInterfaceTilingPass::runOnOperation() {
  FuncOp funcOp = getOperation();
  MLIRContext *context = funcOp.getContext();

  tensor::InsertSliceOp::attachInterface<InsertSliceTiledOpInterface>(*context);

  RewritePatternSet patterns(context);
  patterns.add<LinalgExtTilingPattern<ScatterOp>>(
      context, linalg::LinalgTilingOptions().setTileSizes({10, 20}),
      linalg::LinalgTransformationFilter(
          Identifier::get("tiling_input", context),
          Identifier::get("tiling_output", context)));
  patterns.add<LinalgExtTilingPattern<ScatterOp>>(
      context, linalg::LinalgTilingOptions().setTileSizes(ArrayRef<int64_t>{0}),
      linalg::LinalgTransformationFilter(
          Identifier::get("no_tiling_input", context),
          Identifier::get("no_tiling_output", context)));

  patterns.add<LinalgExtTilingPattern<SortOp>>(
      context, linalg::LinalgTilingOptions().setTileSizes({0, 20}),
      linalg::LinalgTransformationFilter(
          Identifier::get("outer_reduce_input", context),
          Identifier::get("outer_reduce_output", context)));
  patterns.add<LinalgExtTilingPattern<SortOp>>(
      context, linalg::LinalgTilingOptions().setTileSizes({10, 0, 0}),
      linalg::LinalgTransformationFilter(
          Identifier::get("inner_reduce_input", context),
          Identifier::get("inner_reduce_output", context)));

  static linalg::LinalgLoopDistributionOptions workgroupDistributionOptions = {
      [](OpBuilder &builder, Location loc, ArrayRef<Range> parallelLoopRanges) {
        auto numParallelDims = parallelLoopRanges.size();

        SmallVector<linalg::ProcInfo, 3> procInfo(numParallelDims);
        for (size_t dim = 0; dim < numParallelDims; ++dim) {
          procInfo[numParallelDims - dim - 1] = {
              buildFlowWorkgroupInfoOp<IREE::Flow::DispatchWorkgroupIDOp>(
                  builder, dim),
              buildFlowWorkgroupInfoOp<IREE::Flow::DispatchWorkgroupCountOp>(
                  builder, dim)};
        }
        return procInfo;
      },
      {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
       linalg::DistributionMethod::Cyclic},
      DenseMap<StringRef,
               std::function<linalg::ProcInfo(OpBuilder &, Location)>>()};

  patterns
      .add<LinalgExtTilingPattern<ScatterOp>, LinalgExtTilingPattern<SortOp>>(
          context,
          linalg::LinalgTilingOptions()
              .setTileSizes(ArrayRef<int64_t>{10, 0, 30})
              .setDistributionOptions(workgroupDistributionOptions),
          linalg::LinalgTransformationFilter(
              Identifier::get("distribute_input", context),
              Identifier::get("distribute_output", context)));

  patterns.add<InsertSliceTilingPattern>(
      context, linalg::LinalgTilingOptions().setTileSizes({10, 20}),
      linalg::LinalgTransformationFilter(
          Identifier::get("tiling_input", context),
          Identifier::get("tiling_output", context)));

  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<FuncOp>> createTiledOpInterfaceTilingPass() {
  return std::make_unique<TiledOpInterfaceTilingPass>();
}

}  // namespace linalg_ext
}  // namespace iree_compiler
}  // namespace mlir
