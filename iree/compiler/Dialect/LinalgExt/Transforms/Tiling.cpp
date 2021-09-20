// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

/// Returns true if loop is untiled. Only checks if the value is statically
/// zero. It is assumed that a `Value` defined by a constant op is already
/// converted to an `IntegerAttr` of that value. So here just return true if
/// this is an attribute with a zero value.
static bool isUntiledLoop(OpFoldResult valueOrAttr) {
  Optional<int64_t> intVal = getConstantIntValue(valueOrAttr);
  return intVal && *intVal == 0;
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
    TiledOp ret;
    ret.op = tilableOp.getTiledImplementation(builder, outputs, offsets,
                                              tileSizes, ret.results);
    if (!ret.op) {
      return static_cast<LogicalResult>(
          tilableOp.emitOpError("failed to get tiled implementation"));
    }
    return ret;
  }

  // If tile size at this depth is empty, do nothing.
  if (isUntiledLoop(tileSizes[loopDepth])) {
    auto zeroAttr = builder.getI64IntegerAttr(0);
    offsets.push_back(zeroAttr);
    assert(matchPattern(loopBounds[loopDepth].offset, m_Zero()) &&
           "expected loop bounds to have lower bound of zero");
    tileSizes[loopDepth] = getAsOpFoldResult(loopBounds[loopDepth].size);
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
        tileSizes[loopDepth] = getAsOpFoldResult(inBoundsTileSize);
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

FailureOr<TiledOp> tileInterfaceOp(OpBuilder &b, TiledOpInterface tilableOp,
                                   const linalg::LinalgTilingOptions &options) {
  SmallVector<Value> dest = tilableOp.getDestinationOperands();
  if (dest.empty()) {
    return static_cast<LogicalResult>(tilableOp.emitOpError(
        "cannot tile operation without destination operands"));
  }

  SmallVector<StringRef> iteratorTypes = tilableOp.getLoopIteratorTypes();
  SmallVector<Value, 4> tileSizesVals =
      options.tileSizeComputationFunction(b, tilableOp);
  auto zeroAttr = b.getI64IntegerAttr(0);

  // The actual tile sizes used converts `Value` defined as constant 0, to a
  // zero integer attributes. Currently if the iterator type is not "parallel",
  // the tile size is forced to zero as well.
  auto tileSizes = getAsOpFoldResult(tileSizesVals);
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
    return TiledOp{tilableOp, {}, {}};
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
  SmallVector<Value> getDestinationOperands(Operation *op) const {
    SmallVector<Value> dest;
    dest.push_back(cast<tensor::InsertSliceOp>(op).dest());
    return dest;
  }

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

  Operation *getTiledImplementation(Operation *op, OpBuilder &b,
                                    ValueRange outputs,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes,
                                    SmallVectorImpl<Value> &results) const {
    auto insertOp = cast<tensor::InsertSliceOp>(op);
    // Compute a subtensor of the source based on the offsets.
    auto opStrides = insertOp.getMixedStrides();
    if (!llvm::all_of(opStrides, [&](OpFoldResult valueOrAttr) {
          Optional<int64_t> intVal = getConstantIntValue(valueOrAttr);
          return intVal && *intVal == 1;
        })) {
      op->emitOpError("unable to tile operation with non-unit stride");
      return nullptr;
    }
    Location loc = insertOp.getLoc();
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(offsets.size(), oneAttr);
    auto extractSliceOp = b.create<tensor::ExtractSliceOp>(
        loc, insertOp.source(), offsets, sizes, strides);

    // The offsets for the insert is based on the op offsets plus the offsets of
    // the loops passed in.
    auto opOffsets = insertOp.getMixedOffsets();
    auto opSizes = insertOp.getMixedSizes();
    unsigned offsetIndex = 0;
    ArrayRef<int64_t> sourceShape = insertOp.getSourceType().getShape();
    int64_t destRank = insertOp.getType().getRank();
    SmallVector<OpFoldResult> resultOffsets(destRank);
    SmallVector<OpFoldResult> resultSizes(destRank);
    auto zeroAttr = b.getI64IntegerAttr(0);
    for (auto opOffset : llvm::enumerate(opOffsets)) {
      // Check for rank-reducing by checking that
      // 1) The corresponding opSize value is 1
      // 2) The current rank of the source is not 1.
      // Then the opOffset is for the rank-reduced dimension. Skip.
      unsigned opOffsetIndex = opOffset.index();
      Optional<int64_t> opSizeVal = getConstantIntValue(opSizes[opOffsetIndex]);
      if (opSizeVal && *opSizeVal == 1 && sourceShape[offsetIndex] != 1) {
        resultOffsets[opOffsetIndex] = zeroAttr;
        resultSizes[opOffsetIndex] = oneAttr;
        continue;
      }
      OpFoldResult opOffsetVal = opOffset.value();
      OpFoldResult offset = offsets[offsetIndex];
      if (opOffsetVal.is<Attribute>() && offset.is<Attribute>()) {
        resultOffsets[opOffsetIndex] = b.getI64IntegerAttr(
            *getConstantIntValue(opOffsetVal) + *getConstantIntValue(offset));
      } else {
        AffineMap map = AffineMap::get(
            1, 1, {b.getAffineDimExpr(0) + b.getAffineSymbolExpr(0)});
        resultOffsets[opOffsetIndex] =
            b.create<AffineApplyOp>(loc, map,
                                    ValueRange{getValue(b, loc, offset),
                                               getValue(b, loc, opOffsetVal)})
                .getResult();
      }
      resultSizes[opOffsetIndex] = sizes[offsetIndex];
      offsetIndex++;
    }
    SmallVector<OpFoldResult> resultStrides(destRank, oneAttr);
    auto tiledInsertOp = b.create<tensor::InsertSliceOp>(
        loc, extractSliceOp.result(), outputs[0], resultOffsets, resultSizes,
        resultStrides);
    results.push_back(tiledInsertOp.result());
    return extractSliceOp;
  }
};
}  // namespace

LogicalResult TiledOpInterfaceBaseTilingPattern::matchAndRewriteBase(
    TiledOpInterface tilableOp, PatternRewriter &rewriter,
    TiledOp &result) const {
  if (failed(filter.checkAndNotify(rewriter, tilableOp))) return failure();
  if (failed(verifySupportedTilingOptions(rewriter, tilableOp, options))) {
    return failure();
  }

  FailureOr<TiledOp> res = tileInterfaceOp(rewriter, tilableOp, options);
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
                linalg_ext::LinalgExtDialect, memref::MemRefDialect,
                StandardOpsDialect, tensor::TensorDialect, scf::SCFDialect>();
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
  // TODO(ravishankarm): When the interface is added during registration, remove
  // this initialization.
  tensor::InsertSliceOp::attachInterface<InsertSliceTiledOpInterface>(*context);
  return success();
}

void TiledOpInterfaceTilingPass::runOnOperation() {
  FuncOp funcOp = getOperation();
  MLIRContext *context = funcOp.getContext();

  RewritePatternSet patterns(context);
  patterns.add<TiledOpInterfaceTilingPattern>(
      context, linalg::LinalgTilingOptions().setTileSizes({10, 20}),
      linalg::LinalgTransformationFilter(
          Identifier::get("tiling_input", context),
          Identifier::get("tiling_output", context)));
  patterns.add<TiledOpInterfaceTilingPattern>(
      context, linalg::LinalgTilingOptions().setTileSizes(ArrayRef<int64_t>{0}),
      linalg::LinalgTransformationFilter(
          Identifier::get("no_tiling_input", context),
          Identifier::get("no_tiling_output", context)));

  patterns.add<TiledOpInterfaceTilingPattern>(
      context, linalg::LinalgTilingOptions().setTileSizes({0, 20}),
      linalg::LinalgTransformationFilter(
          Identifier::get("outer_reduce_input", context),
          Identifier::get("outer_reduce_output", context)));
  patterns.add<TiledOpInterfaceTilingPattern>(
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

  patterns.add<TiledOpInterfaceTilingPattern>(
      context,
      linalg::LinalgTilingOptions()
          .setTileSizes(ArrayRef<int64_t>{10, 0, 30})
          .setDistributionOptions(workgroupDistributionOptions),
      linalg::LinalgTransformationFilter(
          Identifier::get("distribute_input", context),
          Identifier::get("distribute_output", context)));

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
