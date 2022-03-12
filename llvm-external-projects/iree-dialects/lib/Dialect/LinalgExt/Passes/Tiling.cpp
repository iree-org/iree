// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/Input/InputOps.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
namespace IREE = mlir::iree_compiler::IREE;
using namespace IREE::LinalgExt;

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
    return builder.create<arith::ConstantIndexOp>(
        loc, attr.cast<IntegerAttr>().getInt());
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
  // TODO(#7073): Put the check back. This is required by tiling linalg_ext.fft
  // op. We can put the check back after updating linalg_ext.fft semantics.
  // if (!matchPattern(loopBounds[loopDepth].stride, m_One())) {
  // return static_cast<LogicalResult>(
  // tilableOp.emitOpError("expected stride to be 1"));
  //}
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
  SmallVector<Value> dest = tilableOp.getDestinationOperands(b);
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

  SmallVector<Range> loopBounds = tilableOp.getIterationDomain(b);
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

LogicalResult TiledOpInterfaceBaseTilingPattern::matchAndRewriteBase(
    TiledOpInterface tilableOp, PatternRewriter &rewriter,
    TiledOp &result) const {
  if (failed(filter.checkAndNotify(rewriter, tilableOp))) {
    return failure();
  }
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
    registry.insert<
        AffineDialect, IREE::Input::IREEInputDialect, linalg::LinalgDialect,
        IREE::LinalgExt::IREELinalgExtDialect, memref::MemRefDialect,
        func::FuncDialect, mlir::arith::ArithmeticDialect, math::MathDialect,
        tensor::TensorDialect, scf::SCFDialect>();
  }
  void runOnOperation() override;
};
}  // namespace

template <typename OpTy>
static Value buildFlowWorkgroupInfoOp(OpBuilder &b, unsigned dim) {
  return b.template create<OpTy>(b.getInsertionPoint()->getLoc(), dim);
}

void TiledOpInterfaceTilingPass::runOnOperation() {
  FuncOp funcOp = getOperation();
  MLIRContext *context = funcOp.getContext();

  RewritePatternSet patterns(context);
  patterns.add<TiledOpInterfaceTilingPattern>(
      context, linalg::LinalgTilingOptions().setTileSizes({10, 20}),
      linalg::LinalgTransformationFilter(
          StringAttr::get(context, "tiling_input"),
          StringAttr::get(context, "tiling_output")));
  patterns.add<TiledOpInterfaceTilingPattern>(
      context, linalg::LinalgTilingOptions().setTileSizes(ArrayRef<int64_t>{0}),
      linalg::LinalgTransformationFilter(
          StringAttr::get(context, "no_tiling_input"),
          StringAttr::get(context, "no_tiling_output")));

  patterns.add<TiledOpInterfaceTilingPattern>(
      context, linalg::LinalgTilingOptions().setTileSizes({0, 20}),
      linalg::LinalgTransformationFilter(
          StringAttr::get(context, "outer_reduce_input"),
          StringAttr::get(context, "outer_reduce_output")));
  patterns.add<TiledOpInterfaceTilingPattern>(
      context, linalg::LinalgTilingOptions().setTileSizes({10, 0, 0}),
      linalg::LinalgTransformationFilter(
          StringAttr::get(context, "inner_reduce_input"),
          StringAttr::get(context, "inner_reduce_output")));

  static linalg::LinalgLoopDistributionOptions workgroupDistributionOptions = {
      [](OpBuilder &builder, Location loc, ArrayRef<Range> parallelLoopRanges) {
        auto numParallelDims = parallelLoopRanges.size();

        SmallVector<linalg::ProcInfo, 3> procInfo(numParallelDims);
        for (size_t dim = 0; dim < numParallelDims; ++dim) {
          procInfo[numParallelDims - dim - 1] = {
              buildFlowWorkgroupInfoOp<IREE::Input::DispatchWorkgroupIDOp>(
                  builder, dim),
              buildFlowWorkgroupInfoOp<IREE::Input::DispatchWorkgroupCountOp>(
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
          StringAttr::get(context, "distribute_input"),
          StringAttr::get(context, "distribute_output")));

  patterns.add<TiledOpInterfaceTilingPattern>(
      context,
      linalg::LinalgTilingOptions().setTileSizes(ArrayRef<int64_t>{32}),
      linalg::LinalgTransformationFilter(
          StringAttr::get(context, "tiling_1d_stage5_fft_input"),
          StringAttr::get(context, "tiling_1d_stage5_fft_output")));

  patterns.add<TiledOpInterfaceTilingPattern>(
      context,
      linalg::LinalgTilingOptions().setTileSizes(ArrayRef<int64_t>{10, 32}),
      linalg::LinalgTransformationFilter(
          StringAttr::get(context, "tiling_2d_stage5_fft_input"),
          StringAttr::get(context, "tiling_2d_stage5_fft_output")));

  patterns.add<TiledOpInterfaceTilingPattern>(
      context, linalg::LinalgTilingOptions().setTileSizes({0, 20}),
      linalg::LinalgTransformationFilter(
          StringAttr::get(context, "tiling_repeated_indices_scatter_input"),
          StringAttr::get(context, "tiling_repeated_indices_scatter_output")));

  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<FuncOp>>
IREE::LinalgExt::createTiledOpInterfaceTilingPass() {
  return std::make_unique<TiledOpInterfaceTilingPass>();
}
