// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/Utils.h"

#include "iree/compiler/Codegen/Interfaces/ProcessorOpInterfaces.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "iree-codegen-utils"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions to get entry points
//===----------------------------------------------------------------------===//

FailureOr<IREE::HAL::ExecutableExportOp> getEntryPoint(func::FuncOp funcOp) {
  auto variantOp = funcOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  if (!variantOp) return failure();

  for (auto op : variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
    if (op.getSymName() == funcOp.getName()) {
      return op;
    }
  }
  return failure();
}

FailureOr<IREE::HAL::ExecutableVariantOp> getExecutableVariantOp(
    Operation *op) {
  if (auto result = dyn_cast<IREE::HAL::ExecutableVariantOp>(op)) {
    return result;
  }
  if (auto result = op->getParentOfType<IREE::HAL::ExecutableVariantOp>()) {
    return result;
  }
  return failure();
}

bool isEntryPoint(func::FuncOp func) {
  return func.isPublic() && succeeded(getEntryPoint(func));
}

llvm::StringMap<IREE::HAL::ExecutableExportOp> getAllEntryPoints(
    ModuleOp module) {
  auto variantOp = module->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps;
  for (auto op : variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
    exportOps[op.getSymName()] = op;
  }
  return exportOps;
}

Optional<StringAttr> getConfigStringAttr(
    IREE::HAL::ExecutableTargetAttr targetAttr, StringRef stringAttr) {
  if (!targetAttr) return std::nullopt;
  auto config = targetAttr.getConfiguration();
  if (!config) return std::nullopt;
  auto attr = config.getAs<StringAttr>(stringAttr);
  if (!attr) return std::nullopt;
  return attr;
}

Optional<IntegerAttr> getConfigIntegerAttr(
    IREE::HAL::ExecutableTargetAttr targetAttr, StringRef integerAttr) {
  if (!targetAttr) return std::nullopt;
  auto config = targetAttr.getConfiguration();
  if (!config) return std::nullopt;
  auto attr = config.getAs<IntegerAttr>(integerAttr);
  if (!attr) return std::nullopt;
  return attr;
}

Optional<BoolAttr> getConfigBoolAttr(IREE::HAL::ExecutableTargetAttr targetAttr,
                                     StringRef integerAttr) {
  if (!targetAttr) return std::nullopt;
  auto config = targetAttr.getConfiguration();
  if (!config) return std::nullopt;
  auto attr = config.getAs<BoolAttr>(integerAttr);
  if (!attr) return std::nullopt;
  return attr;
}

Optional<llvm::Triple> getTargetTriple(
    IREE::HAL::ExecutableTargetAttr targetAttr) {
  auto triple = getConfigStringAttr(targetAttr, "target_triple");
  if (!triple) return std::nullopt;
  return llvm::Triple(triple.value().str());
}

bool isVMVXBackend(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return targetAttr && targetAttr.getBackend().getValue().startswith("vmvx");
}

bool hasMicrokernels(IREE::HAL::ExecutableTargetAttr targetAttr) {
  auto enableMicrokernels = getConfigBoolAttr(targetAttr, "ukernels");
  return enableMicrokernels && enableMicrokernels->getValue();
}

bool isReadOnly(Value v) {
  Operation *definingOp = v.getDefiningOp();
  if (!definingOp) return false;
  return TypeSwitch<Operation *, bool>(definingOp)
      .Case<arith::ConstantOp>(
          [&](arith::ConstantOp constantOp) { return true; })
      .Case<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
          [&](auto op) { return isReadOnly(op.getSrc()); })
      .Case<tensor::CastOp, tensor::ExtractSliceOp>(
          [&](auto op) { return isReadOnly(op.getSource()); })
      .Case<IREE::Flow::DispatchTensorLoadOp>(
          [&](IREE::Flow::DispatchTensorLoadOp loadOp) {
            return loadOp.getSource()
                       .getType()
                       .cast<IREE::Flow::DispatchTensorType>()
                       .getAccess() == IREE::Flow::TensorAccess::ReadOnly;
          })
      .Default([&](Operation *op) { return false; });
}

//===----------------------------------------------------------------------===//
// Utility functions to set configurations
//===----------------------------------------------------------------------===//

/// Returns the first of `exprs` which is of the type `T`.
template <typename T>
static AffineExpr getAffineExprOfType(ArrayRef<AffineExpr> exprs) {
  for (auto expr : exprs) {
    if (expr.isa<T>()) return expr;
  }
  return nullptr;
}

/// Returns true if the `expr` is on of the types in {`T1`, `T2`, `T3...`}.
template <typename T>
static bool isaAffineExprOfType(AffineExpr expr) {
  return expr.isa<T>();
}
template <typename T1, typename T2, typename... T3>
static bool isaAffineExprOfType(AffineExpr expr) {
  if (expr.isa<T1>()) {
    return true;
  }
  return isaAffineExprOfType<T2, T3...>(expr);
}

/// Returns a Value that represents the value for symbol or dim expr for the map
/// in the `applyOp`.
static Value getValueForDimOrSymbol(AffineApplyOp applyOp, AffineExpr expr) {
  unsigned numDims = applyOp.getAffineMap().getNumDims();
  if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
    return applyOp.getOperand(dimExpr.getPosition());
  }
  if (auto symbolExpr = expr.dyn_cast<AffineSymbolExpr>()) {
    return applyOp.getOperand(numDims + symbolExpr.getPosition());
  }
  return nullptr;
}
static SmallVector<Value> getValuesForDimsOrSymbols(
    AffineApplyOp applyOp, ArrayRef<AffineExpr> exprs) {
  SmallVector<Value> vals;
  for (auto expr : exprs) {
    vals.push_back(getValueForDimOrSymbol(applyOp, expr));
  }
  return vals;
}

/// Returns the dimension for any operation that implements processor op
/// interfaces.
template <typename T>
static Optional<unsigned> getDimension(Operation *op) {
  if (auto tOp = dyn_cast<T>(op)) {
    return tOp.getDimIndex();
  }
  return std::nullopt;
}
template <typename T1, typename T2, typename... T3>
static Optional<unsigned> getDimension(Operation *op) {
  if (!op) return std::nullopt;
  if (auto dimension = getDimension<T1>(op)) {
    return dimension;
  }
  return getDimension<T2, T3...>(op);
}

/// Checks that all `vals` are defined by some processor id/count/size ops using
/// the same `dimension`. If any element of `vals` is not defined by one of
/// these ops, or the dimensions dont match, returns std::nullopt; oterhwise,
/// returns the dimension.  If `refDimension` is passed checks if the dimension
/// matches the given value.
template <typename... T>
static Optional<unsigned> checkDimensions(
    ArrayRef<Value> vals, Optional<unsigned> refDimension = std::nullopt) {
  for (auto v : vals) {
    auto currDimension = getDimension<T...>(v.getDefiningOp());
    if (!currDimension) return std::nullopt;
    if (refDimension) {
      if (refDimension.value() != currDimension.value()) {
        return std::nullopt;
      }
    } else {
      refDimension = currDimension.value();
    }
  }
  return refDimension;
}

namespace {
/// Visitor to walk `lb` of a distributed loop. Expected the expression to be of
/// the form `a + b * c`, where `a` is the original `lb` and `b`, `c` are either
/// hal.interface.workgroup.id or hal.interface.workgroup.size.
class LowerBoundExprVisitor
    : public AffineExprVisitor<LowerBoundExprVisitor, LogicalResult> {
 public:
  LowerBoundExprVisitor(AffineApplyOp applyOp,
                        LoopTilingAndDistributionInfo &loopInfo)
      : applyOp(applyOp), loopInfo(loopInfo) {}

  LogicalResult visitSymbolExpr(AffineSymbolExpr /*expr*/) { return failure(); }
  LogicalResult visitDimExpr(AffineDimExpr /*expr*/) { return failure(); }
  LogicalResult visitConstantExpr(AffineConstantExpr /*expr*/) {
    return failure();
  }
  LogicalResult visitAffineBinaryOpExpr(AffineBinaryOpExpr /*expr*/) {
    return failure();
  }

  LogicalResult visitAddExpr(AffineBinaryOpExpr expr) {
    AffineExpr offsetExpr =
        getAffineExprOfType<AffineBinaryOpExpr>({expr.getLHS(), expr.getRHS()});
    if (!offsetExpr) {
      // One of the expressions has to be a binary op expr.
      return failure();
    }
    // The other expression must be the undistributed `lb`.
    AffineExpr lbExpr =
        (offsetExpr == expr.getLHS() ? expr.getRHS() : expr.getLHS());
    if (isaAffineExprOfType<AffineDimExpr, AffineSymbolExpr>(lbExpr)) {
      Value v = getValueForDimOrSymbol(applyOp, lbExpr);
      if (!v) {
        return failure();
      }
      loopInfo.untiledLowerBound = getAsOpFoldResult(v);
    } else if (auto constExpr = lbExpr.dyn_cast<AffineConstantExpr>()) {
      loopInfo.untiledLowerBound = IntegerAttr::get(
          IndexType::get(applyOp.getContext()), constExpr.getValue());
    } else {
      return failure();
    }
    return visit(offsetExpr);
  }

  LogicalResult visitMulExpr(AffineBinaryOpExpr expr) {
    SmallVector<Value> vals;
    Optional<unsigned> dimension;
    // workgroupSizeOp may have been folded into a constant expression.
    if (auto wgSize = expr.getRHS().dyn_cast<AffineConstantExpr>()) {
      vals = getValuesForDimsOrSymbols(applyOp, {expr.getLHS()});
      if (vals.size() != 1 || !vals[0]) {
        return failure();
      }
      loopInfo.tileSize = wgSize.getValue();
      dimension = checkDimensions<ProcessorIDInterface>(vals);
    } else {
      vals = getValuesForDimsOrSymbols(applyOp, {expr.getLHS(), expr.getRHS()});
      if (vals.size() != 2 || !vals[0] || !vals[1]) {
        return failure();
      }
      IntegerAttr tileSizeAttr;
      if (matchPattern(vals[1], m_Constant(&tileSizeAttr))) {
        loopInfo.tileSize = tileSizeAttr.getInt();
        dimension = checkDimensions<ProcessorIDInterface>(vals[0]);
      } else {
        dimension =
            checkDimensions<ProcessorIDInterface, ProcessorTileSizeInterface>(
                vals);
      }
    }
    if (!dimension) {
      return failure();
    }
    loopInfo.processorDistributionDim = dimension.value();
    if (!loopInfo.untiledLowerBound) {
      loopInfo.untiledLowerBound =
          IntegerAttr::get(IndexType::get(applyOp.getContext()), 0);
    }
    return success();
  }

 private:
  AffineApplyOp applyOp;
  LoopTilingAndDistributionInfo &loopInfo;
};

/// Visitor to walk the `step` of a distributed loop. Expected the expression to
/// be of the form `a * b * c`, where they could be the dynamic `step` or
/// defined by `hal.interface.workgroup.size`/`hal.interface.workgroup.count`
/// operation.
class StepExprVisitor
    : public AffineExprVisitor<StepExprVisitor, LogicalResult> {
 public:
  StepExprVisitor(AffineApplyOp applyOp,
                  LoopTilingAndDistributionInfo &loopInfo)
      : applyOp(applyOp), loopInfo(loopInfo) {}

  LogicalResult visitSymbolExpr(AffineSymbolExpr /*expr*/) { return failure(); }
  LogicalResult visitDimExpr(AffineDimExpr /*expr*/) { return failure(); }
  LogicalResult visitConstantExpr(AffineConstantExpr /*expr*/) {
    return failure();
  }
  LogicalResult visitAffineBinaryOpExpr(AffineBinaryOpExpr /*expr*/) {
    return failure();
  }

  LogicalResult visitMulExpr(AffineBinaryOpExpr expr) {
    // Check if one of the operands is a binary op expr.
    SmallVector<AffineExpr> sentinels;
    if (auto e = getAffineExprOfType<AffineBinaryOpExpr>(
            {expr.getLHS(), expr.getRHS()})) {
      AffineExpr otherExpr =
          (e == expr.getLHS() ? expr.getRHS() : expr.getLHS());
      if (failed(processSentinel(otherExpr, sentinels))) {
        return failure();
      }
      expr = e.cast<AffineBinaryOpExpr>();
    } else {
      // Check if the workgroup tile size is folded into the affine map itself.
      if (loopInfo.tileSize) {
        if (auto stepCst = expr.getRHS().dyn_cast<AffineConstantExpr>()) {
          loopInfo.untiledStep =
              IntegerAttr::get(IndexType::get(applyOp.getContext()),
                               stepCst.getValue() / *loopInfo.tileSize);
        } else {
          auto stepValue = getValueForDimOrSymbol(applyOp, expr.getRHS());
          IntegerAttr tileSizeAttr;
          if (stepValue && matchPattern(stepValue, m_Constant(&tileSizeAttr))) {
            loopInfo.untiledStep =
                IntegerAttr::get(IndexType::get(applyOp.getContext()),
                                 tileSizeAttr.getInt() / *loopInfo.tileSize);
          }
        }
      } else {
        loopInfo.untiledStep =
            IntegerAttr::get(IndexType::get(applyOp.getContext()), 1);
      }
    }

    if (failed(processSentinel(expr.getLHS(), sentinels)) ||
        (!loopInfo.tileSize &&
         failed(processSentinel(expr.getRHS(), sentinels)))) {
      return failure();
    }
    // Either there are 3 sentinels and step isnt set, or there are two
    // sentinels and the step is set.
    if (sentinels.size() == 3) {
      if (loopInfo.untiledStep) {
        return failure();
      }
      auto it = sentinels.begin();
      for (auto ie = sentinels.end(); it != ie; ++it) {
        Value v = getValueForDimOrSymbol(applyOp, *it);
        if (!v.getDefiningOp<IREE::HAL::InterfaceWorkgroupSizeOp>() &&
            !v.getDefiningOp<IREE::HAL::InterfaceWorkgroupCountOp>()) {
          loopInfo.untiledStep = getAsOpFoldResult(v);
          break;
        }
      }
      if (it != sentinels.end()) {
        sentinels.erase(it);
      }
    }

    if ((sentinels.size() != 2 || !loopInfo.untiledStep) &&
        (sentinels.size() != 1 || !loopInfo.tileSize)) {
      return failure();
    }
    SmallVector<Value> vals = getValuesForDimsOrSymbols(applyOp, sentinels);

    if ((loopInfo.tileSize && !checkDimensions<ProcessorCountInterface>(
                                  vals, loopInfo.processorDistributionDim)) ||
        (!loopInfo.tileSize &&
         !checkDimensions<ProcessorCountInterface, ProcessorTileSizeInterface>(
             vals, loopInfo.processorDistributionDim))) {
      return failure();
    }
    return success();
  }

 private:
  LogicalResult processSentinel(AffineExpr e,
                                SmallVectorImpl<AffineExpr> &sentinels) {
    if (isaAffineExprOfType<AffineDimExpr, AffineSymbolExpr>(e)) {
      sentinels.push_back(e);
      return success();
    } else if (auto constExpr = e.dyn_cast<AffineConstantExpr>()) {
      if (loopInfo.untiledStep) {
        return failure();
      }
      loopInfo.untiledStep = IntegerAttr::get(
          IndexType::get(applyOp.getContext()), constExpr.getValue());
      return success();
    }
    return failure();
  }

  AffineApplyOp applyOp;
  LoopTilingAndDistributionInfo &loopInfo;
};
}  // namespace

template <typename OpTy>
static Optional<unsigned> getInterfaceWorkgroupOpDim(Value value) {
  if (auto op = value.getDefiningOp<OpTy>()) {
    return op.getDimension().getZExtValue();
  }
  return std::nullopt;
}

/// Checks if the `forOp` is a tiled + distributed op. Looks for the op of this
/// form
/// ```
///   %dim = arith.constant ... : index
///   %id = flow.dispatch.workgroup.id[%dim]
///   %count = flow.dispatch.workgroup.count[%dim]
///   %size = flow.dispatch.workgroup.size[%dim]
///   %offset = affine.apply
///     affine_map<(d0)[s0, s1] -> (d0 + s0 * s1)>(%lb)[%id, %size]
///   %new_step = affine.apply
///     affine_map<(d0)[s0, s1] -> (d0 * s0 * s1)>(%step)[%id, %size]
///   scf.for %iv = %offset to %ub step %new_step { ... }
/// ```
Optional<LoopTilingAndDistributionInfo> isTiledAndDistributedLoop(
    scf::ForOp forOp) {
  LoopTilingAndDistributionInfo loopInfo;
  loopInfo.loop = forOp;
  loopInfo.untiledUpperBound = getAsOpFoldResult(forOp.getUpperBound());

  auto lbApplyOp = forOp.getLowerBound().getDefiningOp<AffineApplyOp>();
  auto stepApplyOp = forOp.getStep().getDefiningOp<AffineApplyOp>();

  if (!lbApplyOp || !stepApplyOp) {
    // Try to see if this is a specical case where we have:
    //   scf.for %iv = %id to %ub step %count
    Optional<unsigned> idDim;
    if (auto ifx = dyn_cast_or_null<ProcessorIDInterface>(
            forOp.getLowerBound().getDefiningOp())) {
      idDim = ifx.getDimIndex();
    }

    Optional<unsigned> countDim;
    if (auto ifx = dyn_cast_or_null<ProcessorCountInterface>(
            forOp.getStep().getDefiningOp())) {
      countDim = ifx.getDimIndex();
    }

    if (!idDim || !countDim) return std::nullopt;

    Builder b(forOp.getContext());
    loopInfo.untiledLowerBound = b.getIndexAttr(0);
    loopInfo.untiledStep = b.getIndexAttr(1);
    loopInfo.processorDistributionDim = idDim.value();
    // For such special case, the tile size is 1.
    loopInfo.tileSize = 1;
    return loopInfo;
  }

  LowerBoundExprVisitor lbVisitor(lbApplyOp, loopInfo);
  StepExprVisitor stepVisitor(stepApplyOp, loopInfo);

  if (failed(lbVisitor.visit(lbApplyOp.getAffineMap().getResults()[0]))) {
    return std::nullopt;
  }
  if (failed(stepVisitor.visit(stepApplyOp.getAffineMap().getResults()[0]))) {
    return std::nullopt;
  }
  if (!loopInfo.untiledLowerBound || !loopInfo.untiledStep) {
    return std::nullopt;
  }
  return loopInfo;
}

LogicalResult getFilteredOps(
    func::FuncOp funcOp, RootOpFilteringFn filteringFn,
    SmallVectorImpl<Operation *> &filteredOps,
    SmallVectorImpl<LoopTilingAndDistributionInfo> &tiledLoops) {
  Region &region = funcOp.getFunctionBody();
  if (!llvm::hasSingleElement(region)) {
    return funcOp.emitError("unable dispatch function with multiple blocks");
  }
  Block *body = &region.front();
  auto forOps = body->getOps<scf::ForOp>();
  while (!forOps.empty()) {
    if (!llvm::hasSingleElement(forOps)) return failure();
    scf::ForOp forOp = *(forOps.begin());
    if (auto tiledLoopInfo = isTiledAndDistributedLoop(forOp)) {
      tiledLoops.emplace_back(std::move(tiledLoopInfo.value()));
    }
    body = forOp.getBody();
    forOps = body->getOps<scf::ForOp>();
  }
  for (Operation &op : body->getOperations()) {
    if (filteringFn(&op)) {
      filteredOps.push_back(&op);
    }
  }
  return success();
}

LogicalResult getComputeOps(
    func::FuncOp funcOp, SmallVectorImpl<Operation *> &computeOps,
    SmallVectorImpl<LoopTilingAndDistributionInfo> &tiledLoops) {
  if (failed(getFilteredOps(
          funcOp,
          [](Operation *op) {
            return isa<linalg::LinalgOp, TilingInterface>(op);
          },
          computeOps, tiledLoops))) {
    return failure();
  }
  return success();
}

SmallVector<LoopTilingAndDistributionInfo> getTiledAndDistributedLoopInfo(
    func::FuncOp funcOp) {
  SmallVector<LoopTilingAndDistributionInfo> info;
  funcOp.walk([&](scf::ForOp forOp) {
    if (auto tiledLoopInfo = isTiledAndDistributedLoop(forOp)) {
      info.emplace_back(std::move(tiledLoopInfo.value()));
    }
  });
  return info;
}

/// Create a linalg::GenericOp version of an n-D copy that can further tile,
/// lower to loops or vectorize, unlike the current implementation of
/// memref::CopyOp.
Operation *createLinalgCopyOp(OpBuilder &b, Location loc, Value from, Value to,
                              ArrayRef<NamedAttribute> attributes) {
  auto memrefTypeFrom = from.getType().dyn_cast<MemRefType>();
  auto memrefTypeTo = to.getType().dyn_cast<MemRefType>();
  if (!memrefTypeFrom || !memrefTypeTo ||
      memrefTypeFrom.getRank() != memrefTypeTo.getRank()) {
    mlir::emitError(
        loc, "unable to generate copy op within bufferization from type ")
        << memrefTypeFrom << " to " << memrefTypeTo;
    return nullptr;
  }
  AffineMap id =
      AffineMap::getMultiDimIdentityMap(memrefTypeTo.getRank(), b.getContext());
  SmallVector<utils::IteratorType> iteratorTypes(memrefTypeTo.getRank(),
                                                 utils::IteratorType::parallel);
  return b.create<linalg::GenericOp>(
      loc,
      /*inputs=*/from,
      /*outputs=*/to,
      /*indexingMaps=*/llvm::ArrayRef({id, id}),
      /*iteratorTypes=*/iteratorTypes,
      [](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args.front());
      },
      attributes);
}

template <typename OpTy>
static Value buildHALWorkgroupInfoOp(OpBuilder &b, unsigned dim) {
  return b.template create<OpTy>(b.getInsertionPoint()->getLoc(), dim);
}

linalg::LinalgLoopDistributionOptions getIREELinalgLoopDistributionOptions(
    const SmallVector<int64_t> &tileSizes) {
  return {
      [&tileSizes](OpBuilder &builder, Location loc,
                   ArrayRef<Range> parallelLoopRanges) {
        SmallVector<int64_t> nonZeroTileSizes;
        for (int64_t size : tileSizes) {
          if (size != 0) nonZeroTileSizes.push_back(size);
        }
        auto numParallelDims = parallelLoopRanges.size();

        SmallVector<linalg::ProcInfo, 3> procInfo(numParallelDims);
        Value splitDim;
        for (size_t dim = 0; dim < numParallelDims; ++dim) {
          if (numParallelDims > kNumMaxParallelDims &&
              dim >= kNumMaxParallelDims - 1) {
            if (!splitDim) {
              splitDim =
                  buildHALWorkgroupInfoOp<IREE::HAL::InterfaceWorkgroupIDOp>(
                      builder, 2);
            }
            Value size = getValueOrCreateConstantIndexOp(
                builder, loc,
                parallelLoopRanges[numParallelDims - dim - 1].size);
            Value offset = getValueOrCreateConstantIndexOp(
                builder, loc,
                parallelLoopRanges[numParallelDims - dim - 1].offset);
            AffineExpr d0, d1;
            int64_t tileSize = nonZeroTileSizes[numParallelDims - dim - 1];
            bindSymbols(builder.getContext(), d0, d1);
            Value numTiles = makeComposedAffineApply(
                builder, loc, (d0 - d1).ceilDiv(tileSize), {size, offset});
            Value dimValue;
            if (dim == numParallelDims - 1)
              dimValue = splitDim;
            else {
              dimValue =
                  builder.create<arith::RemUIOp>(loc, splitDim, numTiles);
              splitDim =
                  builder.create<arith::DivUIOp>(loc, splitDim, numTiles);
            }
            procInfo[numParallelDims - dim - 1] = {
                dimValue,
                numTiles,
                linalg::DistributionMethod::Cyclic,
            };
            continue;
          }
          procInfo[numParallelDims - dim - 1] = {
              buildHALWorkgroupInfoOp<IREE::HAL::InterfaceWorkgroupIDOp>(
                  builder, dim),
              buildHALWorkgroupInfoOp<IREE::HAL::InterfaceWorkgroupCountOp>(
                  builder, dim),
              linalg::DistributionMethod::Cyclic};
        }
        return procInfo;
      }};
}

void replaceMemrefUsesAndPropagateType(Operation *oldOp, Value val,
                                       OpBuilder &builder) {
  SmallVector<Operation *> opToDelete;
  SmallVector<OpOperand *> operandsToReplace;
  for (OpOperand &use : oldOp->getUses()) {
    auto subviewUse = dyn_cast<memref::SubViewOp>(use.getOwner());
    if (!subviewUse) {
      // Save the operand to and replace outside the loop to not invalidate the
      // iterator.
      operandsToReplace.push_back(&use);
      continue;
    }
    builder.setInsertionPoint(subviewUse);
    Type newType = memref::SubViewOp::inferRankReducedResultType(
        subviewUse.getType().getShape(), val.getType().cast<MemRefType>(),
        subviewUse.getStaticOffsets(), subviewUse.getStaticSizes(),
        subviewUse.getStaticStrides());
    Value newSubview = builder.create<memref::SubViewOp>(
        subviewUse->getLoc(), newType.cast<MemRefType>(), val,
        subviewUse.getMixedOffsets(), subviewUse.getMixedSizes(),
        subviewUse.getMixedStrides());
    replaceMemrefUsesAndPropagateType(subviewUse, newSubview, builder);
    opToDelete.push_back(use.getOwner());
  }
  for (OpOperand *operand : operandsToReplace) operand->set(val);
  // Clean up old subview ops.
  for (Operation *op : opToDelete) op->erase();
}

}  // namespace iree_compiler
}  // namespace mlir
