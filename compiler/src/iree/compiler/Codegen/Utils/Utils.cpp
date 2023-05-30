// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/Utils.h"

#include "iree/compiler/Codegen/Interfaces/ProcessorOpInterfaces.h"
#include "iree/compiler/Codegen/Interfaces/UKernelOpInterface.h"
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

std::optional<StringAttr> getConfigStringAttr(
    IREE::HAL::ExecutableTargetAttr targetAttr, StringRef stringAttr) {
  if (!targetAttr) return std::nullopt;
  auto config = targetAttr.getConfiguration();
  if (!config) return std::nullopt;
  auto attr = config.getAs<StringAttr>(stringAttr);
  if (!attr) return std::nullopt;
  return attr;
}

std::optional<IntegerAttr> getConfigIntegerAttr(
    IREE::HAL::ExecutableTargetAttr targetAttr, StringRef integerAttr) {
  if (!targetAttr) return std::nullopt;
  auto config = targetAttr.getConfiguration();
  if (!config) return std::nullopt;
  auto attr = config.getAs<IntegerAttr>(integerAttr);
  if (!attr) return std::nullopt;
  return attr;
}

std::optional<BoolAttr> getConfigBoolAttr(
    IREE::HAL::ExecutableTargetAttr targetAttr, StringRef integerAttr) {
  if (!targetAttr) return std::nullopt;
  auto config = targetAttr.getConfiguration();
  if (!config) return std::nullopt;
  auto attr = config.getAs<BoolAttr>(integerAttr);
  if (!attr) return std::nullopt;
  return attr;
}

std::optional<llvm::Triple> getTargetTriple(
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
            return llvm::cast<IREE::Flow::DispatchTensorType>(
                       loadOp.getSource().getType())
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
static Value getValueForDimOrSymbol(affine::AffineApplyOp applyOp,
                                    AffineExpr expr) {
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
    affine::AffineApplyOp applyOp, ArrayRef<AffineExpr> exprs) {
  SmallVector<Value> vals;
  for (auto expr : exprs) {
    vals.push_back(getValueForDimOrSymbol(applyOp, expr));
  }
  return vals;
}

/// Returns the dimension for any operation that implements processor op
/// interfaces.
template <typename T>
static std::optional<unsigned> getDimension(Operation *op) {
  if (auto tOp = dyn_cast<T>(op)) {
    return tOp.getDimIndex();
  }
  return std::nullopt;
}
template <typename T1, typename T2, typename... T3>
static std::optional<unsigned> getDimension(Operation *op) {
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
static std::optional<unsigned> checkDimensions(
    ArrayRef<Value> vals, std::optional<unsigned> refDimension = std::nullopt) {
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
  LowerBoundExprVisitor(affine::AffineApplyOp applyOp,
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
    std::optional<unsigned> dimension;
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
  affine::AffineApplyOp applyOp;
  LoopTilingAndDistributionInfo &loopInfo;
};

/// Visitor to walk the `step` of a distributed loop. Expected the expression to
/// be of the form `a * b * c`, where they could be the dynamic `step` or
/// defined by `hal.interface.workgroup.size`/`hal.interface.workgroup.count`
/// operation.
class StepExprVisitor
    : public AffineExprVisitor<StepExprVisitor, LogicalResult> {
 public:
  StepExprVisitor(affine::AffineApplyOp applyOp,
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

  affine::AffineApplyOp applyOp;
  LoopTilingAndDistributionInfo &loopInfo;
};
}  // namespace

template <typename OpTy>
static std::optional<unsigned> getInterfaceWorkgroupOpDim(Value value) {
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
std::optional<LoopTilingAndDistributionInfo> isTiledAndDistributedLoop(
    scf::ForOp forOp) {
  LoopTilingAndDistributionInfo loopInfo;
  loopInfo.loop = forOp;
  loopInfo.untiledUpperBound = getAsOpFoldResult(forOp.getUpperBound());

  auto lbApplyOp = forOp.getLowerBound().getDefiningOp<affine::AffineApplyOp>();
  auto stepApplyOp = forOp.getStep().getDefiningOp<affine::AffineApplyOp>();

  if (!lbApplyOp || !stepApplyOp) {
    // Try to see if this is a specical case where we have:
    //   scf.for %iv = %id to %ub step %count
    std::optional<unsigned> idDim;
    if (auto ifx = dyn_cast_or_null<ProcessorIDInterface>(
            forOp.getLowerBound().getDefiningOp())) {
      idDim = ifx.getDimIndex();
    }

    std::optional<unsigned> countDim;
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

SmallVector<Operation *> getComputeOps(func::FuncOp funcOp) {
  SmallVector<Operation *> computeOps;
  funcOp.walk([&](Operation *op) {
    if (isa<TilingInterface, IREE::Codegen::UKernelOpInterface>(op)) {
      computeOps.push_back(op);
    }
  });
  return computeOps;
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
  auto memrefTypeFrom = llvm::dyn_cast<MemRefType>(from.getType());
  auto memrefTypeTo = llvm::dyn_cast<MemRefType>(to.getType());
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
    const SmallVector<int64_t> &tileSizes,
    linalg::DistributionMethod distributionMethod,
    int32_t maxWorkgroupParallelDims) {
  return {[&tileSizes, distributionMethod, maxWorkgroupParallelDims](
              OpBuilder &builder, Location loc,
              ArrayRef<Range> parallelLoopRanges) {
    SmallVector<int64_t> nonZeroTileSizes;
    for (int64_t size : tileSizes) {
      if (size != 0) nonZeroTileSizes.push_back(size);
    }
    auto numParallelDims = parallelLoopRanges.size();

    SmallVector<linalg::ProcInfo, 3> procInfo(numParallelDims);
    Value splitDim;
    for (size_t dim = 0; dim < numParallelDims; ++dim) {
      if (numParallelDims > maxWorkgroupParallelDims &&
          dim >= maxWorkgroupParallelDims - 1) {
        if (!splitDim) {
          splitDim = buildHALWorkgroupInfoOp<IREE::HAL::InterfaceWorkgroupIDOp>(
              builder, maxWorkgroupParallelDims - 1);
        }
        Value size = getValueOrCreateConstantIndexOp(
            builder, loc, parallelLoopRanges[numParallelDims - dim - 1].size);
        Value offset = getValueOrCreateConstantIndexOp(
            builder, loc, parallelLoopRanges[numParallelDims - dim - 1].offset);
        AffineExpr d0, d1;
        int64_t tileSize = nonZeroTileSizes[numParallelDims - dim - 1];
        bindSymbols(builder.getContext(), d0, d1);
        Value numTiles = affine::makeComposedAffineApply(
            builder, loc, (d0 - d1).ceilDiv(tileSize), {size, offset});
        Value dimValue;
        if (dim == numParallelDims - 1)
          dimValue = splitDim;
        else {
          dimValue = affine::makeComposedAffineApply(builder, loc, (d0 % d1),
                                                     {splitDim, numTiles});
          splitDim = affine::makeComposedAffineApply(
              builder, loc, (d0).floorDiv(d1), {splitDim, numTiles});
        }
        procInfo[numParallelDims - dim - 1] = {dimValue, numTiles,
                                               distributionMethod};
        continue;
      }
      procInfo[numParallelDims - dim - 1] = {
          buildHALWorkgroupInfoOp<IREE::HAL::InterfaceWorkgroupIDOp>(builder,
                                                                     dim),
          buildHALWorkgroupInfoOp<IREE::HAL::InterfaceWorkgroupCountOp>(builder,
                                                                        dim),
          distributionMethod};
    }
    return procInfo;
  }};
}

//===---------------------------------------------------------------------===//
// Misc. utility functions
//===---------------------------------------------------------------------===//

OpFoldResult convertByteOffsetToElementOffset(RewriterBase &rewriter,
                                              Location loc,
                                              OpFoldResult byteOffset,
                                              Type elementType) {
  OpFoldResult elementWidth =
      TypeSwitch<Type, OpFoldResult>(elementType)
          .Case<ComplexType, FloatType, IntegerType, VectorType>(
              [&](auto type) -> OpFoldResult {
                return rewriter.getIndexAttr(
                    IREE::Util::getRoundedElementByteWidth(elementType));
              })
          .Default([&](Type t) -> OpFoldResult {
            return rewriter.create<IREE::Util::SizeOfOp>(loc, elementType)
                .getResult();
          });
  AffineExpr s0, s1;
  bindSymbols(rewriter.getContext(), s0, s1);
  return affine::makeComposedFoldedAffineApply(rewriter, loc, s0.floorDiv(s1),
                                               {byteOffset, elementWidth});
}

//===---------------------------------------------------------------------===//
// Replace Memref users (transitively)
//===---------------------------------------------------------------------===//

/// Replaces a `use` with the `replacement` for cases where a simple substition
/// might lead to verification errors.
static std::optional<SmallVector<Value>> replaceNonTrivialUse(
    RewriterBase &rewriter, Location loc, OpOperand &use, Value replacement) {
  Operation *user = use.getOwner();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(user);

  LLVM_DEBUG({
    llvm::dbgs() << "\tReplacing in user by creating new user : ";
    user->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
    llvm::dbgs() << "\n";
  });

  if (auto castOp = dyn_cast<memref::CastOp>(user)) {
    auto replacementType = llvm::cast<MemRefType>(replacement.getType());
    auto currentResultType =
        llvm::cast<MemRefType>(castOp.getResult().getType());
    if (replacementType == currentResultType) {
      // Cast is a no op, just return the replacement.
      return SmallVector<Value>{replacement};
    }
    auto newResultType = MemRefType::get(
        currentResultType.getShape(), currentResultType.getElementType(),
        replacementType.getLayout(), replacementType.getMemorySpace());
    auto newCastOp =
        rewriter.create<memref::CastOp>(loc, newResultType, replacement);

    LLVM_DEBUG({
      llvm::dbgs() << "\t\tNew user : ";
      newCastOp->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
      llvm::dbgs() << "\n";
    });
    return SmallVector<Value>(newCastOp->result_begin(),
                              newCastOp->result_end());
  }
  if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
    auto currResultType =
        llvm::cast<MemRefType>(subviewOp.getResult().getType());
    auto newSourceType = llvm::cast<MemRefType>(replacement.getType());
    SmallVector<OpFoldResult> offsets = subviewOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = subviewOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = subviewOp.getMixedStrides();
    MemRefType newResultType =
        (currResultType.getRank() != newSourceType.getRank()
             ? llvm::cast<MemRefType>(
                   memref::SubViewOp::inferRankReducedResultType(
                       currResultType.getShape(), newSourceType, offsets, sizes,
                       strides))
             : nullptr);
    auto newSubviewOp = rewriter.create<memref::SubViewOp>(
        loc, newResultType, replacement, offsets, sizes, strides);

    LLVM_DEBUG({
      llvm::dbgs() << "\t\tNew user : ";
      newSubviewOp->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
      llvm::dbgs() << "\n";
    });
    return SmallVector<Value>(newSubviewOp->result_begin(),
                              newSubviewOp->result_end());
  }
  return std::nullopt;
}

void replaceMemrefUsesAndPropagateType(RewriterBase &rewriter, Location loc,
                                       Value origValue,
                                       Value replacementValue) {
  SmallVector<std::pair<Value, Value>> worklist;
  SmallVector<Operation *> toDeleteUsers;
  worklist.push_back({origValue, replacementValue});

  while (!worklist.empty()) {
    auto [original, replacement] = worklist.pop_back_val();

    LLVM_DEBUG({
      llvm::dbgs() << "//===------------------------------------------===//\n";
      llvm::dbgs() << "Replacing : ";
      original.print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
      llvm::dbgs() << "\n";
    });

    llvm::SmallDenseSet<OpOperand *> preservedUses;

    if (original.getType() != replacement.getType()) {
      for (OpOperand &use : original.getUses()) {
        Operation *user = use.getOwner();
        // Some uses cannot be replaced.
        if (isa<func::ReturnOp, scf::YieldOp>(user)) {
          LLVM_DEBUG({
            llvm::dbgs() << "\tUnhandled user : ";
            user->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
            llvm::dbgs() << "\n";
          });
          preservedUses.insert(&use);
          continue;
        }

        // Some uses might be replace-able but require creating new versions
        // of the users to pass verification.
        std::optional<SmallVector<Value>> nonTrivialUse =
            replaceNonTrivialUse(rewriter, loc, use, replacement);
        if (nonTrivialUse) {
          // Add the results of the new users created as replacements
          // for the old users. Push this back on the to worklist.
          preservedUses.insert(&use);
          for (auto [v1, v2] :
               llvm::zip_equal(user->getResults(), nonTrivialUse.value())) {
            worklist.push_back({v1, v2});
          }
          toDeleteUsers.push_back(user);
          continue;
        }
      }
    }

    // Replace all non-preserved uses.
    rewriter.replaceUsesWithIf(original, replacement, [&](OpOperand &use) {
      if (!preservedUses.count(&use)) {
        LLVM_DEBUG({
          llvm::dbgs() << "\t\tReplacing use in :";
          use.getOwner()->print(llvm::dbgs(),
                                OpPrintingFlags().assumeVerified());
          llvm::dbgs() << "\n";
        });
        return true;
      }
      return false;
    });
  }

  // Iterate over delete-able operations in reverse and delete if
  // there are no users.
  for (auto deleteOp : llvm::reverse(toDeleteUsers)) {
    if (deleteOp->use_empty()) {
      rewriter.eraseOp(deleteOp);
    }
  }
}

void sinkOpsInCFG(const SmallVector<Operation *> &allocs,
                  DominanceInfo &dominators) {
  for (Operation *sinkOp : allocs) {
    Block *dom = nullptr;
    for (Operation *user : sinkOp->getUsers()) {
      if (!dom) {
        dom = user->getBlock();
        // Find the block in the same region.
        while (dom->getParent() != sinkOp->getParentRegion()) {
          dom = dom->getParentOp()->getBlock();
        }
        continue;
      }
      dom = dominators.findNearestCommonDominator(dom, user->getBlock());
    }
    llvm::SmallDenseSet<Operation *> users;
    for (Operation *user : sinkOp->getUsers()) {
      while (user->getParentRegion() != sinkOp->getParentRegion()) {
        user = user->getParentOp();
      }
      users.insert(user);
    }
    Operation *firstUse = dom->getTerminator();
    for (Operation &op : dom->getOperations()) {
      if (users.count(&op)) {
        firstUse = &op;
        break;
      }
    }
    sinkOp->moveBefore(firstUse);
  }
}

/// Infer the number of workgroups from exportOp.
SmallVector<int64_t> getStaticNumWorkgroups(func::FuncOp funcOp) {
  SmallVector<int64_t> result;
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(funcOp);
  if (failed(exportOp)) return result;

  Block *body = exportOp->getWorkgroupCountBody();
  if (!body) return result;

  auto returnOp = cast<IREE::HAL::ReturnOp>(body->getTerminator());
  assert(returnOp.getNumOperands() == 3);

  for (unsigned i = 0; i < 3; ++i) {
    Operation *defOp = returnOp.getOperand(i).getDefiningOp();
    if (auto indexOp = dyn_cast_or_null<arith::ConstantIndexOp>(defOp)) {
      result.push_back(indexOp.value());
    } else {
      return SmallVector<int64_t>();
    }
  }

  return result;
}

}  // namespace iree_compiler
}  // namespace mlir
