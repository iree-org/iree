// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/Utils.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Interfaces/ProcessorOpInterfaces.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"

#define DEBUG_TYPE "iree-codegen-utils"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions to get entry points
//===----------------------------------------------------------------------===//

FailureOr<IREE::HAL::ExecutableEntryPointOp> getEntryPoint(
    func::FuncOp funcOp) {
  auto variantOp = funcOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  if (!variantOp) return failure();

  for (auto op : variantOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
    if (op.sym_name() == funcOp.getName()) {
      return op;
    }
  }
  return failure();
}

bool isEntryPoint(func::FuncOp func) {
  return func.isPublic() && succeeded(getEntryPoint(func));
}

llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> getAllEntryPoints(
    ModuleOp module) {
  auto variantOp = module->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> entryPointOps;
  for (auto op : variantOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
    entryPointOps[op.sym_name()] = op;
  }
  return entryPointOps;
}

/// Returns the StringAttr with the name `stringAttr` in the configuration of
/// `variantOp`, in found.
static Optional<StringAttr> getConfigStringAttr(
    IREE::HAL::ExecutableVariantOp variantOp, StringRef stringAttr) {
  IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.target();
  if (!targetAttr) return llvm::None;
  auto config = targetAttr.getConfiguration();
  if (!config) return llvm::None;
  auto attr = config.getAs<StringAttr>(stringAttr);
  if (!attr) return llvm::None;
  return attr;
}

/// Returns the LLVM Target triple associated with the `hal.executable.variant`
/// operation if set.
static Optional<llvm::Triple> getTargetTriple(
    IREE::HAL::ExecutableVariantOp variantOp) {
  auto triple = getConfigStringAttr(variantOp, "target_triple");
  if (!triple) return llvm::None;
  return llvm::Triple(triple.getValue().str());
}

/// Returns the CPU target features associated with the `hal.executable.variant`
/// operation, if set.
static Optional<StringRef> getCpuFeatures(
    IREE::HAL::ExecutableVariantOp variantOp) {
  auto cpuFeatures = getConfigStringAttr(variantOp, "cpu_features");
  if (!cpuFeatures) return llvm::None;
  return cpuFeatures->getValue();
}

bool isX86(IREE::HAL::ExecutableVariantOp variantOp) {
  Optional<llvm::Triple> triple = getTargetTriple(variantOp);
  return triple && triple.getValue().isX86();
}

// TODO(dcaballe): If we have to check for a significantly large number of
// features in the future, we may want to consider keeping the TTI instance
// alive and query subtarget features data structure.
bool hasAVX2Features(Operation *op) {
  auto variantOp = isa<IREE::HAL::ExecutableVariantOp>(op)
                       ? cast<IREE::HAL::ExecutableVariantOp>(op)
                       : op->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  if (!variantOp) return false;
  Optional<StringRef> features = getCpuFeatures(variantOp);
  if (!features) return false;
  return features->contains("+avx2");
}

bool isRISCV(IREE::HAL::ExecutableVariantOp variantOp) {
  Optional<llvm::Triple> triple = getTargetTriple(variantOp);
  return triple && triple.getValue().isRISCV();
}

bool isReadOnly(Value v) {
  Operation *definingOp = v.getDefiningOp();
  if (!definingOp) return false;
  return TypeSwitch<Operation *, bool>(definingOp)
      .Case<arith::ConstantOp>(
          [&](arith::ConstantOp constantOp) { return true; })
      .Case<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
          [&](auto op) { return isReadOnly(op.src()); })
      .Case<tensor::CastOp, tensor::ExtractSliceOp>(
          [&](auto op) { return isReadOnly(op.source()); })
      .Case<IREE::Flow::DispatchTensorLoadOp>(
          [&](IREE::Flow::DispatchTensorLoadOp loadOp) {
            return loadOp.source()
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
  return llvm::None;
}
template <typename T1, typename T2, typename... T3>
static Optional<unsigned> getDimension(Operation *op) {
  if (!op) return llvm::None;
  if (auto dimension = getDimension<T1>(op)) {
    return dimension;
  }
  return getDimension<T2, T3...>(op);
}

/// Checks that all `vals` are defined by some processor id/count/size ops using
/// the same `dimension`. If any element of `vals` is not defined by one of
/// these ops, or the dimensions dont match, returns llvm::None; oterhwise,
/// returns the dimension.  If `refDimension` is passed checks if the dimension
/// matches the given value.
template <typename... T>
static Optional<unsigned> checkDimensions(
    ArrayRef<Value> vals, Optional<unsigned> refDimension = llvm::None) {
  for (auto v : vals) {
    auto currDimension = getDimension<T...>(v.getDefiningOp());
    if (!currDimension) return llvm::None;
    if (refDimension) {
      if (refDimension.getValue() != currDimension.getValue()) {
        return llvm::None;
      }
    } else {
      refDimension = currDimension.getValue();
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
    loopInfo.processorDistributionDim = dimension.getValue();
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
    return op.dimension().getZExtValue();
  }
  return llvm::None;
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
    // Try to see if this s a specical case where we have:
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

    if (!idDim || !countDim) return llvm::None;

    Builder b(forOp.getContext());
    loopInfo.untiledLowerBound = b.getIndexAttr(0);
    loopInfo.untiledStep = b.getIndexAttr(1);
    loopInfo.processorDistributionDim = idDim.getValue();
    // For such special case, the tile size is 1.
    loopInfo.tileSize = 1;
    return loopInfo;
  }

  LowerBoundExprVisitor lbVisitor(lbApplyOp, loopInfo);
  StepExprVisitor stepVisitor(stepApplyOp, loopInfo);

  if (failed(lbVisitor.visit(lbApplyOp.getAffineMap().getResults()[0]))) {
    return llvm::None;
  }
  if (failed(stepVisitor.visit(stepApplyOp.getAffineMap().getResults()[0]))) {
    return llvm::None;
  }
  if (!loopInfo.untiledLowerBound || !loopInfo.untiledStep) {
    return llvm::None;
  }
  return loopInfo;
}

LogicalResult getFilteredOps(
    func::FuncOp funcOp, RootOpFilteringFn filteringFn,
    SmallVectorImpl<Operation *> &filteredOps,
    SmallVectorImpl<LoopTilingAndDistributionInfo> &tiledLoops) {
  Region &region = funcOp.getBody();
  if (!llvm::hasSingleElement(region)) {
    return funcOp.emitError("unable dispatch function with multiple blocks");
  }
  Block *body = &region.front();
  auto forOps = body->getOps<scf::ForOp>();
  while (!forOps.empty()) {
    if (!llvm::hasSingleElement(forOps)) return failure();
    scf::ForOp forOp = *(forOps.begin());
    if (auto tiledLoopInfo = isTiledAndDistributedLoop(forOp)) {
      tiledLoops.emplace_back(std::move(tiledLoopInfo.getValue()));
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
            return isa<linalg::LinalgOp, IREE::LinalgExt::TiledOpInterface>(op);
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
      info.emplace_back(std::move(tiledLoopInfo.getValue()));
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
  SmallVector<StringRef> iteratorTypes(memrefTypeTo.getRank(),
                                       getParallelIteratorTypeName());
  return b.create<linalg::GenericOp>(
      loc,
      /*inputs=*/from,
      /*outputs=*/to,
      /*indexingMaps=*/llvm::makeArrayRef({id, id}),
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

linalg::LinalgLoopDistributionOptions getIREELinalgLoopDistributionOptions() {
  return {
      [](OpBuilder &builder, Location loc, ArrayRef<Range> parallelLoopRanges) {
        auto numParallelDims = parallelLoopRanges.size();

        SmallVector<linalg::ProcInfo, 3> procInfo(numParallelDims);
        for (size_t dim = 0; dim < numParallelDims; ++dim) {
          procInfo[numParallelDims - dim - 1] = {
              buildHALWorkgroupInfoOp<IREE::HAL::InterfaceWorkgroupIDOp>(
                  builder, dim),
              buildHALWorkgroupInfoOp<IREE::HAL::InterfaceWorkgroupCountOp>(
                  builder, dim)};
        }
        return procInfo;
      },
      {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
       linalg::DistributionMethod::Cyclic},
      DenseMap<StringRef,
               std::function<linalg::ProcInfo(OpBuilder &, Location)>>()};
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
        subviewUse.getType().getRank(), val.getType().cast<MemRefType>(),
        extractFromI64ArrayAttr(subviewUse.static_offsets()),
        extractFromI64ArrayAttr(subviewUse.static_sizes()),
        extractFromI64ArrayAttr(subviewUse.static_strides()));
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
