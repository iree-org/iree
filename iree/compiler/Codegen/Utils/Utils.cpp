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
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"

#define DEBUG_TYPE "iree-codegen-utils"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions to get entry points
//===----------------------------------------------------------------------===//

bool isEntryPoint(FuncOp func) { return func.isPublic(); }

IREE::HAL::ExecutableEntryPointOp getEntryPoint(FuncOp funcOp) {
  auto variantOp = funcOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  for (auto op : variantOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
    if (op.sym_name() == funcOp.getName()) {
      return op;
    }
  }
  return nullptr;
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

/// Returns the LLVM Target triple associated with the `hal.executable.variant`
/// operation if set.
static Optional<llvm::Triple> getTargetTriple(
    IREE::HAL::ExecutableVariantOp variantOp) {
  IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.target();
  if (!targetAttr) return llvm::None;
  auto config = targetAttr.getConfiguration();
  if (!config) return llvm::None;
  auto triple = config.getAs<StringAttr>("target_triple");
  if (!triple) return llvm::None;
  return llvm::Triple(triple.getValue().str());
}

bool isX86(IREE::HAL::ExecutableVariantOp variantOp) {
  Optional<llvm::Triple> triple = getTargetTriple(variantOp);
  return triple && triple.getValue().isX86();
}

//===----------------------------------------------------------------------===//
// Utility functions to get untiled op shapes
//===----------------------------------------------------------------------===//

SmallVector<int64_t> getDistributedTileSizes(
    IREE::Flow::PartitionableLoopsInterface interfaceOp,
    ArrayRef<int64_t> workloadPerWorkgroup) {
  SmallVector<int64_t> tileSizes(interfaceOp.getNumLoops(), 0);
  SmallVector<unsigned> partitionableLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  assert(partitionableLoops.size() == workloadPerWorkgroup.size() &&
         "mismatch in parallelization");
  for (auto it :
       llvm::zip(workloadPerWorkgroup, llvm::reverse(partitionableLoops))) {
    tileSizes[std::get<1>(it)] = std::get<0>(it);
  }
  return tileSizes;
}

/// Walk up the defs of the view, to get the untiled value. Either walks up
/// `ViewOpInterface` op-chains or the `subtensor` op-chains.
static Value getViewSource(Value view) {
  while (true) {
    Operation *definingOp = view.getDefiningOp();
    if (!definingOp) break;
    if (auto viewOp = view.getDefiningOp<ViewLikeOpInterface>()) {
      view = viewOp.getViewSource();
      continue;
    }
    if (auto subTensorOp = view.getDefiningOp<tensor::ExtractSliceOp>()) {
      view = subTensorOp.source();
      continue;
    }
    if (auto dispatchTensorLoadOp =
            view.getDefiningOp<IREE::Flow::DispatchTensorLoadOp>()) {
      view = dispatchTensorLoadOp.source();
      continue;
    }
    break;
  }
  return view;
}

ArrayRef<int64_t> getUntiledShape(Value tiledView) {
  auto type = getViewSource(tiledView).getType();
  return TypeSwitch<Type, ArrayRef<int64_t>>(type)
      .Case<ShapedType, IREE::Flow::DispatchTensorType>(
          [&](auto shapedType) { return shapedType.getShape(); })
      .Default([&](Type type) { return ArrayRef<int64_t>{}; });
}

// TODO(ravishankarm): Using the result shape for vectorization should be
// avoided. Ideally the tile size is enough. But there is a phase ordering issue
// which prevents the tile size from being known at this point.
SmallVector<int64_t> getUntiledResultShape(linalg::LinalgOp linalgOp,
                                           unsigned resultNum) {
  // Get the shape from the `outs` operand.
  SmallVector<int64_t> outputShape =
      llvm::to_vector<4>(getUntiledShape(linalgOp.outputs()[resultNum]));

  // If this is already fully static, it means we didn't tile it at the flow
  // level at all; just return.
  if (llvm::none_of(outputShape, ShapedType::isDynamic)) return outputShape;

  // For Linalg ops with buffer semantics, subview chains should give us enough
  // information; also directly return.
  if (linalgOp.hasBufferSemantics()) return outputShape;

  // For Linalg ops with tensor semantics, we need to correlate how the op
  // should be tiled and the materialized loop nest. The materialized loops'
  // upper bounds should be the original dimension size for the corresponding
  // tiled op shape dimension.
  auto interfaceOp = cast<IREE::Flow::PartitionableLoopsInterface>(*linalgOp);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  SmallVector<LoopTilingAndDistributionInfo> loopInfo =
      getTiledAndDistributedLoopInfo(linalgOp->getParentOfType<FuncOp>());
  // The number of linalg implicit loops to partition and tiled loops
  // surrounding the op should match. Otherwise, something is incorrect.
  assert(partitionedLoops.size() == loopInfo.size());

  // Collect the mapping from the implict loop / iterator indices of the Linalg
  // op to the output shape dimensions. Normally the first batch of iterators of
  // a Linalg op are used to index into the output shape dimensions; but that's
  // not guaranteed. For example, we can see the `linalg.pooling_nhwc_sum` op
  // having the following indexing map for the output:
  //   (N, OH, OW, KH, KW, C) -> (N, OH, OW, C)
  DenseMap<int64_t, int64_t> loopToOutputDimMap;
  auto outputMap =
      linalgOp.getTiedIndexingMap(linalgOp.getOutputOperand(resultNum));
  for (const auto &indexedResult : llvm::enumerate(outputMap.getResults())) {
    if (auto dimExpr = indexedResult.value().dyn_cast<AffineDimExpr>()) {
      loopToOutputDimMap[dimExpr.getPosition()] = indexedResult.index();
    }
  }

  for (auto pair : llvm::zip(llvm::reverse(partitionedLoops), loopInfo)) {
    unsigned loopIndex = std::get<0>(pair);
    const LoopTilingAndDistributionInfo &loopInfo = std::get<1>(pair);
    // If we know the static upper bound of this loop..
    if (Optional<int64_t> attrValue =
            getConstantIntValue(loopInfo.untiledUpperBound)) {
      // ..and it accesses one output dimension..
      if (loopToOutputDimMap.count(loopIndex)) {
        // then we can recover the corresponding shape dimension's size.
        outputShape[loopToOutputDimMap[loopIndex]] = *attrValue;
      }
    }
  }

  return outputShape;
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
    FuncOp funcOp, RootOpFilteringFn filteringFn,
    SmallVectorImpl<Operation *> &filteredOps,
    SmallVectorImpl<LoopTilingAndDistributionInfo> &tiledLoops) {
  Region &region = funcOp.body();
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
    FuncOp funcOp, SmallVectorImpl<Operation *> &computeOps,
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
    FuncOp funcOp) {
  SmallVector<LoopTilingAndDistributionInfo> info;
  funcOp.walk([&](scf::ForOp forOp) {
    if (auto tiledLoopInfo = isTiledAndDistributedLoop(forOp)) {
      info.emplace_back(std::move(tiledLoopInfo.getValue()));
    }
  });
  return info;
}

}  // namespace iree_compiler
}  // namespace mlir
