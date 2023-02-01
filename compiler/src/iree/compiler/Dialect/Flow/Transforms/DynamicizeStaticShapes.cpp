// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-flow-dynamicize-static-shapes"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
struct DynamicizeStaticShapesPass
    : public DynamicizeStaticShapesBase<DynamicizeStaticShapesPass> {
  void runOnOperation() override;
};

/// Returns the corresponding dynamic shaped type of the given static shaped
/// `type`.
RankedTensorType getDynamicTensorType(RankedTensorType type) {
  SmallVector<int64_t, 4> shape;
  shape.resize(type.getRank(), ShapedType::kDynamic);
  return RankedTensorType::get(shape, type.getElementType(),
                               type.getEncoding());
}

/// Goes through the given list of types, return dynamic shaped counterparts for
/// tensor shapes, and return non-tensor types as-is.
SmallVector<Type> getDynamicTensorTypes(Operation::result_type_range types) {
  SmallVector<Type> resultTypes;
  for (Type type : types) {
    if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
      resultTypes.push_back(getDynamicTensorType(tensorType));
    } else {
      resultTypes.push_back(type);
    }
  }
  return resultTypes;
}

/// Returns a vector of SSA values representing each dimension of the given
/// `value`'s shape.
SmallVector<Value, 4> getDimValues(OpBuilder &builder, Location loc,
                                   Value value, ValueRange dynamicDims) {
  auto type = value.getType().cast<ShapedType>();
  SmallVector<Value, 4> dimValues;
  dimValues.reserve(type.getRank());

  int dimIndex = 0;
  for (int64_t dimSize : type.getShape()) {
    if (ShapedType::isDynamic(dimSize)) {
      dimValues.push_back(dynamicDims[dimIndex++]);
    } else {
      dimValues.push_back(
          builder.create<DispatchDynamicizeDimOp>(loc, dimSize));
    }
  }
  assert(dimIndex == dynamicDims.size());
  return dimValues;
}

SmallVector<Value, 4> getTensorEmptyDimValues(OpBuilder &builder,
                                              tensor::EmptyOp emptyOp) {
  SmallVector<Value, 4> dynamicDims;
  int dimIndex = 0;
  for (int64_t dimSize : emptyOp.getType().getShape()) {
    if (ShapedType::isDynamic(dimSize)) {
      dynamicDims.push_back(emptyOp->getOperand(dimIndex));
    } else {
      dynamicDims.push_back(
          builder.create<DispatchDynamicizeDimOp>(emptyOp.getLoc(), dimSize));
    }
  }
  return dynamicDims;
}

void getTensorSliceOffsetsSizesAndDimValues(
    OpBuilder &builder, Location loc, OffsetSizeAndStrideOpInterface op,
    RankedTensorType fullType, RankedTensorType partialType,
    SmallVectorImpl<OpFoldResult> &offsetValues,
    SmallVectorImpl<OpFoldResult> &sizeValues) {
  const SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
  offsetValues.resize(offsets.size());
  for (int i = 0, e = offsets.size(); i < e; ++i) {
    if (auto value = offsets[i].dyn_cast<Value>()) {
      offsetValues[i] = value;
      continue;
    }

    auto attr = offsets[i].get<Attribute>().cast<IntegerAttr>();
    auto dimOp = builder.create<DispatchDynamicizeDimOp>(
        loc, attr.getValue().getSExtValue());
    offsetValues[i] = dimOp.getResult();
  }

  int reducedRank = fullType.getRank() - partialType.getRank();
  assert(reducedRank >= 0);

  const SmallVector<OpFoldResult> sizes = op.getMixedSizes();
  sizeValues.resize(sizes.size());
  for (int i = 0, e = sizes.size(); i < e; ++i) {
    if (auto value = sizes[i].dyn_cast<Value>()) {
      sizeValues[i] = value;
      continue;
    }

    auto attr = sizes[i].get<Attribute>().cast<IntegerAttr>();
    auto value = attr.getValue().getSExtValue();
    if (value == 1 && reducedRank > 0) {
      sizeValues[i] = attr;
      --reducedRank;
    } else {
      auto dimOp = builder.create<DispatchDynamicizeDimOp>(loc, value);
      sizeValues[i] = dimOp.getResult();
    }
  }
}

LogicalResult dynamicizeStaticShapes(IRRewriter &rewriter,
                                     DispatchRegionOp &regionOp) {
  IRRewriter::InsertionGuard guard(rewriter);

  // First get all values defined outside of the dispatch region. Identify
  // tensor typed values and rewrite them to have dynamic shapes in the dispatch
  // region.
  rewriter.setInsertionPoint(regionOp);
  llvm::SetVector<Value> aboveValues;
  mlir::getUsedValuesDefinedAbove(regionOp.getBody(), aboveValues);
  for (int i = 0, e = aboveValues.size(); i < e; ++i) {
    Value value = aboveValues[i];
    auto tensorType = value.getType().dyn_cast<RankedTensorType>();
    if (!tensorType) continue;
    auto newValue =
        DispatchDynamicizeShapeOp::build(rewriter, value.getLoc(), value);
    // Replace all uses in the dispatch region to have dynamic shapes.
    value.replaceUsesWithIf(newValue, [regionOp](OpOperand &operand) {
      return operand.getOwner()->getParentOp() == regionOp;
    });
  }

  // Then update all compute ops in the dispatch region to return values of
  // dynamic shapes.
  Block &block = regionOp.getBody().getBlocks().front();
  SmallVector<Operation *> tensorOps;
  for (Operation &op : block.without_terminator()) {
    auto isTensorType = [](Type type) { return type.isa<RankedTensorType>(); };
    if (llvm::any_of(op.getResultTypes(), isTensorType)) {
      tensorOps.push_back(&op);
    }
  }
  SmallVector<Type, 4> resultTypes;
  for (Operation *op : tensorOps) {
    // For inlined constants, don't do anything.
    if (matchPattern(op, m_Constant())) continue;

    // tensor.empty ops "generate" tensors. Turn those tensors into dynamic
    // shapes by using dimension sizes from above the dispatch region op too.
    if (auto emptyOp = dyn_cast<tensor::EmptyOp>(op)) {
      SmallVector<Value, 4> dynamicDims;
      rewriter.setInsertionPoint(regionOp);
      dynamicDims = getTensorEmptyDimValues(rewriter, emptyOp);
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
          emptyOp, getDynamicTensorType(emptyOp.getType()), dynamicDims);
      continue;
    }

    if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      SmallVector<OpFoldResult, 4> offsets;
      SmallVector<OpFoldResult, 4> sizes;
      rewriter.setInsertionPoint(regionOp);
      getTensorSliceOffsetsSizesAndDimValues(
          rewriter, extractOp.getLoc(), extractOp, extractOp.getSourceType(),
          extractOp.getType(), offsets, sizes);
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          extractOp, getDynamicTensorType(extractOp.getType()),
          extractOp.getSource(), offsets, sizes, extractOp.getMixedStrides());
      continue;
    }

    // For other tensor ops, build new dynamic shaped tensor results.
    rewriter.setInsertionPoint(op);
    resultTypes = getDynamicTensorTypes(op->getResultTypes());
    OperationState state(op->getLoc(), op->getName().getStringRef(),
                         op->getOperands(), resultTypes, op->getAttrs(),
                         op->getSuccessors());
    for (Region &region : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(region, *newRegion, newRegion->begin());
    }
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
  }

  // Finally update the output shape for the region op to match the return
  // inside its region.
  rewriter.setInsertionPoint(regionOp);
  SmallVector<Value> dimValues;
  resultTypes = getDynamicTensorTypes(regionOp.getResultTypes());
  int numResultDims = 0;
  for (Value result : regionOp.getResult()) {
    auto type = result.getType().cast<ShapedType>();
    auto dimRange =
        regionOp.getResultDims().slice(numResultDims, type.getNumDynamicDims());
    dimValues.append(
        getDimValues(rewriter, regionOp.getLoc(), result, dimRange));
    numResultDims += type.getNumDynamicDims();
  }
  auto newOp = rewriter.create<DispatchRegionOp>(
      regionOp.getLoc(), resultTypes, dimValues, regionOp.getWorkload());
  rewriter.inlineRegionBefore(regionOp.getBody(), newOp.getBody(),
                              newOp.getBody().begin());
  SmallVector<Value> results;
  for (auto [oldResult, newResult] :
       llvm::zip(regionOp.getResult(), newOp.getResult())) {
    results.push_back(rewriter.create<tensor::CastOp>(
        regionOp.getLoc(), oldResult.getType(), newResult));
  }
  rewriter.replaceOp(regionOp, results);

  return success();
}
}  // namespace

void DynamicizeStaticShapesPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  IRRewriter rewriter(funcOp->getContext());

  SmallVector<DispatchRegionOp> regionOps;
  funcOp.walk([&](DispatchRegionOp op) { regionOps.push_back(op); });
  for (DispatchRegionOp regionOp : regionOps) {
    if (failed(dynamicizeStaticShapes(rewriter, regionOp)))
      return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createDynamicizeStaticShapesPass() {
  return std::make_unique<DynamicizeStaticShapesPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
