// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Transforms.cpp - Transformations common to all backends ------------===//
//
// Defines transformations that are common to backends
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Transforms/Transforms.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace iree_compiler {

static bool sliceFilter(Operation *op, ValueRange nonIndexComputationOperands,
                        Operation *baseOp) {
  for (auto val : nonIndexComputationOperands) {
    if (op == val.getDefiningOp()) return false;
  }
  if (op->isProperAncestor(baseOp)) return false;
  return !isa<IREE::HAL::InterfaceConstantLoadOp>(op);
}

static SliceAndDynamicDims cloneOffsetsSizesAndStridesImpl(
    OpBuilder &builder, Operation *baseOp,
    ValueRange nonIndexComputationOperands, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
    ValueRange dynamicDims) {
  SetVector<Operation *> slice;
  getBackwardSlice(baseOp, &slice, [&](Operation *op) {
    return sliceFilter(op, nonIndexComputationOperands, baseOp);
  });
  IRMapping bvm;
  for (auto origOp : slice) {
    builder.clone(*origOp, bvm);
  }
  auto remapOpFoldResult = [&bvm](ArrayRef<OpFoldResult> ofrs) {
    SmallVector<OpFoldResult> clonedOfrs;
    clonedOfrs.reserve(ofrs.size());
    for (auto ofr : ofrs) {
      if (ofr.is<Attribute>()) {
        clonedOfrs.push_back(ofr);
      } else {
        clonedOfrs.push_back(bvm.lookupOrDefault(ofr.get<Value>()));
      }
    }
    return clonedOfrs;
  };
  auto remapValues = [&bvm](ValueRange vals) {
    SmallVector<Value> clonedVals;
    clonedVals.reserve(vals.size());
    for (auto val : vals) {
      clonedVals.push_back(bvm.lookupOrDefault(val));
    }
    return clonedVals;
  };

  SliceAndDynamicDims clonedVals;
  clonedVals.offsets = remapOpFoldResult(offsets);
  clonedVals.sizes = remapOpFoldResult(sizes);
  clonedVals.strides = remapOpFoldResult(strides);
  clonedVals.dynamicDims = remapValues(dynamicDims);
  return clonedVals;
}

SliceAndDynamicDims cloneOffsetsSizesAndStrides(
    OpBuilder &builder, IREE::Flow::DispatchTensorStoreOp storeOp) {
  return cloneOffsetsSizesAndStridesImpl(
      builder, storeOp, ValueRange{storeOp.getValue(), storeOp.getTarget()},
      storeOp.getMixedOffsets(), storeOp.getMixedSizes(),
      storeOp.getMixedStrides(), storeOp.getTargetDims());
}

SliceAndDynamicDims cloneOffsetsSizesAndStrides(
    OpBuilder &builder, IREE::Flow::DispatchTensorLoadOp loadOp) {
  return cloneOffsetsSizesAndStridesImpl(
      builder, loadOp, ValueRange{loadOp.getSource()}, loadOp.getMixedOffsets(),
      loadOp.getMixedSizes(), loadOp.getMixedStrides(), loadOp.getSourceDims());
}

/// Hoists `allocaOp` to the entry block of the function if the size is
/// statically bounded. For a static allocation, it returns an allocation
/// of the same size but in the entry basic block. For dynamic (still bounded)
/// allocations creates an allocation, and inserts a subview to match the
/// dynamic shape of the allocation. The method returns a value, but
/// does not replace the uses of the `allocaOp`.
std::optional<Value> hoistStaticallyBoundAllocations(
    func::FuncOp funcOp, OpBuilder &builder, Location loc,
    MemRefType allocaType, ValueRange dynamicSizes,
    std::optional<uint64_t> alignment) {
  IntegerAttr alignmentAttr =
      alignment ? builder.getI64IntegerAttr(alignment.value()) : nullptr;
  // For static case just create a new allocation in the entry block of the
  // same size. No need to insert a subview.
  if (dynamicSizes.empty()) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    Value allocation =
        builder.create<memref::AllocaOp>(loc, allocaType, alignmentAttr);
    return allocation;
  }

  /// For the dynamic but bounded case, insert an allocation
  /// of the shape of the bounds, and a subview of the
  /// required size to be used as a replacement.
  SmallVector<int64_t> staticShape;
  SmallVector<OpFoldResult> subviewSizes;
  staticShape.reserve(allocaType.getRank());
  subviewSizes.reserve(allocaType.getRank());

  int index = 0;
  for (auto dimSize : allocaType.getShape()) {
    if (!ShapedType::isDynamic(dimSize)) {
      staticShape.push_back(dimSize);
      subviewSizes.push_back(builder.getIndexAttr(dimSize));
      continue;
    }
    Value dynamicSize = dynamicSizes[index++];
    auto ub = linalg::getConstantUpperBoundForIndex(dynamicSize);
    if (failed(ub)) {
      return std::nullopt;
    }
    staticShape.push_back(ub.value());
    subviewSizes.push_back(dynamicSize);
  }
  SmallVector<OpFoldResult> offsets(allocaType.getRank(),
                                    builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(allocaType.getRank(),
                                    builder.getIndexAttr(1));

  Value allocation;
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    auto allocationType =
        MemRefType::get(staticShape, allocaType.getElementType());
    allocation =
        builder.create<memref::AllocaOp>(loc, allocationType, alignmentAttr);
  }

  Value subviewOp = builder.create<memref::SubViewOp>(loc, allocation, offsets,
                                                      subviewSizes, strides);
  return subviewOp;
}
std::optional<Value> hoistStaticallyBoundAllocations(
    func::FuncOp funcOp, OpBuilder &builder, memref::AllocaOp allocaOp) {
  builder.setInsertionPoint(allocaOp);
  return hoistStaticallyBoundAllocations(
      funcOp, builder, allocaOp.getLoc(), allocaOp.getType(),
      allocaOp.getDynamicSizes(), allocaOp.getAlignment());
}

}  // namespace iree_compiler
}  // namespace mlir
