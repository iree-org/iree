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

namespace mlir {
namespace iree_compiler {

static void cloneOffsetsSizesAndStridesImpl(
    OpBuilder &builder, Operation *baseOp,
    ValueRange nonIndexComputationOperands, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
    ValueRange dynamicDims, SmallVector<OpFoldResult> &clonedOffsets,
    SmallVector<OpFoldResult> &clonedSizes,
    SmallVector<OpFoldResult> &clonedStrides,
    SmallVector<Value> &clonedDynamicDims) {
  SetVector<Operation *> slice;
  TransitiveFilter filter = [&nonIndexComputationOperands,
                             &baseOp](Operation *op) {
    for (auto val : nonIndexComputationOperands) {
      if (op == val.getDefiningOp()) return false;
    }
    if (op->isProperAncestor(baseOp)) return false;
    return !isa<IREE::HAL::InterfaceConstantLoadOp>(op);
  };
  getBackwardSlice(baseOp, &slice, filter);
  BlockAndValueMapping bvm;
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

  clonedOffsets = remapOpFoldResult(offsets);
  clonedSizes = remapOpFoldResult(sizes);
  clonedStrides = remapOpFoldResult(strides);
  clonedDynamicDims = remapValues(dynamicDims);
}

void cloneOffsetsSizesAndStrides(OpBuilder &builder,
                                 IREE::Flow::DispatchTensorStoreOp storeOp,
                                 SmallVector<OpFoldResult> &offsets,
                                 SmallVector<OpFoldResult> &sizes,
                                 SmallVector<OpFoldResult> &strides,
                                 SmallVector<Value> &dynamicDims) {
  return cloneOffsetsSizesAndStridesImpl(
      builder, storeOp, ValueRange{storeOp.getValue(), storeOp.getTarget()},
      storeOp.getMixedOffsets(), storeOp.getMixedSizes(),
      storeOp.getMixedStrides(), storeOp.getTargetDims(), offsets, sizes,
      strides, dynamicDims);
}

void cloneOffsetsSizesAndStrides(OpBuilder &builder,
                                 IREE::Flow::DispatchTensorLoadOp loadOp,
                                 SmallVector<OpFoldResult> &offsets,
                                 SmallVector<OpFoldResult> &sizes,
                                 SmallVector<OpFoldResult> &strides,
                                 SmallVector<Value> &dynamicDims) {
  return cloneOffsetsSizesAndStridesImpl(
      builder, loadOp, ValueRange{loadOp.getSource()}, loadOp.getMixedOffsets(),
      loadOp.getMixedSizes(), loadOp.getMixedStrides(), loadOp.getSourceDims(),
      offsets, sizes, strides, dynamicDims);
}

}  // namespace iree_compiler
}  // namespace mlir