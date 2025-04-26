// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

//===---------------------------------------------------------------------===//
// MapScatterOp Transforms
//===---------------------------------------------------------------------===//

void insertTransformationAtMapScatterStart(
    OpBuilder &builder, MapScatterOp mapScatterOp,
    function_ref<SmallVector<Value>(ArrayRef<BlockArgument>)>
        transformationBuilder,
    int64_t numSourceIndices) {
  Block &transformBody = mapScatterOp.getTransformationRegion().front();
  SmallVector<BlockArgument> oldSourceIndices(transformBody.getArguments());
  SmallVector<Type> indexTypes(numSourceIndices, builder.getIndexType());
  SmallVector<Location> locs(numSourceIndices, mapScatterOp.getLoc());

  // Create the new block arguments for the new source indices, and transform
  // them using the callback.
  SmallVector<BlockArgument> newSourceIndices(
      transformBody.addArguments(indexTypes, locs));
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&transformBody);
  SmallVector<Value> newSourceIndicesTransformed(
      transformationBuilder(newSourceIndices));

  // Replace the old source indices with the results of the transformation on
  // the new source indices.
  assert(oldSourceIndices.size() == newSourceIndicesTransformed.size() &&
         "expected transformation to produce the same number of Values as the "
         "previous number of source indices.");
  for (auto [oldIdx, newIdx] :
       llvm::zip_equal(oldSourceIndices, newSourceIndicesTransformed)) {
    SmallVector<OpOperand *> uses(llvm::make_pointer_range(oldIdx.getUses()));
    for (OpOperand *use : uses) {
      use->set(newIdx);
    }
  }
  transformBody.eraseArguments(0, oldSourceIndices.size());
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
