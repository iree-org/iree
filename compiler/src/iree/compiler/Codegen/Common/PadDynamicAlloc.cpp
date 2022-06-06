// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace iree_compiler {

static LogicalResult padAlloc(memref::AllocOp allocOp) {
  OpBuilder builder(allocOp);
  SmallVector<int64_t> shape = llvm::to_vector(allocOp.getType().getShape());
  SmallVector<OpFoldResult> sizes;
  size_t dynamicDimIdx = 0;
  for (int64_t &dimSize : shape) {
    if (dimSize != ShapedType::kDynamicSize) {
      sizes.push_back(builder.getIndexAttr(dimSize));
      continue;
    }
    Value dim = allocOp.dynamicSizes()[dynamicDimIdx++];
    auto ub = linalg::getConstantUpperBoundForIndex(dim);
    if (failed(ub)) {
      return allocOp.emitOpError(
          "unexpected allocation without upper bound shapes");
    }
    dimSize = *ub;
    sizes.push_back(dim);
  }
  if (dynamicDimIdx == 0) return success();
  Type elType = allocOp.getType().getElementType();
  MemRefType allocType = MemRefType::get(
      shape, elType, {}, allocOp.getType().getMemorySpaceAsInt());
  Location loc = allocOp.getLoc();
  Value paddedAlloc = builder.create<memref::AllocOp>(loc, allocType);
  SmallVector<OpFoldResult> offsets(shape.size(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(shape.size(), builder.getIndexAttr(1));
  Value subview = builder.create<memref::SubViewOp>(loc, paddedAlloc, offsets,
                                                    sizes, strides);
  replaceMemrefUsesAndPropagateType(allocOp, subview, builder);
  allocOp->erase();
  return success();
}

namespace {

struct PadDynamicAllocPass : public PadDynamicAllocBase<PadDynamicAllocPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<memref::AllocOp> sharedMemAllocs;
    // Collect all the alloc operations.
    funcOp.walk(
        [&](memref::AllocOp allocOp) { sharedMemAllocs.push_back(allocOp); });
    for (memref::AllocOp alloc : sharedMemAllocs) {
      if (failed(padAlloc(alloc))) return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createPadDynamicAlloc() {
  return std::make_unique<PadDynamicAllocPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
