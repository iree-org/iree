// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_PACKSTORAGEPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

static RankedTensorType appendAttributeToTensor(RankedTensorType type) {
  IREE::Encoding::PackedStorageAttr packedAttr;
  return RankedTensorType::get(type.getShape(), type.getElementType(), packedAttr);
}

struct PackStoragePass
    : impl::PackStoragePassBase<PackStoragePass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
  }
  void runOnOperation() override;
};

void PackStoragePass::runOnOperation() {
  [[maybe_unused]]
  auto funcOp = getOperation();
  for (auto & arg : funcOp.getArguments()) {
    if(auto tensorType = dyn_cast<RankedTensorType>(arg.getType())) {
      auto elementType = tensorType.getElementType();
      if (elementType.isIntOrFloat() && elementType.getIntOrFloatBitWidth() == 1) {
        arg.setType(appendAttributeToTensor(tensorType));
      }
    }
  }
}

} // namespace mlir::iree_compiler::GlobalOptimization
