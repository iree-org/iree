// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mlir::iree_compiler {

namespace {

struct GPUSynchronizeBuffers
    : public GPUSynchronizeBuffersBase<GPUSynchronizeBuffers> {

  GPUSynchronizeBuffers() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override;
};

}; // namespace

void GPUSynchronizeBuffers::runOnOperation() {
  Operation *root = getOperation();
  IRRewriter rewriter(root->getContext());

  auto sync = ([](OpBuilder builder) {
    builder.create<gpu::BarrierOp>(builder.getInsertionPoint()->getLoc());
  });

  if (failed(synchronizeBuffers(rewriter, root, sync))) {
    return signalPassFailure();
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createGPUSynchronizeBuffersPass() {
  return std::make_unique<GPUSynchronizeBuffers>();
}

} // namespace mlir::iree_compiler
