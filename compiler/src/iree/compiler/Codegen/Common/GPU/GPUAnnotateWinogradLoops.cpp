// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"

namespace mlir::iree_compiler {

namespace {

class GPUAnnotateWinogradLoopsPass final
    : public GPUAnnotateWinogradLoopsBase<GPUAnnotateWinogradLoopsPass> {
public:
  GPUAnnotateWinogradLoopsPass() = default;
  GPUAnnotateWinogradLoopsPass(const GPUAnnotateWinogradLoopsPass &pass) =
      default;

  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<scf::ForOp> forOps;
    funcOp.walk([&](scf::ForOp forOp) {
      if (!isTiledAndDistributedLoop(forOp))
        forOps.push_back(forOp);
    });

    // MLIRContext *context = &getContext();
    // OpBuilder builder(context);
    // const char *attrName = getSPIRVDistributeAttrName();
    // for (auto [index, forOp] : llvm::enumerate(forOps)) {
    //   if (index > kNumGPUDims)
    //     break;
    //   forOp->setAttr(attrName, builder.getIndexAttr(index));
    // }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUAnnotateWinogradLoopsPass() {
  return std::make_unique<GPUAnnotateWinogradLoopsPass>();
}

} // namespace mlir::iree_compiler
