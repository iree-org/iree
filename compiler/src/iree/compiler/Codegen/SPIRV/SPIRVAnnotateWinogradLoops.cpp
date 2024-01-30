// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"

namespace mlir::iree_compiler {

namespace {

class SPIRVAnnotateWinogradLoopsPass final
    : public SPIRVAnnotateWinogradLoopsBase<SPIRVAnnotateWinogradLoopsPass> {
public:
  SPIRVAnnotateWinogradLoopsPass() = default;
  SPIRVAnnotateWinogradLoopsPass(const SPIRVAnnotateWinogradLoopsPass &pass) =
      default;

  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<scf::ForOp> forOps;
    funcOp.walk([&](scf::ForOp forOp) {
      if (!isTiledAndDistributedLoop(forOp))
        forOps.push_back(forOp);
    });

    MLIRContext *context = &getContext();
    OpBuilder builder(context);
    const char *attrName = getSPIRVDistributeAttrName();
    for (auto [index, forOp] : llvm::enumerate(forOps)) {
      if (index > kNumGPUDims)
        break;
      forOp->setAttr(attrName, builder.getIndexAttr(index));
    }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createSPIRVAnnotateWinogradLoopsPass() {
  return std::make_unique<SPIRVAnnotateWinogradLoopsPass>();
}

} // namespace mlir::iree_compiler
