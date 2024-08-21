// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPIRVANNOTATEWINOGRADLOOPSPASS
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"

namespace {

class SPIRVAnnotateWinogradLoopsPass final
    : public impl::SPIRVAnnotateWinogradLoopsPassBase<
          SPIRVAnnotateWinogradLoopsPass> {
public:
  using impl::SPIRVAnnotateWinogradLoopsPassBase<
      SPIRVAnnotateWinogradLoopsPass>::SPIRVAnnotateWinogradLoopsPassBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<scf::ForOp> forOps;
    funcOp.walk([&](scf::ForOp forOp) {
      if (!isTiledAndDistributedLoop(forOp))
        forOps.push_back(forOp);
    });

    MLIRContext *context = &getContext();
    OpBuilder builder(context);
    const char *attrName = getGPUDistributeAttrName();
    for (auto [index, forOp] : llvm::enumerate(forOps)) {
      if (index > kNumGPUDims)
        break;
      forOp->setAttr(attrName, builder.getIndexAttr(index));
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
