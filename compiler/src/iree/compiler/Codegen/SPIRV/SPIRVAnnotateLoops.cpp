// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-annotate-loops"

namespace mlir {
namespace iree_compiler {

namespace {

class SPIRVAnnotateLoopsPass final
    : public SPIRVAnnotateLoopsBase<SPIRVAnnotateLoopsPass> {
 public:
  SPIRVAnnotateLoopsPass() = default;
  SPIRVAnnotateLoopsPass(const SPIRVAnnotateLoopsPass &pass) = default;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    SmallVector<scf::ForOp, 4> forOps;
    funcOp.walk([&](scf::ForOp forOp) {
      if (!isTiledAndDistributedLoop(forOp)) forOps.push_back(forOp);
    });

    MLIRContext *context = &getContext();
    OpBuilder builder(context);
    const char *attrName = getSPIRVDistributeAttrName();
    for (auto forOp : llvm::enumerate(forOps)) {
      if (forOp.index() > kNumGPUDims) break;
      forOp.value()->setAttr(attrName, builder.getIndexAttr(forOp.index()));
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVAnnotateLoopsPass() {
  return std::make_unique<SPIRVAnnotateLoopsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
