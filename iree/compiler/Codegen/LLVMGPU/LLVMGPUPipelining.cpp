// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

//====---------------------------------------------------------------------===//
// Pass to pipeline copy to shared memory for matmul op.
//====---------------------------------------------------------------------===//

namespace mlir {
namespace iree_compiler {

static const StringLiteral kPipeliningLoopMarker = "__pipelining_K_loop__";
static const StringLiteral kPipeliningGlobalLoad = "__pipelining_global_load__";

/// Helper to recursively add operation dependencies within `block` to `dep`
/// set.
static void addDepOps(llvm::SmallDenseSet<Operation*>& dep, Operation* op,
                      Block* block) {
  if (!dep.insert(op).second) return;
  for (Value operand : op->getOperands()) {
    Operation* defOp = operand.getDefiningOp();
    if (defOp && defOp->getBlock() == block) addDepOps(dep, defOp, block);
  }
}

/// Assign stages to the loop ops. Simple logic for now, put load from global
/// memory in stage 0 and the rest in stage 1.
static void getPipelineStages(
    scf::ForOp forOp, std::vector<std::pair<Operation*, unsigned>>& ops) {
  if (!forOp->hasAttr(kPipeliningLoopMarker)) return;

  // Track dependencies of the global memory load.
  llvm::SmallDenseSet<Operation*> loadDep;
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (op.hasAttr(kPipeliningGlobalLoad)) {
      addDepOps(loadDep, &op, forOp.getBody());
    }
  }
  // Create a modulo schedule with loads from global memory and the operations
  // it depends on in stage 0. Store to shared memory and computation are in
  // stage 1. In order to have a correct scheduling even with back edges we
  // order stages in decreasing order.
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (!loadDep.count(&op) && !isa<scf::YieldOp>(op))
      ops.push_back(std::make_pair(&op, 1));
  }
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (loadDep.count(&op)) ops.push_back(std::make_pair(&op, 0));
  }
}

namespace {
struct LLVMGPUPipeliningPass
    : public LLVMGPUPipeliningBase<LLVMGPUPipeliningPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext* context = &getContext();
    // Mark the loop with shared memory copy for pipelining.
    funcOp.walk([](scf::ForOp forOp) {
      bool copyToWorkgroupMemory = false;
      OpBuilder builder(forOp.getContext());
      for (Operation& op : forOp.getBody()->getOperations()) {
        // Pipeline the most inner for op that should be a flat region.
        if (op.getNumRegions() > 0) return;
        auto ld = dyn_cast<vector::TransferReadOp>(op);
        if (!ld) continue;
        unsigned ldAddSpace =
            ld.source().getType().cast<MemRefType>().getMemorySpaceAsInt();
        if (ldAddSpace != 0 || !ld->hasOneUse()) continue;
        auto st =
            dyn_cast<vector::TransferWriteOp>(ld->use_begin()->getOwner());
        if (!st) continue;
        unsigned stAddSpace =
            st.source().getType().cast<MemRefType>().getMemorySpaceAsInt();
        if (stAddSpace != 3) continue;
        copyToWorkgroupMemory = true;
        ld->setAttr(kPipeliningGlobalLoad, builder.getUnitAttr());
      }
      if (copyToWorkgroupMemory) {
        forOp->setAttr(kPipeliningLoopMarker, builder.getUnitAttr());
      }
    });
    scf::PipeliningOption options;
    options.getScheduleFn = getPipelineStages;
    RewritePatternSet pipeliningPatterns(context);
    scf::populateSCFLoopPipeliningPatterns(pipeliningPatterns, options);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(pipeliningPatterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLLVMGPUPipeliningPass() {
  return std::make_unique<LLVMGPUPipeliningPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
