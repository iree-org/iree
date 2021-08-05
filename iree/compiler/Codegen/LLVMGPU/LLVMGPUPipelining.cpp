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

  // Track dependencies of transfer_read op from global memory.
  llvm::SmallDenseSet<Operation*> loadDep;
  for (Operation& op : forOp.getBody()->getOperations()) {
    auto ld = dyn_cast<vector::TransferReadOp>(op);
    if (ld &&
        ld.source().getType().cast<MemRefType>().getMemorySpaceAsInt() == 0) {
      addDepOps(loadDep, ld, forOp.getBody());
    }
  }
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
    // Mark the loop we want to pipeline.
    funcOp.walk([](scf::ForOp forOp) {
      bool hasBarrier = false;
      for (Operation& op : forOp.getBody()->getOperations()) {
        // Pipeline the most inner for op that should be a flat region.
        if (op.getNumRegions() > 0) return;
        if (isa<gpu::BarrierOp>(op)) hasBarrier = true;
      }
      if (hasBarrier) {
        OpBuilder builder(forOp.getContext());
        forOp->setAttr(kPipeliningLoopMarker, builder.getUnitAttr());
      }
    });
    scf::PipeliningOption options;
    options.getScheduleFn = getPipelineStages;
    RewritePatternSet pipeliningPatterns(context);
    scf::populateSCFLoopPipeliningPatterns(pipeliningPatterns, options);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(pipeliningPatterns));
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLLVMGPUPipeliningPass() {
  return std::make_unique<LLVMGPUPipeliningPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
