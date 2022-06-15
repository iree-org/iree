// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/NVGPUDialect.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
static void getPipelineStages(scf::ForOp forOp,
                              std::vector<std::pair<Operation*, unsigned>>& ops,
                              unsigned depth) {
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
  // stage `maxDepth`. In order to have a correct scheduling even with back
  // edges we order stages in decreasing order.
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (!loadDep.count(&op) && !isa<scf::YieldOp>(op))
      ops.push_back(std::make_pair(&op, depth));
  }
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (loadDep.count(&op)) ops.push_back(std::make_pair(&op, 0));
  }
}

static void setAsyncAnnotations(Operation* op,
                                scf::PipeliningOption::PipelinerPart part,
                                unsigned iteration, unsigned depth) {
  auto waitOp = dyn_cast<nvgpu::DeviceAsyncWaitOp>(op);
  if (!waitOp || waitOp.numGroups()) return;
  int numGroupInFlight = 0;
  if (part == scf::PipeliningOption::PipelinerPart::Kernel) {
    numGroupInFlight = depth - 1;
  } else {
    // By construction there should be no wait op in the prologue as all the
    // wait should be in the last stage.
    assert(part == scf::PipeliningOption::PipelinerPart::Epilogue);
    // Based on the schedule we pick we know how many groups are in flight for
    // each iteration of the epilogue.
    numGroupInFlight = depth - 1 - iteration;
  }
  OpBuilder b(op);
  waitOp->setAttr(waitOp.numGroupsAttrName(),
                  b.getI32IntegerAttr(numGroupInFlight));
}

namespace {
struct GPUPipeliningPass : public GPUPipeliningBase<GPUPipeliningPass> {
  GPUPipeliningPass(unsigned depth) : depth(depth) {}
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext* context = &getContext();
    // Mark the loop with shared memory copy for pipelining.
    funcOp.walk([](scf::ForOp forOp) {
      bool copyToWorkgroupMemory = false;
      OpBuilder builder(forOp.getContext());
      SmallVector<Operation*> barriers;
      for (Operation& op : forOp.getBody()->getOperations()) {
        // Pipeline the most inner for op that should be a flat region.
        if (op.getNumRegions() > 0) return;
        if (isa<gpu::BarrierOp>(op)) {
          barriers.push_back(&op);
        }
        if (isa<nvgpu::DeviceAsyncCopyOp, nvgpu::DeviceAsyncCreateGroupOp>(
                op)) {
          copyToWorkgroupMemory = true;
          op.setAttr(kPipeliningGlobalLoad, builder.getUnitAttr());
          // async copy ops need to be moved along with previous barrier.
          for (Operation* barrier : barriers) {
            barrier->setAttr(kPipeliningGlobalLoad, builder.getUnitAttr());
          }
          barriers.clear();
          continue;
        }
        auto ld = dyn_cast<vector::TransferReadOp>(op);
        if (!ld) continue;
        unsigned ldAddSpace =
            ld.getSource().getType().cast<MemRefType>().getMemorySpaceAsInt();
        if (ldAddSpace != 0 || !ld->hasOneUse()) continue;
        auto st =
            dyn_cast<vector::TransferWriteOp>(ld->use_begin()->getOwner());
        if (!st) continue;
        unsigned stAddSpace =
            st.getSource().getType().cast<MemRefType>().getMemorySpaceAsInt();
        if (stAddSpace != 3) continue;
        copyToWorkgroupMemory = true;
        ld->setAttr(kPipeliningGlobalLoad, builder.getUnitAttr());
      }
      if (copyToWorkgroupMemory) {
        forOp->setAttr(kPipeliningLoopMarker, builder.getUnitAttr());
      }
    });
    scf::PipeliningOption options;
    unsigned maxDepth = depth;
    auto getSchedule = [maxDepth](
                           scf::ForOp forOp,
                           std::vector<std::pair<Operation*, unsigned>>& ops) {
      return getPipelineStages(forOp, ops, maxDepth);
    };
    auto setAnnotation = [maxDepth](Operation* op,
                                    scf::PipeliningOption::PipelinerPart part,
                                    unsigned iteration) {
      return setAsyncAnnotations(op, part, iteration, maxDepth);
    };
    options.getScheduleFn = getSchedule;
    options.annotateFn = setAnnotation;
    RewritePatternSet pipeliningPatterns(context);
    scf::populateSCFLoopPipeliningPatterns(pipeliningPatterns, options);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(pipeliningPatterns)))) {
      return signalPassFailure();
    }
  }

 private:
  unsigned depth;
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createGPUPipeliningPass(
    unsigned depth) {
  return std::make_unique<GPUPipeliningPass>(depth);
}

}  // namespace iree_compiler
}  // namespace mlir
