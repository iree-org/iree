// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

//====---------------------------------------------------------------------===//
// Pass to pipeline copy to shared memory for matmul op.
//====---------------------------------------------------------------------===//

namespace mlir {
namespace iree_compiler {

static const StringLiteral kPipeliningLoopMarker = "__pipelining_K_loop__";
static const StringLiteral kPipeliningFirstStage = "__pipelining_first_stage__";
static const StringLiteral kPipeliningExtraBarrier =
    "__pipelining_extra_barrier__";

/// Returns true if the given `memrefType` has the default numeric address space
/// 0 or a HAL descriptor type address space.
static bool hasDefaultOrHALAddressSpace(MemRefType memrefType) {
  Attribute addrSpace = memrefType.getMemorySpace();
  if (!addrSpace) return true;
  auto intAttr = addrSpace.dyn_cast<IntegerAttr>();
  // Accept both default numeric address space and HAL descriptor type address
  // space--the former is used by LLVMGPU while the latter is used by SPIR-V.
  if (intAttr && intAttr.getInt() == 0) return true;
  return addrSpace.isa<IREE::HAL::DescriptorTypeAttr>();
}

// Returns a new predicated operation to support unpeeled epilogue. Unpeeled
// epilogue needs to handle the last iterations within the mainloop which
// requires predicating operations, for e.g., OOB global memory access. This
// helper function predicates operations (where predication is avialable),
// checks if unpredicated operations are side-effect free and acceptable to
// execute speculatively.
static Operation* replaceOpWithPredicatedOp(Operation* op, Value pred,
                                            PatternRewriter& rewriter) {
  // Predication is only supported for AsyncCopyOp. Thus, for operations which
  // are *not* AsyncCopyOp additional checks are requrired in order to be issued
  // speculatively.
  if (!isa<nvgpu::DeviceAsyncCopyOp>(op)) {
    // Return/execute the op if it is a side effect free.
    if (mlir::isMemoryEffectFree(op)) return op;
    // Return/execute the op if it is barrier, commit group, or ldmatrix op.
    if (isa<gpu::BarrierOp, nvgpu::DeviceAsyncCreateGroupOp, nvgpu::LdMatrixOp,
            nvgpu::DeviceAsyncWaitOp>(op))
      return op;
    // Return/execute the op if it is a shared memory load.
    if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
      auto loadBaseType = loadOp.getBase().getType().cast<MemRefType>();
      if (hasSharedMemoryAddressSpace(loadBaseType)) return op;
    }
    // If we are here that means the operation does not have predication support
    // and cannot be speculatively executed. Thus, unpeeled epilogue is not
    // supported.
    assert(false &&
           "Unpeeled epilogue not supported with a side-effect instruction "
           "with no predication.");
  }

  // Replace mainloop AsyncCopy with AsyncCopy(zfill) inline asm.
  auto asyncCopyOp = dyn_cast<nvgpu::DeviceAsyncCopyOp>(op);
  auto loc = asyncCopyOp->getLoc();

  // Create srcElement Value based on the pred.
  // The next few lins generate the below code:
  // srcElement = (pred) ?  dstElements : 0;
  Value dstElements =
      rewriter.create<arith::ConstantOp>(loc, asyncCopyOp.getDstElementsAttr());
  Value c0Index = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto srcElements =
      rewriter.create<arith::SelectOp>(loc, pred, dstElements, c0Index);
  auto asyncCopyZfillOp = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
      loc, nvgpu::DeviceAsyncTokenType::get(asyncCopyOp.getContext()),
      asyncCopyOp.getDst(), asyncCopyOp.getDstIndices(), asyncCopyOp.getSrc(),
      asyncCopyOp.getSrcIndices(), asyncCopyOp.getDstElements(), srcElements,
      UnitAttr());

  rewriter.eraseOp(asyncCopyOp);

  // Return the newly create predicated AsyncCopyZfillOp.
  return asyncCopyZfillOp;
}

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

/// Add an op and its dependency to `ops` set and skip operations contained into
/// filter. This also adds all the ops to filter so that they don't get matched
/// again.
static void addOpsAndDeps(llvm::SmallDenseSet<Operation*>& filter,
                          llvm::SmallDenseSet<Operation*>& ops, Operation* op,
                          Block* block) {
  if (!filter.insert(op).second) return;
  ops.insert(op);
  for (Value operand : op->getOperands()) {
    Operation* defOp = operand.getDefiningOp();
    if (defOp && defOp->getBlock() == block)
      addOpsAndDeps(filter, ops, defOp, block);
  }
}

/// Add an op and its dependency to `ops` set and skip operations contained into
/// filter. This also adds all the ops to filter so that they don't get matched
/// again.
static void addOpsAndDepsWithFilter(llvm::SmallDenseSet<Operation*>& filter,
                                    llvm::SmallDenseSet<Operation*>& skippedOps,
                                    llvm::SmallDenseSet<Operation*>& ops,
                                    Operation* op, Block* block) {
  if (!filter.insert(op).second || skippedOps.count(op)) return;
  ops.insert(op);
  for (Value operand : op->getOperands()) {
    Operation* defOp = operand.getDefiningOp();
    if (defOp && defOp->getBlock() == block)
      addOpsAndDepsWithFilter(filter, skippedOps, ops, defOp, block);
  }
}

/// Assign stages to the loop ops. Simple logic by default, put load from global
/// memory in stage 0 and the rest in stage 1. If store_stage = 0 then put store
/// to shared memory in stage 0 as well.
static void getPipelineStages(scf::ForOp forOp,
                              std::vector<std::pair<Operation*, unsigned>>& ops,
                              unsigned depth) {
  if (!forOp->hasAttr(kPipeliningLoopMarker)) return;

  // Track dependencies of stage 0 ops.
  llvm::SmallDenseSet<Operation*> loadDep;
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (op.hasAttr(kPipeliningFirstStage)) {
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
                                unsigned iteration, unsigned depth,
                                PipeliningSchedulingStrategy schedule) {
  if (auto waitOp = dyn_cast<nvgpu::DeviceAsyncWaitOp>(op)) {
    // Based on the order copies within the loop we need to adjust the number of
    // copies in flight.
    bool copyBeforeLoad =
        schedule == PipeliningSchedulingStrategy::nvidiaTensorCore;
    if (waitOp.getNumGroups()) return;
    int numGroupInFlight = 0;
    if (part == scf::PipeliningOption::PipelinerPart::Kernel ||
        part == scf::PipeliningOption::PipelinerPart::Prologue) {
      numGroupInFlight = copyBeforeLoad ? depth - 2 : depth - 1;
    } else {
      // By construction there should be no wait op in the prologue as all the
      // wait should be in the last stage.
      assert(part == scf::PipeliningOption::PipelinerPart::Epilogue);
      // Based on the schedule we pick we know how many groups are in flight for
      // each iteration of the epilogue.
      numGroupInFlight = depth - 1 - iteration;
    }
    OpBuilder b(op);
    waitOp->setAttr(waitOp.getNumGroupsAttrName(),
                    b.getI32IntegerAttr(numGroupInFlight));
  } else if (auto barrierOp = dyn_cast<gpu::BarrierOp>(op)) {
    unsigned pipelineStoreStage =
        schedule == PipeliningSchedulingStrategy::loadStoreStage0 ? 0 : 1;
    if (pipelineStoreStage != 0 ||
        part != mlir::scf::PipeliningOption::PipelinerPart::Prologue ||
        iteration >= depth - 1)
      return;
    OpBuilder b(op);
    barrierOp->setAttr(kPipeliningExtraBarrier, b.getUnitAttr());
  }
}

/// Check if the for operations contains a shared memory copy that can be
/// pipelined and annotate operations with stage information if this is the
/// case.
static bool setPipeliningMarkers(scf::ForOp forOp, bool pipelineStoreStage) {
  bool copyToWorkgroupMemory = false;
  OpBuilder builder(forOp.getContext());
  SmallVector<Operation*> barriers;
  for (Operation& op : forOp.getBody()->getOperations()) {
    // Pipeline the most inner for op that should be a flat region.
    if (op.getNumRegions() > 0) return false;
    if (isa<gpu::BarrierOp>(op)) {
      barriers.push_back(&op);
      if (pipelineStoreStage == 0)
        op.setAttr(kPipeliningFirstStage, builder.getUnitAttr());
    }
    if (isa<nvgpu::DeviceAsyncCopyOp, nvgpu::DeviceAsyncCreateGroupOp>(op)) {
      copyToWorkgroupMemory = true;
      op.setAttr(kPipeliningFirstStage, builder.getUnitAttr());
      // async copy ops need to be moved along with previous barrier.
      for (Operation* barrier : barriers) {
        barrier->setAttr(kPipeliningFirstStage, builder.getUnitAttr());
      }
      barriers.clear();
      continue;
    }
    auto ld = dyn_cast<vector::TransferReadOp>(op);
    if (!ld) continue;
    auto ldSrcType = ld.getSource().getType().cast<MemRefType>();
    if (!hasDefaultOrHALAddressSpace(ldSrcType) || !ld->hasOneUse()) continue;
    auto st = dyn_cast<vector::TransferWriteOp>(ld->use_begin()->getOwner());
    if (!st) continue;
    auto stSrcType = st.getSource().getType().cast<MemRefType>();
    if (!hasSharedMemoryAddressSpace(stSrcType)) continue;
    copyToWorkgroupMemory = true;
    ld->setAttr(kPipeliningFirstStage, builder.getUnitAttr());
    if (pipelineStoreStage == 0)
      st->setAttr(kPipeliningFirstStage, builder.getUnitAttr());
  }
  if (copyToWorkgroupMemory) {
    forOp->setAttr(kPipeliningLoopMarker, builder.getUnitAttr());
    if (pipelineStoreStage == 0 && !barriers.empty()) {
      barriers.front()->erase();
    }
  }
  return copyToWorkgroupMemory;
}

/// Return true if the op is used as operand `index` of an mma.sync op.
static bool isMMAOperand(Operation* op, int64_t index) {
  OpOperand* use = &(*((op->getUses()).begin()));
  if (auto extract = dyn_cast<vector::ExtractStridedSliceOp>(use->getOwner())) {
    use = &(*((extract->getUses()).begin()));
  }
  if (!isa<nvgpu::MmaSyncOp>(use->getOwner())) return false;
  return use->getOperandNumber() == index;
}

/// Return true if the op is used as operand A of an mma.sync op.
static bool isAOperand(Operation* op) { return isMMAOperand(op, 0); }
/// Return true if the op is used as operand B of an mma.sync op.
static bool isBOperand(Operation* op) { return isMMAOperand(op, 1); }

/// Loads from Shared Memory and the MMA operations on registers for a kgroup.
struct WarpMmaOp {
  // Load matrixA from Shared Memory to registers.
  llvm::SetVector<Operation*> loadOperationsA;
  // Load matrixB from Shared Memory to registers.
  llvm::SetVector<Operation*> loadOperationsB;
  // Warp-level MMA operations on registers.
  llvm::SetVector<Operation*> mmaOperations;
};

/// Structure to hold the matmul's mainloop information:
/// Seperates the mmasync operations into kgroups and collects the Shared Memory
/// loads for each kgroup. This information is used to pipeline the mainloop and
/// to generate an optimal schedule interleaving Global Memory loads, Shared
/// Memory loads, and math operations.
struct MainLoopInfo {
  //
  // Data members
  //

  // Mainloop operations GlobalMemory -> SharedMemory
  SmallVector<Operation*> copyGlobalToSharedOps;
  SmallVector<Operation*> barrierOps;
  SmallVector<Operation*> asyncWaitOps;

  // Warp-level MMA operations SharedMemory -> Registers -> MMA.
  // Map: kgroupIdx -> WarpMmaOp
  std::unordered_map<int, WarpMmaOp> warpOperations;

  //
  // Methods
  //

  // Backtrack from the MmaSyncOp operand (mlir::OpOperand) to its defining
  // `mlir::Operation` to find the ldmatrix or ld.shared operations that load
  // MmaSyncOp operands.
  void backtrackToFindSmemLoad(Operation* op,
                               llvm::SetVector<Operation*>& loadOperations,
                               Block* block) {
    if (!op) return;

    if (isa<nvgpu::LdMatrixOp>(op)) {
      if (op->getBlock() == block) loadOperations.insert(op);
      return;
    }

    // Recurse upwards towards the definition until a load is found.
    // Assumption here is that only single operand operations are leading up to
    // LdMatrix.
    Operation* defOp = op->getOperand(0).getDefiningOp();

    backtrackToFindSmemLoad(defOp, loadOperations, block);
  }

  // Recursively traverse the chain of mma operations for all kgroups from 0
  // 0 (start) to numKgroups (ends scf.yield).
  void vistMmaSyncOp(Operation* op, int kgroupIdx,
                     llvm::SmallDenseSet<Operation*>& seenMmaSyncOps) {
    // if the operation in an `scf.yield`, we reached the end of MmaSyncOp chain
    // return.
    if (seenMmaSyncOps.count(op) || isa<scf::YieldOp>(op)) return;

    seenMmaSyncOps.insert(op);
    warpOperations[kgroupIdx].mmaOperations.insert(op);

    backtrackToFindSmemLoad(op->getOperand(0).getDefiningOp(),
                            warpOperations[kgroupIdx].loadOperationsA,
                            op->getBlock());

    backtrackToFindSmemLoad(op->getOperand(1).getDefiningOp(),
                            warpOperations[kgroupIdx].loadOperationsB,
                            op->getBlock());

    vistMmaSyncOp((op->getUses().begin())->getOwner(), ++kgroupIdx,
                  seenMmaSyncOps);
  }

  // ctor
  MainLoopInfo() {}

  // Iterate through the mainloop and collect the async.copy, barrier, and
  // async.wait operations. These operations are used to pipeline the
  // mainloop. Additionally, collect the mma.sync operations and separate them
  // into kgroups. These operations are used to generate an optimal
  // finer-grained schedule for Shared Memory loads, Global Memory loads, and
  // math operations.
  void collect(scf::ForOp forOp) {
    llvm::SmallDenseSet<Operation*> seenMmaSyncOps;
    std::cout << "Collecting MainLoopInfo: " << std::endl;
    for (Operation& op : forOp.getBody()->getOperations()) {
      op.dump();
      if (isa<nvgpu::MmaSyncOp>(op)) {
        vistMmaSyncOp(&op, 0 /*kgroup = 0*/, seenMmaSyncOps);
      }
      if (isa<nvgpu::DeviceAsyncCopyOp, nvgpu::DeviceAsyncCreateGroupOp>(op)) {
        copyGlobalToSharedOps.push_back(&op);
      }
      if (isa<gpu::BarrierOp>(op)) {
        barrierOps.push_back(&op);
      }
      if (isa<nvgpu::DeviceAsyncWaitOp>(op)) {
        asyncWaitOps.push_back(&op);
      }
    }

    // Assert that barrierOps and asyncWaitOps have only 1 occurance in
    // un-pipelined mainloop.
    assert(barrierOps.size() == 1 && "Expected only one barrier op");
    assert(asyncWaitOps.size() == 1 && "Expected only one async.wait op");

#if 0
    // Obtain the list of all the MmaSyncOps within the mainloop.
    auto mmaSyncOps = forOp.getBody()->getOps<nvgpu::MmaSyncOp>();

    for (auto mmaSyncOp : mmaSyncOps) {
      vistMmaSyncOp(mmaSyncOp.getOperation(), 0, visited);
      // mmaSyncOp.dump();
    }
#endif
  }

  // Returns the number of kgroups in the Warp-level MMA operations.
  int getNumKGroups() { return warpOperations.size(); }

  // Dump the Mainloop info collected.
  void dump() {
    // Debug prints
    for (auto warpOp : warpOperations) {
      std::cout << "Load operations for operandA kGroup (" << warpOp.first
                << ")" << std::endl;
      for (auto loadOp : warpOp.second.loadOperationsA) {
        loadOp->dump();
      }
      std::cout << "Load operations for operandB kGroup (" << warpOp.first
                << ")" << std::endl;
      for (auto loadOp : warpOp.second.loadOperationsB) {
        loadOp->dump();
      }
      std::cout << "Mma Sync kGroup (" << warpOp.first << ")" << std::endl;
      for (auto mmaOp : warpOp.second.mmaOperations) {
        mmaOp->dump();
      }
    }

    std::cout << "Copy Global to Shared Memory" << std::endl;
    for (auto copyOp : copyGlobalToSharedOps) {
      copyOp->dump();
    }

    std::cout << "Barrier" << std::endl;
    for (auto barrierOp : barrierOps) {
      barrierOp->dump();
    }

    std::cout << "Async Wait" << std::endl;
    for (auto asyncWaitOp : asyncWaitOps) {
      asyncWaitOp->dump();
    }
  }
};

/// Return a pipelining schedule that gives good performance on Nvidia
/// Ampere target.
static void getNvidiaTensorCorePipeline(
    scf::ForOp forOp, std::vector<std::pair<Operation*, unsigned>>& ops,
    unsigned depth) {
  bool loopCanBePipelined = false;
  // TODO: Tune this and make it a more fine grain logic.
  static constexpr int64_t numPrefetchSmemLoadPerOperand = 4;
  SmallVector<Operation*> stageCopyToSharedMemory;
  SmallVector<Operation*> stagePrefetch;
  SmallVector<Operation*> stageCompute;
  int64_t numLoadA = 0;
  int64_t numLoadB = 0;
  for (Operation& op : forOp.getBody()->getOperations()) {
    // Pipeline the most inner for op that should be a flat region.
    if (op.getNumRegions() > 0) {
      loopCanBePipelined = false;
      break;
    }
    if (isa<gpu::BarrierOp, nvgpu::DeviceAsyncWaitOp>(op)) {
      stagePrefetch.push_back(&op);
    }
    if (isa<nvgpu::MmaSyncOp>(op)) {
      stageCompute.push_back(&op);
    }
    if (isa<nvgpu::DeviceAsyncCopyOp, nvgpu::DeviceAsyncCreateGroupOp>(op)) {
      stageCopyToSharedMemory.push_back(&op);
    }
    if (isa<nvgpu::LdMatrixOp>(op)) {
      // Prefecth some of the ldmatrix.
      if (isAOperand(&op)) {
        numLoadA++;
        if (numLoadA <= numPrefetchSmemLoadPerOperand) {
          stagePrefetch.push_back(&op);
          continue;
        }
      }
      if (isBOperand(&op)) {
        numLoadB++;
        if (numLoadB <= numPrefetchSmemLoadPerOperand) {
          stagePrefetch.push_back(&op);
          continue;
        }
      }
      // If not prefected go in the last stage.
      stageCompute.push_back(&op);
    }
  }

  // Return an empty schedule if the loop is not a candidate to be pipelined.
  if (loopCanBePipelined || stageCopyToSharedMemory.empty() ||
      stageCompute.empty())
    return;

  // Add all the dependencies of the operations in the stages.
  llvm::SmallDenseSet<Operation*> deps;
  llvm::SmallDenseSet<Operation*> stageCopyToSharedMemoryDeps;
  llvm::SmallDenseSet<Operation*> stageNMinusOneDeps;
  llvm::SmallDenseSet<Operation*> stageNDeps;
  for (Operation* op : stageCopyToSharedMemory) {
    addOpsAndDeps(deps, stageCopyToSharedMemoryDeps, op, forOp.getBody());
  }

  for (Operation* op : stagePrefetch) {
    addOpsAndDeps(deps, stageNMinusOneDeps, op, forOp.getBody());
  }

  for (Operation* op : stageCompute) {
    addOpsAndDeps(deps, stageNDeps, op, forOp.getBody());
  }

  // Schedule Compute and dependent load operations in stage (depth - 1).
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (stageNDeps.count(&op)) ops.push_back(std::make_pair(&op, depth - 1));
  }
  // Schedule copy to shared memory in stage 0.
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (stageCopyToSharedMemoryDeps.count(&op))
      ops.push_back(std::make_pair(&op, 0));
  }
  // Schedule prefetch data into register in stage (depth - 2).
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (stageNMinusOneDeps.count(&op))
      ops.push_back(std::make_pair(&op, depth - 2));
  }

  // Print schedule
  std::cout << "getNvidiaTensorCorePipeline Schedule" << std::endl;
  for (auto op : ops) {
    std::cout << " Stage: " << op.second;
    std::cout << " Operation: ";
    op.first->dump();
    std::cout << std::endl;
  }

  // Create schedule and assign software pipelining stages kgroup-by-kgroup.
}

/// Schedule operations and assign software pipelining stages.
static void scheduleOperations(
    std::vector<std::pair<Operation*, unsigned>>& ops,
    llvm::SmallDenseSet<Operation*> dependentOps, unsigned pipelineDepth) {
  for (Operation* op : dependentOps) {
    std::cout << "Stage : " << pipelineDepth << " ";
    std::cout << "Schedule operation: ";
    op->dump();
    std::cout << std::flush;
    std::cout << std::endl;
    std::cout << std::endl;
    ops.push_back(std::make_pair(op, pipelineDepth));
  }
}

/// Return a pipelining schedule that gives good performance on Nvidia
/// Ampere target.
static void getNvidiaTensorCoreScheduleAndPipeline(
    scf::ForOp forOp, std::vector<std::pair<Operation*, unsigned>>& ops,
    unsigned depth) {
  MainLoopInfo info;
  info.collect(forOp);
  info.dump();

  // Schedule the mainloop kgroup-by-kgroup.
  llvm::SmallDenseSet<Operation*> scheduledOps;
  llvm::SmallDenseSet<Operation*> skippedOps;

  for (int kgroup = 0; kgroup < info.getNumKGroups(); kgroup++) {
    // Schedule warp-level mma.sync operations.
    for (auto& mmaOp : info.warpOperations[kgroup].mmaOperations) {
      skippedOps.insert(info.warpOperations[0].loadOperationsA.begin(),
                        info.warpOperations[0].loadOperationsA.end());
      skippedOps.insert(info.warpOperations[0].loadOperationsB.begin(),
                        info.warpOperations[0].loadOperationsB.end());

      llvm::SmallDenseSet<Operation*> dependentOps;

      addOpsAndDepsWithFilter(scheduledOps, skippedOps, dependentOps, mmaOp,
                              forOp.getBody());
      scheduleOperations(ops, dependentOps, depth - 1);
    }

    skippedOps.clear();

    // Schedule warp-level ldmatrix or ld.shared operations for operandA.
    for (auto& ldmatrixOp : info.warpOperations[kgroup].loadOperationsA) {
      llvm::SmallDenseSet<Operation*> dependentOps;
      addOpsAndDepsWithFilter(scheduledOps, skippedOps, dependentOps,
                              ldmatrixOp, forOp.getBody());
      int pipelineStage = (kgroup == 0) ? (depth - 2) : (depth - 1);
      scheduleOperations(ops, dependentOps, pipelineStage);
    }

    // Schedule warp-level ldmatrix or ld.shared operations for operandB.
    for (auto& ldmatrixOp : info.warpOperations[kgroup].loadOperationsB) {
      llvm::SmallDenseSet<Operation*> dependentOps;
      addOpsAndDepsWithFilter(scheduledOps, skippedOps, dependentOps,
                              ldmatrixOp, forOp.getBody());
      int pipelineStage = (kgroup == 0) ? (depth - 2) : (depth - 1);
      scheduleOperations(ops, dependentOps, pipelineStage);
    }

#if 0  // TODO: Distribute cp.async operations across.
    // Schedule async copy from Global Memory to Shared Memory.
    int copyIdxStart =
        kgroup * (info.copyGlobalToSharedOps.size() / info.getNumKGroups());
    int copyIdxEnd = (kgroup + 1) *
                     (info.copyGlobalToSharedOps.size() / info.getNumKGroups());

    for (int idx = copyIdxStart; idx < copyIdxEnd; idx++) {
      llvm::SmallDenseSet<Operation*> dependentOps;
      addOpsAndDeps(scheduledOps, dependentOps, info.copyGlobalToSharedOps[idx],
                    forOp.getBody());
      scheduleOperations(ops, dependentOps, 0);
    }
#endif

    if (kgroup == info.getNumKGroups() - 1) {
      // Schedule any remaining cp.async or cp.async.commit_group operations.
      int idx = 0;

      while (idx < info.copyGlobalToSharedOps.size()) {
        info.copyGlobalToSharedOps[idx]->dump();
        llvm::SmallDenseSet<Operation*> dependentOps;
        addOpsAndDepsWithFilter(scheduledOps, skippedOps, dependentOps,
                                info.copyGlobalToSharedOps[idx],
                                forOp.getBody());
        scheduleOperations(ops, dependentOps, 0);
        idx++;
      }
      // Schedule cp.async.wait_group and bar.sync 0 for after the last
      // cp.async.
      llvm::SmallDenseSet<Operation*> asyncWaitDependentOps;
      addOpsAndDepsWithFilter(scheduledOps, skippedOps, asyncWaitDependentOps,
                              info.asyncWaitOps[0], forOp.getBody());
      scheduleOperations(ops, asyncWaitDependentOps, 0);

      llvm::SmallDenseSet<Operation*> barrierDependentOps;
      addOpsAndDepsWithFilter(scheduledOps, skippedOps, barrierDependentOps,
                              info.barrierOps[0], forOp.getBody());

      scheduleOperations(ops, barrierDependentOps, 0);
    }
  }
}

/// Apply pipeline rewrite pattern assuming the operations were already
/// annotated with stage information.
// TODO: move away from using attribute annotations.
static FailureOr<scf::ForOp> applyPipelining(
    scf::ForOp forOp, int64_t depth, bool epiloguePeeling,
    PipeliningSchedulingStrategy schedule) {
  // TODO: Refactor schedules to not rely on markers.
  if (schedule == PipeliningSchedulingStrategy::loadGlobalStage0 ||
      schedule == PipeliningSchedulingStrategy::loadStoreStage0) {
    unsigned pipelineStoreStage =
        schedule == PipeliningSchedulingStrategy::loadGlobalStage0;
    if (!setPipeliningMarkers(forOp, pipelineStoreStage)) {
      return failure();
    }
  }

  scf::PipeliningOption options;
  unsigned maxDepth = depth;
  auto getSchedule = [maxDepth, schedule](
                         scf::ForOp forOp,
                         std::vector<std::pair<Operation*, unsigned>>& ops) {
    if (schedule == PipeliningSchedulingStrategy::nvidiaTensorCore) {
      return getNvidiaTensorCoreScheduleAndPipeline(forOp, ops, maxDepth);
      // return getNvidiaTensorCorePipeline(forOp, ops, maxDepth);
    }
    return getPipelineStages(forOp, ops, maxDepth);
  };
  auto setAnnotation = [maxDepth, schedule](
                           Operation* op,
                           scf::PipeliningOption::PipelinerPart part,
                           unsigned iteration) {
    return setAsyncAnnotations(op, part, iteration, maxDepth, schedule);
  };
  options.getScheduleFn = getSchedule;
  options.annotateFn = setAnnotation;

  // Use un-peeled epilogue (i.e. epiloguePeeling=flase) only when predication
  // is avialable a.k.a. AsyncCopyOp.
  if (!epiloguePeeling) {
    options.peelEpilogue = false;
    options.predicateFn = [](Operation* op, Value pred,
                             PatternRewriter& rewriter) {
      return replaceOpWithPredicatedOp(op, pred, rewriter);
    };
  }
  scf::ForLoopPipeliningPattern pattern(options, forOp->getContext());
  transform::TrivialPatternRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  return pattern.returningMatchAndRewrite(forOp, rewriter);
}
namespace {
struct GPUPipeliningPass : public GPUPipeliningBase<GPUPipeliningPass> {
  GPUPipeliningPass(bool epiloguePeeling, int64_t depth,
                    PipeliningSchedulingStrategy schedule)
      : depth(depth), schedule(schedule), epiloguePeeling(epiloguePeeling) {}
  void initOptions() {
    if (GPUPipeliningBase::depth.hasValue())
      depth = GPUPipeliningBase::depth.getValue();
    if (GPUPipeliningBase::epiloguePeeling.hasValue())
      epiloguePeeling = GPUPipeliningBase::epiloguePeeling.getValue();
    if (GPUPipeliningBase::scheduleIndex.hasValue())
      schedule = (PipeliningSchedulingStrategy)
                     GPUPipeliningBase::scheduleIndex.getValue();
  }

  void runOnOperation() override {
    initOptions();
    auto funcOp = getOperation();
    SmallVector<scf::ForOp> forOps;
    // Mark the loop with shared memory copy for pipelining.
    funcOp.walk([&forOps](scf::ForOp forOp) { forOps.push_back(forOp); });
    for (scf::ForOp forOp : forOps) {
      (void)applyPipelining(forOp, depth, epiloguePeeling, schedule);
    }
    // Remove extra barriers from the prologue assuming appropriate
    // multi-buffering.
    funcOp.walk([](gpu::BarrierOp barrierOp) {
      if (barrierOp->hasAttr(kPipeliningExtraBarrier)) barrierOp->erase();
    });
  }

 private:
  int64_t depth;
  PipeliningSchedulingStrategy schedule;
  bool epiloguePeeling;
};
}  // namespace

FailureOr<scf::ForOp> pipelineSharedMemoryCopy(
    scf::ForOp forOp, PipeliningSchedulingStrategy startegy, bool peelEpilogue,
    int64_t depth, PatternRewriter& rewriter) {
  return applyPipelining(forOp, depth, peelEpilogue, startegy);
}

/// Pass options
/// epiloguePeeling - try enable/disable epilogue peeling.
/// true  : Peel epilogue (no additional checks required)
/// false : Try and use unpeeled epilogue (check if predication is supported
/// is avialable)
std::unique_ptr<OperationPass<func::FuncOp>> createGPUPipeliningPass(
    bool epiloguePeeling, unsigned depth,
    PipeliningSchedulingStrategy schedule) {
  return std::make_unique<GPUPipeliningPass>(epiloguePeeling, depth, schedule);
}

}  // namespace iree_compiler
}  // namespace mlir
