// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
  // Type definitions
  //
  using KgroupToWarpMmaMap = std::unordered_map<int, WarpMmaOp>;

  //
  // Data members
  //

  // Mainloop operations GlobalMemory -> SharedMemory
  SetVector<Operation*> copyGlobalToSharedOps;
  SetVector<Operation*> asyncCreateGroupOp;
  SetVector<Operation*> barrierOps;
  SetVector<Operation*> asyncWaitOps;

  // Warp-level MMA operations SharedMemory -> Registers -> MMA.
  // Map: kgroupIdx -> WarpMmaOp
  KgroupToWarpMmaMap warpOperations;

  std::vector<llvm::SetVector<Operation*>> ldmatrixOpDeps;
  std::vector<llvm::SetVector<Operation*>> mmasyncOpDeps;
  llvm::SetVector<Operation*> copyGlobalToSharedOpDeps;

  // Map: Operation -> kgroupIdx
  std::unordered_map<Operation*, int> opToKgroupMap;

  // Set to track the dependencies already seen to a backward slice.
  llvm::SetVector<Operation*> seenDepOps;

  // Set to track the mma operations in forward slice to count kgroups and
  // populate the warp-level WarpMmaOp kgroupIdx -> WarpMmaOp map.
  llvm::SetVector<Operation*> seenMmaOps;

  //
  // Methods
  //

  // Populates the dependent operations in ``dependentOps`` for the given a op
  // recursively that are in the same block and not added to the backward slice
  // of some other op.
  void backwardSliceOfDependentOps(llvm::SetVector<Operation*>& dependentOps,
                                   Operation* op, Block* block) {
    if (!seenDepOps.insert(op)) return;
    // Add the unseen op to the dependentOps and recurse on its operands.
    dependentOps.insert(op);
    for (Value operand : op->getOperands()) {
      Operation* defOp = operand.getDefiningOp();
      if (defOp && defOp->getBlock() == block)
        backwardSliceOfDependentOps(dependentOps, defOp, block);
    }
  }

  // Backtrack from the MmaSyncOp operand (mlir::OpOperand) to its defining
  // `mlir::Operation` to find the ldmatrix or ld.shared operations that load
  // MmaSyncOp operands.
  void backtrackToFindSmemLoad(Operation* op,
                               llvm::SetVector<Operation*>& loadOperations,
                               Block* block) {
    if (!op) return;

    if (isa<nvgpu::LdMatrixOp>(op)) {
      if (op->getBlock() == block) {
        loadOperations.insert(op);
      }
      return;
    }

    // Recurse upwards towards the definition until a Shared Memory load is
    // found. Assumption here is that only single operand operations are
    // leading up to LdMatrix.
    Operation* defOp = op->getOperand(0).getDefiningOp();

    backtrackToFindSmemLoad(defOp, loadOperations, block);
  }

  // Recursively traverse the chain of mma operations for all kgroups from 0
  // (start) to numKgroups (ends scf.yield).
  void vistMmaSyncOp(Operation* op, int kgroupIdx) {
    // if the operation in an `scf.yield`, we reached the end of MmaSyncOp chain
    // return.
    if (seenMmaOps.count(op) || isa<scf::YieldOp>(op)) return;

    seenMmaOps.insert(op);
    warpOperations[kgroupIdx].mmaOperations.insert(op);

    backtrackToFindSmemLoad(op->getOperand(0).getDefiningOp(),
                            warpOperations[kgroupIdx].loadOperationsA,
                            op->getBlock());

    backtrackToFindSmemLoad(op->getOperand(1).getDefiningOp(),
                            warpOperations[kgroupIdx].loadOperationsB,
                            op->getBlock());

    vistMmaSyncOp((op->getUses().begin())->getOwner(), ++kgroupIdx);
  }

  // Ctor.
  MainLoopInfo(scf::ForOp forOp) { analyze(forOp); }

  // Iterate through the mainloop and collect the `cp.async`,
  // `cp.commit_group`, `cp.wait_group`, and `barrier` operations. These
  // operations are used to pipeline the mainloop. Additionally, collect the
  // `mma.sync` and `ldmatrix`/`ld.shared` operations and separate them into
  // kgroups. These operations are used to generate an optimal *finer-grained*
  // schedule of global memory loads, shared memory loads, and math operations.
  void analyze(scf::ForOp forOp) {
    for (Operation& op : forOp.getBody()->getOperations()) {
      // Collect the async.copy, async.wait, and barrier operations for
      // coarse-grained instruction pipelining.
      if (isa<nvgpu::DeviceAsyncCopyOp>(op)) {
        copyGlobalToSharedOps.insert(&op);
      }
      if (isa<nvgpu::DeviceAsyncCreateGroupOp>(op)) {
        asyncCreateGroupOp.insert(&op);
      }
      if (isa<gpu::BarrierOp>(op)) {
        barrierOps.insert(&op);
      }
      if (isa<nvgpu::DeviceAsyncWaitOp>(op)) {
        asyncWaitOps.insert(&op);
      }
      // Collect the warp-level mma.sync and load operations (smem -> registers)
      // and separate them into kgroups for fine-grained instruction scheduling.
      if (isa<nvgpu::MmaSyncOp>(op)) {
        vistMmaSyncOp(&op, 0 /*kgroup = 0*/);
      }
    }

    // Assert that cp.async.commit_group, cp.async.wait_group, and bar.sync have
    // only 1 occurance in un-pipelined mainloop (forOp).
    assert(asyncCreateGroupOp.size() == 1 &&
           "Expected only one async.create.group op");
    assert(asyncWaitOps.size() == 1 && "Expected only one async.wait op");
    assert(barrierOps.size() == 1 && "Expected only one barrier op");

    // Obtain the reverse map: operation -> kgroup.
    for (auto& warpOp : warpOperations) {
      for (auto mmaOp : warpOp.second.mmaOperations) {
        opToKgroupMap[mmaOp] = warpOp.first;
      }
      for (auto loadOp : warpOp.second.loadOperationsA) {
        opToKgroupMap[loadOp] = warpOp.first;
      }
      for (auto loadOp : warpOp.second.loadOperationsB) {
        opToKgroupMap[loadOp] = warpOp.first;
      }
    }

    // Collect the dependent operations for cp.async for couarse-grained
    // instruction scheduling.
    // Dependent operations for cp.async in loop order.
    for (Operation& op : forOp.getBody()->getOperations()) {
      if (isa<nvgpu::DeviceAsyncCopyOp>(&op)) {
        backwardSliceOfDependentOps(copyGlobalToSharedOpDeps, &op,
                                    forOp.getBody());
      }
    }

    // Collect the dependent operations for mma.sync and ldmatix operations
    // seperated by kgroups for fine-grained instruction scheduling.
    // Resize to accomodate the number of kgroups.
    ldmatrixOpDeps.resize(warpOperations.size());
    mmasyncOpDeps.resize(warpOperations.size());

    for (int kgroup = 0; kgroup < getNumberOfKgroups(); ++kgroup) {
      // Dependent operations for ldmatrix for the kgroup in loop order.
      for (Operation& op : forOp.getBody()->getOperations()) {
        if (isa<nvgpu::LdMatrixOp>(&op)) {
          if (opToKgroupMap[&op] == kgroup) {
            backwardSliceOfDependentOps(ldmatrixOpDeps[kgroup], &op,
                                        forOp.getBody());
          }
        }
      }

      // Dependent operations for mma.sync the kgroup in loop order.
      for (Operation& op : forOp.getBody()->getOperations()) {
        if (isa<nvgpu::MmaSyncOp>(&op)) {
          if (opToKgroupMap[&op] == kgroup) {
            backwardSliceOfDependentOps(mmasyncOpDeps[kgroup], &op,
                                        forOp.getBody());
          }
        }
      }
    }
  }

  // Returns the number of kgroups in the Warp-level MMA operations.
  int getNumberOfKgroups() { return warpOperations.size(); }
};

/// This function returns an instruction schedule and stage assignment for the
/// mainloop that gives good performance on Nvidia Ampere architecture using
/// Ampere-style multi-staged pipeline.
///
/// @param forOp the main loop operation to pipeline and schedule.
/// @param ops a vector of pairs of operations and their assigned pipeline
/// stage.
/// @param numStages the total number of pipeline stages used for pipelining the
/// mainloop.
static void getMultiStagedPipelineSchedule(
    scf::ForOp forOp, std::vector<std::pair<Operation*, unsigned>>& ops,
    unsigned numStages) {
  // Analyze the main loop and obtain information for coarse-grained pipelining
  // and fine-grained instruction scheduling.
  MainLoopInfo info(forOp);
  int numKgroups = info.getNumberOfKgroups();

  // Assert the requirements for pipelining the mainloop targeting NVIDIA
  // Tensor Cores using multistaged pipeline.
  assert(numKgroups > 1 && "Number of kgroups should be 2 ore more");
  assert(numStages > 2 && "Number of stages should be 3 ore more");

  // Start pipelining and scheduling the main loop, all kgroups but the last
  // one.
  for (int kgroup = 0; kgroup < numKgroups - 1; kgroup++) {
    // Fine-grained instruction scheduling: interleave Shared Memory loads
    // into and mma.sync operations to hide load latencies.

    // Load the next kgroup into registers.
    for (Operation& op : forOp.getBody()->getOperations()) {
      if (info.ldmatrixOpDeps[kgroup + 1].count(&op))
        ops.push_back(std::make_pair(&op, numStages - 1));
    }

    // Issue mma.sync on previous loaded kgroup.
    for (Operation& op : forOp.getBody()->getOperations()) {
      if (info.mmasyncOpDeps[kgroup].count(&op))
        ops.push_back(std::make_pair(&op, numStages - 1));
    }
  }

  // Coarse-grained instruction pipelining: pipeline Global Memory
  // transfer (GMEM -> SMEM) several stages in advance.

  // Schedule all cp.async and one cp.async.commit_group.
  // TODO: Distribute cp.async throughout the main loop and do not concentrate
  // it at one place.
  // Schedule all cp.async and one cp.async.commit_group.
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (info.copyGlobalToSharedOpDeps.count(&op))
      ops.push_back(std::make_pair(&op, 0 /*pipelineStage*/));
  }
  ops.push_back(
      std::make_pair(info.asyncCreateGroupOp[0], 0 /*pipelineStage*/));

  // Schedule and pipeline all async.wait and barrier
  ops.push_back(std::make_pair(info.asyncWaitOps[0], numStages - 2));
  ops.push_back(std::make_pair(info.barrierOps[0], numStages - 2));
  //////////////////////////////////////////////////////////////////////////////

  // Coarse-grained instruction pipelining: pipeline Shared Memory loads
  // (SMEM -> Registers) for the first kgroup (kgroup = 0) one stage in advance.

  // Schedule the Shared Memory loads for the first kgroup and pipeline them
  // into one stage ahead.
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (info.ldmatrixOpDeps[0].count(&op))
      ops.push_back(std::make_pair(&op, numStages - 2));
  }

  // Issue mma.sync on for the last kgroup at the end of the mainloop.
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (info.mmasyncOpDeps[numKgroups - 1].count(&op))
      ops.push_back(std::make_pair(&op, numStages - 1));
  }
}

// Apply pipeline rewrite pattern assuming the operations were already
// annotated with stage information.
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
      return getMultiStagedPipelineSchedule(forOp, ops, maxDepth);
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
