// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/CommonGPUPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-pipelining"

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
  auto intAttr = llvm::dyn_cast<IntegerAttr>(addrSpace);
  // Accept both default numeric address space and HAL descriptor type address
  // space--the former is used by LLVMGPU while the latter is used by SPIR-V.
  if (intAttr && intAttr.getInt() == 0) return true;
  return llvm::isa<IREE::HAL::DescriptorTypeAttr>(addrSpace);
}

/// Returns a new predicated operation to support unpeeled epilogue. Unpeeled
/// epilogue needs to handle the last iterations within the mainloop which
/// requires predicating operations, for e.g., OOB global memory access. This
/// helper function predicates operations (where predication is avialable),
/// checks if unpredicated operations are side-effect free and acceptable to
/// execute speculatively.
static Operation* replaceOpWithPredicatedOp(RewriterBase& rewriter,
                                            Operation* op, Value pred) {
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
      auto loadBaseType = llvm::cast<MemRefType>(loadOp.getBase().getType());
      if (hasSharedMemoryAddressSpace(loadBaseType)) return op;
    }
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      auto loadBaseType = loadOp.getMemRefType();
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
  // srcElement = (pred) ?  prevSrcElements : 0;
  Value dstElements =
      rewriter.create<arith::ConstantOp>(loc, asyncCopyOp.getDstElementsAttr());
  Value originalSrcElement =
      asyncCopyOp.getSrcElements() ? asyncCopyOp.getSrcElements() : dstElements;
  Value c0Index = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto srcElements =
      rewriter.create<arith::SelectOp>(loc, pred, originalSrcElement, c0Index);
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
    auto ldSrcType = llvm::cast<MemRefType>(ld.getSource().getType());
    if (!hasDefaultOrHALAddressSpace(ldSrcType) || !ld->hasOneUse()) continue;
    auto st = dyn_cast<vector::TransferWriteOp>(ld->use_begin()->getOwner());
    if (!st) continue;
    auto stSrcType = llvm::cast<MemRefType>(st.getSource().getType());
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

/// Warp-level TensorOp.
/// The data structure holds the warp-level Tensor Core (mma.sync) operations
/// and their dependencies for a kgroup.
struct WarpMmaOp {
  // Defining op and its dependencies for mma.sync's lhs/matrixA/OperandA.
  llvm::SetVector<Operation*> lhsOperations;
  // Defining op and its dependencies for mma.sync's rhs/matrixB/OperandB.
  llvm::SetVector<Operation*> rhsOperations;
  // Warp-level Tensor Core operations on operands in registers.
  llvm::SetVector<Operation*> mmaOperations;
};

/// Structure to hold the matmul's mainloop information:
/// Seperates the mma operations into kgroups and collects the Shared Memory
/// loads for each kgroup. This information is used to pipeline the mainloop and
/// to generate an optimal schedule; interleaving Global Memory loads, Shared
/// Memory loads, and math operations.
struct MainLoopInfo {
  // Mainloop asyncronous copy operations:
  // `cp.async` GlobalMemory -> SharedMemory
  llvm::SetVector<Operation*> copyGlobalToSharedOps;
  llvm::SetVector<Operation*> asyncCreateGroupOp;
  llvm::SetVector<Operation*> barrierOps;
  llvm::SetVector<Operation*> asyncWaitOps;

  // Mainloop asyncronous copy operations dependencies
  llvm::SetVector<Operation*> copyGlobalToSharedOpDeps;

  // Warp-level syncronous operations:
  // `ldmatrix, ld.shared` SharedMemory -> Registers
  // `mma.sync` Registers -> Tensor Cores.
  llvm::SmallVector<WarpMmaOp, 4> warpOperations;

  // Set to track the dependencies already seen to a backward slice.
  llvm::SetVector<Operation*> seenDepOps;

  // Set to track the mma operations in forward slice to count kgroups and
  // populate the warp-level warpOperations
  llvm::SetVector<Operation*> seenMmaOps;

  // Boolen to store if the mainloop can be pipelined (coarse-grained
  // scheduling) and the instructions can be interleaved (fine-grained
  // scheduling).
  bool isSchedulable = false;

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

  // Obtains nvgpu.ldmatrix, memref.load, vector.extract_strided_slice, or
  // vector.insert operations that is the defining operations of the mma.sync
  // operand. The operations are added to a set of specific kgroup operations.
  void mmaOperandDefOperation(Operation* op,
                              llvm::SetVector<Operation*>& defOperation,
                              Block* block) {
    if (!op) return;

    // If the operations defining the mma.sync's operand is one of the
    // qualifying operations, add the operations to the current kgroup defining
    // operations set.
    if (isa<nvgpu::LdMatrixOp, memref::LoadOp, vector::ExtractStridedSliceOp,
            vector::InsertOp>(op)) {
      if (op->getBlock() == block) {
        defOperation.insert(op);
      }
      return;
    }
  }

  // Recursively traverse the chain of mma operations for all kgroups from 0
  // (start) to numKgroups (ends scf.yield).
  // Assumption: The mma operations are in a chain of monotonicaly increasing
  // kgroup order.
  void vistMmaSyncOp(Operation* op, int kgroup) {
    // if the operation in an `scf.yield`, we reached the end of MmaSyncOp chain
    // return.
    if (seenMmaOps.count(op) || isa<scf::YieldOp>(op)) return;

    seenMmaOps.insert(op);

    // If the kgroup is not in the vector, create a new WarpMmaOp.
    if (warpOperations.size() < kgroup + 1)
      warpOperations.push_back(WarpMmaOp());

    mmaOperandDefOperation(op->getOperand(0).getDefiningOp(),
                           warpOperations[kgroup].lhsOperations,
                           op->getBlock());

    mmaOperandDefOperation(op->getOperand(1).getDefiningOp(),
                           warpOperations[kgroup].rhsOperations,
                           op->getBlock());

    warpOperations[kgroup].mmaOperations.insert(op);

    vistMmaSyncOp((op->getUses().begin())->getOwner(), ++kgroup);
  }

  MainLoopInfo(scf::ForOp forOp) : isSchedulable(true) { analyze(forOp); }

  // Iterate through the mainloop and collect `cp.async`, `cp.commit_group`,
  // `cp.wait_group`, and `barrier` operations. These operations are used to
  // pipeline the mainloop and cheorograph asyncroncy for a *coarse-grained*
  // schedule. Additionally, collect the `mma.sync` and `ldmatrix`/`ld.shared`
  // operations and separate them into kgroups. The information is helpful in
  // generating an optimal *finer-grained* instruction interleaving of global
  // memory loads, shared memory loads, and math operations.
  void analyze(scf::ForOp forOp) {
    for (Operation& op : forOp.getBody()->getOperations()) {
      if (op.getNumRegions() > 0) {
        // Pipeline and schedule the most inner for op ,i.e., the mainloop that
        // should be a flat region.
        isSchedulable = false;
        return;
      }
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
      if (isa<nvgpu::MmaSyncOp>(op)) {
        // MmaSyncOp visitor traverses the chain of mma operations and separates
        // them into kgroups.
        vistMmaSyncOp(&op, 0 /*kgroup=0*/);
      }
    }

    // Debug print warpOperations for kgroup-by-kgroup.
    LLVM_DEBUG({
      for (int i = 0; i < warpOperations.size(); ++i) {
        llvm::dbgs() << "kgroup: " << i << "\n";
        llvm::dbgs() << "mma.sync: \n";
        for (auto op : warpOperations[i].mmaOperations) {
          op->dump();
        }
        llvm::dbgs() << "\n";
        llvm::dbgs() << "defining operations for lhs: \n";
        for (auto op : warpOperations[i].lhsOperations) {
          op->dump();
        }
        llvm::dbgs() << "\n";
        llvm::dbgs() << "defining operations for rhs: \n";
        for (auto op : warpOperations[i].rhsOperations) {
          op->dump();
        }
        llvm::dbgs() << "\n";
      }
    });

    // If one of the ingredients (`cp.async`, `cp.commit_group`,
    // `cp.wait_group`, `bar.sync`, `mma.sync`, `ldmatrix` or `ld.shared`) for
    // scheduling is missing, the mainloop cannot be scheduled.
    if (copyGlobalToSharedOps.empty() || asyncCreateGroupOp.empty() ||
        asyncWaitOps.empty() || barrierOps.empty() || warpOperations.empty()) {
      isSchedulable = false;
      return;
    }

    // Collect the dependent operations for `cp.async` in the mainloop order for
    // coarse-grained software pipeling. The deps are collected in stage order,
    // i.e., `cp.async`'s deps in stage 0 are collected first.
    for (Operation& op : forOp.getBody()->getOperations()) {
      if (isa<nvgpu::DeviceAsyncCopyOp>(&op)) {
        backwardSliceOfDependentOps(copyGlobalToSharedOpDeps, &op,
                                    forOp.getBody());
      }
    }

    // Collect the dependent operations for `mma.sync`, lhs, and rhs defining
    // operations. The operation and their dependencies are seperated by kgroups
    // for fine-grained instruction scheduling.
    for (int kgroup = 0; kgroup < getNumberOfKgroups(); ++kgroup) {
      for (Operation& op : forOp.getBody()->getOperations()) {
        if (isa<nvgpu::LdMatrixOp, memref::LoadOp,
                vector::ExtractStridedSliceOp, vector::InsertOp>(&op)) {
          if (warpOperations[kgroup].lhsOperations.count(&op)) {
            backwardSliceOfDependentOps(warpOperations[kgroup].lhsOperations,
                                        &op, forOp.getBody());
          }
          if (warpOperations[kgroup].rhsOperations.count(&op)) {
            backwardSliceOfDependentOps(warpOperations[kgroup].rhsOperations,
                                        &op, forOp.getBody());
          }
        }
      }
      for (Operation& op : forOp.getBody()->getOperations()) {
        if (isa<nvgpu::MmaSyncOp>(&op)) {
          if (warpOperations[kgroup].mmaOperations.count(&op)) {
            backwardSliceOfDependentOps(warpOperations[kgroup].mmaOperations,
                                        &op, forOp.getBody());
          }
        }
      }
    }
  }

  // Returns the number of kgroups in the Warp-level MMA operations.
  int getNumberOfKgroups() { return warpOperations.size(); }
};

/// Prints the given `funcOp` after a leading `step` comment header.
static void debugMainloopSchedule(
    MainLoopInfo& mainloop, int numStages,
    std::vector<std::pair<Operation*, unsigned>>& ops) {
  LLVM_DEBUG({
    llvm::dbgs() << "//--- Mainloop schedule generated for Nvidia Ampere "
                    "mma.sync TensorCore Pipeline. ---//\n";
    llvm::dbgs() << " Number of stages: " << numStages << "\n";
    llvm::dbgs() << " Number of kgroups: " << mainloop.getNumberOfKgroups()
                 << "\n";
    llvm::dbgs() << " Number of mainloop instructions " << ops.size() << "\n";
    llvm::dbgs() << " Mainloop instructions schedule and stage assignment: \n";
    for (auto& stage_op_pair : ops) {
      llvm::dbgs() << " Stage (" << stage_op_pair.second << ") , Operation: ";
      stage_op_pair.first->dump();
    }
    llvm::dbgs() << "\n\n";
  });
}

/// This function returns an *coarse-grained* stage assignment for software
/// pipelining of the mainloop and a *fine-grained* instruction interleaving.
/// The schedule provides good performance on Nvidia Ampere architecture using
/// Ampere-style multi-staged pipeline.
///
/// @param forOp the mainloop to pipeline and schedule.
/// @param ops a vector of pairs: [(operations, pipeline_stage)].
/// @param numStages the total number of pipeline stages used for multi-buffer.
static void getNvidiaAmpereTensorCorePipeline(
    scf::ForOp forOp, std::vector<std::pair<Operation*, unsigned>>& ops,
    unsigned numStages) {
  // Analyze the main loop and obtain information for coarse-grained pipelining
  // and fine-grained instruction scheduling.
  MainLoopInfo mainloop(forOp);

  // If the mainloop is not schedulable, return an empty schedule.
  if (!mainloop.isSchedulable) return;

  // NVIDIA Ampere Tensor Core multi-staged pipeline requires at least 2 kgroups
  // and 3 software pipeline stages. If the conditions are not met, return an
  // empty schedule.
  int numKgroups = mainloop.getNumberOfKgroups();
  if (numKgroups < 2 || numStages < 3) {
    return;
  }

  // Un-pipelined mainloop should have only one occurance of
  // cp.async.commit_group and cp.async.wait_group. Additionally, two barrier
  // ops are inserted around each staged copy. The barrier op before the copy is
  // un-necessary and will be removed. If the conditions are not met, return an
  // empty schedule.
  if (!(mainloop.asyncCreateGroupOp.size() == 1) ||
      !(mainloop.asyncWaitOps.size() == 1) ||
      !(mainloop.barrierOps.size() == 2)) {
    return;
  }

  // Start pipelining and scheduling the main loop, all kgroups but the last
  // one.
  for (int kgroup = 0; kgroup < numKgroups - 1; kgroup++) {
    // Fine-grained instruction scheduling: interleave Shared Memory loads
    // into and mma.sync operations to hide load latencies.

    // Load the next kgroup into registers.
    for (Operation& op : forOp.getBody()->getOperations()) {
      if (mainloop.warpOperations[kgroup + 1].lhsOperations.count(&op) ||
          mainloop.warpOperations[kgroup + 1].rhsOperations.count(&op)) {
        ops.push_back(std::make_pair(&op, numStages - 1));
      }
    }

    // Issue mma.sync on previous loaded kgroup.
    for (Operation& op : forOp.getBody()->getOperations()) {
      if (mainloop.warpOperations[kgroup].mmaOperations.count(&op))
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
    if (mainloop.copyGlobalToSharedOpDeps.count(&op))
      ops.push_back(std::make_pair(&op, 0 /*pipelineStage*/));
  }
  ops.push_back(
      std::make_pair(mainloop.asyncCreateGroupOp[0], 0 /*pipelineStage*/));

  // Schedule and pipeline all async.wait and barrier
  ops.push_back(std::make_pair(mainloop.asyncWaitOps[0], numStages - 2));
  mainloop.barrierOps[0]->erase();
  ops.push_back(std::make_pair(mainloop.barrierOps[1], numStages - 2));
  //////////////////////////////////////////////////////////////////////////////

  // Coarse-grained instruction pipelining: pipeline Shared Memory loads
  // (SMEM -> Registers) for the first kgroup (kgroup = 0) one stage in
  // advance.

  // Schedule the Shared Memory loads for the first kgroup and pipeline them
  // into one stage ahead.
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (mainloop.warpOperations[0].lhsOperations.count(&op) ||
        mainloop.warpOperations[0].rhsOperations.count(&op))
      ops.push_back(std::make_pair(&op, numStages - 2));
  }

  // Issue mma.sync on for the last kgroup at the end of the mainloop.
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (mainloop.warpOperations[numKgroups - 1].mmaOperations.count(&op))
      ops.push_back(std::make_pair(&op, numStages - 1));
  }

  // Prints the mainloop schedule generated for NVIDIA Ampere through native
  // Tensor Core operations (asyncronous copy, load matrix, and mma.sync).
  debugMainloopSchedule(mainloop, numStages, ops);
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
      return getNvidiaAmpereTensorCorePipeline(forOp, ops, maxDepth);
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
    options.predicateFn = [](RewriterBase& rewriter, Operation* op,
                             Value pred) {
      return replaceOpWithPredicatedOp(rewriter, op, pred);
    };
  }
  scf::ForLoopPipeliningPattern pattern(options, forOp->getContext());
  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  return pipelineForLoop(rewriter, forOp, options);
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
    RewriterBase& rewriter, scf::ForOp forOp,
    PipeliningSchedulingStrategy strategy, bool peelEpilogue, int64_t depth) {
  return applyPipelining(forOp, depth, peelEpilogue, strategy);
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
