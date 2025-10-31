// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/SCF/Transforms/Transforms.h"

#define DEBUG_TYPE "iree-codegen-llvmgpu-prefetch-shared-memory-copy"

namespace mlir::iree_compiler {

namespace {

class LoopPrefetcher {
public:
  /// Creates an instance that plans the given scf.for |op| to be ready for
  /// prefetching. Returns failure if unable to support the given |op|.
  static FailureOr<LoopPrefetcher> get(scf::ForOp op) {
    if (!op.getOps<scf::ForOp>().empty()) {
      LDBG() << "Loop prefetcher does not support nested loops yet";
      return failure();
    }

    LoopPrefetcher prefetcher;
    prefetcher.forOp = op;
    prefetcher.lb = prefetcher.ub = prefetcher.step = 0;

    if (failed(prefetcher.initializeLoopInfo())) {
      LDBG() << "Failed to initialize loop info (unsupported loop)";
      return failure();
    }

    if (failed(prefetcher.initializeStages())) {
      LDBG() << "Failed to initialize stage info (unsupported loop)";
      return failure();
    }

    return prefetcher;
  }

  // Ops in the original scf.for loop that belongs to different classes.
  SmallVector<Operation *> readStage;
  SmallVector<Operation *> writeStage;
  SmallVector<Operation *> computeStage;

private:
  LogicalResult initializeLoopInfo() {
    std::optional<int64_t> lbCst = getConstantIndex(forOp.getLowerBound());
    std::optional<int64_t> ubCst = getConstantIndex(forOp.getUpperBound());
    std::optional<int64_t> stepCst = getConstantIndex(forOp.getStep());
    if (!lbCst || !ubCst || !stepCst)
      return failure();

    lb = *lbCst;
    ub = *ubCst;
    step = *stepCst;

    int64_t numIters = llvm::divideCeil(ub - lb, step);
    if (numIters <= 2)
      return failure();

    return success();
  }

  void getValueDependencies(Operation *op, DenseSet<Operation *> &dependencies,
                            bool noTransferReads = false) {
    if (!op || dependencies.contains(op)) {
      return;
    }

    if (!forOp->isProperAncestor(op)) {
      return;
    }

    dependencies.insert(op);
    op->walk([&](Operation *nested) {
      for (Value val : nested->getOperands()) {
        getValueDependencies(val.getDefiningOp(), dependencies,
                             noTransferReads);
      }
    });
  }

  void getValueDependenciesForIf(scf::IfOp ifOp,
                                 DenseSet<Operation *> &readDependencies,
                                 DenseSet<Operation *> &writeDependencies) {
    // scf.if with results should be supported directly through the usual
    // handling so bail-out here in that case
    if (ifOp->getNumResults() != 0)
      return;
    bool hasGlobalRead = false;
    bool hasSharedWrite = false;
    bool hasPrivateOrSharedWrite = false;
    // Else region not yet supported.
    if (!ifOp.getElseRegion().empty()) {
      return;
    }
    ifOp->walk([&](Operation *op) {
      if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
        auto sourceType = dyn_cast<MemRefType>(readOp.getBase().getType());
        if (hasGlobalMemoryAddressSpace(sourceType)) {
          hasGlobalRead = true;
        }
      }
      if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
        auto dstType = dyn_cast<MemRefType>(writeOp.getBase().getType());
        if (hasSharedMemoryAddressSpace(dstType)) {
          hasSharedWrite = true;
        }
        if (!hasGlobalMemoryAddressSpace(dstType)) {
          hasPrivateOrSharedWrite = true;
        }
      }
    });
    // if op has both read and write stages and hence we cannot do prefetching.
    if (hasGlobalRead && hasSharedWrite) {
      return;
    }
    // Note that the order matters here, if we have a global read and a private
    // write the global read getting assigned to read stage takes precedence.
    // But private write by itself will be assigned write stage. This is becuase
    // private writes are transient values which are typically produced by the
    // read stage and consumed  by the write stage and moving this to write
    // stage makes sure the read stage doesnt get blocked.
    if (hasGlobalRead) {
      getValueDependencies(ifOp, readDependencies);
    } else if (hasPrivateOrSharedWrite) {
      getValueDependencies(ifOp, writeDependencies);
    }
    // Bail-out for unahndled if ops.
  }

  // We only support loops whose bodies can be divided into 3 stages (read,
  // write, compute). If there are any remaining ops with side effects (except
  // for gpu.barrier and certain scf.if ops), the loop is not supported.
  // For scf.if we only support them if we can analyze the region and
  // identify which stage the if op should belong to.
  LogicalResult initializeStages() {
    DenseSet<Operation *> readDependencies;
    DenseSet<Operation *> writeDependencies;
    DenseSet<Operation *> computeDependencies;

    for (Operation &op : forOp.getBody()->getOperations()) {
      if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
        auto srcType = dyn_cast<MemRefType>(read.getBase().getType());
        // only global memory reads should be selected as root ops of read
        // stage. Reads from shared memory are picked by compute stage and reads
        // from private memory can belong to any stage.
        if (!hasGlobalMemoryAddressSpace(srcType)) {
          continue;
        }
        getValueDependencies(read, readDependencies);
      } else if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
        getValueDependencies(write, writeDependencies);
      } else if (auto compute = dyn_cast<scf::YieldOp>(op)) {
        getValueDependencies(compute, computeDependencies);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        getValueDependenciesForIf(ifOp, readDependencies, writeDependencies);
      }
    }
    // If `scf.yeild` is the only compute op then there is no value in doing
    // prefetching.
    if (computeDependencies.size() == 1) {
      LDBG() << "Loop does not have compute so not doing prefetching." << forOp;
      return failure();
    }

    // Restore the original order.
    for (auto &op : forOp.getBody()->getOperations()) {
      bool hasStage = false;
      if (readDependencies.contains(&op)) {
        readStage.push_back(&op);
        hasStage = true;
      }
      // We do not need to duplicate read stage ops in write stage as the
      // iteration count of both stages is always same.
      if (writeDependencies.contains(&op) && !hasStage) {
        writeStage.push_back(&op);
        hasStage = true;
      }
      if (computeDependencies.contains(&op)) {
        computeStage.push_back(&op);
        hasStage = true;
      }

      // Ops with a stage will be cloned over to the new for op, while barriers
      // will be re-written.
      if (!hasStage && !isa<gpu::BarrierOp>(op)) {
        if (!isPure(&op)) {
          LDBG() << "Found a non-pure loop body op not assigned to any stage "
                    "(unsupported loop): "
                 << op;
          return failure();
        }
      }
    }

    LLVM_DEBUG({
      // Stages cannot have overlapping operations.
      llvm::dbgs() << "--- Read Stage ---\n";
      for (Operation *op : readStage)
        llvm::dbgs() << *op << "\n";
      llvm::dbgs() << "--- Write Stage ---\n";
      for (Operation *op : writeStage)
        llvm::dbgs() << *op << "\n";
      llvm::dbgs() << "--- Compute Stage ---\n";
      for (Operation *op : computeStage)
        llvm::dbgs() << *op << "\n";
    });

    return success();
  }

private:
  // The original scf.for loop to prefetch shared memory copy from.
  scf::ForOp forOp;
  // Original static loop range and step.
  int64_t lb, ub, step;
};

} // namespace

// Populates the opToStage map by assigning stages to operations based on
// prefetcher groups and number of stages.
static void
populateOpToStageMap(LoopPrefetcher &prefetcher, scf::ForOp forOp,
                     unsigned numStages,
                     llvm::DenseMap<Operation *, unsigned> &opToStage) {
  auto assignOp = [&](Operation *op, unsigned stage) {
    if (!op || isa<scf::YieldOp>(op))
      return;
    opToStage[op] = stage;
  };

  if (numStages == 1) {
    // Single-stage pipelining: all ops in stage 0.
    for (Operation &op : forOp.getBody()->without_terminator()) {
      assignOp(&op, /*stage=*/0);
    }
  } else if (numStages == 2) {
    // Two-stage pipelining: readStage in stage 0, compute+write in stage 1.
    for (Operation *op : prefetcher.readStage)
      assignOp(op, /*stage=*/0);
    for (Operation *op : prefetcher.writeStage)
      assignOp(op, /*stage=*/0);
    for (Operation *op : prefetcher.computeStage)
      assignOp(op, /*stage=*/1);
  } else {
    for (Operation *op : prefetcher.readStage)
      assignOp(op, /*stage=*/0);
    for (Operation *op : prefetcher.writeStage)
      assignOp(op, /*stage=*/1);
    for (Operation *op : prefetcher.computeStage)
      assignOp(op, /*stage=*/2);
  }

  LDBG() << "Pipelining schedule for loop:\n";
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (opToStage.count(&op)) {
      LDBG() << "  stage " << opToStage[&op] << " : " << op;
    }
  }
}

// Builds the final schedule and creates the PipeliningOption with
// getScheduleFn.
static FailureOr<scf::PipeliningOption>
buildPipeliningOption(LoopPrefetcher &prefetcher,
                      const llvm::DenseMap<Operation *, unsigned> &opToStage) {
  std::vector<std::pair<Operation *, unsigned>> finalSchedule;
  finalSchedule.reserve(opToStage.size());

  // Build the schedule in desired execution order: read -> compute -> write
  for (Operation *op : prefetcher.readStage) {
    if (opToStage.count(op))
      finalSchedule.push_back({op, opToStage.lookup(op)});
  }
  for (Operation *op : prefetcher.computeStage) {
    if (opToStage.count(op))
      finalSchedule.push_back({op, opToStage.lookup(op)});
  }
  for (Operation *op : prefetcher.writeStage) {
    if (opToStage.count(op))
      finalSchedule.push_back({op, opToStage.lookup(op)});
  }

  if (finalSchedule.empty()) {
    return failure();
  }

  scf::PipeliningOption options;
  options.getScheduleFn =
      [finalSchedule](scf::ForOp loop,
                      std::vector<std::pair<Operation *, unsigned>> &outSched) {
        outSched = finalSchedule;
      };

  options.peelEpilogue = true;
  options.supportDynamicLoops = false;
  options.predicateFn = nullptr;

  return options;
}

// Invokes the pipelineForLoop transformation.
static FailureOr<scf::ForOp>
invokePipelineForLoop(scf::ForOp forOp, const scf::PipeliningOption &options) {
  IRRewriter irRewriter(forOp);
  bool modifiedIR = false;
  FailureOr<scf::ForOp> newForOpOrFail =
      scf::pipelineForLoop(irRewriter, forOp, options, &modifiedIR);

  if (failed(newForOpOrFail)) {
    return failure();
  }

  return *newForOpOrFail;
}

// Post-processes the pipelined loop by inserting barriers.
static void postprocessPipelinedLoop(RewriterBase &rewriter,
                                     scf::ForOp newForOp) {
  // Helper to check for shared memory
  auto hasSharedMemory = [](Value val) -> bool {
    auto memrefType = dyn_cast<MemRefType>(val.getType());
    if (!memrefType)
      return false;
    auto addrSpace =
        dyn_cast_or_null<gpu::AddressSpaceAttr>(memrefType.getMemorySpace());
    return addrSpace && addrSpace.getValue() == gpu::AddressSpace::Workgroup;
  };

  // Helper to check if operation or its nested ops have shared memory reads
  auto hasNestedSharedRead = [hasSharedMemory](Operation *op) -> bool {
    bool found = false;
    op->walk([&](vector::TransferReadOp readOp) {
      if (hasSharedMemory(readOp.getBase())) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return found;
  };

  // Helper to check if operation or its nested ops have shared memory writes
  auto hasNestedSharedWrite = [hasSharedMemory](Operation *op) -> bool {
    bool found = false;
    op->walk([&](vector::TransferWriteOp writeOp) {
      if (hasSharedMemory(writeOp.getBase())) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return found;
  };

  // Insert barriers in the kernel loop
  rewriter.setInsertionPointToStart(newForOp.getBody());
  bool insertedComputeBarrier = false;
  bool insertedWriteBarrier = false;

  for (Operation &op :
       llvm::make_early_inc_range(newForOp.getBody()->without_terminator())) {
    // Insert barrier before first shared memory read (compute stage)
    if (!insertedComputeBarrier && hasNestedSharedRead(&op)) {
      rewriter.setInsertionPoint(&op);
      gpu::BarrierOp::create(rewriter, newForOp.getLoc());
      insertedComputeBarrier = true;
    }
    // Insert barriers before first shared memory write (write stage)
    if (!insertedWriteBarrier && hasNestedSharedWrite(&op)) {
      rewriter.setInsertionPoint(&op);
      gpu::BarrierOp::create(rewriter, newForOp.getLoc());
      amdgpu::SchedBarrierOp::create(
          rewriter, newForOp.getLoc(),
          amdgpu::sched_barrier_opt_enumAttr::get(
              rewriter.getContext(), amdgpu::sched_barrier_opt_enum::none));
      insertedWriteBarrier = true;
    }
  }

  // Insert barrier before epilogue (after the new loop)
  rewriter.setInsertionPointAfter(newForOp);
  gpu::BarrierOp::create(rewriter, newForOp.getLoc());
}

FailureOr<scf::ForOp> prefetchSharedMemoryCopy(RewriterBase &rewriter,
                                               scf::ForOp forOp,
                                               unsigned numStages) {
  auto prefetcherOr = LoopPrefetcher::get(forOp);
  if (failed(prefetcherOr))
    return failure();
  LoopPrefetcher &prefetcher = *prefetcherOr;

  llvm::DenseMap<Operation *, unsigned> opToStage;
  populateOpToStageMap(prefetcher, forOp, numStages, opToStage);

  FailureOr<scf::PipeliningOption> optionsOr =
      buildPipeliningOption(prefetcher, opToStage);
  if (failed(optionsOr))
    return failure();

  FailureOr<scf::ForOp> newForOpOr = invokePipelineForLoop(forOp, *optionsOr);
  if (failed(newForOpOr))
    return failure();

  scf::ForOp newForOp = *newForOpOr;
  postprocessPipelinedLoop(rewriter, newForOp);

  return newForOp;
}
} // namespace mlir::iree_compiler
