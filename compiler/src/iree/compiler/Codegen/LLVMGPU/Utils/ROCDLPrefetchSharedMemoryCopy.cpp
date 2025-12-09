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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Analysis/SliceAnalysis.h"
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

#define DEBUG_TYPE "iree-codegen-llvmgpu-prefetch-shared-memory-copy"

namespace mlir::iree_compiler {

namespace {
// Structure to hold the stage classification result.
struct StageClassification {
  SmallVector<Operation *> readStage;
  SmallVector<Operation *> writeStage;
  SmallVector<Operation *> computeStage;
};
} // namespace

// Helper function to check if a transfer_read is from global memory.
static bool isGlobalMemoryRead(vector::TransferReadOp read) {
  auto srcType = dyn_cast<MemRefType>(read.getBase().getType());
  return hasGlobalMemoryAddressSpace(srcType);
}

// Helper function to check if a transfer_write is to shared memory.
static bool isSharedMemoryWrite(vector::TransferWriteOp write) {
  auto dstType = dyn_cast<MemRefType>(write.getBase().getType());
  return hasSharedMemoryAddressSpace(dstType);
}

// Analyzes the contents of an scf.if operation in a single pass.
// Identifies what types of operations exist, adds the appropriate root
// operation, and validates that the scf.if doesn't have conflicting operations.
// Returns failure if the scf.if has both global reads and shared writes.
static LogicalResult analyzeIfOp(scf::IfOp ifOp,
                                 SmallVector<Operation *> &readRoots,
                                 SmallVector<Operation *> &writeRoots) {
  bool hasGlobalRead = false;
  bool hasSharedWrite = false;
  bool hasPrivateOrSharedWrite = false;

  // Walk once to collect flags and candidate operations
  ifOp->walk([&](Operation *nestedOp) {
    if (auto read = dyn_cast<vector::TransferReadOp>(nestedOp)) {
      if (!hasGlobalRead && isGlobalMemoryRead(read)) {
        hasGlobalRead = true;
        readRoots.push_back(nestedOp);
        LDBG() << "  Found global read in scf.if: " << *nestedOp;
      }
    } else if (auto write = dyn_cast<vector::TransferWriteOp>(nestedOp)) {
      if (!hasSharedWrite && isSharedMemoryWrite(write)) {
        hasSharedWrite = true;
      }
      auto dstType = dyn_cast<MemRefType>(write.getBase().getType());
      if (!hasGlobalMemoryAddressSpace(dstType) && !hasPrivateOrSharedWrite) {
        hasPrivateOrSharedWrite = true;
        writeRoots.push_back(nestedOp);
        LDBG() << "  Found private/shared write in scf.if: " << *nestedOp;
      }
    }
  });

  // Validate: Cannot pipeline if scf.if has both global read and shared write
  if (hasGlobalRead && hasSharedWrite) {
    LDBG() << "  ERROR: scf.if has both global read and shared write - "
           << "unpipelineable: " << ifOp;
    return failure();
  }

  return success();
}

// Checks if a loop has sufficient iterations for prefetching.
static LogicalResult checkLoopIterations(scf::ForOp forOp) {
  std::optional<int64_t> lbCst = getConstantIndex(forOp.getLowerBound());
  std::optional<int64_t> ubCst = getConstantIndex(forOp.getUpperBound());
  std::optional<int64_t> stepCst = getConstantIndex(forOp.getStep());

  if (!lbCst || !ubCst || !stepCst) {
    LDBG() << "Loop bounds are not constant";
    return failure();
  }

  int64_t numIters = llvm::divideCeil(*ubCst - *lbCst, *stepCst);
  if (numIters <= 2) {
    LDBG() << "Loop has too few iterations: " << numIters;
    return failure();
  }

  return success();
}

// Step 1: Identify root operations for each stage.
// Root operations are the anchors from which we compute backward slices.
// This function looks inside scf.if blocks to find roots, so that backward
// slice computation works naturally without special handling.
// Returns failure if any scf.if has conflicting operations (both global reads
// and shared writes).
static LogicalResult
identifyRootOperations(scf::ForOp forOp, SmallVector<Operation *> &readRoots,
                       SmallVector<Operation *> &writeRoots,
                       SmallVector<Operation *> &computeRoots) {

  LDBG() << "\n=== Step 1: Identifying Root Operations ===";

  for (Operation &op : forOp.getBody()->getOperations()) {
    // Read stage roots: vector.transfer_read from global memory
    if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
      if (isGlobalMemoryRead(read)) {
        readRoots.push_back(&op);
        LDBG() << "  Read root: " << op;
      }
    }
    // Write stage roots: all vector.transfer_write operations
    else if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
      writeRoots.push_back(&op);
      LDBG() << "  Write root: " << op;
    }
    // Compute stage roots: scf.yield (carries loop-carried dependencies)
    else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      computeRoots.push_back(&op);
      LDBG() << "  Compute root: " << op;
    }
    // Look inside scf.if blocks to find nested transfer operations
    // Treat the scf.if as a single unit - add it to only one stage
    else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // Analyze the scf.if contents and add roots in a single pass
      // The scf.if itself will be added to slices via parent walking
      if (failed(analyzeIfOp(ifOp, readRoots, writeRoots))) {
        return failure();
      }
    }
  }

  LDBG() << "Found " << readRoots.size() << " read roots, " << writeRoots.size()
         << " write roots, " << computeRoots.size() << " compute roots";

  // Validate that we have at least one read root - pipelining requires
  // global memory reads to prefetch
  if (readRoots.empty()) {
    LDBG() << "No global memory reads found - cannot pipeline";
    return failure();
  }

  return success();
}

// Step 2: Compute backward slice for a set of root operations.
// Returns failure if unsupported nested operations are encountered.
static LogicalResult computeBackwardSlice(ArrayRef<Operation *> roots,
                                          scf::ForOp forOp,
                                          SetVector<Operation *> &slice) {

  BackwardSliceOptions options;
  // Only include operations within the loop body
  options.filter = [&](Operation *op) { return forOp->isProperAncestor(op); };

  for (Operation *root : roots) {
    SetVector<Operation *> rootSlice;
    (void)getBackwardSlice(root, &rootSlice, options);
    // getBackwardSlice doesn't include the root itself, so add it explicitly
    slice.insert(root);
    slice.insert_range(rootSlice);

    // Also add any parent scf.if operations that contain this root
    // This is necessary because roots inside scf.if need the if to be scheduled
    // We also need to compute backward slices of ALL operations inside the
    // scf.if to capture dependencies like memref.alloca that nested ops use
    Operation *parent = root->getParentOp();
    while (parent != forOp.getOperation()) {
      if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
        slice.insert(ifOp.getOperation());

        // Compute backward slice OF the scf.if itself to capture its condition
        SetVector<Operation *> ifSlice;
        (void)getBackwardSlice(ifOp.getOperation(), &ifSlice, options);
        slice.insert_range(ifSlice);

        // Compute backward slices of all nested operations to get dependencies
        // This approach ensures correctness by capturing all transitive
        // dependencies like memref.alloca that nested operations may use.
        ifOp.getOperation()->walk([&](Operation *nestedOp) {
          if (nestedOp != ifOp.getOperation()) {
            SetVector<Operation *> nestedSlice;
            (void)getBackwardSlice(nestedOp, &nestedSlice, options);
            slice.insert_range(nestedSlice);
          }
        });
      } else if (parent->getNumRegions() > 0) {
        // If we encounter a region-bearing op that's not scf.if, fail
        LDBG() << "  ERROR: Unsupported nested region-bearing operation: "
               << parent->getName();
        return failure();
      }
      // else: Non-region-bearing intermediate operation, continue walking up
      parent = parent->getParentOp();
    }
  }

  return success();
}

// Step 3: Classify all operations into stages and restore original order.
static LogicalResult classifyOperationsIntoStages(
    scf::ForOp forOp, const SetVector<Operation *> &readSlice,
    const SetVector<Operation *> &writeSlice,
    const SetVector<Operation *> &computeSlice, StageClassification &result) {

  LDBG() << "\n=== Step 3: Classifying Operations into Stages ===";
  LDBG() << "  Read slice size: " << readSlice.size();
  LDBG() << "  Write slice size: " << writeSlice.size();
  LDBG() << "  Compute slice size: " << computeSlice.size();

  // If compute slice only has the yield, there's no real compute
  if (computeSlice.size() == 1) {
    LDBG() << "Loop has no meaningful compute operations";
    return failure();
  }

  // Restore original order while assigning to stages
  for (Operation &op : forOp.getBody()->getOperations()) {
    bool assigned = false;

    if (readSlice.contains(&op)) {
      result.readStage.push_back(&op);
      assigned = true;
      LDBG() << "  READ: " << op;
    }

    // Don't duplicate ops that are in read stage
    if (writeSlice.contains(&op) && !assigned) {
      result.writeStage.push_back(&op);
      assigned = true;
      LDBG() << "  WRITE: " << op;
    }

    // Don't duplicate ops already assigned to read or write stages. The compute
    // slice (from scf.yield) naturally includes all operations, but we want to
    // classify them based on their primary purpose
    if (computeSlice.contains(&op) && !assigned) {
      result.computeStage.push_back(&op);
      assigned = true;
      LDBG() << "  COMPUTE: " << op;
    }

    // Check for unassigned operations with side effects
    if (!assigned && !isPure(&op) && !isa<gpu::BarrierOp>(op)) {
      LDBG() << "  ERROR: Unassigned op with side effects: " << op;
      return failure();
    }
  }

  LDBG() << "\n=== Final Stage Classification ===";
  LDBG() << "--- Read Stage (" << result.readStage.size() << " ops) ---";
  for (Operation *op : result.readStage)
    LDBG() << *op;
  LDBG() << "--- Write Stage (" << result.writeStage.size() << " ops) ---";
  for (Operation *op : result.writeStage)
    LDBG() << *op;
  LDBG() << "--- Compute Stage (" << result.computeStage.size() << " ops) ---";
  for (Operation *op : result.computeStage)
    LDBG() << *op;

  return success();
}

// Main function to compute stage classification for a loop.
static FailureOr<StageClassification>
computeStageClassification(scf::ForOp forOp) {
  LDBG() << "\n=== Computing Stage Classification for Loop ===";

  // Check for nested loops
  if (!forOp.getOps<scf::ForOp>().empty()) {
    LDBG() << "Nested loops not supported";
    return failure();
  }

  // Check loop has sufficient iterations
  if (failed(checkLoopIterations(forOp))) {
    return failure();
  }

  // Step 1: Identify root operations (with inline validation)
  SmallVector<Operation *> readRoots, writeRoots, computeRoots;
  if (failed(
          identifyRootOperations(forOp, readRoots, writeRoots, computeRoots))) {
    return failure();
  }

  // Step 2: Compute backward slices
  LDBG() << "\n=== Step 2: Computing Backward Slices ===";
  SetVector<Operation *> readSlice, writeSlice, computeSlice;

  if (failed(computeBackwardSlice(readRoots, forOp, readSlice))) {
    return failure();
  }
  LDBG() << "  Read slice: " << readSlice.size() << " operations";

  if (failed(computeBackwardSlice(writeRoots, forOp, writeSlice))) {
    return failure();
  }
  LDBG() << "  Write slice: " << writeSlice.size() << " operations";

  if (failed(computeBackwardSlice(computeRoots, forOp, computeSlice))) {
    return failure();
  }
  LDBG() << "  Compute slice: " << computeSlice.size() << " operations";

  // Step 3: Classify operations into stages
  StageClassification result;
  if (failed(classifyOperationsIntoStages(forOp, readSlice, writeSlice,
                                          computeSlice, result))) {
    return failure();
  }

  return result;
}

// Removes all barrier operations from the loop body.
// Original barriers are designed for the unpipelined loop structure and will
// be replaced with barriers appropriate for the pipelined structure.
static void removeBarriers(scf::ForOp forOp) {
  SmallVector<Operation *> toErase;
  // Only remove barriers from the direct loop body, not from nested regions.
  // Nested loops might not be pipelined and should keep their barriers.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<gpu::BarrierOp>(op)) {
      toErase.push_back(&op);
    }
  }
  for (Operation *op : toErase) {
    op->erase();
  }
}

// Populates the opToStage map by assigning stages to operations based on
// stage classification and number of stages.
static void
populateOpToStageMap(const StageClassification &stages, scf::ForOp forOp,
                     unsigned numStages,
                     llvm::DenseMap<Operation *, unsigned> &opToStage) {
  auto assignOp = [&](Operation *op, unsigned stage) {
    if (!op || isa<scf::YieldOp>(op))
      return;
    opToStage[op] = stage;
  };

  if (numStages == 2) {
    // Two-stage pipelining: read+write in stage 0, compute in stage 1.
    for (Operation *op : stages.readStage)
      assignOp(op, /*stage=*/0);
    for (Operation *op : stages.writeStage)
      assignOp(op, /*stage=*/0);
    for (Operation *op : stages.computeStage)
      assignOp(op, /*stage=*/1);
  } else {
    // Three-stage pipelining: read in stage 0, write in stage 1, compute in
    // stage 2.
    for (Operation *op : stages.readStage)
      assignOp(op, /*stage=*/0);
    for (Operation *op : stages.writeStage)
      assignOp(op, /*stage=*/1);
    for (Operation *op : stages.computeStage)
      assignOp(op, /*stage=*/2);
  }
}

// Populates cluster IDs for each operation based on stage groupings.
// Cluster ordering determines execution order within each iteration:
// - 2-stage pipeline: read -> compute -> write
//   (read i+1, compute i, write i+1)
// - 3-stage pipeline: compute -> write -> read
//   (compute i, write i+1, read i+2)
static void
populateOpToClusterMap(const StageClassification &stages, unsigned numStages,
                       llvm::DenseMap<Operation *, unsigned> &opToCluster) {
  unsigned clusterID = 0;

  if (numStages == 2) {
    // 2-stage pipeline: read first, then compute, then write
    // This allows reading for next iteration while computing current
    for (Operation *op : stages.readStage) {
      opToCluster[op] = clusterID;
    }
    ++clusterID;

    for (Operation *op : stages.computeStage) {
      opToCluster[op] = clusterID;
    }
    ++clusterID;

    for (Operation *op : stages.writeStage) {
      opToCluster[op] = clusterID;
    }
    ++clusterID;
  } else {
    // 3-stage pipeline: compute first, then write, then read
    // This maximizes distance between read and use
    for (Operation *op : stages.computeStage) {
      opToCluster[op] = clusterID;
    }
    ++clusterID;

    for (Operation *op : stages.writeStage) {
      opToCluster[op] = clusterID;
    }
    ++clusterID;

    for (Operation *op : stages.readStage) {
      opToCluster[op] = clusterID;
    }
    ++clusterID;
  }

  LDBG() << "Built opToCluster map with " << clusterID << " clusters "
         << "(numStages=" << numStages << ")";
}

// Builds the final schedule using opToCluster and opToStage mappings.
// Sorts operations by cluster ID, maintaining original order within each
// cluster.
static void buildFinalSchedule(
    const StageClassification &stages,
    const llvm::DenseMap<Operation *, unsigned> &opToStage,
    const llvm::DenseMap<Operation *, unsigned> &opToCluster,
    std::vector<std::pair<Operation *, unsigned>> &finalSchedule) {

  // Collect all operations from all stages with their cluster IDs
  SmallVector<Operation *> allOps;
  allOps.append(stages.readStage.begin(), stages.readStage.end());
  allOps.append(stages.computeStage.begin(), stages.computeStage.end());
  allOps.append(stages.writeStage.begin(), stages.writeStage.end());

  // Sort by cluster ID, maintaining original order within each cluster
  llvm::stable_sort(allOps, [&](Operation *a, Operation *b) {
    unsigned clusterA = opToCluster.lookup(a);
    unsigned clusterB = opToCluster.lookup(b);
    return clusterA < clusterB;
  });

  // Build the final schedule from the sorted operations
  for (Operation *op : allOps) {
    if (opToStage.count(op)) {
      unsigned stage = opToStage.lookup(op);
      finalSchedule.push_back({op, stage});
    }
  }

  LDBG() << "Built final schedule with " << finalSchedule.size()
         << " operations";
}

// Builds the PipeliningOption with getScheduleFn.
static scf::PipeliningOption buildPipeliningOption(
    const std::vector<std::pair<Operation *, unsigned>> &finalSchedule) {
  scf::PipeliningOption options;
  options.getScheduleFn =
      [finalSchedule](scf::ForOp loop,
                      std::vector<std::pair<Operation *, unsigned>> &outSched) {
        outSched = finalSchedule;
      };

  options.peelEpilogue = true;
  // TODO: Enable dynamic trip count support. We'll want to run in-range
  // analysis to restrict to cases where we know K is large enough to benefit
  // from pipelining.
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

// Helper to check for shared memory.
static bool hasSharedMemory(Value val) {
  auto memrefType = dyn_cast<MemRefType>(val.getType());
  if (!memrefType)
    return false;
  auto addrSpace =
      dyn_cast_if_present<gpu::AddressSpaceAttr>(memrefType.getMemorySpace());
  return addrSpace && addrSpace.getValue() == gpu::AddressSpace::Workgroup;
}

// Helper to check if operation or its nested ops have shared memory reads.
static bool hasNestedSharedRead(Operation *op) {
  bool found = false;
  op->walk([&](vector::TransferReadOp readOp) {
    if (hasSharedMemory(readOp.getBase())) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

// Helper to check if operation or its nested ops have shared memory writes.
static bool hasNestedSharedWrite(Operation *op) {
  bool found = false;
  op->walk([&](vector::TransferWriteOp writeOp) {
    if (hasSharedMemory(writeOp.getBase())) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

struct SharedBarrierState {
  bool needBarrierBeforeWrite = false;
  bool needBarrierBeforeRead = false;
};

// Inserts synchronization barriers before shared memory accesses in the given
// range using a running `SharedBarrierState`. Conceptually, we track whether
// the next shared read (or write) must be preceded by a barrier, and only emit
// one when that flag is set. For example, if the iteration sequence observes a
// shared read (R) followed by another read, nothing is inserted; the state only
// toggles `needBarrierBeforeWrite`, so the next shared write (W) will emit a
// barrier before it. Conversely, seeing a write first toggles
// `needBarrierBeforeRead`, so the following read receives the barrier. This
// keeps the minimum number of synchronizations while still enforcing the R↔W
// ordering required by the pipelined schedule.
static SharedBarrierState insertBarriersInRange(RewriterBase &rewriter,
                                                Location loc,
                                                Block::iterator begin,
                                                Block::iterator end,
                                                SharedBarrierState state) {
  for (auto it = begin; it != end; ++it) {
    Operation &op = *it;
    bool hasSharedRead = hasNestedSharedRead(&op);
    bool hasSharedWrite = hasNestedSharedWrite(&op);

    if (hasSharedRead && state.needBarrierBeforeRead) {
      rewriter.setInsertionPoint(&op);
      gpu::BarrierOp::create(rewriter, loc);
      state.needBarrierBeforeRead = false;
    }

    if (hasSharedWrite && state.needBarrierBeforeWrite) {
      rewriter.setInsertionPoint(&op);
      gpu::BarrierOp::create(rewriter, loc);
      amdgpu::SchedBarrierOp::create(
          rewriter, loc,
          amdgpu::sched_barrier_opt_enumAttr::get(
              rewriter.getContext(), amdgpu::sched_barrier_opt_enum::none));
      state.needBarrierBeforeWrite = false;
    }

    if (hasSharedRead)
      state.needBarrierBeforeWrite = true;
    if (hasSharedWrite)
      state.needBarrierBeforeRead = true;
  }

  return state;
}

// Check if the operation is inside a loop (scf.for, scf.while, etc.)
static bool isInsideLoop(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent) {
    if (isa<scf::ForOp, scf::WhileOp, scf::ParallelOp>(parent)) {
      return true;
    }
    parent = parent->getParentOp();
  }
  return false;
}

// Inserts synchronization barriers in the pipelined loop.
static void insertPipelineBarriers(RewriterBase &rewriter,
                                   scf::ForOp newForOp) {
  Block *parentBlock = newForOp->getBlock();
  Location loc = newForOp.getLoc();
  SharedBarrierState state;

  // Check if the pipelined loop is nested inside another loop.
  // If nested, we need prologue barriers because:
  //   - The epilogue of outer iteration N writes to shared memory
  //   - The prologue of outer iteration N+1 reads from shared memory
  //   - Without a barrier, there's a data race between iterations
  bool isNested = isInsideLoop(newForOp);

  if (isNested) {
    // Nested loop: insert barriers in prologue for correctness.
    // Start with needBarrierBeforeWrite=true because the epilogue of the
    // previous outer iteration may have read from shared memory.
    state.needBarrierBeforeWrite = true;
    state = insertBarriersInRange(rewriter, loc, parentBlock->begin(),
                                  newForOp->getIterator(), state);
  } else {
    // Non-nested loop: skip prologue barriers for performance.
    // The prologue contains global->shared memory copies that don't require
    // barriers between them since there are no shared memory reads in prologue.
    // We only need to set up the state for the loop body: since the prologue
    // writes to shared memory, subsequent shared reads in the loop body will
    // need a barrier.
    state.needBarrierBeforeRead = true;
  }

  // Loop body (exclude terminator).
  Block *body = newForOp.getBody();
  state = insertBarriersInRange(rewriter, loc, body->begin(),
                                std::prev(body->end()), state);

  // Epilogue (operations after the loop).
  Block::iterator epilogueStart = std::next(newForOp->getIterator());
  insertBarriersInRange(rewriter, loc, epilogueStart, parentBlock->end(),
                        state);
}

// Dumps the planned schedule before pipelining for debugging purposes.
static void
dumpSchedule(const std::vector<std::pair<Operation *, unsigned>> &finalSchedule,
             const llvm::DenseMap<Operation *, unsigned> &opToCluster) {
  LDBG() << "\n=== Planned Pipeline Schedule (before pipelining) ===";
  LDBG() << "Total operations in schedule: " << finalSchedule.size();

  for (size_t i = 0; i < finalSchedule.size(); ++i) {
    Operation *op = finalSchedule[i].first;
    unsigned stage = finalSchedule[i].second;
    unsigned cluster = opToCluster.lookup(op);

    LDBG() << "  [" << i << "] cluster=" << cluster << " stage=" << stage
           << " op: " << *op;
  }

  LDBG() << "=== End Planned Schedule ===\n";
}

FailureOr<scf::ForOp> prefetchSharedMemoryCopy(RewriterBase &rewriter,
                                               scf::ForOp forOp,
                                               unsigned numStages) {
  // No prefetching needed for single-stage pipelining.
  if (numStages <= 1) {
    return forOp;
  }

  // For global->shared->register data flow, we have 3 operation groups (read,
  // write, compute), so 3 stages is the maximum meaningful pipeline depth.
  if (numStages > 3) {
    LDBG() << "numStages=" << numStages
           << " requested but capping to 3 (maximum for read, write, compute)";
    numStages = 3;
  }

  // Compute stage classification using the new refactored approach
  auto stagesOr = computeStageClassification(forOp);
  if (failed(stagesOr)) {
    return failure();
  }
  const StageClassification &stages = *stagesOr;

  // For multi-stage pipelining, remove original barriers as they're designed
  // for unpipelined structure. New barriers will be inserted appropriately.
  removeBarriers(forOp);

  llvm::DenseMap<Operation *, unsigned> opToStage;
  populateOpToStageMap(stages, forOp, numStages, opToStage);

  // Step 1: Populate standalone opToCluster map
  llvm::DenseMap<Operation *, unsigned> opToCluster;
  populateOpToClusterMap(stages, numStages, opToCluster);

  // Step 2: Use opToCluster + opToStage to build the final schedule
  std::vector<std::pair<Operation *, unsigned>> finalSchedule;
  finalSchedule.reserve(opToStage.size());
  buildFinalSchedule(stages, opToStage, opToCluster, finalSchedule);

  if (finalSchedule.empty()) {
    return failure();
  }

  // Dump the planned schedule before pipelining (while ops are still valid)
  LLVM_DEBUG(dumpSchedule(finalSchedule, opToCluster));

  scf::PipeliningOption options = buildPipeliningOption(finalSchedule);

  FailureOr<scf::ForOp> newForOpOr = invokePipelineForLoop(forOp, options);
  if (failed(newForOpOr)) {
    return failure();
  }

  scf::ForOp newForOp = *newForOpOr;

  // Insert barriers in the pipelined loop
  insertPipelineBarriers(rewriter, newForOp);

  return newForOp;
}
} // namespace mlir::iree_compiler
