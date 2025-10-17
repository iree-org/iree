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
    prefetcher.mapping = SmallVector<IRMapping>(4);
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

  // Emits the prologue before the main pipelined loop.
  void emitPrologue(RewriterBase &rewriter) {
    Location loc = forOp.getLoc();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, lb);
    // Directly write in the prologue and use the shared memory to communicate
    // data instead of the loop carried values. Read (0)
    emitRead(mapping[0], rewriter, zero);
    // Write(0)
    emitWrite(mapping[0], rewriter, zero);
  }

  /// Emits the main pipelined loop structure.
  scf::ForOp createKernelLoop(RewriterBase &rewriter) {
    Location loc = forOp.getLoc();
    int64_t newUpperBound = ub - step;
    auto newUb = arith::ConstantIndexOp::create(rewriter, loc, newUpperBound);

    // Keep original iter args and then add some for what's being loaded to
    // registers.
    auto iterArgs = llvm::to_vector_of<Value>(forOp.getInitArgs());
    auto newForOp = scf::ForOp::create(rewriter, loc, forOp.getLowerBound(),
                                       newUb, forOp.getStep(), iterArgs);

    // When there are no iter args, the loop body terminator will be created.
    // Since we always create it below, remove the terminator if it was created.
    if (!newForOp.getBody()->empty())
      rewriter.eraseOp(newForOp.getBody()->getTerminator());

    return newForOp;
  }

  /// Emits contents into the main pipelined loop structure.
  void createKernel(RewriterBase &rewriter, scf::ForOp newForOp) {
    rewriter.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
    Location loc = forOp.getLoc();
    Value indVar = newForOp.getInductionVar();
    Value increment = arith::ConstantIndexOp::create(rewriter, loc, step);
    Value iPlusOne = arith::AddIOp::create(rewriter, loc, indVar, increment,
                                           arith::IntegerOverflowFlags::nsw);

    for (int i = 0; i < 3; ++i) {
      for (auto [idx, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
        mapping[i].map(arg, newForOp.getRegionIterArgs()[idx]);
      }
    }

    emitRead(mapping[1], rewriter, iPlusOne);
    emitBarrier(loc, rewriter);
    emitCompute(mapping[0], rewriter, indVar);
    emitBarrier(loc, rewriter);
    emitSchedBarrier(loc, rewriter);
    emitWrite(mapping[1], rewriter, iPlusOne);
    updateYield(mapping[0], rewriter);
    return;
  }

  // Emits the epilogue after the main pipelined loop and returns the final
  // results to replace the original loop results main loop.
  SmallVector<Value> emitEpilogue(RewriterBase &rewriter, scf::ForOp newForOp) {
    rewriter.setInsertionPointAfter(newForOp);
    Location loc = forOp.getLoc();
    Value nMinusOne =
        arith::ConstantIndexOp::create(rewriter, loc, ub - 1 * step);

    // Map iter_args to results of newForOp.
    for (unsigned i = 0, e = forOp->getNumResults(); i != e; ++i) {
      mapping[0].map(forOp.getRegionIterArg(i), newForOp.getResult(i));
    }

    emitBarrier(loc, rewriter);
    return emitCompute(mapping[0], rewriter, nMinusOne);
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

  /// Creates all read stage ops for a loop iteration with |rewriter| and maps
  /// the original loop induction variable to |iv| in |mapping|.
  void emitRead(IRMapping &mapping, RewriterBase &rewriter, Value iv) {
    // Map the original loop induction variable to |iv| for later op rewrites.
    mapping.map(forOp.getInductionVar(), iv);

    for (Operation *op : readStage) {
      rewriter.clone(*op, mapping);
    }
  }

  /// Creates all write stage ops for a loop iteration with |rewriter| and maps
  /// the original loop induction variable to |iv| in |mapping|.
  void emitWrite(IRMapping &mapping, RewriterBase &rewriter, Value iv) {
    // Map the original loop induction variable to |iv| for later op rewrites.
    mapping.map(forOp.getInductionVar(), iv);

    for (Operation *op : writeStage) {
      rewriter.clone(*op, mapping);
    }
  }

  /// Creates a gpu.barrier op with |rewriter|.
  void emitBarrier(Location loc, RewriterBase &rewriter) {
    gpu::BarrierOp::create(rewriter, loc);
  }

  void emitSchedBarrier(Location loc, RewriterBase &rewriter) {
    amdgpu::SchedBarrierOp::create(
        rewriter, loc,
        amdgpu::sched_barrier_opt_enumAttr::get(
            rewriter.getContext(), amdgpu::sched_barrier_opt_enum::none));
  }

  /// Creates all compute stage ops for a loop iteration with |rewriter| and
  /// maps the original loop induction variable to |iv| in |mapping|.
  SmallVector<Value> emitCompute(IRMapping &mapping, RewriterBase &rewriter,
                                 Value indVar) {
    // Map the original loop induction variable to |iv| for later op rewrites.
    mapping.map(forOp.getInductionVar(), indVar);

    SmallVector<Value> results;
    for (Operation *op : computeStage) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        // On yield, return the operands of the yield converted.
        results =
            llvm::map_to_vector<4>(yieldOp.getOperands(), [&](Value operand) {
              return mapping.lookup(operand);
            });
        break;
      }

      rewriter.clone(*op, mapping);
    }

    return results;
  }

  void updateYield(IRMapping &mapping, RewriterBase &rewriter) {
    for (Operation *op : computeStage) {
      if (auto yield = dyn_cast<scf::YieldOp>(op)) {
        rewriter.clone(*op, mapping);
        break;
      }
    }
  }

private:
  // Mapping for each pipelined stage.
  SmallVector<IRMapping, 4> mapping;
  // The original scf.for loop to prefetch shared memory copy from.
  scf::ForOp forOp;
  // Original static loop range and step.
  int64_t lb, ub, step;
};

} // namespace

// FailureOr<scf::ForOp> prefetchSharedMemoryCopy(RewriterBase &rewriter,
//                                                scf::ForOp forOp) {
//   rewriter.setInsertionPoint(forOp);
//
//   auto prefetcherOr = LoopPrefetcher::get(forOp);
//   if (failed(prefetcherOr))
//     return failure();
//   LoopPrefetcher &prefetcher = *prefetcherOr;
//
//   prefetcher.emitPrologue(rewriter);
//   scf::ForOp newForOp = prefetcher.createKernelLoop(rewriter);
//   prefetcher.createKernel(rewriter, newForOp);
//
//   SmallVector<Value> results = prefetcher.emitEpilogue(rewriter, newForOp);
//   rewriter.replaceOp(forOp, results);
//   return newForOp;
// }

// Insert/replace in the same file where the LoopPrefetcher and IREE pass live.
// Assumes the same includes as in your file (gpu, amdgpu, scf, arith, etc).
// Also assumes Triton's schedule attribute names are available:
//   mlir::triton::kLoopStageAttrName
//   mlir::triton::kLoopClusterAttrName
//
// This function replaces the old pattern-based prologue/kernel/epilogue
// emission with a call to triton::pipelineForLoop while preserving the
// loop-analysis and op grouping from LoopPrefetcher.

FailureOr<scf::ForOp> prefetchSharedMemoryCopy(RewriterBase &rewriter,
                                               scf::ForOp forOp,
                                               unsigned numStages) {
  // 1) Analyze and classify ops into stages using the existing helper.
  auto prefetcherOr = LoopPrefetcher::get(forOp);
  if (failed(prefetcherOr))
    return failure();
  LoopPrefetcher &prefetcher = *prefetcherOr;

  // Build maps from original op -> stage
  llvm::DenseMap<Operation *, unsigned> opToStage;
  auto assignOp = [&](Operation *op, unsigned stage) {
    if (!op)
      return;
    opToStage[op] = stage;
  };

  // Assign stages based on prefetcher groups.
  // Important: build the schedule grouped by stage (all readStage then
  // computeStage then writeStage). This guides pipelineForLoop to emit the
  // prologue/kernel/epilogue in the intended shape.
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

  // Convenience references
  auto &computeOps = prefetcher.computeStage;
  auto &writeOps = prefetcher.writeStage;
  SmallVector<Operation *> syntheticBarriers;
  if (!computeOps.empty()) {
    // barrierBeforeCompute: place right before the first compute op in the
    // original body.
    Operation *firstCompute = computeOps.front();
    {
      OpBuilder b(forOp.getContext());
      auto loc = firstCompute->getLoc();
      b.setInsertionPoint(firstCompute);
      auto barrierBefore = gpu::BarrierOp::create(b, loc);
      syntheticBarriers.push_back(barrierBefore);
      // assign this barrier to the same stage as the write-stage or whichever
      // you prefer since you asked: "for 2-stage kernel put barrier before and
      // after compute" choose stage = (numStages == 2 ? 1 :  (numStages >=3 ? 1
      // : 0))
      unsigned stage = (numStages == 1 ? 0 : (numStages == 2 ? 1 : 2));
      opToStage[barrierBefore] = stage;
    }

    // barrierAfterCompute: place right after last compute op
    // Operation *lastCompute = computeOps.back();
    //{
    //  OpBuilder b(forOp.getContext());
    //  if (lastCompute != lastCompute->getBlock()->getTerminator()){
    //    b.setInsertionPointAfter(lastCompute);
    //  } else {
    //    // This means last compute is the terminator itself (scf.yield).
    //    b.setInsertionPoint(lastCompute);
    //  }
    //
    //  auto loc = lastCompute->getLoc();
    //  auto barrierAfter =gpu::BarrierOp::create(b, loc);
    //  syntheticBarriers.push_back(barrierAfter);
    //  unsigned stage= (numStages == 1 ? 0 : (numStages == 2 ? 1 : 2));
    //  opToStage[barrierAfter] = stage;
    //}
    Operation *firstWrite = writeOps.front();
    {
      OpBuilder b(forOp.getContext());
      auto loc = firstWrite->getLoc();
      b.setInsertionPoint(firstWrite);
      auto barrierBefore = gpu::BarrierOp::create(b, loc);
      syntheticBarriers.push_back(barrierBefore);
      // assign this barrier to the same stage as the write-stage or whichever
      // you prefer
      unsigned stage = (numStages == 1 ? 0 : (numStages == 2 ? 0 : 1));
      opToStage[barrierBefore] = stage;
    }

    //  // optionally add amdgpu.sched_barrier after compute (example)
    //  //{
    //  //  OpBuilder b(lastCompute->getNextNode());
    //  //  auto loc = lastCompute->getLoc();
    //  //  auto sched = b.create<amdgpu::SchedBarrierOp>(
    //  //      loc, amdgpu::sched_barrier_opt_enumAttr::get(forOp.getContext(),
    //  // amdgpu::sched_barrier_opt_enum::none));
    //  //  syntheticBarriers.push_back(sched);
    //  //  // typically scheduling barrier belongs to stage that sits between
    //  load and local_load;
    //  //  // choose stageAfter (above).
    //  //  opToStage[sched] = (numStages == 1 ? 0 : 1);
    //  //}
  }

  // 2) Build finalSchedule: a vector of (originalOp, stage) in *the* order
  // we want pipelineForLoop to consider. We supply grouped-by-stage order,
  // which mirrors the manual IREE flow (read->compute->write).
  std::vector<std::pair<Operation *, unsigned>> finalSchedule;
  finalSchedule.reserve(opToStage.size());
  // append read stage ops in the original in-body order
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (opToStage.count(&op) && opToStage[&op] == 0)
      finalSchedule.push_back({&op, 0});
  }
  // append compute stage
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (opToStage.count(&op) && opToStage[&op] == 2)
      finalSchedule.push_back({&op, 2});
  }
  // append write stage
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (opToStage.count(&op) && opToStage[&op] == 1)
      finalSchedule.push_back({&op, 1});
  }
  // for (Operation *b : syntheticBarriers) {
  //   unsigned stage = opToStage.count(b) ? opToStage[b] : 0u;
  //   finalSchedule.emplace_back(b, stage);
  // }

  if (finalSchedule.empty()) {
    return failure();
  }

  // 3) Prepare the PipeliningOption
  scf::PipeliningOption options;

  // supply the schedule to the pipeliner
  options.getScheduleFn =
      [finalSchedule](scf::ForOp loop,
                      std::vector<std::pair<Operation *, unsigned>> &outSched) {
        outSched = finalSchedule;
      };

  options.annotateFn = nullptr;
  options.peelEpilogue = true;
  options.supportDynamicLoops = false;
  options.predicateFn = nullptr;

  // 4) Invoke the pipeliner
  IRRewriter irRewriter(forOp);
  bool modifiedIR = false;
  FailureOr<scf::ForOp> newForOpOrFail =
      scf::pipelineForLoop(irRewriter, forOp, options, &modifiedIR);

  if (failed(newForOpOrFail)) {
    return failure();
  }

  // Return the new loop generated by the pipeliner.
  return *newForOpOrFail;
}
} // namespace mlir::iree_compiler
