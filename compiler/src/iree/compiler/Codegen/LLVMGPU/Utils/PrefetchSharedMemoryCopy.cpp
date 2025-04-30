// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define DEBUG_TYPE "iree-codegen-llvmgpu-prefetch-shared-memory-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

namespace {

class LoopPrefetcher {
public:
  /// Creates an instance that plans the given scf.for |op| to be ready for
  /// prefetching. Returns failure if unable to support the given |op|.
  static FailureOr<LoopPrefetcher> get(scf::ForOp op) {
    if (!op.getOps<scf::ForOp>().empty()) {
      LDBG("Loop prefetcher does not support nested loops yet");
      return failure();
    }

    LoopPrefetcher prefetcher;
    prefetcher.mapping = SmallVector<IRMapping>(4);
    prefetcher.forOp = op;
    prefetcher.lb = prefetcher.ub = prefetcher.step = 0;

    if (failed(prefetcher.initializeLoopInfo())) {
      LDBG("Failed to initialize loop info (unsupported loop)");
      return failure();
    }

    if (failed(prefetcher.initializeStages())) {
      LDBG("Failed to initialize stage info (unsupported loop)");
      return failure();
    }

    return prefetcher;
  }

  // Emits the prologue before the main pipelined loop.
  void emitPrologue(RewriterBase &rewriter) {
    Location loc = forOp.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, lb);
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
    auto newUb = rewriter.create<arith::ConstantIndexOp>(loc, newUpperBound);

    // Keep original iter args and then add some for what's being loaded to
    // registers.
    auto iterArgs = llvm::to_vector_of<Value>(forOp.getInitArgs());
    auto newForOp = rewriter.create<scf::ForOp>(
        loc, forOp.getLowerBound(), newUb, forOp.getStep(), iterArgs);

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
    Value increment = rewriter.create<arith::ConstantIndexOp>(loc, step);
    Value iPlusOne = rewriter.create<arith::AddIOp>(loc, indVar, increment);

    for (int i = 0; i < 3; ++i) {
      for (auto [idx, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
        mapping[i].map(arg, newForOp.getRegionIterArgs()[idx]);
      }
    }

    emitRead(mapping[1], rewriter, iPlusOne);
    emitBarrier(loc, rewriter);
    emitCompute(mapping[0], rewriter, indVar);
    emitBarrier(loc, rewriter);
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
        rewriter.create<arith::ConstantIndexOp>(loc, ub - 1 * step);

    // Map iter_args to results of newForOp.
    for (unsigned i = 0, e = forOp->getNumResults(); i != e; ++i) {
      mapping[0].map(forOp.getRegionIterArg(i), newForOp.getResult(i));
    }

    emitBarrier(loc, rewriter);
    return emitCompute(mapping[0], rewriter, nMinusOne);
  }

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

    if (noTransferReads && isa<vector::TransferReadOp>(op)) {
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

  // We only support loops whose bodies can be divided into 3 stages (read,
  // write, compute). If there are any remaining ops with side effects (except
  // for gpu.barrier), the loop is not supported.
  LogicalResult initializeStages() {
    DenseSet<Operation *> readDependencies;
    DenseSet<Operation *> writeDependencies;
    DenseSet<Operation *> computeDependencies;

    for (Operation &op : forOp.getBody()->getOperations()) {
      if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
        getValueDependencies(read, readDependencies);
      } else if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
        getValueDependencies(write, writeDependencies,
                             /*noTransferReads=*/true);
      } else if (auto compute = dyn_cast<scf::YieldOp>(op)) {
        getValueDependencies(compute, computeDependencies);
      }
    }
    // If `scf.yeild` is the only compute op then there is no value in doing
    // prefetching.
    if (computeDependencies.size() == 1) {
      LDBG("Loop does not have compute so not doing prefetching." << forOp);
      return failure();
    }

    // Restore the original order.
    for (auto &op : forOp.getBody()->getOperations()) {
      bool hasStage = false;
      if (readDependencies.contains(&op)) {
        readStage.push_back(&op);
        hasStage = true;
      }
      if (writeDependencies.contains(&op)) {
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
          LDBG("Found a non-pure loop body op not assigned to any stage "
               "(unsupported loop): "
               << op);
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
    rewriter.create<gpu::BarrierOp>(loc);
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

  // Ops in the original scf.for loop that belongs to different classes.
  SmallVector<Operation *> readStage;
  SmallVector<Operation *> writeStage;
  SmallVector<Operation *> computeStage;
};

} // namespace

FailureOr<scf::ForOp> prefetchSharedMemoryCopy(RewriterBase &rewriter,
                                               scf::ForOp forOp) {
  rewriter.setInsertionPoint(forOp);

  auto prefetcherOr = LoopPrefetcher::get(forOp);
  if (failed(prefetcherOr))
    return failure();
  LoopPrefetcher &prefetcher = *prefetcherOr;

  prefetcher.emitPrologue(rewriter);
  scf::ForOp newForOp = prefetcher.createKernelLoop(rewriter);
  prefetcher.createKernel(rewriter, newForOp);

  SmallVector<Value> results = prefetcher.emitEpilogue(rewriter, newForOp);
  rewriter.replaceOp(forOp, results);
  return newForOp;
}

} // namespace mlir::iree_compiler
