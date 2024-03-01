// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-prefetching"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

namespace {

/// Returns true if the given `memrefType` has the default numeric address space
/// 0 or a HAL descriptor type address space.
static bool hasDefaultOrHALAddressSpace(MemRefType memrefType) {
  Attribute addrSpace = memrefType.getMemorySpace();
  if (!addrSpace)
    return true;

  auto intAttr = dyn_cast<IntegerAttr>(memrefType.getMemorySpace());
  // Accept both default numeric address space and HAL descriptor type address
  // space--the former is used by LLVMGPU while the latter is used by SPIR-V.
  if (intAttr && intAttr.getInt() == 0)
    return true;

  return isa<IREE::HAL::DescriptorTypeAttr>(addrSpace);
}

static FailureOr<int64_t> getConstantIdx(Value v) {
  if (!isa<IndexType>(v.getType()))
    return failure();

  APInt val;
  if (!matchPattern(v, m_ConstantInt(&val)))
    return failure();

  return val.getSExtValue();
}

class LoopPrefetcher {
public:
  static FailureOr<LoopPrefetcher> getLoopPrefetcher(scf::ForOp op) {
    LoopPrefetcher prefetcher;
    prefetcher.singleStage = true;
    prefetcher.mapping = SmallVector<IRMapping>(4);
    prefetcher.forOp = op;
    FailureOr<int64_t> upperBoundCst = getConstantIdx(op.getUpperBound());
    FailureOr<int64_t> lowerBoundCst = getConstantIdx(op.getLowerBound());
    FailureOr<int64_t> stepCst = getConstantIdx(op.getStep());
    if (failed(upperBoundCst) || failed(lowerBoundCst) || failed(stepCst)) {
      return failure();
    }

    prefetcher.ub = *upperBoundCst;
    prefetcher.lb = *lowerBoundCst;
    prefetcher.step = *stepCst;

    int64_t numIters = ceilDiv(prefetcher.ub - prefetcher.lb, prefetcher.step);
    if (numIters <= 2)
      return failure();

    return prefetcher;
  }

  enum class Stage { Read, Compute, Write, None };

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

  void initializeStages() {
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

    // Restore the original order.
    for (auto &op : forOp.getBody()->getOperations()) {
      if (readDependencies.contains(&op)) {
        readStage.push_back(&op);
      }
      if (writeDependencies.contains(&op)) {
        writeStage.push_back(&op);
      }
      if (computeDependencies.contains(&op)) {
        computeStage.push_back(&op);
      }
    }

    LLVM_DEBUG({
      // Stages cannot have overlapping operations.
      llvm::dbgs() << "--- Read Stage ---\n";
      for (Operation *op : readStage) {
        op->dump();
      }
      llvm::dbgs() << "--- Write Stage ---\n";
      for (Operation *op : writeStage) {
        op->dump();
      }
      llvm::dbgs() << "--- Compute Stage ---\n";
      for (Operation *op : computeStage) {
        op->dump();
      }
    });
  }

  /// Clone `op` and call `callback` on the cloned op's oeprands as well as any
  /// operands of nested ops that:
  /// 1) aren't defined within the new op or
  /// 2) are block arguments.
  static Operation *
  cloneAndUpdateOperands(RewriterBase &rewriter, Operation *op,
                         function_ref<void(OpOperand *newOperand)> callback) {
    Operation *clone = rewriter.clone(*op);
    for (OpOperand &operand : clone->getOpOperands())
      callback(&operand);
    return clone;
  }

  SmallVector<Value> emitRead(IRMapping &mapping, RewriterBase &rewriter,
                              Value iv) {
    mapping.map(forOp.getInductionVar(), iv);
    SmallVector<Value> results;
    for (Operation *op : readStage) {
      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            if (mapping.contains(newOperand->get())) {
              newOperand->set(mapping.lookup(newOperand->get()));
            }
          });

      if (isa<vector::TransferReadOp>(newOp)) {
        llvm::append_range(results, newOp->getResults());
      }

      // Map read operations to new read operations.
      for (unsigned i = 0, e = op->getNumResults(); i != e; ++i) {
        mapping.map(op->getResult(i), newOp->getResult(i));
      }
    }
    return results;
  }

  void emitWrite(IRMapping &mapping, RewriterBase &rewriter, Value iv) {
    mapping.map(forOp.getInductionVar(), iv);
    for (Operation *op : writeStage) {
      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            if (mapping.contains(newOperand->get())) {
              newOperand->set(mapping.lookup(newOperand->get()));
            }
          });

      // If a mapping for any results already exists, move on, otherwise,
      // add a new mapping.
      for (unsigned i = 0, e = op->getNumResults(); i != e; ++i) {
        if (!mapping.contains(op->getResult(i))) {
          mapping.map(op->getResult(i), newOp->getResult(i));
        }
      }
    }
  }

  void emitBarrier(Location loc, RewriterBase &rewriter) {
    rewriter.create<gpu::BarrierOp>(loc);
  }

  SmallVector<Value> emitCompute(IRMapping &mapping, RewriterBase &rewriter,
                                 Value indVar) {
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

      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            if (mapping.contains(newOperand->get())) {
              newOperand->set(mapping.lookup(newOperand->get()));
            }
          });
      results = newOp->getResults();

      // Map compute operations to new compute operations.
      for (unsigned i = 0, e = op->getNumResults(); i != e; ++i) {
        mapping.map(op->getResult(i), newOp->getResult(i));
      }
    }

    return results;
  }

  void updateYield(IRMapping &mapping, SmallVector<Value> &readValues,
                   RewriterBase &rewriter) {
    for (Operation *op : computeStage) {
      if (auto yield = dyn_cast<scf::YieldOp>(op)) {
        cloneAndUpdateOperands(rewriter, yield, [&](OpOperand *newOperand) {
          if (mapping.contains(newOperand->get())) {
            newOperand->set(mapping.lookup(newOperand->get()));
          }
        });

        break;
      }
    }
    // Remove the invalid link to yield
    // computeStage.pop_back();
  }

  std::tuple<SmallVector<Value>, SmallVector<Value>>
  emitPrologue(RewriterBase &rewriter) {
    Location loc = forOp.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, lb);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, lb + step);
    SmallVector<Value> iterArgs;
    SmallVector<Value> readResults;
    SmallVector<Value> writeArgs;

    if (singleStage) {
      // Read (0)
      emitRead(mapping[0], rewriter, zero);
      // Write(0)
      emitWrite(mapping[0], rewriter, zero);
      return {iterArgs, writeArgs};
    }

    // Read(0).
    iterArgs = emitRead(mapping[0], rewriter, zero);
    // Read(1).
    readResults = emitRead(mapping[1], rewriter, one);
    llvm::append_range(iterArgs, readResults);

    // Collect the values to be used as write args.
    for (Operation *op : readStage) {
      if (isa<vector::TransferReadOp>(op)) {
        auto transferReadOp = cast<vector::TransferReadOp>(op);
        for (Operation *user : transferReadOp.getResult().getUsers()) {
          if (auto writeOp = dyn_cast<vector::TransferWriteOp>(user)) {
            writeArgs.push_back(writeOp.getVector());
          }
        }
      }
    }

    return {iterArgs, writeArgs};
  }

  static void createWriteMappings(SmallVector<Value> &srcValues,
                                  SmallVector<Value> &targetValues,
                                  IRMapping &mapping) {
    for (auto [src, tgt] : llvm::zip_equal(srcValues, targetValues)) {
      mapping.map(src, tgt);
    }
  }

  scf::ForOp createKernelLoop(RewriterBase &rewriter,
                              SmallVector<Value> &newIterArgs,
                              SmallVector<Value> &writeArgs) {
    Location loc = forOp.getLoc();
    int64_t newUpperBound = singleStage ? (ub - step) : (ub - 2 * step);
    auto newUb = rewriter.create<arith::ConstantIndexOp>(loc, newUpperBound);

    // Keep original iter args and then add some for what's being loaded to
    // registers.
    auto iterArgs = llvm::to_vector_of<Value>(forOp.getInitArgs());
    llvm::append_range(iterArgs, newIterArgs);

    Value newStep = singleStage ? forOp.getStep()
                                : rewriter.create<arith::AddIOp>(
                                      loc, forOp.getStep(), forOp.getStep());
    auto newForOp = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                                newUb, newStep, iterArgs);

    // When there are no iter args, the loop body terminator will be created.
    // Since we always create it below, remove the terminator if it was created.
    if (!newForOp.getBody()->empty())
      rewriter.eraseOp(newForOp.getBody()->getTerminator());

    if (singleStage)
      return newForOp;

    SmallVector<Value> targetValues(writeArgs.size());
    for (size_t i = 0, e = writeArgs.size(); i != e; ++i)
      targetValues[i] = newForOp.getRegionIterArg(i + 1);

    createWriteMappings(writeArgs, targetValues, mapping[0]);

    for (size_t i = 0, e = writeArgs.size(); i != e; ++i)
      targetValues[i] = newForOp.getRegionIterArg(i + e + 1);

    createWriteMappings(writeArgs, targetValues, mapping[1]);
    return newForOp;
  }

  void createKernel(scf::ForOp newForOp, RewriterBase &rewriter) {
    rewriter.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
    Location loc = forOp.getLoc();
    Value indVar = newForOp.getInductionVar();
    Value increment = rewriter.create<arith::ConstantIndexOp>(loc, step);
    Value iPlusOne = rewriter.create<arith::AddIOp>(loc, indVar, increment);
    Value iPlusTwo = rewriter.create<arith::AddIOp>(loc, iPlusOne, increment);
    Value iPlusThree = rewriter.create<arith::AddIOp>(loc, iPlusTwo, increment);

    for (int i = 0; i < 3; ++i) {
      for (auto [idx, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
        mapping[i].map(arg, newForOp.getRegionIterArgs()[idx]);
      }
    }

    SmallVector<Value> readRegisters, moreRegisters;
    if (singleStage) {
      emitRead(mapping[1], rewriter, iPlusOne);
      emitBarrier(loc, rewriter);
      emitCompute(mapping[0], rewriter, indVar);
      emitBarrier(loc, rewriter);
      emitWrite(mapping[1], rewriter, iPlusOne);
      updateYield(mapping[0], readRegisters, rewriter);
      return;
    }

    emitWrite(mapping[0], rewriter, indVar);
    readRegisters = emitRead(mapping[2], rewriter, iPlusTwo);
    emitBarrier(loc, rewriter);
    auto computeResults = emitCompute(mapping[0], rewriter, indVar);
    mapping[0].map(forOp.getRegionIterArg(0), computeResults[0]);
    emitBarrier(loc, rewriter);
    emitWrite(mapping[1], rewriter, iPlusOne);
    moreRegisters = emitRead(mapping[3], rewriter, iPlusThree);
    emitBarrier(loc, rewriter);
    emitCompute(mapping[0], rewriter, iPlusOne);
    emitBarrier(loc, rewriter);
    readRegisters.append(moreRegisters.begin(), moreRegisters.end());
    updateYield(mapping[0], readRegisters, rewriter);
  }

  SmallVector<Value> emitEpilogue(scf::ForOp newForOp, RewriterBase &rewriter,
                                  SmallVector<Value> &writeArgs) {
    Location loc = forOp.getLoc();
    Value nMinusTwo =
        rewriter.create<arith::ConstantIndexOp>(loc, ub - 2 * step);
    Value nMinusOne =
        rewriter.create<arith::ConstantIndexOp>(loc, ub - 1 * step);

    // Map iter_args to results of newForOp.
    for (unsigned i = 0, e = forOp->getNumResults(); i != e; ++i) {
      mapping[0].map(forOp.getRegionIterArg(i), newForOp.getResult(i));
    }

    if (singleStage) {
      emitBarrier(loc, rewriter);
      return emitCompute(mapping[0], rewriter, nMinusOne);
    }

    SmallVector<Value> targetValues(writeArgs.size());
    for (size_t i = 0, e = writeArgs.size(); i != e; ++i)
      targetValues[i] = newForOp.getResult(i + 1);

    createWriteMappings(writeArgs, targetValues, mapping[2]);

    for (size_t i = 0, e = writeArgs.size(); i != e; ++i)
      targetValues[i] = newForOp.getResult(i + e + 1);

    createWriteMappings(writeArgs, targetValues, mapping[3]);

    emitWrite(mapping[2], rewriter, nMinusTwo);
    emitBarrier(loc, rewriter);
    SmallVector<Value> computeResults =
        emitCompute(mapping[0], rewriter, nMinusTwo);
    mapping[0].map(forOp.getRegionIterArg(0), computeResults[0]);
    emitBarrier(loc, rewriter);
    emitWrite(mapping[3], rewriter, nMinusOne);
    emitBarrier(loc, rewriter);
    computeResults = emitCompute(mapping[0], rewriter, nMinusOne);
    return computeResults;
  }

private:
  SmallVector<IRMapping, 4> mapping;
  scf::ForOp forOp;
  int64_t ub, lb, step;
  bool singleStage;

  SmallVector<Operation *> readStage;
  SmallVector<Operation *> writeStage;
  SmallVector<Operation *> computeStage;
};

FailureOr<scf::ForOp> applyPrefetching(RewriterBase &rewriter,
                                       scf::ForOp forOp) {
  rewriter.setInsertionPoint(forOp);

  auto prefetcherOr = LoopPrefetcher::getLoopPrefetcher(forOp);
  if (failed(prefetcherOr)) {
    return failure();
  }
  LoopPrefetcher &prefetcher = *prefetcherOr;

  prefetcher.initializeStages();
  auto [iterArgs, writeArgs] = prefetcher.emitPrologue(rewriter);

  scf::ForOp newForOp =
      prefetcher.createKernelLoop(rewriter, iterArgs, writeArgs);
  prefetcher.createKernel(newForOp, rewriter);

  rewriter.setInsertionPointAfter(newForOp);
  SmallVector<Value> result =
      prefetcher.emitEpilogue(newForOp, rewriter, writeArgs);

  rewriter.replaceOp(forOp, result);
  return newForOp;
}

struct LLVMGPUPrefetchSharedMemoryPass final
    : LLVMGPUPrefetchSharedMemoryBase<LLVMGPUPrefetchSharedMemoryPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    // Collect for ops.
    SmallVector<scf::ForOp> forOps;
    funcOp.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

    if (forOps.empty())
      return;

    if (failed(prefetchSharedMemoryCopy(rewriter, forOps.front()))) {
      return signalPassFailure();
    }
  }
};

} // namespace

FailureOr<scf::ForOp> prefetchSharedMemoryCopy(RewriterBase &rewriter,
                                               scf::ForOp forOp) {
  return applyPrefetching(rewriter, forOp);
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUPrefetchSharedMemoryPass() {
  return std::make_unique<LLVMGPUPrefetchSharedMemoryPass>();
}

} // namespace mlir::iree_compiler
