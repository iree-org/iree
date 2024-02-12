// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-prefetching"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace iree_compiler {

/// Returns true if the given `memrefType` has the default numeric address space
/// 0 or a HAL descriptor type address space.
static bool hasDefaultOrHALAddressSpace(MemRefType memrefType) {
  Attribute addrSpace = memrefType.getMemorySpace();
  if (!addrSpace)
    return true;
  auto intAttr = llvm::dyn_cast<IntegerAttr>(addrSpace);
  // Accept both default numeric address space and HAL descriptor type address
  // space--the former is used by LLVMGPU while the latter is used by SPIR-V.
  if (intAttr && intAttr.getInt() == 0)
    return true;
  return llvm::isa<IREE::HAL::DescriptorTypeAttr>(addrSpace);
}

class LoopPrefetcher {
public:
  bool initializeLoopInfo(scf::ForOp op) {
    singleStage = true;
    mapping = SmallVector<IRMapping>(4);
    forOp = op;
    auto upperBoundCst =
        forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    auto lowerBoundCst =
        forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto stepCst = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
    if (!upperBoundCst || !lowerBoundCst || !stepCst) {
      return false;
    }
    ub = upperBoundCst.value();
    lb = lowerBoundCst.value();
    step = stepCst.value();

    int64_t numIteration = ceilDiv(ub - lb, step);

    if (numIteration <= 2) {
      return false;
    }

    return true;
  }

  enum class Stage { Read, Compute, Write, None };

  void getReachable(Operation *op, DenseSet<Operation *> &reachable,
                    bool noTransferReads = false) {
    if (!op)
      return;

    if (reachable.contains(op)) {
      return;
    }

    if (noTransferReads) {
      if (isa<vector::TransferReadOp>(op))
        return;
    }

    if (!forOp->isProperAncestor(op)) {
      return;
    }

    op->walk([&](Operation *nested) {
      reachable.insert(op);
      for (Value val : nested->getOperands()) {
        getReachable(val.getDefiningOp(), reachable, noTransferReads);
      }
    });
  }

  void initializeStages() {
    DenseSet<Operation *> readReachable;
    DenseSet<Operation *> writeReachable;
    DenseSet<Operation *> computeReachable;

    for (Operation &op : forOp.getBody()->getOperations()) {
      if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
        getReachable(read, readReachable);
      }
      if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
        getReachable(write, writeReachable, true);
      }
      if (auto compute = dyn_cast<scf::YieldOp>(op)) {
        getReachable(compute, computeReachable);
      }
    }

    for (auto &op : forOp.getBody()->getOperations()) {
      if (readReachable.contains(&op)) {
        readStage.push_back(&op);
      }
      if (writeReachable.contains(&op)) {
        writeStage.push_back(&op);
      }
      if (computeReachable.contains(&op)) {
        computeStage.push_back(&op);
      }
    }

    LLVM_DEBUG({
      // Stages cannot have overlapping operations
      llvm::dbgs() << "--- Read Stage ---\n";
      for (auto &op : readStage) {
        op->dump();
      }
      llvm::dbgs() << "--- Write Stage ---\n";
      for (auto &op : writeStage) {
        op->dump();
      }
      llvm::dbgs() << "--- Compute Stage ---\n";
      for (auto &op : computeStage) {
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
        results.append(newOp->getResults().begin(), newOp->getResults().end());
      }

      // Map read operations to new read operations.
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
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
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
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
                                 Value iv) {
    mapping.map(forOp.getInductionVar(), iv);
    SmallVector<Value> results;
    for (Operation *op : computeStage) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        // On yield, return the operands of the yield converted.
        results = llvm::to_vector<4>(
            llvm::map_range(yieldOp.getOperands(), [&](Value operand) {
              return mapping.lookup(operand);
            }));
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
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        mapping.map(op->getResult(i), newOp->getResult(i));
      }
    }

    return results;
  }

  void updateYield(IRMapping &mapping, SmallVector<Value> &readValues,
                   RewriterBase &rewriter) {
    for (auto op : computeStage) {
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
    Value zero = rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), lb);
    Value one =
        rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), lb + step);
    SmallVector<Value> iterArgs, readResults;
    SmallVector<Value> writeArgs;

    if (singleStage) {
      // Read (0)
      emitRead(mapping[0], rewriter, zero);
      // Write(0)
      emitWrite(mapping[0], rewriter, zero);
    } else {
      // Read(0).
      iterArgs = emitRead(mapping[0], rewriter, zero);
      // Read(1).
      readResults = emitRead(mapping[1], rewriter, one);
      iterArgs.append(readResults.begin(), readResults.end());

      // Collect the values to be used as write args
      for (Operation *op : readStage) {
        if (isa<vector::TransferReadOp>(op)) {
          auto transferReadOp = llvm::cast<vector::TransferReadOp>(op);
          for (Operation *user : transferReadOp.getResult().getUsers()) {
            if (auto writeOp = llvm::cast<vector::TransferWriteOp>(user)) {
              writeArgs.push_back(writeOp.getVector());
            }
          }
        }
      }
    }

    return std::make_tuple(iterArgs, writeArgs);
  }

  static void createWriteMappings(RewriterBase &rewriter,
                                  SmallVector<Value> &srcValues,
                                  SmallVector<Value> &targetValues,
                                  IRMapping &mapping) {
    for (int i = 0; i < srcValues.size(); i++) {
      mapping.map(srcValues[i], targetValues[i]);
    }
  }

  scf::ForOp createKernelLoop(RewriterBase &rewriter,
                              SmallVector<Value> &newIterArgs,
                              SmallVector<Value> &writeArgs) {
    int64_t newUpperBound = singleStage ? ub - step : ub - 2 * step;
    auto newUb =
        rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), newUpperBound);

    // Keep original iter args and then add some for what's being loaded to
    // registers
    SmallVector<Value> iterArgs(forOp.getInitArgs().begin(),
                                forOp.getInitArgs().end());
    if (!newIterArgs.empty())
      iterArgs.append(newIterArgs.begin(), newIterArgs.end());

    Value newStep = singleStage
                        ? forOp.getStep()
                        : rewriter.create<arith::AddIOp>(
                              forOp.getLoc(), forOp.getStep(), forOp.getStep());
    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), newUb, newStep, iterArgs);

    // When there are no iter args, the loop body terminator will be created.
    // Since we always create it below, remove the terminator if it was created.
    if (!newForOp.getBody()->empty())
      rewriter.eraseOp(newForOp.getBody()->getTerminator());

    if (!singleStage) {
      SmallVector<Value> targetValues(writeArgs.size());
      for (int i = 1, j = 0; j < writeArgs.size(); i++, j++)
        targetValues[j] = newForOp.getRegionIterArg(i);
      createWriteMappings(rewriter, writeArgs, targetValues, mapping[0]);
      for (int i = writeArgs.size() + 1, j = 0; j < writeArgs.size(); i++, j++)
        targetValues[j] = newForOp.getRegionIterArg(i);
      createWriteMappings(rewriter, writeArgs, targetValues, mapping[1]);
    }

    return newForOp;
  }

  void createKernel(scf::ForOp newForOp, RewriterBase &rewriter) {

    rewriter.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
    Value i = newForOp.getInductionVar();
    Value increment =
        rewriter.create<arith::ConstantIndexOp>(forOp.getLoc(), step);
    Value iPlusOne =
        rewriter.create<arith::AddIOp>(forOp.getLoc(), i, increment);
    Value iPlusTwo =
        rewriter.create<arith::AddIOp>(forOp.getLoc(), iPlusOne, increment);
    Value iPlusThree =
        rewriter.create<arith::AddIOp>(forOp.getLoc(), iPlusTwo, increment);
    Location loc = forOp.getLoc();

    for (int i = 0; i < 3; i++) {
      for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs())) {
        mapping[i].map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
      }
    }

    SmallVector<Value> readRegisters, moreRegisters;
    if (singleStage) {
      emitRead(mapping[1], rewriter, iPlusOne);
      emitBarrier(loc, rewriter);
      emitCompute(mapping[0], rewriter, i);
      emitBarrier(loc, rewriter);
      emitWrite(mapping[1], rewriter, iPlusOne);
      updateYield(mapping[0], readRegisters, rewriter);
    } else {
      emitWrite(mapping[0], rewriter, i);
      readRegisters = emitRead(mapping[2], rewriter, iPlusTwo);
      emitBarrier(loc, rewriter);
      auto computeResults = emitCompute(mapping[0], rewriter, i);
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
  }

  SmallVector<Value> emitEpilogue(scf::ForOp newForOp, RewriterBase &rewriter,
                                  SmallVector<Value> &writeArgs) {
    Location loc = forOp.getLoc();
    Value nMinusTwo =
        rewriter.create<arith::ConstantIndexOp>(loc, ub - 2 * step);
    Value nMinusOne =
        rewriter.create<arith::ConstantIndexOp>(loc, ub - 1 * step);

    // Map iter_args to results of newForOp.
    for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
      mapping[0].map(forOp.getRegionIterArg(i), newForOp.getResult(i));
    }

    SmallVector<Value> computeResults;
    if (singleStage) {
      emitBarrier(loc, rewriter);
      computeResults = emitCompute(mapping[0], rewriter, nMinusOne);
    } else {
      SmallVector<Value> targetValues(writeArgs.size());
      for (int i = 1, j = 0; j < writeArgs.size(); i++, j++)
        targetValues[j] = newForOp.getResult(i);
      createWriteMappings(rewriter, writeArgs, targetValues, mapping[2]);
      for (int i = writeArgs.size() + 1, j = 0; j < writeArgs.size(); i++, j++)
        targetValues[j] = newForOp.getResult(i);
      createWriteMappings(rewriter, writeArgs, targetValues, mapping[3]);

      emitWrite(mapping[2], rewriter, nMinusTwo);
      emitBarrier(loc, rewriter);
      computeResults = emitCompute(mapping[0], rewriter, nMinusTwo);
      mapping[0].map(forOp.getRegionIterArg(0), computeResults[0]);
      emitBarrier(loc, rewriter);
      emitWrite(mapping[3], rewriter, nMinusOne);
      emitBarrier(loc, rewriter);
      computeResults = emitCompute(mapping[0], rewriter, nMinusOne);
    }
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

namespace {

FailureOr<scf::ForOp> applyPrefetching(scf::ForOp forOp) {
  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);

  LoopPrefetcher prefetcher;
  if (!prefetcher.initializeLoopInfo(forOp)) {
    return failure();
  }
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

} // namespace

FailureOr<scf::ForOp> prefetchSharedMemoryCopy(RewriterBase &rewriter,
                                               scf::ForOp forOp) {
  return applyPrefetching(forOp);
}

} // namespace iree_compiler
} // namespace mlir
