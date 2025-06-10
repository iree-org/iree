// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-fission-transfer-ops-in-control-flow"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FISSIONTRANSFEROPSINCONTROLFLOWPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

/// Replaces all occurrences of `oldVal` in the operands of `op` with `newVal`.
static void replaceOperand(Operation *op, Value oldVal, Value newVal) {
  for (auto &operand : op->getOpOperands()) {
    if (operand.get() == oldVal) {
      operand.set(newVal);
    }
  }
}

/// Collects a backward slice of operations within the same control flow scope
/// (i.e., with the same parent) from the specified operation. This is useful
/// for identifying dependencies within a block.
static SetVector<Operation *>
collectBackwardSliceInControlFlow(Operation *op, Operation *parentOp) {
  BackwardSliceOptions options;
  options.inclusive = false;
  options.filter = [&](Operation *op) { return parentOp == op->getParentOp(); };
  SetVector<Operation *> slice;
  LogicalResult result = getBackwardSlice(op, &slice, options);
  if (failed(result)) {
    return {};
  }
  return slice;
}

/// Clones a slice of operations, remapping their operands.
static void cloneSliceIntoLoop(IRRewriter &rewriter,
                               SetVector<Operation *> &slice,
                               IRMapping &mapping) {
  for (Operation *op : slice) {
    rewriter.clone(*op, mapping);
  }
}

/// Creates a new loop with the same iteration parameters (bounds, step) as the
/// given loop.
static scf::ForOp createNewLoop(IRRewriter &rewriter, scf::ForOp forOp,
                                Location loc) {
  return rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                     forOp.getUpperBound(), forOp.getStep(),
                                     forOp.getRegionIterArgs());
}

/// Creates an alloca operation for the intermediate results of the transfer.
/// %alloca_size = (%upper_bound - %lower_bound) / %step
/// Note the division is a ceildiv to ensure enough space is allocated.
static memref::AllocaOp createAlloca(IRRewriter &rewriter,
                                     vector::TransferReadOp readOp,
                                     scf::ForOp forOp) {
  auto loc = forOp.getLoc();
  auto allocaSize = rewriter.create<arith::CeilDivUIOp>(
      loc,
      rewriter.create<arith::SubIOp>(loc, forOp.getUpperBound(),
                                     forOp.getLowerBound()),
      forOp.getStep());

  auto vectorType = cast<VectorType>(readOp.getVectorType());
  SmallVector<int64_t> memrefShape(vectorType.getShape());
  memrefShape.insert(memrefShape.begin(), ShapedType::kDynamic);
  auto privateAddrSpaceAttr = gpu::AddressSpaceAttr::get(
      rewriter.getContext(), gpu::GPUDialect::getPrivateAddressSpace());
  auto memrefType = MemRefType::get(memrefShape, vectorType.getElementType(),
                                    AffineMap{}, privateAddrSpaceAttr);

  return rewriter.create<memref::AllocaOp>(loc, memrefType,
                                           ValueRange{allocaSize});
}

/// Creates an index for accessing the memref in the loop. This index is
/// normalized into step of one in order to access the correct element from the
/// alloca. %index = (%loop_index - %loop_lower_bound) / %loop_step
static Value createMemrefAccessIndex(IRRewriter &rewriter, scf::ForOp forOp) {
  auto subIOp = rewriter.create<arith::SubIOp>(
      forOp.getLoc(), forOp.getInductionVar(), forOp.getLowerBound());
  auto divUIOp =
      rewriter.create<arith::DivUIOp>(forOp.getLoc(), subIOp, forOp.getStep());
  return divUIOp.getResult();
}

/// Sets up the read loop for the transfer read operation.
static void setupReadLoop(IRRewriter &rewriter, vector::TransferReadOp readOp,
                          scf::ForOp forOp, memref::AllocaOp alloca) {
  scf::ForOp readLoop = createNewLoop(rewriter, forOp, readOp.getLoc());
  rewriter.setInsertionPointToStart(readLoop.getBody());

  IRMapping readMapping;
  readMapping.map(forOp.getInductionVar(), readLoop.getInductionVar());
  SetVector<Operation *> readSlice =
      collectBackwardSliceInControlFlow(readOp, forOp);
  cloneSliceIntoLoop(rewriter, readSlice, readMapping);
  Operation *lastRead = rewriter.clone(*readOp, readMapping);

  auto readLoopIndex = createMemrefAccessIndex(rewriter, readLoop);
  SmallVector<Value> readLoopIndices = {readLoopIndex};
  auto readOpIndicesSize = readOp.getIndices().size();
  auto constantZero =
      rewriter.create<arith::ConstantIndexOp>(readOp.getLoc(), 0);
  auto zeroIndices = SmallVector<Value>(readOpIndicesSize, constantZero);
  readLoopIndices.insert(readLoopIndices.end(), zeroIndices.begin(),
                         zeroIndices.end());

  rewriter.create<vector::TransferWriteOp>(
      readOp.getLoc(), lastRead->getResult(0), alloca, readLoopIndices);

  LDBG("Read loop: \n" << readLoop << "\n");
  rewriter.setInsertionPointAfter(readLoop.getOperation());
}

/// Sets up the write loop for the transfer write operation.
static void setupWriteLoop(IRRewriter &rewriter, vector::TransferReadOp readOp,
                           vector::TransferWriteOp writeOp, scf::ForOp forOp,
                           memref::AllocaOp alloca) {
  scf::ForOp writeLoop = createNewLoop(rewriter, forOp, writeOp.getLoc());
  rewriter.setInsertionPointToStart(writeLoop.getBody());

  auto writeLoopIndex = createMemrefAccessIndex(rewriter, writeLoop);
  SmallVector<Value> writeLoopIndices = {writeLoopIndex};
  auto zeroIndicesSize = writeOp.getIndices().size();
  auto constantZero =
      rewriter.create<arith::ConstantIndexOp>(readOp.getLoc(), 0);
  SmallVector<Value> zeroIndices(zeroIndicesSize, constantZero);
  writeLoopIndices.insert(writeLoopIndices.end(), zeroIndices.begin(),
                          zeroIndices.end());

  vector::TransferReadOp newReadOp = rewriter.create<vector::TransferReadOp>(
      writeOp.getLoc(), writeOp.getVectorType(), alloca, writeLoopIndices);

  IRMapping writeMapping;
  writeMapping.map(forOp.getInductionVar(), writeLoop.getInductionVar());
  SetVector<Operation *> writeSlice =
      collectBackwardSliceInControlFlow(writeOp, forOp);
  cloneSliceIntoLoop(rewriter, writeSlice, writeMapping);

  rewriter.clone(*writeOp, writeMapping);
  for (auto &op : writeLoop.getBody()->getOperations()) {
    replaceOperand(&op, writeMapping.lookup(readOp.getResult()), newReadOp);
  }

  LDBG("Write loop: \n" << writeLoop << "\n");
}

/// Splits transfer read and write operations from a control flow Operation
/// (forOp) into separate loops.
///
/// For example, given a loop with transfer read and write operations:
///   scf.for %i = 0 to 10 {
///     %read = vector.transfer_read ...
///     vector.transfer_write %read ...
///   }
///
/// This function will transform it into:
///   %alloca = memref.alloca ...  // Alloca for intermediate results
///   scf.for %i = 0 to 10 {
///     %read = vector.transfer_read ...
///     vector.transfer_write %read %alloca
///   }
///   scf.for %j = 0 to 10 {
///     %read = vector.transfer_read %alloca
///     vector.transfer_write %read ...
///   }
static void splitTransferOpsFromControlFlow(IRRewriter &rewriter,
                                            vector::TransferReadOp readOp,
                                            vector::TransferWriteOp writeOp,
                                            scf::ForOp forOp) {
  LDBG("Splitting transfer ops from control flow: \n"
       << "For Op: " << forOp << "\n");

  rewriter.setInsertionPoint(forOp);
  memref::AllocaOp alloca = createAlloca(rewriter, readOp, forOp);

  setupReadLoop(rewriter, readOp, forOp, alloca);
  setupWriteLoop(rewriter, readOp, writeOp, forOp, alloca);

  rewriter.eraseOp(forOp);
}

/// Checks if the transfer read and write operations are legal for fissioning.
static bool isLegal(vector::TransferReadOp readOp,
                    vector::TransferWriteOp writeOp, scf::ForOp forOp) {
  if (readOp->getParentOp() != forOp || writeOp->getParentOp() != forOp) {
    return false;
  }
  if (!hasGlobalMemoryAddressSpace(
          cast<MemRefType>(readOp.getBase().getType()))) {
    return false;
  }
  if (hasGlobalMemoryAddressSpace(
          cast<MemRefType>(writeOp.getBase().getType()))) {
    return false;
  }
  return true;
}

namespace {
struct FissionTarget {
  scf::ForOp parent;
  vector::TransferReadOp readOp;
  vector::TransferWriteOp writeOp;
};
} // namespace

/// Populates a FissionTarget from a scf::ForOp by checking if it contains a
/// transfer read and write operation that can be legally fissioned.
static FailureOr<FissionTarget> populateFissionTarget(scf::ForOp forOp) {
  // Fission Loop always has a transfer_write as the last operation.
  auto lastOp = forOp.getBody()->getTerminator()->getPrevNode();
  if (!isa<vector::TransferWriteOp>(lastOp)) {
    return failure();
  }

  vector::TransferWriteOp writeOp = cast<vector::TransferWriteOp>(lastOp);
  SetVector<Operation *> writeSlice =
      collectBackwardSliceInControlFlow(writeOp, forOp.getOperation());
  for (Operation *op : writeSlice) {
    if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
      if (!isLegal(readOp, writeOp, forOp)) {
        continue;
      }

      FissionTarget fissionTarget = {forOp, readOp, writeOp};
      return fissionTarget;
    }
  }
  return failure();
}

struct FissionTransferOpsInControlFlowPass final
    : impl::FissionTransferOpsInControlFlowPassBase<
          FissionTransferOpsInControlFlowPass> {
  using impl::FissionTransferOpsInControlFlowPassBase<
      FissionTransferOpsInControlFlowPass>::
      FissionTransferOpsInControlFlowPassBase;
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    SmallVector<scf::ForOp> loops;
    funcOp.walk([&loops](scf::ForOp forOp) { loops.push_back(forOp); });

    SmallVector<FissionTarget> fissionTargets;
    for (scf::ForOp forOp : loops) {
      auto result = populateFissionTarget(forOp);
      if (failed(result)) {
        continue;
      }
      // When not doing multi-trip fission if we have even one multi-trip loop
      // we bail-out from this pass and dont do fission as we wont be doing any
      // prefetching which is the point of doing fission.
      if (!FissionMultiTrip && !neverRunsSecondIteration(forOp)) {
        return;
      }
      fissionTargets.push_back(result.value());
    }

    for (const FissionTarget &target : fissionTargets) {
      splitTransferOpsFromControlFlow(rewriter, target.readOp, target.writeOp,
                                      target.parent);
    }
  }
};

} // namespace mlir::iree_compiler
