// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-fission-transfer-ops-in-control-flow"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FISSIONTRANSFEROPSINCONTROLFLOWPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

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

namespace {
struct FissionTarget {
  scf::ForOp parent;
  SmallVector<vector::TransferReadOp> readOps;
  SmallVector<vector::TransferWriteOp> writeOps;
};
} // namespace

/// Sets up the read loop for the transfer read operation.
static void setupReadLoop(IRRewriter &rewriter, const FissionTarget &target,
                          ArrayRef<memref::AllocaOp> allocaOps) {
  // Create a copy of the original loop, without the 'transfer_write's.
  IRMapping mapping;
  auto readLoop = cast<scf::ForOp>(rewriter.clone(*target.parent, mapping));
  for (auto writeOp : target.writeOps) {
    rewriter.eraseOp(mapping.lookup(writeOp));
  }

  rewriter.setInsertionPoint(readLoop.getBody()->getTerminator());
  auto allocaIndex = createMemrefAccessIndex(rewriter, readLoop);
  auto constantZero =
      rewriter.create<arith::ConstantIndexOp>(readLoop.getLoc(), 0);

  // Store 'transfer_read' results into the corresponding 'alloca'.
  for (size_t i = 0; i < allocaOps.size(); i++) {
    memref::AllocaOp allocaOp = allocaOps[i];
    auto readOp = cast<vector::TransferReadOp>(
        mapping.lookup<Operation *>(target.readOps[i]));

    SmallVector<Value> indices = {allocaIndex};
    indices.append(allocaOp.getType().getShape().size() - 1, constantZero);
    rewriter.create<vector::TransferWriteOp>(readOp.getLoc(), readOp, allocaOp,
                                             indices);
  }

  LDBG() << "Read loop: \n" << readLoop << "\n";
  rewriter.setInsertionPointAfter(readLoop);
}

/// Sets up the write loop for the transfer write operation.
static void setupWriteLoop(IRRewriter &rewriter, const FissionTarget &target,
                           ArrayRef<memref::AllocaOp> allocaOps) {
  // Create a copy of the original loop, where 'transfer_read's are replaced
  // with reads from the corresponding 'alloca'.
  IRMapping mapping;
  auto writeLoop = cast<scf::ForOp>(rewriter.clone(*target.parent, mapping));

  rewriter.setInsertionPointToStart(writeLoop.getBody());
  auto allocaIndex = createMemrefAccessIndex(rewriter, writeLoop);
  auto constantZero =
      rewriter.create<arith::ConstantIndexOp>(writeLoop.getLoc(), 0);
  for (size_t i = 0; i < allocaOps.size(); i++) {
    memref::AllocaOp allocaOp = allocaOps[i];
    auto readOp = cast<vector::TransferReadOp>(
        mapping.lookup<Operation *>(target.readOps[i]));

    rewriter.setInsertionPointAfter(readOp);
    SmallVector<Value> indices = {allocaIndex};
    indices.append(allocaOp.getType().getShape().size() - 1, constantZero);
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        readOp, readOp.getVectorType(), allocaOp, indices, readOp.getPadding());
  }

  LDBG() << "Write loop: \n" << writeLoop << "\n";
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
                                            const FissionTarget &target) {
  LDBG() << "Splitting transfer ops from control flow: \n"
         << "For Op: " << target.parent << "\n";

  rewriter.setInsertionPoint(target.parent);
  SmallVector<memref::AllocaOp> allocaOps;
  for (auto readOp : target.readOps) {
    allocaOps.push_back(createAlloca(rewriter, readOp, target.parent));
  }

  setupReadLoop(rewriter, target, allocaOps);
  setupWriteLoop(rewriter, target, allocaOps);

  rewriter.eraseOp(target.parent);
}

/// Populates a FissionTarget from a scf::ForOp by checking if it contains
/// transfer read and write operations that can be legally fissioned.
static FailureOr<FissionTarget> populateFissionTarget(scf::ForOp forOp) {
  SmallVector<vector::TransferReadOp> readOps;
  SmallVector<vector::TransferWriteOp> writeOps;
  for (Operation &op : forOp.getOps()) {
    if (auto readOp = dyn_cast<vector::TransferReadOp>(&op)) {
      if (!hasGlobalMemoryAddressSpace(
              cast<MemRefType>(readOp.getBase().getType()))) {
        return failure();
      }
      readOps.push_back(readOp);
    } else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(&op)) {
      if (hasGlobalMemoryAddressSpace(
              cast<MemRefType>(writeOp.getBase().getType()))) {
        return failure();
      }
      writeOps.push_back(writeOp);
    } else if (!mlir::isPure(&op)) {
      // Only the read/write ops may have side effects, since we assume we can
      // re-order/erase the other ops freely.
      return failure();
    }
  }

  if (readOps.empty() || writeOps.empty() ||
      readOps.size() != writeOps.size()) {
    return failure();
  }
  return FissionTarget{forOp, readOps, writeOps};
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
      splitTransferOpsFromControlFlow(rewriter, target);
    }

    // Cleanup dead ops.
    (void)applyPatternsGreedily(funcOp, {});
  }
};

} // namespace mlir::iree_compiler
