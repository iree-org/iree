//===-- LinalgExtBufferization.cpp - Linalg Extension bufferization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/LinalgExtBufferization.h"

#include <mlir/IR/BuiltinOps.h>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

/// Return the destinations that an InParallelOp is inserting into. One per
/// ParallelInsertSliceOp.
static SmallVector<OpOperand *> getInsertionDest(InParallelOp inParallelOp) {
  Operation *terminator = inParallelOp.region().front().getTerminator();
  auto performConcOp = dyn_cast<PerformConcurrentlyOp>(terminator);
  assert(performConcOp && "expected PerformConcurrentlyOp as terminator");

  SmallVector<OpOperand *> result;
  performConcOp.walk([&](ParallelInsertSliceOp insertOp) {
    result.push_back(&insertOp->getOpOperand(1) /*dest*/);
  });

  return result;
}

namespace mlir {

using bufferization::BufferizableOpInterface;
using bufferization::BufferizationState;
using bufferization::BufferRelation;
using bufferization::getMemRefType;
using bufferization::replaceOpWithBufferizedValues;
using bufferization::replaceOpWithNewBufferizedOp;
using tensor::ExtractSliceOp;

namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

/// Bufferization of InParallelOp. This also bufferizes the terminator of the
/// region. There are op interfaces for the terminators (PerformConcurrentlyOp
/// and ParallelInsertSliceOp), but these are only used during analysis. Not
/// for bufferization.
struct InParallelOpInterface
    : public BufferizableOpInterface::ExternalModel<InParallelOpInterface,
                                                    InParallelOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(
      Operation *op, OpResult opResult, const BufferizationState &state) const {
    // Get OpOperand (dest) from corresponding ParallelInsertSliceOp.
    auto inParallelOp = cast<InParallelOp>(op);
    return {getInsertionDest(inParallelOp)[opResult.getResultNumber()]};
  }

  bool isMemoryWrite(Operation *op, OpResult opResult,
                     const BufferizationState &state) const {
    // This op is a memory write. Stop lookup here to avoid finding false
    // conflicts involving this op and one of the ops in the region. This is
    // similar to how scf.if ops are analyzed.
    return true;
  }

  bool isAllocationHoistingBarrier(Operation *op) const { return true; }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &b,
                          const BufferizationState &state) const {
    OpBuilder::InsertionGuard g(b);
    auto inParallelOp = cast<InParallelOp>(op);
    Block *body = &inParallelOp.region().front();
    Operation *oldTerminator = body->getTerminator();
    assert(isa<PerformConcurrentlyOp>(oldTerminator) &&
           "unexpected terminator");

    // Gather new results of the InParallelOp.
    SmallVector<Value> newResults;
    for (OpResult opResult : inParallelOp->getOpResults()) {
      SmallVector<OpOperand *> insertDestOperands =
          state.getAliasingOpOperand(opResult);
      assert(insertDestOperands.size() == 1 &&
             "expected exactly one aliasing OpOperand");
      // Insert copies right before the PerformConcurrentlyOp terminator. They
      // should not be inside terminator (which would be the default insertion
      // point).
      Value buffer = *state.getBuffer(
          b, *insertDestOperands.front(), /*forceInPlace=*/false,
          /*customCopyInsertionPoint=*/oldTerminator);
      newResults.push_back(buffer);
      Value destTensor = insertDestOperands.front()->get();

      // Replace all uses of the insert dest tensor inside the InParallelOp
      // with the result buffer.
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(body);
      Value toTensorOp =
          b.create<bufferization::ToTensorOp>(inParallelOp.getLoc(), buffer);
      for (OpOperand &use : destTensor.getUses())
        if (body->findAncestorOpInBlock(*use.getOwner()))
          // This is a use inside the InParallelOp.
          use.set(toTensorOp);
    }

    // Create new InParallelOp without any results.
    TypeRange newResultTypes;
    auto newInParallelOp = b.create<InParallelOp>(
        inParallelOp.getLoc(), newResultTypes, inParallelOp.num_threads());

    // Delete terminator.
    newInParallelOp.getBody()->getTerminator()->erase();

    // Move over block contents of the old op.
    b.mergeBlocks(inParallelOp.getBody(), newInParallelOp.getBody(),
                  {newInParallelOp.getBody()->getArgument(0)});

    // Bufferize terminator.
    auto performConcurrentlyOp =
        cast<PerformConcurrentlyOp>(newInParallelOp.getBody()->getTerminator());
    b.setInsertionPoint(performConcurrentlyOp);
    WalkResult walkResult =
        performConcurrentlyOp.walk([&](ParallelInsertSliceOp insertOp) {
          Location loc = insertOp.getLoc();
          Type srcType = getMemRefType(
              insertOp.source().getType().cast<RankedTensorType>(),
              state.getOptions());
          Type destType =
              getMemRefType(insertOp.dest().getType().cast<RankedTensorType>(),
                            state.getOptions());
          // ParallelInsertSliceOp bufferizes to a copy.
          auto srcMemref = b.create<bufferization::ToMemrefOp>(
              loc, srcType, insertOp.source());
          auto destMemref = b.create<bufferization::ToMemrefOp>(
              loc, destType, insertOp.dest());
          Value subview = b.create<memref::SubViewOp>(
              loc, destMemref, insertOp.getMixedOffsets(),
              insertOp.getMixedSizes(), insertOp.getMixedStrides());
          // This memcpy will fold away if everything bufferizes in-place.
          if (failed(createMemCpy(b, insertOp.getLoc(), srcMemref, subview,
                                  state.getOptions())))
            return WalkResult::interrupt();
          b.eraseOp(insertOp);
          return WalkResult::advance();
        });
    if (walkResult.wasInterrupted()) return failure();

    // Replace the op.
    replaceOpWithBufferizedValues(b, op, newResults);

    return success();
  }
};

/// Nothing to do for PerformConcurrentlyOp.
struct PerformConcurrentlyOpInterface
    : public BufferizableOpInterface::ExternalModel<
          PerformConcurrentlyOpInterface, PerformConcurrentlyOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &b,
                          const BufferizationState &state) const {
    llvm_unreachable("op does not have any tensor OpOperands / OpResults");
    return failure();
  }
};

/// Return true if the (ExtractSliceOp, ParallelInsertSliceOp) pair match (i.e.
/// equivalent operand / result and same offset/sizes/strides specification).
static bool areEquivalentExtractSliceOps(const BufferizationState &state,
                                         ExtractSliceOp st,
                                         ParallelInsertSliceOp sti) {
  if (!st || !sti) return false;
  if (st != sti &&
      !state.areEquivalentBufferizedValues(st.source(), sti.dest()))
    return false;
  if (!sameOffsetsSizesAndStrides(st, sti, isEqualConstantIntOrValue))
    return false;
  return true;
}

/// Return true if `value` is originating from an ExtractSliceOp that matches
/// the given InsertSliceOp.
static bool hasMatchingExtractSliceOp(const BufferizationState &state,
                                      Value value,
                                      ParallelInsertSliceOp insertOp) {
  auto condition = [&](Value val) {
    if (auto extractOp = val.getDefiningOp<ExtractSliceOp>())
      if (areEquivalentExtractSliceOps(state, extractOp, insertOp)) return true;
    return false;
  };

  return llvm::all_of(state.findValueInReverseUseDefChain(value, condition),
                      condition);
}

/// Analysis of ParallelInsertSliceOp.
struct ParallelInsertSliceOpInterface
    : public BufferizableOpInterface::ExternalModel<
          ParallelInsertSliceOpInterface, ParallelInsertSliceOp> {
  SmallVector<OpResult> getAliasingOpResult(
      Operation *op, OpOperand &opOperand,
      const BufferizationState &state) const {
    if (&opOperand != &op->getOpOperand(1) /*dest*/) return {};

    // ParallelInsertSliceOp itself has no results. Tensors are returned via
    // the parent op.
    auto inParallelOp = op->getParentOfType<InParallelOp>();
    assert(inParallelOp &&
           "could not find valid owner of parallel_insert_slice");

    // The i-th ParallelInsertSliceOp result is returned via the i-th OpResult
    // of the parent InParallelOp.
    Block *block = op->getBlock();
    unsigned int opIdx = 0;
    for (ParallelInsertSliceOp insertOp :
         block->getOps<ParallelInsertSliceOp>()) {
      if (insertOp.getOperation() == op) break;
      ++opIdx;
    }
    assert(opIdx < inParallelOp->getNumResults() &&
           "could not find op inside terminator op");

    return {inParallelOp->getResult(opIdx)};
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return &opOperand == &op->getOpOperand(1) /*dest*/;
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &b,
                          const BufferizationState &state) const {
    // Will be bufferized as part of InParallelOp.
    return failure();
  }

  // TODO: This is copied from TensorInterfaceImpl.cpp. Find a way to share
  // the code.
  bool isNotConflicting(Operation *op, OpOperand *uRead,
                        OpOperand *uConflictingWrite,
                        const BufferizationState &state) const {
    Operation *readingOp = uRead->getOwner();
    Operation *conflictingWritingOp = uConflictingWrite->getOwner();

    // Special rules for matching ExtractSliceOp/InsertSliceOp pairs. If
    // uRead is an InsertSliceOp...
    if (auto insertSliceOp = dyn_cast<ParallelInsertSliceOp>(readingOp)) {
      // As an example, consider the following IR.
      //
      // %0 = tensor.extract_slice %t[%a, %b][%c, %d][1, 1] {inplace = [true] }
      // %1 = linalg.fill %cst, %0 {inplace= [true] }
      // %2 = tensor.insert_slice %1 into %t[%a, %b][%c, %d][1, 1]
      //     {inplace= [true] }

      // TODO: Use insertSliceOp.getDestOpOperand etc. when available.
      if (uRead == &insertSliceOp->getOpOperand(1) /*dest*/ &&
          hasMatchingExtractSliceOp(state, uConflictingWrite->get(),
                                    insertSliceOp))
        // Case 1: The main insight is that InsertSliceOp reads only part of
        // the destination tensor. The overwritten area is not read. If
        // uConflictingWrite writes into exactly the memory location that is
        // being read by uRead, this is not a conflict.
        //
        // In the above example:
        // uRead             = OpOperand 1 (%t) of tensor.insert_slice
        // uConflictingWrite = OpOperand 1 (%0) of linalg.fill
        //
        // The read of %t does not conflict with the write of the FillOp
        // (same aliases!) because the area that the FillOp operates on is
        // exactly the one that is *not* read via %t.
        return true;

      if (uRead == &insertSliceOp->getOpOperand(0) /*source*/ &&
          uConflictingWrite == &insertSliceOp->getOpOperand(1) /*dest*/ &&
          hasMatchingExtractSliceOp(state, uRead->get(), insertSliceOp))
        // Case 2: The read of the source tensor and the write to the dest
        // tensor via an InsertSliceOp is not a conflict if the read is
        // reading exactly that part of an equivalent tensor that the
        // InsertSliceOp is writing.
        //
        // In the above example:
        // uRead             = OpOperand 0 (%1) of tensor.insert_slice
        // uConflictingWrite = OpOperand 1 (%t) of tensor.insert_slice
        return true;
    }

    // If uConflictingWrite is an InsertSliceOp...
    if (auto insertSliceOp =
            dyn_cast<ParallelInsertSliceOp>(conflictingWritingOp))
      // As an example, consider the following IR.
      //
      // %0 = tensor.extract_slice %t[%a, %b][%c, %d][1, 1] {inplace = [true] }
      // %1 = linalg.fill %cst, %0 {inplace= [true] }
      // %2 = tensor.insert_slice %1 into %t[%a, %b][%c, %d][1, 1]
      //     {inplace= [true] }
      // %3 = vector.transfer_read %1, %cst
      //
      // In the above example:
      // uRead             = OpOperand 0 (%1) of vector.transfer_read
      // uConflictingWrite = OpOperand 1 (%t) of tensor.insert_slice
      // lastWrite         = %1
      //
      // This is not a conflict because the InsertSliceOp overwrites the
      // memory segment of %1 with the exact same data. (Effectively, there
      // is no memory write here.)
      if (uConflictingWrite == &insertSliceOp->getOpOperand(1) /*dest*/ &&
          state.areEquivalentBufferizedValues(uRead->get(),
                                              insertSliceOp.source()) &&
          hasMatchingExtractSliceOp(state, insertSliceOp.source(),
                                    insertSliceOp))
        return true;

    return false;
  }
};
}  // namespace LinalgExt
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

void mlir::iree_compiler::IREE::LinalgExt::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<InParallelOp, InParallelOpInterface>();
  registry
      .addOpInterface<PerformConcurrentlyOp, PerformConcurrentlyOpInterface>();
  registry
      .addOpInterface<ParallelInsertSliceOp, ParallelInsertSliceOpInterface>();
}
