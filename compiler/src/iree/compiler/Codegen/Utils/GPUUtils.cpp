// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/GPUUtils.h"

#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/SideEffectUtils.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// GPU processor IDs and sizes
//===----------------------------------------------------------------------===//

llvm::SmallVector<mlir::linalg::ProcInfo, 2> getGPUThreadIdsAndCounts(
    mlir::OpBuilder &builder, mlir::Location loc, unsigned numDims,
    llvm::ArrayRef<int64_t> workgroupSize) {
  assert(numDims <= kNumGPUDims);
  llvm::SmallVector<mlir::linalg::ProcInfo, 2> procInfo(numDims);
  std::array<gpu::Dimension, kNumGPUDims> dimAttr{
      gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
  mlir::Type indexType = builder.getIndexType();
  for (unsigned i = 0; i < numDims; ++i) {
    procInfo[numDims - 1 - i] = {
        builder.create<mlir::gpu::ThreadIdOp>(loc, indexType, dimAttr[i]),
        builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(workgroupSize[i])),
        linalg::DistributionMethod::Cyclic};
  }
  return procInfo;
}

llvm::SmallVector<mlir::linalg::ProcInfo, 2> getSubgroupIdsAndCounts(
    mlir::OpBuilder &builder, mlir::Location loc, unsigned warpSize,
    unsigned numDims, llvm::ArrayRef<int64_t> numSubgroups) {
  assert(numDims <= kNumGPUDims);
  llvm::SmallVector<mlir::linalg::ProcInfo, 2> procInfo(numDims);
  std::array<gpu::Dimension, kNumGPUDims> dimAttr{
      gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
  mlir::Type indexType = builder.getIndexType();
  for (unsigned i = 0; i < numDims; ++i) {
    mlir::Value subgroupId =
        builder.create<mlir::gpu::ThreadIdOp>(loc, indexType, dimAttr[i]);
    if (i == 0) {
      mlir::AffineExpr d0 = builder.getAffineDimExpr(0);
      subgroupId = mlir::makeComposedAffineApply(
          builder, loc, d0.floorDiv(builder.getAffineConstantExpr(warpSize)),
          {subgroupId});
    }
    procInfo[numDims - 1 - i] = {
        subgroupId,
        builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(numSubgroups[i])),
        linalg::DistributionMethod::Cyclic};
  }
  return procInfo;
}

std::array<int64_t, 3> getWorkgroupSize(mlir::func::FuncOp funcOp) {
  std::array<int64_t, 3> workgroupSize;
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp =
      mlir::iree_compiler::getEntryPoint(funcOp);
  llvm::Optional<mlir::ArrayAttr> workgroupSizeAttr =
      exportOp->getWorkgroupSize();
  assert(workgroupSizeAttr.has_value());
  for (auto it : llvm::enumerate(workgroupSizeAttr.value())) {
    workgroupSize[it.index()] =
        it.value().cast<mlir::IntegerAttr>().getValue().getZExtValue();
  }
  return workgroupSize;
}

//===----------------------------------------------------------------------===//
// GPU vectorization
//===----------------------------------------------------------------------===//

bool canPerformVectorAccessUsingAllThreads(ArrayRef<int64_t> shape,
                                           int64_t threadCount,
                                           int64_t vectorSize) {
  // Verify that each dimension of the shape can be distributed on the
  // threads
  int64_t threadsAvailable = threadCount;
  for (auto &dim : llvm::enumerate(llvm::reverse(shape))) {
    int64_t numElementPerThread = dim.index() == 0 ? vectorSize : 1;
    int64_t numThreads = dim.value() / numElementPerThread;
    if (numThreads == 0) return false;
    if (numThreads > threadsAvailable) {
      // If there are no enough remaining threads to distribute the current
      // dimension, try to use all remaining threads. But we still need to make
      // sure all work can be distributed to these threads evenly.
      if (numThreads % threadsAvailable != 0) return false;
      numThreads = threadsAvailable;
    }
    if (threadsAvailable % numThreads != 0) return false;
    threadsAvailable = threadsAvailable / numThreads;
    if (threadsAvailable == 1) break;
  }
  return threadsAvailable == 1;
}

//===----------------------------------------------------------------------===//
// GPU workgroup memory
//===----------------------------------------------------------------------===//

Optional<Value> allocateWorkgroupMemory(OpBuilder &builder,
                                        memref::SubViewOp subview,
                                        ArrayRef<Value> sizeBounds,
                                        DataLayout &) {
  OpBuilder::InsertionGuard guard(builder);

  func::FuncOp funcOp = subview->getParentOfType<func::FuncOp>();
  if (!funcOp) return llvm::None;

  // The subview size bounds are expected to be constant; they specify the shape
  // of the allocation.
  SmallVector<int64_t, 2> shape;
  for (Value bound : sizeBounds) {
    APInt value;
    if (!matchPattern(bound, m_ConstantInt(&value))) return llvm::None;
    shape.push_back(value.getSExtValue());
  }

  builder.setInsertionPoint(&funcOp.front(), funcOp.front().begin());
  auto type = MemRefType::get(shape, subview.getType().getElementType(), {},
                              gpu::GPUDialect::getWorkgroupAddressSpace());
  Value buffer = builder.create<memref::AllocOp>(funcOp.getLoc(), type);
  return buffer;
}

LogicalResult deallocateWorkgroupMemory(OpBuilder &, Value /*buffer*/) {
  return success();
}

LogicalResult copyToWorkgroupMemory(OpBuilder &b, Value src, Value dst) {
  Operation *copyOp = b.create<memref::CopyOp>(src.getLoc(), src, dst);
  setMarker(copyOp, getCopyToWorkgroupMemoryMarker());
  return success();
}

static bool propagateCopyDestIntoProducerFill(memref::CopyOp copyOp) {
  // Look for a fill Op writing into the copyOp source.
  Operation *prevOp = copyOp->getPrevNode();
  while (prevOp) {
    if (isMemoryEffectFree(prevOp)) {
      prevOp = prevOp->getPrevNode();
      continue;
    }

    auto fillOp = dyn_cast<linalg::FillOp>(prevOp);
    if (!fillOp) break;
    if (fillOp.output() != copyOp.getSource()) break;
    // Move the fillOp and change the destination to the copy destination.
    fillOp->moveBefore(copyOp);
    fillOp.getOutputsMutable().assign(copyOp.getTarget());
    return true;
  }
  return false;
}

// Split input/output operand from copy from shared memory into a separate
// input.
static void insertInputValueIntoGeneric(Value source, linalg::GenericOp op) {
  SmallVector<Value> newOperands;
  SmallVector<AffineMap> maps;
  for (OpOperand *in : op.getDpsInputOperands()) {
    newOperands.push_back(in->get());
    maps.push_back(op.getMatchingIndexingMap(in));
  }
  newOperands.push_back(source);
  assert(op.getNumDpsInits() == 1);
  OpOperand *outOperand = op.getDpsInitOperand(0);
  maps.push_back(op.getMatchingIndexingMap(outOperand));
  maps.push_back(op.getMatchingIndexingMap(outOperand));
  Location loc = op.getLoc();
  SmallVector<StringRef> iterTypes(op.getNumLoops(),
                                   getParallelIteratorTypeName());
  OpBuilder builder(op);
  auto newOp = builder.create<linalg::GenericOp>(
      loc, newOperands, outOperand->get(), maps, iterTypes);
  newOp.getRegion().getBlocks().splice(newOp.getRegion().begin(),
                                       op.getRegion().getBlocks());

  Block &payload = newOp.getRegion().front();
  payload.addArgument(payload.getArguments().back().getType(), loc);
  setMarker(newOp, getCopyToWorkgroupMemoryMarker());
}

/// Propagate the shared memory copy into the consumer op if it's a fully
/// parallel linalg.generic.
static bool propagateCopySourceIntoConsumerGeneric(
    memref::CopyOp copyOp, SmallVector<Operation *> &toDelete) {
  // Look for a generic Op reading the copyOp target.
  Operation *nextOp = copyOp->getNextNode();
  while (nextOp) {
    if (isMemoryEffectFree(nextOp)) {
      nextOp = nextOp->getNextNode();
      continue;
    }
    auto consumer = dyn_cast<linalg::GenericOp>(nextOp);
    if (!consumer || consumer.getNumDpsInits() != 1 ||
        !consumer.getMatchingIndexingMap(consumer.getDpsInitOperand(0))
             .isIdentity())
      break;
    if (*consumer.getOutputs().begin() != copyOp.getTarget()) break;
    insertInputValueIntoGeneric(copyOp.getSource(), consumer);
    toDelete.push_back(consumer);
    return true;
  }
  return false;
}

/// This is needed because we are doing promotion to shared memory on buffers.
/// This is a fragile and temporary solution until we move to be able to do this
/// kind of transformations on tensors.
void propagateSharedMemoryCopy(func::FuncOp funcOp) {
  SmallVector<Operation *> toDelete;
  funcOp.walk([&toDelete](memref::CopyOp copyOp) {
    if (hasMarker(copyOp, getCopyToWorkgroupMemoryMarker())) {
      if (propagateCopyDestIntoProducerFill(copyOp) ||
          propagateCopySourceIntoConsumerGeneric(copyOp, toDelete))
        toDelete.push_back(copyOp.getOperation());
    }
  });
  for (Operation *op : toDelete) op->erase();
}

void insertBarriersAroundSharedMemoryCopy(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  // Insert barriers before and after copies to workgroup memory and skip
  // insert barriers between back to back copy to workgroup memory.
  funcOp.walk([&builder](Operation *copyOp) {
    if (hasMarker(copyOp, getCopyToWorkgroupMemoryMarker())) {
      Operation *prevOp = copyOp->getPrevNode();
      if (!prevOp || !hasMarker(prevOp, getCopyToWorkgroupMemoryMarker())) {
        builder.setInsertionPoint(copyOp);
        builder.create<gpu::BarrierOp>(copyOp->getLoc());
      }
      Operation *nextOp = copyOp->getNextNode();
      if (!nextOp || !hasMarker(nextOp, getCopyToWorkgroupMemoryMarker())) {
        builder.setInsertionPointAfter(copyOp);
        builder.create<gpu::BarrierOp>(copyOp->getLoc());
      }
    }
  });
}

}  // namespace iree_compiler
}  // namespace mlir
