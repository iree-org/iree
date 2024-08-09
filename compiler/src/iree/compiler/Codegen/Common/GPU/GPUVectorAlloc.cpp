// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUVECTORALLOCPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

// For optimal performance we always want to copy 128 bits.
constexpr int copyVectorNumBits = 128;

/// Filter to decide which contraction ops need allocations.
static bool contractOpFilter(Operation *op) {
  auto contractOp = dyn_cast<vector::ContractionOp>(op);
  if (!contractOp) {
    return false;
  }
  SmallVector<unsigned> dims;
  for (auto [idx, type] : llvm::enumerate(contractOp.getIteratorTypesArray())) {
    if (type == vector::IteratorType::parallel) {
      dims.push_back(idx);
    }
  }
  SmallVector<int64_t> shapes;
  contractOp.getIterationBounds(shapes);
  // Don't promote vector*matrix kind of case.
  int numNonUnitParallelLoop = 0;
  for (unsigned parallelDim : dims) {
    if (shapes[parallelDim] != 1) {
      numNonUnitParallelLoop++;
    }
  }
  // TODO: Relax this constraint.
  return numNonUnitParallelLoop > 1 && dims.size() >= 2 && dims.size() <= 3;
}

// Allocates a tensor to copy the vector into a la bufferization.alloc_tensor.
// This allocation is always static as vectors are currently always static
// where this is used.
static FailureOr<Value> allocateTensorForVector(OpBuilder &b, Location loc,
                                                Value vector) {
  VectorType vectorType = llvm::cast<VectorType>(vector.getType());
  if (vectorType.isScalable()) {
    return failure();
  }

  Attribute sharedMemoryAddrSpace = gpu::AddressSpaceAttr::get(
      b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());

  RankedTensorType tensorType =
      RankedTensorType::get(vectorType.getShape(), vectorType.getElementType(),
                            sharedMemoryAddrSpace);
  // Vectors are always statically shaped.
  auto allocTensorOp = b.create<bufferization::AllocTensorOp>(
      loc, tensorType, ValueRange{}, Value());
  allocTensorOp.setMemorySpaceAttr(sharedMemoryAddrSpace);

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  Value copied = b.create<vector::TransferWriteOp>(loc, vector, allocTensorOp,
                                                   indices, inBounds)
                     .getResult();
  // Create a marker for bufferization to keep this tensor in place. This
  // prevents read/write forwarding of the transfers used to do the copy.
  return b
      .create<bufferization::MaterializeInDestinationOp>(copied.getLoc(),
                                                         copied, copied)
      ->getResult(0);
}

static Value readVectorFromTensor(OpBuilder &b, VectorType vectorType,
                                  Value tensor) {
  Value c0 = b.create<arith::ConstantIndexOp>(tensor.getLoc(), 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  return b
      .create<vector::TransferReadOp>(tensor.getLoc(), vectorType, tensor,
                                      indices, inBounds)
      .getResult();
}

struct GPUVectorAllocPass final
    : impl::GPUVectorAllocPassBase<GPUVectorAllocPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    SmallVector<IREE::VectorExt::ToLayoutOp> opsToPromote;
    funcOp.walk([&](IREE::VectorExt::ToLayoutOp op) {
      if (op->hasAttr("shared_memory_conversion")) {
        opsToPromote.push_back(op);
      }
    });

    for (IREE::VectorExt::ToLayoutOp op : opsToPromote) {
      OpBuilder builder(op);

      // HACK: Until proper barrier placement is handled later we have to
      // synchronize explicitly in this pass.

      // Synchronize before the write to shared memory to avoid stepping over
      // reads in the previous iteration of a loop.
      builder.create<gpu::BarrierOp>(op->getLoc());

      // Promote both of the input operands, excluding the accumulator.
      OpOperand &operand = op.getInputMutable();
      FailureOr<Value> ret =
          allocateTensorForVector(builder, op->getLoc(), operand.get());
      if (failed(ret)) {
        return signalPassFailure();
      }

      // Synchronize after the write to shared memory before we read from it.
      builder.create<gpu::BarrierOp>(op->getLoc());

      VectorType inputTy = cast<VectorType>(op.getType());
      Value read = readVectorFromTensor(builder, inputTy, *ret);
      operand.set(read);

      // Remove the shared_memory_conversion attribute from the to_layout
      // operation.
      op->removeAttr("shared_memory_conversion");
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
