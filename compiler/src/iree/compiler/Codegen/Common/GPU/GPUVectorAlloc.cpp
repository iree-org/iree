// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
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
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-gpu-vector-alloc"

namespace mlir::iree_compiler {

// For optimal performance we always want to copy 128 bits.
static constexpr int copyVectorNumBits = 128;

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

namespace {

struct GPUVectorAllocPass : public GPUVectorAllocBase<GPUVectorAllocPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
    registry.insert<gpu::GPUDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();

    SmallVector<vector::ContractionOp> opsToPromote;
    funcOp.walk([&](vector::ContractionOp op) {
      // Today we only do promotion for certain contractions.
      if (contractOpFilter(op))
        opsToPromote.push_back(op);
    });
    for (vector::ContractionOp contractOp : opsToPromote) {
      OpBuilder builder(contractOp);
      // Promote both of the input operands, excluding the accumulator.
      OpOperand &lhs = contractOp.getLhsMutable();
      FailureOr<Value> lhsRet =
          allocateTensorForVector(builder, contractOp->getLoc(), lhs.get());
      if (failed(lhsRet)) {
        return signalPassFailure();
      }

      OpOperand &rhs = contractOp.getRhsMutable();
      FailureOr<Value> rhsRet =
          allocateTensorForVector(builder, contractOp->getLoc(), rhs.get());
      if (failed(rhsRet)) {
        return signalPassFailure();
      }

      // HACK: Until proper barrier placement is handled later we have to
      // synchronize here.
      builder.create<gpu::BarrierOp>(contractOp->getLoc());

      Value lhsVec =
          readVectorFromTensor(builder, contractOp.getLhsType(), *lhsRet);
      Value rhsVec =
          readVectorFromTensor(builder, contractOp.getRhsType(), *rhsRet);
      lhs.set(lhsVec);
      rhs.set(rhsVec);
    }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUVectorAlloc() {
  return std::make_unique<GPUVectorAllocPass>();
}

} // namespace mlir::iree_compiler
