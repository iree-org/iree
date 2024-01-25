// Copyright 2022 The IREE Authors
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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-gpu-vector-alloc"

namespace mlir::iree_compiler {

// For optimal performance we always want to copy 128 bits
static constexpr int copyVectorNumBits = 128;

/// Filter to decide which contract ops need allocations.
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
  // HACK: Until proper barrier placement is handled later we have to
  // synchronize here.
  b.create<gpu::BarrierOp>(loc);
  return b
      .create<vector::TransferReadOp>(loc, vectorType, copied, indices,
                                      inBounds)
      .getResult();
}

namespace {

struct GPUVectorAllocPass : public GPUVectorAllocBase<GPUVectorAllocPass> {
private:
  GPUPromoteSharedMemPattern promoteSharedMemPattern =
      GPUPromoteSharedMemPattern::ContractionOpPattern;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();

    SmallVector<Operation *> opsToPromote;
    funcOp.walk([&](Operation *op) {
      // Today we only do promotion for contractions.
      if (contractOpFilter(op))
        opsToPromote.push_back(op);
    });
    for (Operation *op : opsToPromote) {
      OpBuilder builder(op);
      if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
        bufferization::BufferizationOptions options;
        // Promote all the input operands
        OpOperand &lhs = contractOp.getLhsMutable();
        FailureOr<Value> ret =
            allocateTensorForVector(builder, op->getLoc(), lhs.get());
        if (failed(ret)) {
          return signalPassFailure();
        }
        lhs.set(ret.value());

        OpOperand &rhs = contractOp.getLhsMutable();
        ret = allocateTensorForVector(builder, op->getLoc(), rhs.get());
        if (failed(ret)) {
          return signalPassFailure();
        }
        lhs.set(ret.value());
      }
    }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUVectorAlloc() {
  return std::make_unique<GPUVectorAllocPass>();
}

} // namespace mlir::iree_compiler
