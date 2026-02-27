// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUVECTORALLOCPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

/// Convert stretching broadcasts (broadcasting a non-unit dim from 1) into
/// broadcast + transpose so the layout analysis can handle them.
/// Copied from LLVMGPUVectorDistribute.cpp::RemoveUnitDimStrechingBroadcast.
struct RemoveStretchingBroadcast final : OpRewritePattern<vector::BroadcastOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::BroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    SetVector<int64_t> stretchedDims = broadcastOp.computeBroadcastedUnitDims();
    if (stretchedDims.empty()) {
      return failure();
    }

    VectorType srcTy = cast<VectorType>(broadcastOp.getSource().getType());
    VectorType dstTy = broadcastOp.getResultVectorType();
    int64_t numLeadingBroadcastDims = dstTy.getRank() - srcTy.getRank();
    SmallVector<int64_t> broadcastedUnitDims;
    for (auto i : llvm::seq<int64_t>(numLeadingBroadcastDims)) {
      if (dstTy.getShape()[i] == 1) {
        broadcastedUnitDims.push_back(i);
      }
    }
    if (stretchedDims.size() > broadcastedUnitDims.size()) {
      return failure();
    }

    auto perm = llvm::to_vector(llvm::seq<int64_t>(dstTy.getRank()));
    for (auto [stretchedDim, broadcastedUnitDim] :
         llvm::zip(stretchedDims.getArrayRef(), broadcastedUnitDims)) {
      std::swap(perm[stretchedDim], perm[broadcastedUnitDim]);
    }
    VectorType permutedBroadcastTy = VectorType::get(
        applyPermutation(dstTy.getShape(), perm), dstTy.getElementType());
    Value permutedBroadcast = vector::BroadcastOp::create(
        rewriter, broadcastOp.getLoc(), permutedBroadcastTy,
        broadcastOp.getSource());
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(broadcastOp,
                                                     permutedBroadcast, perm);
    return success();
  }
};

// Allocates a tensor to copy the vector into a la bufferization.alloc_tensor.
// This allocation is always static as vectors are currently always static
// where this is used.
static FailureOr<Value> allocateTensorForVector(OpBuilder &b, Location loc,
                                                Value vector) {
  VectorType vectorType = cast<VectorType>(vector.getType());
  if (vectorType.isScalable()) {
    return failure();
  }

  Attribute sharedMemoryAddrSpace = gpu::AddressSpaceAttr::get(
      b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());

  RankedTensorType tensorType =
      RankedTensorType::get(vectorType.getShape(), vectorType.getElementType(),
                            sharedMemoryAddrSpace);
  // Vectors are always statically shaped.
  auto allocTensorOp = bufferization::AllocTensorOp::create(
      b, loc, tensorType, ValueRange{}, Value());
  allocTensorOp.setMemorySpaceAttr(sharedMemoryAddrSpace);

  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  Value copied = vector::TransferWriteOp::create(b, loc, vector, allocTensorOp,
                                                 indices, inBounds)
                     .getResult();
  return copied;
}

static Value readVectorFromTensor(OpBuilder &b, VectorType vectorType,
                                  Value tensor) {
  Value c0 = arith::ConstantIndexOp::create(b, tensor.getLoc(), 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  return vector::TransferReadOp::create(b, tensor.getLoc(), vectorType, tensor,
                                        indices, /*padding=*/std::nullopt,
                                        inBounds)
      .getResult();
}

/// Materialize shared memory for all to_layout ops marked with
/// shared_memory_conversion. Clears the attribute after materialization.
static LogicalResult
materializeSharedMemoryConversions(FunctionOpInterface funcOp) {
  SmallVector<IREE::VectorExt::ToLayoutOp> opsToPromote;
  funcOp.walk([&](IREE::VectorExt::ToLayoutOp op) {
    if (op.getSharedMemoryConversion()) {
      opsToPromote.push_back(op);
    }
  });

  for (IREE::VectorExt::ToLayoutOp op : opsToPromote) {
    OpBuilder builder(op);

    // HACK: Until proper barrier placement is handled later we have to
    // synchronize explicitly in this pass.

    // Synchronize before the write to shared memory to avoid stepping over
    // reads in the previous iteration of a loop. We set this barrier
    // at the start of this block.
    builder.setInsertionPointToStart(op->getBlock());
    gpu::BarrierOp::create(builder, op->getLoc(), gpu::AddressSpace::Workgroup);

    // Promote both of the input operands, excluding the accumulator.
    builder.setInsertionPoint(op);
    OpOperand &operand = op.getInputMutable();
    FailureOr<Value> ret =
        allocateTensorForVector(builder, op->getLoc(), operand.get());
    if (failed(ret)) {
      return failure();
    }

    // Synchronize after the write to shared memory before we read from it.
    auto synced =
        IREE::GPU::ValueBarrierOp::create(builder, op->getLoc(), *ret);

    VectorType inputTy = cast<VectorType>(op.getType());
    Value read = readVectorFromTensor(builder, inputTy, synced.getResult(0));
    operand.set(read);

    // Remove the shared_memory_conversion attribute from the to_layout
    // operation.
    op.setSharedMemoryConversion(false);
  }
  return success();
}

struct GPUVectorAllocPass final
    : impl::GPUVectorAllocPassBase<GPUVectorAllocPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    // Remove stretching broadcasts before layout analysis — the analysis
    // asserts that broadcasts don't stretch.
    {
      RewritePatternSet patterns(funcOp.getContext());
      patterns.add<RemoveStretchingBroadcast>(funcOp.getContext());
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Run layout analysis to find additional conflict points.
    // The analysis sees the materialized shared memory roundtrips and
    // only detects genuinely new conflicts.
    llvm::MapVector<Value, IREE::VectorExt::VectorLayoutInterface> layouts;
    propagateVectorLayoutInfo(funcOp, layouts);

    // Mark newly-inserted to_layout ops where input/output layouts don't
    // match — these are genuine conflicts needing shared memory.
    funcOp.walk([&](IREE::VectorExt::ToLayoutOp op) {
      auto inputLayout = layouts.lookup(op.getInput());
      auto outputLayout = layouts.lookup(op.getResult());
      if (inputLayout && outputLayout &&
          inputLayout.needsSharedMemoryForConversion(outputLayout)) {
        op.setSharedMemoryConversion(true);
      }
    });

    // Phase 3: Materialize any newly-found conflicts.
    if (failed(materializeSharedMemoryConversions(funcOp))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
