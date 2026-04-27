// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
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
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUVECTORALLOCPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

// Off by default: swizzle is an opt-in optimization. When disabled the
// GPUVectorAlloc pass behaves identically to its original form (no
// SwizzleHintOp emitted, no flat-1D / expand_shape wrapping).
llvm::cl::opt<bool> clEnableVectorAllocSwizzle(
    "iree-codegen-gpu-enable-vector-alloc-swizzle",
    llvm::cl::desc("Enable XOR swizzle hint creation in GPUVectorAlloc to "
                   "eliminate LDS bank conflicts. Mutually exclusive with the "
                   "GPUReduceBankConflicts padding pass."),
    llvm::cl::init(false));

namespace {

// Allocates a tensor to copy the vector into a la bufferization.alloc_tensor.
// This allocation is always static as vectors are currently always static
// where this is used. When |swizzle| is provided, wraps the allocation with
// a SwizzleHintOp; FlattenSwizzleHintAllocsPass later flattens the multi-D
// alloc + hint into the flat-1D + expand_shape form that ResolveSwizzleHints
// expects.
static FailureOr<Value>
allocateTensorForVector(OpBuilder &b, Location loc, Value vector,
                        std::optional<IREE::Codegen::XORShuffleAttr> swizzle) {
  VectorType vectorType = cast<VectorType>(vector.getType());
  if (vectorType.isScalable()) {
    return failure();
  }

  Attribute sharedMemoryAddrSpace = gpu::AddressSpaceAttr::get(
      b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());

  RankedTensorType tensorType =
      RankedTensorType::get(vectorType.getShape(), vectorType.getElementType(),
                            sharedMemoryAddrSpace);
  auto allocTensorOp = bufferization::AllocTensorOp::create(
      b, loc, tensorType, ValueRange{}, Value());
  allocTensorOp.setMemorySpaceAttr(sharedMemoryAddrSpace);

  Value dest = allocTensorOp;
  if (swizzle) {
    // Let FlattenSwizzleHintAllocsPass handle the flat-1D + expand_shape
    // rewrite post-bufferization. Creating the hint on the multi-D tensor
    // keeps operand and result types identical, so bufferization produces
    // matching memref types on both sides.
    dest = IREE::Codegen::SwizzleHintOp::create(b, loc, tensorType,
                                                allocTensorOp, *swizzle);
  }

  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  Value copied =
      vector::TransferWriteOp::create(b, loc, vector, dest, indices, inBounds)
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

/// Compute XOR swizzle parameters from the writer/reader layouts.
///
/// All target-specific numbers are parameters (numBanks comes from the GPU
/// target attribute; bank byte width is an IREE-wide convention). No
/// per-shape hardcoding: accessWidth is derived from the propagated
/// NestedLayoutAttr element tiles, innerDim comes from the vector type.
static std::optional<IREE::Codegen::XORShuffleAttr>
computeSwizzleFromLayouts(MLIRContext *ctx, VectorType vectorType,
                          IREE::VectorExt::VectorLayoutInterface inputLayout,
                          IREE::VectorExt::VectorLayoutInterface outputLayout,
                          int64_t numBanks) {
  if (vectorType.getRank() == 0) {
    return std::nullopt;
  }

  auto inNested = dyn_cast<IREE::VectorExt::NestedLayoutAttr>(inputLayout);
  auto outNested = dyn_cast<IREE::VectorExt::NestedLayoutAttr>(outputLayout);
  if (!inNested || !outNested) {
    return std::nullopt;
  }
  if (inNested.getElementTile().empty() || outNested.getElementTile().empty()) {
    return std::nullopt;
  }

  int64_t innerDim = vectorType.getShape().back();
  int64_t elemBytes = (vectorType.getElementTypeBitWidth() + 7) / 8;

  int64_t writerAccess = inNested.getElementTile().back();
  int64_t readerAccess = outNested.getElementTile().back();
  int64_t accessWidth = std::max(writerAccess, readerAccess);
  if (accessWidth == 0 || innerDim % accessWidth != 0) {
    return std::nullopt;
  }

  // When an element is at least a full bank wide (elemBytes >= bankBytes,
  // i.e. f32/f64) and the reader is narrower than the writer (typical MFMA:
  // vector<k> stores, scalar reads), picking accessWidth = max(w, r) yields
  // an XOR step of `accessWidth * elemBytes` bytes that may not be coprime
  // with the bank-cycle byte count. The number of unique bank positions the
  // step produces is `numBanks / gcd(step_banks, numBanks)`. If this is
  // less than the number of threads fighting for the same column, reads
  // conflict.
  //
  // Capping accessWidth to 2 sets step_banks = 2 (since elemBytes >=
  // bankBytes implies each element spans at least one bank). For numBanks
  // that are a power of 2 (typical: 32), gcd(2, numBanks) = 2, giving
  // numBanks/2 unique positions — enough for a half-wave bijection on
  // layouts that have up to numBanks/2 row-threads per column-thread.
  //
  // This is the minimum cap that eliminates the common-case 2-way conflict;
  // going lower (accessWidth=1) produces even fewer unique positions due
  // to XOR output-width truncation.
  if (elemBytes >= kSharedMemoryBankWidthBytes && readerAccess < writerAccess &&
      accessWidth > 2 && innerDim % 2 == 0) {
    accessWidth = 2;
  }

  int64_t maxPhase = innerDim / accessWidth;
  if (maxPhase <= 1) {
    return std::nullopt;
  }

  // If the row stride (in banks) is coprime with numBanks, consecutive
  // rows already land on distinct banks — XOR swizzle wouldn't add value.
  int64_t rowBytes = innerDim * elemBytes;
  int64_t rowStrideBanks = (rowBytes / kSharedMemoryBankWidthBytes) % numBanks;
  if (rowStrideBanks != 0 && std::gcd(rowStrideBanks, numBanks) == 1) {
    return std::nullopt;
  }

  return IREE::Codegen::XORShuffleAttr::get(ctx, innerDim, accessWidth,
                                            innerDim, /*per_phase=*/1);
}

/// Materialize shared memory for all to_layout ops marked with
/// shared_memory_conversion. Clears the attribute after materialization.
static LogicalResult materializeSharedMemoryConversions(
    FunctionOpInterface funcOp,
    const llvm::MapVector<Value, IREE::VectorExt::VectorLayoutInterface>
        &layouts) {
  SmallVector<IREE::VectorExt::ToLayoutOp> opsToPromote;
  funcOp.walk([&](IREE::VectorExt::ToLayoutOp op) {
    if (op.getSharedMemoryConversion()) {
      opsToPromote.push_back(op);
    }
  });

  // Query the LDS bank count from the GPU target attribute. Fall back to
  // the common value 32 if the target doesn't expose it (disables the
  // bank-alias bail-out but keeps swizzle functional).
  int64_t numBanks = 32;
  if (clEnableVectorAllocSwizzle) {
    if (IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp)) {
      if (auto bc = target.getWgp().getWorkgroupMemoryBankCount()) {
        numBanks = *bc;
      }
    }
  }

  OpBuilder builder(funcOp);
  for (IREE::VectorExt::ToLayoutOp op : opsToPromote) {
    // HACK: Until proper barrier placement is handled later we have to
    // synchronize explicitly in this pass.

    // Synchronize before the write to shared memory to avoid stepping over
    // reads in the previous iteration of a loop. We set this barrier
    // at the start of this block.
    builder.setInsertionPointToStart(op->getBlock());
    gpu::BarrierOp::create(builder, op->getLoc(), gpu::AddressSpace::Workgroup);

    builder.setInsertionPoint(op);
    OpOperand &operand = op.getInputMutable();
    // Compute XOR swizzle from input/output layouts to eliminate bank
    // conflicts. The input layout (writer) comes from layout propagation;
    // the output layout (reader) is on the to_layout op itself.
    std::optional<IREE::Codegen::XORShuffleAttr> swizzle;
    if (clEnableVectorAllocSwizzle) {
      auto inputLayout = layouts.lookup(op.getInput());
      auto outputLayout = layouts.lookup(op.getResult());
      // If the input value doesn't have a propagated layout, use the
      // to_layout's own layout as the output and treat the input as having
      // the same element tile (conservative: accessWidth = reader's width).
      if (!inputLayout && outputLayout) {
        inputLayout = outputLayout;
      }
      if (inputLayout && outputLayout) {
        swizzle = computeSwizzleFromLayouts(
            funcOp.getContext(), cast<VectorType>(op.getType()), inputLayout,
            outputLayout, numBanks);
      }
    }

    FailureOr<Value> ret =
        allocateTensorForVector(builder, op->getLoc(), operand.get(), swizzle);
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
      populateVectorLayoutCanonicalizations(patterns);
      walkAndApplyPatterns(funcOp, std::move(patterns));
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
    if (failed(materializeSharedMemoryConversions(funcOp, layouts))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
