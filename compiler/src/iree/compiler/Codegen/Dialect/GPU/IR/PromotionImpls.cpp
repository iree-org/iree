
// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "iree-codegen-promotion-utils"

namespace mlir::iree_compiler::IREE::GPU {

/// Helper to insert copy with the specified attr.
Value promoteValue(OpBuilder &builder, Location loc, Value v, Attribute attr) {
  auto tensorType = cast<RankedTensorType>(v.getType());
  SmallVector<OpFoldResult> mixedSizes = tensor::getMixedSizes(builder, loc, v);
  Value empty = tensor::EmptyOp::create(builder, loc, mixedSizes,
                                        tensorType.getElementType());
  auto copy = linalg::CopyOp::create(builder, loc, v, empty);
  setLoweringConfig(copy, attr);
  return copy.getResult(0);
}

// Helper to insert a swizzle hint op and flatten the associated alloc.
Value swizzlePromoteValue(OpBuilder &builder, Location loc, Value v,
                          Attribute attr, int64_t rowWidth,
                          int64_t accessWidth) {
  auto tensorType = cast<RankedTensorType>(v.getType());
  SmallVector<OpFoldResult> mixedSizes = tensor::getMixedSizes(builder, loc, v);
  if (!tensorType.hasStaticShape()) {
    LDBG() << "Cannot create swizzle_hint op for non static shapes";
    Value empty = tensor::EmptyOp::create(builder, loc, mixedSizes,
                                          tensorType.getElementType());
    auto copy = linalg::CopyOp::create(builder, loc, v, empty);
    if (attr) {
      setLoweringConfig(copy, attr);
    }
    return copy.getResult(0);
  }
  int64_t numElements = tensorType.getNumElements();
  Value empty = tensor::EmptyOp::create(builder, loc, {numElements},
                                        tensorType.getElementType());
  Value swizzled = IREE::Codegen::SwizzleHintOp::create(
      builder, loc, empty,
      IREE::Codegen::XORShuffleAttr::get(builder.getContext(), rowWidth,
                                         accessWidth, int64_t(), int64_t()));
  Value expanded = tensor::ExpandShapeOp::create(
      builder, loc, tensorType, swizzled,
      {llvm::to_vector(llvm::seq(tensorType.getRank()))});
  auto copy = linalg::CopyOp::create(builder, loc, v, expanded);
  if (attr) {
    setLoweringConfig(copy, attr);
  }
  return copy.getResult(0);
}

std::optional<Value> promotionImpl(OpBuilder &builder, OpOperand &operand,
                                   Attribute attr) {
  if (auto producer = operand.get().getDefiningOp<TilingInterface>()) {
    // Skip promotion of fills.
    if (isa<linalg::FillOp>(producer)) {
      return operand.get();
    }
    if (auto generic = dyn_cast<linalg::GenericOp>(&*producer)) {
      if (linalg::isaFillOpInterface(generic)) {
        return operand.get();
      }
    }

    // We only support thread tile size derivation of linalgOp and Im2colOp for
    // now.
    if (isa<linalg::LinalgOp, IREE::LinalgExt::Im2colOp>(
            producer.getOperation())) {
      setLoweringConfig(producer, attr);
      return operand.get();
    }
  }

  auto tensorType = dyn_cast<RankedTensorType>(operand.get().getType());
  if (!tensorType) {
    return operand.get();
  }
  return std::nullopt;
}

/// Inserts a `linalg.copy` directly before the given operation on the
/// specified operand, for example with operand index = 1:
///
///   %2 = linalg.matmul ins(%0, %1)
///
/// becomes
///
///   %empty = tensor.empty()
///   %copy = linalg.copy %1 to %empty {
///     lowering_config = #iree_gpu.{derived_thread_config|use_global_dma}}
///   linalg.matmul ins(%0, %copy)
///
/// If the producer is already a tilable op, the producer is just annotated with
/// the underlying attribute.
/// Additionally we can also promote results so in above example we will
/// generate for index = 2 :
///   %out_buffer = bufferization.alloc_tensor
///   %copy1 = linalg.copy %2 to %out_buffer
///   %copy2 = linalg.copy %copy1 to %empty {
///     lowering_config = #iree_gpu.derived_thread_config}
Value defaultPromotionImpl(OpBuilder &builder, OpOperand &operand,
                           Attribute attr) {
  std::optional<Value> promotedValue = promotionImpl(builder, operand, attr);
  if (promotedValue.has_value()) {
    return promotedValue.value();
  }
  return promoteValue(builder, operand.getOwner()->getLoc(), operand.get(),
                      attr);
}

/// Inserts a `linalg.copy` directly before the given operation on the
/// specified operand, similar to the defaultPromotionImpl.
/// The difference is this also assigns a `iree_codegen.swizzle_hint` op
/// to the generated `tensor.empty` op.
/// For example:
///
///   %2 = linalg.matmul ins(%0, %1)
///
/// becomes
///
///   %empty = tensor.empty()
///   %swizzle = iree_codegen.swizzle_hint %empty[...]
///   %copy = linalg.copy %1 to %swizzle {
///     lowering_config = #iree_gpu.{derived_thread_config|use_global_dma}}
///   linalg.matmul ins(%0, %copy)
Value swizzlePromotionImpl(OpBuilder &builder, OpOperand &operand,
                           Attribute attr, int64_t rowWidth,
                           int64_t accessWidth) {
  std::optional<Value> promotedValue = promotionImpl(builder, operand, attr);
  if (promotedValue.has_value()) {
    return promotedValue.value();
  }
  return swizzlePromoteValue(builder, operand.getOwner()->getLoc(),
                             operand.get(), attr, rowWidth, accessWidth);
}

/// Inserts a `linalg.copy` directly before the given operation on the
/// specified operand, and also inserts a buffer_resource_cast on the producing
/// dispatch input if possible.
Value cacheSwizzlePromotionImpl(OpBuilder &builder, OpOperand &operand,
                                Attribute attr) {
  Value promotedValue = defaultPromotionImpl(builder, operand, attr);
  OpOperand *bufferCastOperand = nullptr;
  // We only support looking through copies and Im2Col for now.
  if (auto producer =
          promotedValue.getDefiningOp<IREE::LinalgExt::Im2colOp>()) {
    bufferCastOperand = &producer.getInputMutable();

  } else if (auto producer = promotedValue.getDefiningOp<linalg::CopyOp>()) {
    bufferCastOperand = producer.getDpsInputOperand(0);
  } else {
    return promotedValue;
  }

  Value bufferCastValue = bufferCastOperand->get();

  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointAfterValue(bufferCastValue);
  auto tensorType = cast<RankedTensorType>(bufferCastValue.getType());
  if (tensorType.getRank() == 0) {
    // Skip rank 0 inputs, there is no point in cache swizzling these.
    return promotedValue;
  }

  Location loc = promotedValue.getLoc();
  // Use the size in bytes of the inner most dimension as the cache swizzle
  // value. This is a very rudimentary choice, but functions well enough as a
  // default.
  AffineExpr s0, s1;
  bindSymbols(builder.getContext(), s0, s1);
  Value dtype =
      arith::ConstantIndexOp::create(
          builder, loc, tensorType.getElementType().getIntOrFloatBitWidth())
          ->getResult(0);

  OpFoldResult dim = tensor::getMixedSize(builder, loc, bufferCastValue,
                                          tensorType.getRank() - 1);
  Value zero =
      getValueOrCreateConstantIntOp(builder, loc, builder.getIndexAttr(0));
  Value strideBytes = getValueOrCreateConstantIndexOp(
      builder, loc,
      affine::makeComposedFoldedAffineApply(builder, loc, (s0 * s1).ceilDiv(8),
                                            {dim, dtype}));
  Value strideBitsMod8 = getValueOrCreateConstantIntOp(
      builder, loc,
      affine::makeComposedFoldedAffineApply(builder, loc, (s0 * s1) % 8,
                                            {dim, dtype}));
  // If the stride in bits is not a multiple of 8, set the value to 0. This will
  // be ignored by cacheSwizzleStride.
  Value cmp = arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, strideBitsMod8, zero);
  Value cacheSwizzleVal =
      arith::SelectOp::create(builder, loc, cmp, strideBytes, zero).getResult();
  // Insert the resource cast optimistically. If the input is not castable
  // (e.g. another producer) later patterns will drop it anyway as it is treated
  // like a hint.
  auto resourceCast = IREE::GPU::BufferResourceCastOp::create(
      builder, loc, tensorType, bufferCastValue, cacheSwizzleVal);
  bufferCastOperand->assign(resourceCast);

  return promotedValue;
}

} // namespace mlir::iree_compiler::IREE::GPU
