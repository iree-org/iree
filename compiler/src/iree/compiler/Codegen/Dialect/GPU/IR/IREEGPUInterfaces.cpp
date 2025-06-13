// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::iree_compiler::IREE::GPU {

/// Helper to insert copy with the specified attr.
Value promoteValue(OpBuilder &builder, Location loc, Value v, Attribute attr) {
  auto tensorType = cast<RankedTensorType>(v.getType());
  SmallVector<OpFoldResult> mixedSizes = tensor::getMixedSizes(builder, loc, v);

  Value empty = builder.create<tensor::EmptyOp>(loc, mixedSizes,
                                                tensorType.getElementType());
  auto copy = builder.create<linalg::CopyOp>(loc, v, empty);
  setLoweringConfig(copy, attr);
  return copy.getResult(0);
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

  return promoteValue(builder, operand.getOwner()->getLoc(), operand.get(),
                      attr);
}

} // namespace mlir::iree_compiler::IREE::GPU

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.cpp.inc"
