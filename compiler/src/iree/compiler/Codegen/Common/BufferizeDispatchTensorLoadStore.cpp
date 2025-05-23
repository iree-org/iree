// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SubsetOpInterface.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_BUFFERIZEDISPATCHTENSORLOADSTOREPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

static Value getBufferizedSubspanOrSubview(
    RewriterBase &rewriter, Location loc,
    IREE::HAL::InterfaceBindingSubspanOp subspanOp, RankedTensorType tensorType,
    SmallVector<OpFoldResult> offsets, SmallVector<OpFoldResult> sizes,
    SmallVector<OpFoldResult> strides) {
  auto dispatchTensorType =
      cast<IREE::TensorExt::DispatchTensorType>(subspanOp->getResultTypes()[0]);
  SmallVector<Value> subspanDynamicSizes = subspanOp.getDynamicDims();
  Value subspanMemref = findOrCreateSubspanBuffer(rewriter, subspanOp);
  SmallVector<Value> dynamicSizes;
  std::tie(std::ignore, dynamicSizes) = decomposeMixedValues(sizes);
  if (equalTensorShape(tensorType, dynamicSizes, dispatchTensorType,
                       subspanDynamicSizes)) {
    return subspanMemref;
  }
  MemRefType subviewMemRefType = memref::SubViewOp::inferRankReducedResultType(
      tensorType.getShape(), cast<MemRefType>(subspanMemref.getType()), offsets,
      sizes, strides);
  return rewriter.create<memref::SubViewOp>(
      loc, subviewMemRefType, subspanMemref, offsets, sizes, strides);
}

static void
bufferizeDispatchTensorLoad(RewriterBase &rewriter,
                            IREE::TensorExt::DispatchTensorLoadOp loadOp) {
  Value source = loadOp.getSource();
  auto subspanOp = source.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  if (!subspanOp) {
    return;
  }
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(loadOp);
  Value sourceBuffer = getBufferizedSubspanOrSubview(
      rewriter, loadOp.getLoc(), subspanOp, loadOp.getType(),
      loadOp.getMixedOffsets(), loadOp.getMixedSizes(),
      loadOp.getMixedStrides());
  rewriter.replaceOpWithNewOp<IREE::Codegen::LoadFromBufferOp>(
      loadOp, loadOp.getType(), sourceBuffer);
}

static void
bufferizeDispatchTensorStore(RewriterBase &rewriter,
                             IREE::TensorExt::DispatchTensorStoreOp storeOp) {
  Value target = storeOp.getTarget();
  auto subspanOp = target.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  if (!subspanOp) {
    return;
  }
  // For the store_to_buffer op, generate any subviews as early as possible in
  // the IR. This opens more opportunities for using the store_to_buffer op's
  // SubsetInsertionOpInterface, since equivalent subset extractions can only be
  // created after the store_to_buffer op's output (the subspan or subview) in
  // the IR.
  OpBuilder::InsertionGuard g(rewriter);
  (void)setInsertionPointAfterLastNeededValue(
      rewriter, cast<SubsetInsertionOpInterface>(storeOp.getOperation()));
  Value outputBuffer = getBufferizedSubspanOrSubview(
      rewriter, storeOp.getLoc(), subspanOp, storeOp.getValueType(),
      storeOp.getMixedOffsets(), storeOp.getMixedSizes(),
      storeOp.getMixedStrides());

  Value tensor = storeOp.getValue();
  rewriter.setInsertionPoint(storeOp);
  rewriter.replaceOpWithNewOp<IREE::Codegen::StoreToBufferOp>(storeOp, tensor,
                                                              outputBuffer);
}

namespace {

struct BufferizeDispatchTensorLoadStorePass final
    : impl::BufferizeDispatchTensorLoadStorePassBase<
          BufferizeDispatchTensorLoadStorePass> {
  using BufferizeDispatchTensorLoadStorePassBase::
      BufferizeDispatchTensorLoadStorePassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();
    SmallVector<IREE::TensorExt::DispatchTensorLoadOp> loadOps(
        funcOp.getFunctionBody()
            .getOps<IREE::TensorExt::DispatchTensorLoadOp>());
    SmallVector<IREE::TensorExt::DispatchTensorStoreOp> storeOps(
        funcOp.getFunctionBody()
            .getOps<IREE::TensorExt::DispatchTensorStoreOp>());

    IRRewriter rewriter(context);
    for (IREE::TensorExt::DispatchTensorLoadOp loadOp : loadOps) {
      bufferizeDispatchTensorLoad(rewriter, loadOp);
    }
    for (IREE::TensorExt::DispatchTensorStoreOp storeOp : storeOps) {
      bufferizeDispatchTensorStore(rewriter, storeOp);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
