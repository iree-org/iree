// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_BITCASTUNSUPPORTEDELEMENTTYPESPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

/// Helper to get the target type for bitcasting if required. Returns failure
/// if bitcasting is not possible, and returns a null type if not required.
static FailureOr<RankedTensorType>
getBitcastRequiredType(Builder &b, Operation *root,
                       RankedTensorType tensorType) {
  Type elementType = tensorType.getElementType();
  // TODO: Support bitcasting non-floats (complex/structs/tuples).
  if (!elementType.isIntOrFloat()) {
    return RankedTensorType();
  }

  // All integer and byte aligned types are legal.
  int64_t bitwidth = elementType.getIntOrFloatBitWidth();
  if (elementType.isInteger() || bitwidth % 8 == 0) {
    return RankedTensorType();
  }

  int64_t innerSize = tensorType.getShape().back();
  int64_t innerNumBits = innerSize * bitwidth;
  if (ShapedType::isDynamic(innerSize) || innerNumBits % 8 != 0) {
    return root->emitOpError(
        "Unsupported tensor type unable to pack to bytes.");
  }

  SmallVector<int64_t> newShape(tensorType.getShape());
  newShape.back() = innerNumBits / 8;
  return RankedTensorType::get(newShape, b.getI8Type());
}

static LogicalResult bitcastBlockArgUser(RewriterBase &rewriter,
                                         Operation *root, Operation *user,
                                         Value arg,
                                         RankedTensorType originalType,
                                         RankedTensorType targetType) {
  int64_t innerSize = originalType.getShape().back();
  if (auto load = dyn_cast<IREE::TensorExt::DispatchTensorLoadOp>(user)) {
    SmallVector<int64_t> newSizes(load.getStaticSizes());
    if (newSizes.back() != innerSize || load.getStaticOffsets().back() != 0) {
      return root->emitOpError("non-full load of unsupported element type.");
    }
    newSizes.back() = targetType.getShape().back();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(load);

    // Load in the new type -> bitcast back to the original type.
    auto newLoad = rewriter.create<IREE::TensorExt::DispatchTensorLoadOp>(
        load.getLoc(), targetType, arg, load.getSourceDims(), load.getOffsets(),
        load.getSizes(), load.getStrides(), load.getStaticOffsets(), newSizes,
        load.getStaticStrides());
    rewriter.replaceOpWithNewOp<IREE::TensorExt::BitCastOp>(
        load, originalType, newLoad, load.getSizes(), load.getSizes());
    return success();
  } else if (auto store =
                 dyn_cast<IREE::TensorExt::DispatchTensorStoreOp>(user)) {
    SmallVector<int64_t> newSizes(store.getStaticSizes());
    if (newSizes.back() != innerSize || store.getStaticOffsets().back() != 0) {
      return root->emitOpError("non-full store of unsupported element type.");
    }
    newSizes.back() = targetType.getShape().back();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(store);

    auto bitcast = rewriter.create<IREE::TensorExt::BitCastOp>(
        store.getLoc(), targetType, store.getValue(), store.getSizes(),
        store.getSizes());
    // Bitcast to the target type -> store in the new type.
    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorStoreOp>(
        store, TypeRange{}, bitcast, arg, store.getTargetDims(),
        store.getOffsets(), store.getSizes(), store.getStrides(),
        store.getStaticOffsets(), newSizes, store.getStaticStrides());
    return success();
  }

  return root->emitOpError(
      "non-tensor load or store user of unsupported element type.");
}

/// Bitcast unsupported input tensor types to i8s in place. Bitcasts inside the
/// dispatch use `iree_tensor_ext.bitcast` and outside it uses
/// `flow.tensor.bitcast`.
static LogicalResult
bitcastWorkgroupsInputs(RewriterBase &rewriter,
                        IREE::Flow::DispatchWorkgroupsOp workgroupsOp) {
  for (auto [index, operand] :
       llvm::enumerate(workgroupsOp.getArgumentsMutable())) {
    // Skip tied operands. They will be processed separately.
    if (workgroupsOp.isOperandTied(operand.getOperandNumber())) {
      continue;
    }

    auto tensorType = dyn_cast<RankedTensorType>(operand.get().getType());
    // TODO: Figure out what to do about encodings.
    if (!tensorType || tensorType.getEncoding()) {
      continue;
    }

    auto maybeCastedTensorType =
        getBitcastRequiredType(rewriter, workgroupsOp, tensorType);
    if (failed(maybeCastedTensorType)) {
      return failure();
    }

    RankedTensorType castedTensorType = *maybeCastedTensorType;
    if (!castedTensorType) {
      continue;
    }

    BlockArgument blockArg = workgroupsOp.getInputBlockArgument(index);
    auto dispatchType =
        cast<IREE::TensorExt::DispatchTensorType>(blockArg.getType());

    auto castedDispatchType = IREE::TensorExt::DispatchTensorType::get(
        dispatchType.getAccess(), castedTensorType);

    rewriter.setInsertionPoint(workgroupsOp);
    ValueRange sourceDims =
        workgroupsOp.getOperandDynamicDims(operand.getOperandNumber());
    auto inputBitcast = rewriter.create<IREE::Flow::TensorBitCastOp>(
        workgroupsOp.getLoc(), castedTensorType, operand.get(), sourceDims,
        sourceDims);
    operand.assign(inputBitcast);

    blockArg.setType(castedDispatchType);
    // Copy the list of users since they will be updated while bitcasting.
    SmallVector<Operation *> users(blockArg.getUsers());
    for (auto user : users) {
      if (failed(bitcastBlockArgUser(rewriter, workgroupsOp, user, blockArg,
                                     tensorType, castedTensorType))) {
        return failure();
      }
    }
  }
  return success();
}

/// Bitcast unsupported output and tied input tensor types to bytes.
static LogicalResult
bitcastWorkgroupsOutputs(RewriterBase &rewriter,
                         IREE::Flow::DispatchWorkgroupsOp workgroupsOp) {
  // Try to keep the check for whether bitcasting is required fast as that is
  // the dramatically more common case.
  SmallVector<std::pair<int64_t, RankedTensorType>> resultsToCast;
  for (OpResult result : workgroupsOp->getOpResults()) {
    auto tensorType = dyn_cast<RankedTensorType>(result.getType());
    // TODO: Figure out what to do about encodings.
    if (!tensorType || tensorType.getEncoding()) {
      continue;
    }

    auto maybeCastedTensorType =
        getBitcastRequiredType(rewriter, workgroupsOp, tensorType);
    if (failed(maybeCastedTensorType)) {
      return failure();
    }

    RankedTensorType castedTensorType = *maybeCastedTensorType;
    if (castedTensorType) {
      resultsToCast.emplace_back(result.getResultNumber(), castedTensorType);
    }
  }

  // Nothing to do.
  if (resultsToCast.empty()) {
    return success();
  }

  SmallVector<Type> newResultTypes(workgroupsOp->getResultTypes());
  for (auto [resultIndex, castedTensorType] : resultsToCast) {
    OpResult originalResult = workgroupsOp->getOpResult(resultIndex);
    auto tensorType = cast<RankedTensorType>(originalResult.getType());
    newResultTypes[resultIndex] = castedTensorType;

    BlockArgument blockArg = workgroupsOp.getOutputBlockArgument(resultIndex);
    auto dispatchType =
        cast<IREE::TensorExt::DispatchTensorType>(blockArg.getType());
    auto castedDispatchType = IREE::TensorExt::DispatchTensorType::get(
        dispatchType.getAccess(), castedTensorType);

    // Cast the input to the dispatch if the corresponding result is tied.
    if (OpOperand *tiedOperand =
            workgroupsOp.getTiedResultOpOperand(originalResult)) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(workgroupsOp);
      ValueRange dynamicDims =
          workgroupsOp.getOperandDynamicDims(tiedOperand->getOperandNumber());

      auto inputBitcast = rewriter.create<IREE::Flow::TensorBitCastOp>(
          workgroupsOp.getLoc(), castedTensorType, tiedOperand->get(),
          dynamicDims, dynamicDims);
      tiedOperand->assign(inputBitcast);
    }

    // Update the type of the block argument and bitcast all load/store users.
    blockArg.setType(castedDispatchType);
    SmallVector<Operation *> users(blockArg.getUsers());
    for (auto user : users) {
      if (failed(bitcastBlockArgUser(rewriter, workgroupsOp, user, blockArg,
                                     tensorType, castedTensorType))) {
        return failure();
      }
    }
  }

  // Clone the op with new result types and steal the body.
  rewriter.setInsertionPoint(workgroupsOp);
  auto newWorkgroupsOp = rewriter.create<IREE::Flow::DispatchWorkgroupsOp>(
      workgroupsOp->getLoc(), workgroupsOp.getWorkload(), newResultTypes,
      workgroupsOp.getResultDims(), workgroupsOp.getArguments(),
      workgroupsOp.getArgumentDims(),
      workgroupsOp.getTiedResultOperandIndices(), workgroupsOp->getAttrs());
  newWorkgroupsOp->setDialectAttrs(workgroupsOp->getDialectAttrs());
  auto &newBody = newWorkgroupsOp.getClosureBodyRegion();
  newBody.takeBody(workgroupsOp.getClosureBodyRegion());

  // Copy the workgroup_count region.
  auto &workgroupCountRegion = workgroupsOp.getWorkgroupCount();
  if (!workgroupCountRegion.empty()) {
    auto &newWorkgroupCountRegion = newWorkgroupsOp.getWorkgroupCount();
    newWorkgroupCountRegion.takeBody(workgroupCountRegion);
  }

  // Bitcast back to the original types for replacement.
  SmallVector<Value> replacements(newWorkgroupsOp.getResults());
  for (auto [resultIndex, castedTensorType] : resultsToCast) {
    auto originalType =
        cast<RankedTensorType>(workgroupsOp.getResult(resultIndex).getType());
    ValueRange dynamicDims = newWorkgroupsOp.getResultDynamicDims(resultIndex);
    replacements[resultIndex] = rewriter.create<IREE::Flow::TensorBitCastOp>(
        workgroupsOp.getLoc(), originalType, replacements[resultIndex],
        dynamicDims, dynamicDims);
  }

  rewriter.replaceOp(workgroupsOp, replacements);
  return success();
}

namespace {
struct BitcastUnsupportedElementTypesPass
    : public impl::BitcastUnsupportedElementTypesPassBase<
          BitcastUnsupportedElementTypesPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void BitcastUnsupportedElementTypesPass::runOnOperation() {
  Operation *op = getOperation();

  IRRewriter rewriter(&getContext());

  WalkResult res = op->walk([&](IREE::Flow::DispatchWorkgroupsOp workgroupsOp) {
    // First update bitcast all non-tied input operands in place.
    if (failed(bitcastWorkgroupsInputs(rewriter, workgroupsOp))) {
      return WalkResult::interrupt();
    }
    // Bitcast all required outputs (including tied operands).
    if (failed(bitcastWorkgroupsOutputs(rewriter, workgroupsOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (res.wasInterrupted()) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
