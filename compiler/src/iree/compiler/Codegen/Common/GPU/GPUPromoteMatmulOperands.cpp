// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPROMOTEMATMULOPERANDSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

void promoteOperand(OpBuilder &builder, Operation *op, unsigned index) {
  Value operand = op->getOperand(index);

  if (auto producer = operand.getDefiningOp<TilingInterface>()) {
    setLoweringConfig(producer, IREE::GPU::DerivedThreadConfigAttr::get(
                                    builder.getContext()));
    return;
  }

  auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
  if (!tensorType) {
    return;
  }

  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, op->getLoc(), operand);
  Value empty = builder.create<tensor::EmptyOp>(op->getLoc(), mixedSizes,
                                                tensorType.getElementType());
  auto copy = builder.create<linalg::CopyOp>(op->getLoc(), operand, empty);
  setLoweringConfig(
      copy, IREE::GPU::DerivedThreadConfigAttr::get(builder.getContext()));
  op->setOperand(index, copy.getResult(0));
}

struct GPUPromoteMatmulOperandsPass final
    : impl::GPUPromoteMatmulOperandsPassBase<GPUPromoteMatmulOperandsPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    OpBuilder builder(funcOp);
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      if (!isMatmulOrBatchMatmul(linalgOp)) {
        return;
      }
      builder.setInsertionPoint(linalgOp);
      promoteOperand(builder, linalgOp, 0);
      promoteOperand(builder, linalgOp, 1);
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
