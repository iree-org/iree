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

/// Inserts a `linalg.copy` directly before the given operation on the
/// specified operand, for example with operand index = 1:
///
///   linalg.matmul ins(%0, %1)
///
/// becomes
///
///   %empty = tensor.empty()
///   %copy = linalg.copy %1 to %empty {
///     lowering_config = #iree_gpu.derived_thread_config}
///   linalg.matmul ins(%0, %copy)
///
/// If the producer is already a tilable op, the producer is just annotated with
/// #iree_gpu.derived_thread_config to indicate that it should be distributed
/// to threads independently of the matmul.
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
    funcOp.walk([&](Operation *op) {
      auto loweringConfig =
          getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
      if (!loweringConfig) {
        return;
      }

      std::optional<SmallVector<int64_t>> promotedOperands =
          loweringConfig.getPromotedOperandList();
      if (!promotedOperands) {
        return;
      }

      builder.setInsertionPoint(op);
      for (auto operand : promotedOperands.value()) {
        promoteOperand(builder, op, operand);
      }
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
