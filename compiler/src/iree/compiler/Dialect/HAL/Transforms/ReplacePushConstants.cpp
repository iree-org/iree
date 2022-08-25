// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {
namespace {

// These must match what the runtime uses.
#define IREE_PUSH_CONSTANT_EMULATION_SET 3
#define IREE_PUSH_CONSTANT_EMULATION_BINDING 0

static void replaceConstantLoadOp(IREE::HAL::InterfaceConstantLoadOp op) {
  OpBuilder builder(op);

  // hal.interface.binding.subspan -> !flow.dispatch.tensor<readonly:i32>
  auto opFlowTensorType = IREE::Flow::DispatchTensorType::get(
      IREE::Flow::TensorAccess::ReadOnly, {}, op.getType());
  // Offset by the op's index.
  auto offsetValue = builder.createOrFold<arith::ConstantIndexOp>(
      op.getLoc(), op.getIndex().getZExtValue());
  SmallVector<Value> dynamicDims;
  // Keep the op's alignment hint.
  auto alignmentAttr = op.getAlignmentAttr();
  // Note: we're ignoring the op's potential 'values' hint (if provided) -
  // InterfaceBindingSubspanOp has no matching concept and we assume that any
  // analysis using the hint should have been performed by earlier passes.
  auto subspanOp = builder.create<IREE::HAL::InterfaceBindingSubspanOp>(
      op.getLoc(), opFlowTensorType,
      /*set=*/APInt(64, IREE_PUSH_CONSTANT_EMULATION_SET),
      /*binding=*/APInt(64, IREE_PUSH_CONSTANT_EMULATION_BINDING),
      IREE::HAL::DescriptorType::StorageBuffer, offsetValue, dynamicDims,
      alignmentAttr);

  // flow.dispatch.tensor.load -> tensor<i32>
  auto tensorType = RankedTensorType::get({}, op.getType());
  auto loadOp = builder.create<IREE::Flow::DispatchTensorLoadOp>(
      op.getLoc(), tensorType, subspanOp, dynamicDims);

  // tensor.extract -> i32
  auto extractOp = builder.create<tensor::ExtractOp>(op.getLoc(), loadOp);

  op.replaceAllUsesWith(extractOp.getResult());
  op.erase();
}

class ReplacePushConstantsPass
    : public PassWrapper<ReplacePushConstantsPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplacePushConstantsPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                    mlir::tensor::TensorDialect, IREE::Flow::FlowDialect,
                    IREE::HAL::HALDialect>();
  }

  StringRef getArgument() const override {
    return "iree-hal-replace-push-constants";
  }

  StringRef getDescription() const override {
    return "Replaces push constant loads with binding loads for when using HAL "
           "devices that do not support push constants.";
  }

  void runOnOperation() override {
    auto parentOp = getOperation();
    auto constantLoadOps = llvm::to_vector<4>(
        parentOp.getOps<IREE::HAL::InterfaceConstantLoadOp>());

    for (auto constantLoadOp : constantLoadOps) {
      replaceConstantLoadOp(constantLoadOp);
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createReplacePushConstantsPass() {
  return std::make_unique<ReplacePushConstantsPass>();
}

static PassRegistration<ReplacePushConstantsPass> pass;

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
