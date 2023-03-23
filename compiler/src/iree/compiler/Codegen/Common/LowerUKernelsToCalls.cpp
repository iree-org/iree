// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/UKernelOps.h"
#include "iree/compiler/Codegen/Interfaces/UKernelOpInterface.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct LowerUKernelOpsToCallsPass
    : LowerUKernelOpsToCallsBase<LowerUKernelOpsToCallsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, func::FuncDialect>();
  }
  void runOnOperation() override;
};
}  // namespace

static bool castOperands(IRRewriter &rewriter,
                         mlir::Operation::operand_range operands,
                         mlir::MutableOperandRange mutables) {
  bool changed = false;
  SmallVector<Value> new_operands;
  for (auto operand : operands) {
    if (auto memrefType = dyn_cast<mlir::MemRefType>(operand.getType())) {
      auto addressSpace =
          memrefType.getMemorySpace().dyn_cast_or_null<gpu::AddressSpaceAttr>();
      if (addressSpace && addressSpace.getValue() ==
                              gpu::GPUDialect::getWorkgroupAddressSpace()) {
        changed = true;
        mlir::MemRefType new_memrefType = mlir::MemRefType::get(
            memrefType.getShape(), memrefType.getElementType(),
            memrefType.getLayout());
        operand = rewriter.create<memref::MemorySpaceCastOp>(
            operand.getLoc(), new_memrefType, operand);
      }
    }
    new_operands.push_back(operand);
  }
  mutables.assign(new_operands);
  return changed;
}

static void castAddressSpace(IRRewriter &rewriter,
                             IREE::Codegen::UKernelGenericOp op) {
  castOperands(rewriter, op.getInputs(), op.getInputsMutable());
  castOperands(rewriter, op.getOutputs(), op.getOutputsMutable());
  castOperands(rewriter, op.getOtherOperands(), op.getOtherOperandsMutable());
}

void LowerUKernelOpsToCallsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  SmallVector<Operation *> toDelete;
  Operation *errorOp = nullptr;
  IRRewriter rewriter(context);

  // LLVM ABI doesn't allow address space, so we cast operands back to generic
  // address space. Otherwise, inlining doesn't work since ABI of callee and
  // caller don't match
  getOperation().walk([&](IREE::Codegen::UKernelGenericOp microKernelOp) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(microKernelOp);
    castAddressSpace(rewriter, microKernelOp);
  });

  WalkResult result = getOperation().walk(
      [&](IREE::Codegen::UKernelOpInterface microKernelOp) -> WalkResult {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(microKernelOp);
        FailureOr<func::CallOp> callOp =
            microKernelOp.lowerToFunctionCall(rewriter);
        if (failed(callOp)) {
          errorOp = microKernelOp;
          return WalkResult::interrupt();
        }
        toDelete.push_back(microKernelOp);
        return WalkResult::advance();
      });
  if (result.wasInterrupted()) {
    errorOp->emitOpError(
        "failed to lower micro kernel operation to function call");
    return signalPassFailure();
  }
  for (auto op : toDelete) {
    op->erase();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createLowerUKernelOpsToCallsPass() {
  return std::make_unique<LowerUKernelOpsToCallsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
