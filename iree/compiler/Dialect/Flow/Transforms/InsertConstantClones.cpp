// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Inserts tensor clones of constant values that escape a block either via
// function returns or branches. This is a workaround for not having a full
// DFA pass that attributes tensors with their usage information.
//
// TODO(#7277): remove when switched to streams (happens there now).
class InsertConstantClonesPass
    : public InsertConstantClonesBase<InsertConstantClonesPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect, IREE::Flow::FlowDialect>();
  }

  void runOnOperation() override {
    for (auto &block : getOperation()) {
      if (auto op = dyn_cast<mlir::ReturnOp>(block.getTerminator())) {
        tryCloneConstantValues(op);
      } else if (auto op = dyn_cast<mlir::BranchOp>(block.getTerminator())) {
        tryCloneConstantValues(op);
      } else if (auto op =
                     dyn_cast<mlir::CondBranchOp>(block.getTerminator())) {
        tryCloneConstantValues(op);
      }
    }
  }

 private:
  // Heuristics to try to determine if a tensor will be sourced from a constant.
  // This really is just a guess and a real solution requires proper DFA.
  bool isLikelyConstant(Value value) {
    auto op = value.getDefiningOp();
    if (!op) return false;
    if (op->hasTrait<OpTrait::ConstantLike>() ||
        isa<IREE::Util::UnfoldableConstantOp>(op)) {
      return true;
    } else if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadOp>(op)) {
      return loadOp.isGlobalImmutable();
    } else if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadIndirectOp>(op)) {
      return true;  // can't know indirect variable behavior (without DFA)
    } else if (auto sliceOp = dyn_cast<IREE::Flow::TensorSliceOp>(op)) {
      return isLikelyConstant(sliceOp.source());
    } else if (auto reshapeOp = dyn_cast<IREE::Flow::TensorReshapeOp>(op)) {
      return isLikelyConstant(reshapeOp.source());
    }
    return false;
  }

  void replaceWithClone(Value value, Operation *useOp, OpBuilder &builder) {
    auto cloneOp =
        builder.create<IREE::Flow::TensorCloneOp>(value.getLoc(), value);
    value.replaceUsesWithIf(cloneOp.result(), [&](OpOperand &operand) {
      return operand.getOwner() == useOp;
    });
  }

  void tryCloneConstantValues(mlir::ReturnOp op) {
    OpBuilder builder(op);
    for (auto value : op.getOperands()) {
      if (value.getType().isa<TensorType>() && isLikelyConstant(value)) {
        replaceWithClone(value, op, builder);
      }
    }
  }

  void tryCloneConstantValues(mlir::BranchOp op) {
    OpBuilder builder(op);
    for (auto value : op.getOperands()) {
      if (value.getType().isa<TensorType>() && isLikelyConstant(value)) {
        replaceWithClone(value, op, builder);
      }
    }
  }

  void tryCloneConstantValues(mlir::CondBranchOp op) {
    OpBuilder builder(op);
    for (unsigned i = 0; i < op.getNumSuccessors(); ++i) {
      auto operands = op.getSuccessorOperands(i);
      if (!operands.hasValue()) continue;
      for (auto value : operands.getValue()) {
        if (value.getType().isa<TensorType>() && isLikelyConstant(value)) {
          replaceWithClone(value, op, builder);
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<mlir::FuncOp>> createInsertConstantClonesPass() {
  return std::make_unique<InsertConstantClonesPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
