// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <iterator>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

static bool isStreamableOp(Operation *op) {
  if (isa<StreamableOpInterface>(op)) {
    return true;
  }
  if (llvm::isa<Shape::TieShapeOp>(op)) {
    return true;
  }
  return false;
}

static bool isReturned(Operation *op) {
  for (auto result : op->getResults()) {
    for (auto *user : result.getUsers()) {
      if (user->hasTrait<OpTrait::IsTerminator>()) return true;
    }
  }
  return false;
}

static llvm::SmallVector<Operation *, 16> getOpsToHoist(Block &block) {
  llvm::SmallVector<Operation *, 16> opsToHoist;
  for (Operation &op : block) {
    if (!isStreamableOp(&op) && !op.hasTrait<OpTrait::IsTerminator>() &&
        MemoryEffectOpInterface::hasNoEffect(&op) && !isReturned(&op)) {
      opsToHoist.push_back(&op);
    }
  }
  return opsToHoist;
}

// Returns an operation in |block| that defines |v|, if one exists.
static Operation *getDefiningOpInBlock(Value v, Block &block) {
  if (OpResult opResult = v.dyn_cast<OpResult>()) {
    if (opResult.getOwner()->getBlock() == &block) {
      return opResult.getOwner();
    }
  }
  return nullptr;
}

namespace {
// Hoist ops that cannot be put into streams as far up in their block as they
// can go. This aims to improve stream creation by clustering streamable ops
// together.
//
// This pass shares similar goals to HoistShapeCalculationsPass, but is not
// limited to shape calculation operations.
class HoistUnstreamableOpsPass
    : public HoistUnstreamableOpsBase<HoistUnstreamableOpsPass> {
 public:
  void runOnOperation() override {
    auto func = getOperation();
    for (Block &block : func) {
      // TODO(gcmn): isBeforeInBlock is O(n) with repeated block modification,
      // making this quadratic.
      for (Operation *op : getOpsToHoist(block)) {
        Operation *moveAfter = nullptr;
        for (Value operand : op->getOperands()) {
          if (Operation *definingOp = getDefiningOpInBlock(operand, block)) {
            if (moveAfter == nullptr ||
                moveAfter->isBeforeInBlock(definingOp)) {
              moveAfter = definingOp;
            }
          }
        }
        if (moveAfter != nullptr) {
          op->moveAfter(moveAfter);
        } else {
          op->moveBefore(&block, block.begin());
        }
      }
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<mlir::FuncOp>> createHoistUnstreamableOpsPass() {
  return std::make_unique<HoistUnstreamableOpsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
