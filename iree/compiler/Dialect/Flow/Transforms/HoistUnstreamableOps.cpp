// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <iterator>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

static bool isStreamableOp(Operation *op) {
  if (auto streamableOp = dyn_cast<StreamableOpInterface>(op)) {
    return streamableOp.isUsableInStream();
  }
  if (llvm::isa<Shape::TieShapeOp>(op)) {
    return true;
  }
  return false;
}

static llvm::SmallVector<Operation *, 16> getOpsToHoist(Block &block) {
  llvm::SmallVector<Operation *, 16> opsToHoist;
  for (Operation &op : block) {
    if (!isStreamableOp(&op) && !op.isKnownTerminator() &&
        MemoryEffectOpInterface::hasNoEffect(&op)) {
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
class HoistUnstreamableOps
    : public PassWrapper<HoistUnstreamableOps, FunctionPass> {
 public:
  void runOnFunction() override {
    auto func = getFunction();
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

std::unique_ptr<OperationPass<FuncOp>> createHoistUnstreamableOpsPass() {
  return std::make_unique<HoistUnstreamableOps>();  // NOLINT
}

static PassRegistration<HoistUnstreamableOps> pass(
    "iree-flow-hoist-unstreamable-ops",
    "Hoist ops that cannot be captured in streams to the top of their block.");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
