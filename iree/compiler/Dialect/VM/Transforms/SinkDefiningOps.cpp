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

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

class SinkDefiningOpsPass
    : public PassWrapper<SinkDefiningOpsPass, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    for (auto funcOp : getOperation().getOps<FuncOp>()) {
      DominanceInfo domInfo(funcOp);

      // Consider only those constant ops in the entry block.
      SmallVector<std::pair<Operation *, Operation *>, 8> sinks;
      for (auto &op : funcOp.getBlocks().front()) {
        if (op.getNumResults() != 1 ||
            !MemoryEffectOpInterface::hasNoEffect(&op)) {
          // Probably not safe to move.
          continue;
        }

        auto users = llvm::to_vector<4>(op.getUsers());
        if (users.empty()) {
          // No users (probably leftover needing DCE).
          continue;
        } else if (users.size() == 1) {
          // Only a single user; safe to move.
          sinks.push_back(std::make_pair(&op, users.front()));
          continue;
        }

        // Find the common dominator block across all uses. This may be the
        // entry block itself.
        Block *commonDominator = users.front()->getBlock();
        for (auto user : users) {
          commonDominator = domInfo.findNearestCommonDominator(
              commonDominator, user->getBlock());
        }

        // Find the first use within the dominator block (if any) so that we
        // can sink down to it.
        Operation *firstUserInDominator = commonDominator->getTerminator();
        for (auto user : users) {
          if (user->getBlock() == commonDominator) {
            if (user->isBeforeInBlock(firstUserInDominator)) {
              firstUserInDominator = user;
            }
          }
        }

        sinks.push_back(std::make_pair(&op, firstUserInDominator));
      }

      // Sink values after iterating.
      for (auto &sink : sinks) {
        sink.first->moveBefore(sink.second);
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createSinkDefiningOpsPass() {
  return std::make_unique<SinkDefiningOpsPass>();
}

static PassRegistration<SinkDefiningOpsPass> pass(
    "iree-vm-sink-defining-ops",
    "Sinks defining ops with few uses to their use-sites.");

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
