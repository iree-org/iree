// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::VM {

class SinkDefiningOpsPass
    : public PassWrapper<SinkDefiningOpsPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override { return "iree-vm-sink-defining-ops"; }

  StringRef getDescription() const override {
    return "Sinks defining ops with few uses to their use-sites.";
  }

  void runOnOperation() override {
    for (auto funcOp : getOperation().getOps<FuncOp>()) {
      DominanceInfo domInfo(funcOp);

      // Consider only those constant ops in the entry block.
      SmallVector<std::pair<Operation *, Operation *>, 8> sinks;
      for (auto &op : funcOp.getBlocks().front()) {
        if (op.getNumResults() != 1 || !isMemoryEffectFree(&op)) {
          // Probably not safe to move.
          continue;
        }

        auto users = llvm::to_vector(op.getUsers());
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

static PassRegistration<SinkDefiningOpsPass> pass;

} // namespace mlir::iree_compiler::IREE::VM
