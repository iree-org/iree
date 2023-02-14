// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-codegen-hoist-statically-bound-allocations"

namespace mlir {
namespace iree_compiler {

namespace {

struct HoistStaticallyBoundAllocationsPass
    : HoistStaticallyBoundAllocationsBase<HoistStaticallyBoundAllocationsPass> {
  void runOnOperation() override;
};

}  // namespace

/// Some uses of a `memref.alloca` can be replaced with a `memref.subview`
/// easily. Other uses (like a use in a `scf.yield` or `func.return`) are
/// non-trivial because of compatibility between types of different SSA values.
static bool isUseReplacableWithSubview(OpOperand &use) {
  Operation *user = use.getOwner();
  return isa<linalg::LinalgOp, memref::StoreOp, memref::SubViewOp>(user);
}

void HoistStaticallyBoundAllocationsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  SmallVector<memref::AllocaOp> allocaOps;

  // Collect all allocas that are hoistable.
  funcOp.walk([&](memref::AllocaOp allocaOp) {
    if (allocaOp->getBlock() == &funcOp.getBody().front()) return;
    if (allocaOp.getDynamicSizes().empty()) {
      allocaOps.push_back(allocaOp);
      return;
    }
    if (llvm::all_of(allocaOp->getUses(), [](OpOperand &use) {
          return isUseReplacableWithSubview(use);
        })) {
      allocaOps.push_back(allocaOp);
      return;
    }
  });

  // Hoist the allocas and replace all uses.
  OpBuilder builder(&getContext());
  for (auto allocaOp : allocaOps) {
    LLVM_DEBUG({
      llvm::dbgs() << "Alloca Op : ";
      allocaOp->dump();
      int numUses = std::distance(allocaOp.getResult().use_begin(),
                                  allocaOp.getResult().use_end());
      llvm::dbgs() << " num Uses : " << numUses;
    });
    std::optional<Value> replacement =
        hoistStaticallyBoundAllocations(funcOp, builder, allocaOp);
    if (!replacement) continue;
    LLVM_DEBUG({
      llvm::dbgs() << "Replacement : ";
      replacement->dump();
    });
    Value replacementVal = replacement.value();
    allocaOp.getResult().replaceAllUsesWith(replacementVal);
    allocaOp->erase();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createHoistStaticallyBoundAllocationsPass() {
  return std::make_unique<HoistStaticallyBoundAllocationsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
