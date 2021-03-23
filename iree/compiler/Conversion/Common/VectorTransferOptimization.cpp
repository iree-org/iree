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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

// Return true if all the uses of op are either Store/transfer_write.
// There can be SubviewOp users as long as all its users are also
// StoreOp/transfer_write. If return true it also fills out the uses, if it
// returns false uses is unchanged.
static bool allUsesAreStores(Operation* op, std::vector<Operation*>& uses) {
  std::vector<Operation*> opUses;
  for (OpOperand& use : op->getUses()) {
    Operation* useOp = use.getOwner();
    if (isa<vector::TransferWriteOp, memref::StoreOp>(useOp) ||
        (isa<memref::SubViewOp>(useOp) && allUsesAreStores(useOp, opUses))) {
      opUses.push_back(useOp);
      continue;
    }
    return false;
  }
  uses.insert(uses.end(), opUses.begin(), opUses.end());
  return true;
}

// Track temporary allocations that are never read from. If this is the case
// it means both the allocations and associated stores can be removed.
static void eraseDeadAllocAndStores(FuncOp funcOp) {
  std::vector<Operation*> opToErase;
  funcOp.walk([&](memref::AllocOp op) {
    if (allUsesAreStores(op, opToErase)) {
      opToErase.push_back(op.getOperation());
    }
  });
  for (Operation* op : opToErase) {
    op->erase();
  }
}

namespace {

struct VectorTransferOptimizationPass
    : public PassWrapper<VectorTransferOptimizationPass, FunctionPass> {
  void runOnFunction() override {
    FuncOp funcOp = getFunction();
    // Generate vector.shape_cast for dropping leading one dimensions in vector
    // ops. This increases the chance that we can forward more transfer writes
    // to transfer reads.
    OwningRewritePatternList patterns(&getContext());
    mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

    vector::transferOpflowOpt(funcOp);
    // Delete potential dead alloc and associated ops after store to load
    // forwarding.
    eraseDeadAllocAndStores(getFunction());
  }
};

}  // namespace

std::unique_ptr<FunctionPass> createVectorTransferOptimizationPass() {
  return std::make_unique<VectorTransferOptimizationPass>();
}

static PassRegistration<VectorTransferOptimizationPass> pass(
    "iree-codegen-optimize-vector-transfer",
    "Run optimization transformations on vector transfer operations",
    [] { return std::make_unique<VectorTransferOptimizationPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
