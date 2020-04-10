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

#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "mlir/Analysis/Dominance.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {
namespace {

bool isSimpleShapeCalculationOp(Operation *op) {
  // The op must have no side effects.
  if (!MemoryEffectOpInterface::hasNoEffect(op) || op->getNumRegions() != 0) {
    return false;
  }
  // The op should operate on types that are likely shape calculations.
  // The exact predicate used here isn't too important. The main important thing
  // is that we don't want to include ops on tensors.
  for (Type type : op->getOperandTypes()) {
    if (type.isa<TensorType>()) {
      return false;
    }
  }
  return true;
}

// Return an operation in `block` that defines `v`, if one exists.
Operation *getDefiningOpInBlock(Value v, Block &block) {
  if (OpResult opResult = v.dyn_cast<OpResult>()) {
    if (opResult.getOwner()->getBlock() == &block) {
      return opResult.getOwner();
    }
  }
  return nullptr;
}

DenseSet<Operation *> calculateOpsToHoist(Block &block) {
  // Strategy:
  // Backward DFS from shapex.tie_shape shape operands (second
  // operands), staying within the block and not incorporating ops that don't
  // satisfy `isSimpleShapeCalculationOp`.

  SmallVector<Operation *, 16> worklist;
  // The return value, and also used as a "visited" set for our DFS below.
  DenseSet<Operation *> opsToHoistSet;

  // The worklist is initially populated with shape-producing ops defined in
  // this block.
  //
  // We are only hoisting ops within the block, so block arguments and any
  // values defined outside the block (which already dominate the entire block)
  // don't matter.
  for (Operation &op : block) {
    if (auto tieShape = dyn_cast<Shape::TieShapeOp>(op)) {
      if (Operation *op = getDefiningOpInBlock(tieShape.shape(), block)) {
        worklist.push_back(op);
      }
    }
  }
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!isSimpleShapeCalculationOp(op)) {
      continue;
    }
    if (opsToHoistSet.insert(op).second) {
      for (Value v : op->getOperands()) {
        if (Operation *op = getDefiningOpInBlock(v, block)) {
          worklist.push_back(op);
        }
      }
    }
  }
  return opsToHoistSet;
}

void hoistOps(DenseSet<Operation *> opsToHoistSet, Block &block,
              DominanceInfo &domInfo) {
  auto opsToHoist = llvm::to_vector<16>(opsToHoistSet);
  llvm::sort(opsToHoist, [&](Operation *lhs, Operation *rhs) {
    return domInfo.properlyDominates(lhs, rhs);
  });

  for (Operation *op : opsToHoist) {
    Operation *insertAfter = nullptr;
    for (Value operand : op->getOperands()) {
      if (Operation *definingOp = getDefiningOpInBlock(operand, block)) {
        if (insertAfter == nullptr ||
            domInfo.properlyDominates(definingOp, insertAfter)) {
          insertAfter = definingOp;
        }
      }
    }
    if (insertAfter != nullptr) {
      op->moveBefore(&*std::next(insertAfter->getIterator()));
    } else {
      op->moveBefore(&block, block.begin());
    }
  }
}

// Best-effort pass for hoisting shape calculations earlier in the program.
// We currently don't provide any hard guarantees about exactly what invariants
// are established by this pass.
//
// The goal of this pass is to unblock further progress on dynamic shape
// support. One pragmatic thing we observe is that for IREE, dispatch region
// formation requires that when there is a `shapex.tie_shape %tensor, %shape`
// op, to even properly form the dispatch region, IREE needs `%shape` to
// dominate `%tensor` since the dispatch region's "workload" is derived from the
// shape.
//
// This pass doesn't have a cost model, so it shouldn't be considered a generic
// "hoist stuff to make things faster" type of pass. It's strictly a
// best-effort pass to make certain lowerings work, albeit on somewhat shaky
// ground. Longer-term, IREE's dispatch region formation will use a more
// sophisticated algorithm and the analysis/hoisting done here will be a
// byproduct of the dispatch region formation legality analysis/preparation.
class HoistShapeCalculations
    : public PassWrapper<HoistShapeCalculations, FunctionPass> {
 public:
  void runOnFunction() override {
    auto func = getFunction();
    DominanceInfo domInfo(func);
    for (Block &block : func) {
      DenseSet<Operation *> opsToHoist = calculateOpsToHoist(block);
      hoistOps(opsToHoist, block, domInfo);
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createHoistShapeCalculationsPass() {
  return std::make_unique<HoistShapeCalculations>();  // NOLINT
}

static PassRegistration<HoistShapeCalculations> pass(
    "iree-shape-hoist-shape-calculations",
    "Best-effort shape calculation hoisting.");

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
