// Copyright 2019 Google LLC
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

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

#define DEBUG_TYPE "iree-dispatch"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// Replaces |returnOp| with a clone including |newOperands| appended.
LogicalResult appendReturnOperands(IREE::Flow::ReturnOp returnOp,
                                   ArrayRef<Value> newOperands) {
  // Insert prior to the original return.
  OpBuilder builder(returnOp);

  // Clone with new args.
  SmallVector<Value, 8> operands;
  operands.reserve(returnOp.getNumOperands() + newOperands.size());
  operands.append(returnOp.operand_begin(), returnOp.operand_end());
  operands.append(newOperands.begin(), newOperands.end());
  builder.create<IREE::Flow::ReturnOp>(returnOp.getLoc(), operands);

  // Remove original.
  returnOp.erase();

  return success();
}

// Replaces |regionOp| with a clone including |newArgs| and |newResults|.
DispatchRegionOp appendRegionArgsAndResults(DispatchRegionOp &regionOp,
                                            ArrayRef<Value> newArgs,
                                            ArrayRef<Value> newResults,
                                            Location otherLoc) {
  // Insert prior to the original region.
  OpBuilder builder(regionOp);

  // Location is original region + new region location (both probably fused).
  SmallVector<Location, 2> fusedLocs = {regionOp.getLoc(), otherLoc};
  auto fusedLoc = FusedLoc::get(fusedLocs, regionOp.getContext());

  // Clone with new results.
  SmallVector<Value, 8> operands;
  operands.append(regionOp.args().begin(), regionOp.args().end());
  operands.append(newArgs.begin(), newArgs.end());
  SmallVector<Type, 8> resultTypes;
  resultTypes.append(regionOp.result_type_begin(), regionOp.result_type_end());
  for (auto newResult : newResults) {
    resultTypes.push_back(newResult.getType());
  }
  auto newRegionOp = builder.create<DispatchRegionOp>(
      fusedLoc, resultTypes, regionOp.workload(), operands,
      regionOp.getAttrs());
  newRegionOp.body().takeBody(regionOp.body());

  // Replace uses of original values with the new values.
  for (int i = 0; i < regionOp.getNumResults(); ++i) {
    regionOp.getResult(i).replaceAllUsesWith(newRegionOp.getResult(i));
  }

  // Erase the original region.
  regionOp.erase();

  return newRegionOp;
}

// Removes results that are not used from the dispatch region.
// Returns the new operation. There may be unused ops in the region but DCE
// should take care of that later.
DispatchRegionOp removeUnusedResults(DispatchRegionOp regionOp) {
  // Find return value within the region.
  auto &regionBlock = regionOp.body().getBlocks().front();
  auto returnOp = dyn_cast<IREE::Flow::ReturnOp>(regionBlock.getTerminator());
  if (!returnOp) {
    regionBlock.getParent()->getParentOfType<FuncOp>().emitError()
        << "block does not contain an flow.return op";
  }

  // Calculate new return values.
  SmallVector<Type, 8> newReturnTypes;
  SmallVector<Value, 8> newReturnValues;
  SmallVector<Value, 8> newRegionResults;
  for (int i = 0; i < returnOp.getNumOperands(); ++i) {
    auto resultValue = regionOp.getResult(i);
    if (!resultValue.use_empty()) {
      // Still has uses so we will preserve it.
      newReturnTypes.push_back(resultValue.getType());
      newReturnValues.push_back(returnOp.getOperand(i));
      newRegionResults.push_back(resultValue);
    }
  }

  // Update return op operands. We can do this in-place as we are only shrinking
  // the list.
  returnOp.getOperation()->setOperands(newReturnValues);

  // Insert prior to the original region.
  OpBuilder builder(regionOp);

  // Clone with new results.
  auto newRegionOp = builder.create<DispatchRegionOp>(
      regionOp.getLoc(), newReturnTypes, regionOp.workload(), regionOp.args(),
      regionOp.getAttrs());
  newRegionOp.body().takeBody(regionOp.body());

  // Replace uses of original values with the new values.
  for (int i = 0; i < newRegionResults.size(); ++i) {
    newRegionResults[i].replaceAllUsesWith(newRegionOp.getResult(i));
  }

  // Erase the original region.
  regionOp.erase();

  return newRegionOp;
}

// Returns true if |lhs| and |rhs| have either an identical workload or one that
// is compatible.
bool areDispatchRegionWorkloadsCompatible(DispatchRegionOp &lhs,
                                          DispatchRegionOp &rhs) {
  // TODO(benvanik): more sophisticated checking; right now it's just identical.
  return lhs.workload() == rhs.workload();
}

// Returns true if |value| depends in any way on |op| through any path.
bool doesValueDependOnOperation(Value value, Operation *op) {
  if (!value.getDefiningOp()) {
    return false;
  } else if (value.getDefiningOp() == op) {
    return true;
  } else if (value.getDefiningOp()->getBlock() == op->getBlock() &&
             value.getDefiningOp()->isBeforeInBlock(op)) {
    // Can't depend on |op| as it is defined prior to it.
    return false;
  }
  for (auto operand : value.getDefiningOp()->getOperands()) {
    if (doesValueDependOnOperation(operand, op)) {
      return true;
    }
  }
  return true;
}

// Returns true if |rhs| transitively depends on any out of |lhs|.
// |rhs| may depend directly on the results of |lhs| but no other ops in the
// parent block will use the results prior to |rhs|.
bool areDispatchRegionsTransitivelyDependent(DispatchRegionOp &lhs,
                                             DispatchRegionOp &rhs) {
  for (auto arg : rhs.args()) {
    if (arg.getDefiningOp() != lhs && doesValueDependOnOperation(arg, lhs)) {
      // Transitively dependent - boo - can't merge yet.
      return true;
    }
  }
  return false;
}

// Returns true if the dispatch region contains only a single block.
// This is because our merge isn't very smart and will not preserve the CFG
// right now. We can fix this when needed.
bool isDispatchRegionMergable(DispatchRegionOp &regionOp) {
  // Disallow merging of dispatch regions containing matmuls and other big ops.
  // We do this to allow backends to lower the big op as entirely isolated such
  // that substituting library calls is easier.
  for (auto &block : regionOp.body().getBlocks()) {
    for (auto &op : block) {
      // TODO(b/144530470): replace with tablegen attributes/interfaces.
      if (isa<mhlo::ReduceOp>(op) || isa<mhlo::DotOp>(op) ||
          isa<mhlo::ConvOp>(op) || isa<mhlo::ReduceWindowOp>(op) ||
          isa<mhlo::PadOp>(op) || isa<mhlo::TorchIndexSelectOp>(op) ||
          isa<mhlo::SliceOp>(op) || isa<mhlo::ConcatenateOp>(op)) {
        return false;
      }
    }
  }
  return regionOp.body().getBlocks().size() == 1;
}

// Merges |rhs| into |lhs| and returns the new |lhs| op.
// Precondition: !areDispatchRegionsTransitivelyDependent
DispatchRegionOp mergeDispatchRegions(DispatchRegionOp &lhs,
                                      DispatchRegionOp &rhs) {
  auto &lhsBlock = lhs.body().front();
  auto &rhsBlock = rhs.body().front();

  // Find the values used as return values in the lhs.
  // We'll need to replace the uses in rhs with these.
  auto lhsReturnOp = cast<IREE::Flow::ReturnOp>(lhsBlock.getTerminator());
  SmallVector<Value, 8> lhsReturnValues;
  lhsReturnValues.reserve(lhsReturnOp.getNumOperands());
  lhsReturnValues.append(lhsReturnOp.operand_begin(),
                         lhsReturnOp.operand_end());

  // Find the values used as return values in the rhs.
  // We'll add these to the results of the lhs region.
  auto rhsReturnOp = cast<IREE::Flow::ReturnOp>(rhsBlock.getTerminator());
  SmallVector<Value, 8> rhsReturnValues;
  rhsReturnValues.reserve(rhsReturnOp.getNumOperands());
  rhsReturnValues.append(rhsReturnOp.operand_begin(),
                         rhsReturnOp.operand_end());

  // Compute new args.
  BlockAndValueMapping mapping;
  SmallVector<Value, 8> newArgs;
  auto lhsArgs = llvm::to_vector<8>(lhs.args());
  auto rhsArgs = llvm::to_vector<8>(rhs.args());
  for (int rhsOpIdx = 0; rhsOpIdx < rhsArgs.size(); ++rhsOpIdx) {
    bool didElide = false;
    // Find if the rhs arg already exists on the lhs and dedupe.
    for (int lhsOpIdx = 0; lhsOpIdx < lhsArgs.size(); ++lhsOpIdx) {
      if (rhsArgs[rhsOpIdx] == lhsArgs[lhsOpIdx]) {
        mapping.map(rhsBlock.getArgument(rhsOpIdx),
                    lhsBlock.getArgument(lhsOpIdx));
        didElide = true;
        break;
      }
    }
    // Find if the arg has a direct dependency on the results of the lhs.
    for (int lhsResultIdx = 0; lhsResultIdx < lhs.getNumResults();
         ++lhsResultIdx) {
      if (rhsArgs[rhsOpIdx] == lhs.getResult(lhsResultIdx)) {
        // Direct dependency; can elide. We'll skip adding it to the new region
        // args and instead just remap it later.
        mapping.map(rhsBlock.getArgument(rhsOpIdx),
                    lhsReturnValues[lhsResultIdx]);
        didElide = true;
        break;
      }
    }
    if (!didElide) {
      // Add to the lhs block.
      auto oldArg = rhs.getOperand(rhsOpIdx + 1);
      auto newArg = lhsBlock.addArgument(oldArg.getType());
      mapping.map(rhsBlock.getArgument(rhsOpIdx), newArg);
      newArgs.push_back(oldArg);
    }
  }

  OpBuilder regionBuilder = OpBuilder::atBlockEnd(&lhsBlock);

  // Copy ops (replacing any args as needed).
  // Note that we need to insert prior to the terminator.
  regionBuilder.setInsertionPoint(lhsReturnOp);
  for (auto &op : rhsBlock) {
    // Note that this updates the mapping with the new values (so at the end
    // we have those new values).
    //
    // We avoid the return op here as we have already merged it above.
    if (!op.isKnownTerminator()) {
      regionBuilder.clone(op, mapping);
    }
  }

  // Compute new results and add to both region and return op.
  SmallVector<Value, 8> newResults;
  for (auto rhsResult : rhsReturnValues) {
    newResults.push_back(mapping.lookupOrDefault(rhsResult));
  }
  if (failed(appendReturnOperands(lhsReturnOp, newResults))) {
    return nullptr;
  }
  auto newRegionOp =
      appendRegionArgsAndResults(lhs, newArgs, newResults, rhs.getLoc());

  // Replace uses of original values with the new values.
  for (int i = 0; i < rhs.getNumResults(); ++i) {
    rhs.getResult(i).replaceAllUsesWith(
        newRegionOp.getResult(lhsReturnValues.size() + i));
  }

  // Remove rhs region.
  rhs.erase();

  // Remove results from the lhs that aren't used anymore as they may have been
  // elided when we merged as only the rhs was using them.
  newRegionOp = removeUnusedResults(newRegionOp);

  return newRegionOp;
}

// Merges multiple dispatch regions within a block into the same region,
// if possible. Operations may be reordered if it's possible to merge more while
// still obeying data dependencies.
LogicalResult mergeBlockDispatchRegions(FuncOp func, Block *parentBlock) {
  LLVM_DEBUG(llvm::dbgs() << "+++ MERGING BLOCK DISPATCH REGIONS:\n");
  SmallVector<DispatchRegionOp, 8> mergableRegions;
  for (auto &op : *parentBlock) {
    if (auto regionOp = dyn_cast<DispatchRegionOp>(op)) {
      if (isDispatchRegionMergable(regionOp)) {
        LLVM_DEBUG(llvm::dbgs() << "   -REGION MERGABLE-\n");
        mergableRegions.push_back(regionOp);
      } else {
        LLVM_DEBUG(llvm::dbgs() << "   -REGION NOT MERGABLE-\n");
      }
    }
  }
  for (int i = 0; i < mergableRegions.size(); ++i) {
    if (!mergableRegions[i]) continue;
    auto &lhs = mergableRegions[i];
    for (int j = i + 1; j < mergableRegions.size(); ++j) {
      if (!mergableRegions[j]) continue;
      auto &rhs = mergableRegions[j];
      if (!areDispatchRegionWorkloadsCompatible(lhs, rhs) ||
          areDispatchRegionsTransitivelyDependent(lhs, rhs)) {
        LLVM_DEBUG(llvm::dbgs() << "   -REGIONS INCOMPATIBLE-\n");
        continue;
      }
      if (!isDispatchRegionMergable(rhs)) {
        // TODO(b/134675461): support non-trivial control flow.
        LLVM_DEBUG(llvm::dbgs()
                   << "   -REGION CONTAINS NON-TRIVIAL CONTROL FLOW-\n");
      }
      mergableRegions[i] = mergeDispatchRegions(lhs, rhs);
      if (!mergableRegions[i]) {
        return failure();
      }
      mergableRegions[j] = nullptr;
      --i;  // Try again to see if there are subsequent regions to merge.
      LLVM_DEBUG(llvm::dbgs() << "   -> MERGED REGIONS\n");
      break;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "--- MERGED BLOCK DISPATCH REGIONS\n");
  return success();
}

}  // namespace

// Identifies dispatch regions that have compatible workloads and folds them.
// This relies on CSE having deduped workloads to simplify the logic to simply
// looking for dispatch regions using the same values.
class FoldCompatibleDispatchRegionsPass
    : public PassWrapper<FoldCompatibleDispatchRegionsPass, FunctionPass> {
 public:
  void runOnFunction() override {
    auto func = getFunction();
    for (auto &block : func) {
      if (failed(mergeBlockDispatchRegions(func, &block))) {
        return signalPassFailure();
      }
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>>
createFoldCompatibleDispatchRegionsPass() {
  return std::make_unique<FoldCompatibleDispatchRegionsPass>();
}

static PassRegistration<FoldCompatibleDispatchRegionsPass> pass(
    "iree-flow-fold-compatible-dispatch-regions",
    "Folds dispatch regions that have compatible workloads");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
