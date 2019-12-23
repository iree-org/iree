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

#include "iree/compiler/Utils/DispatchUtils.h"

#include <numeric>

#include "iree/compiler/IR/Ops.h"
#include "iree/compiler/Utils/TypeConversionUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

void calculateWorkload(ArrayRef<int64_t> shape,
                       std::array<int32_t, 3> &workload) {
  // Drop the trailing ones from the shape.
  while (shape.size() > 1 && shape.back() == 1) {
    shape = shape.drop_back();
  }
  if (shape.size() <= 3) {
    // Maps to XYZ (possibly with 1's for unused dimensions).
    for (auto dim : enumerate(shape)) {
      workload[shape.size() - 1 - dim.index()] = dim.value();
    }
  } else {
    // Need to flatten the shape to fit XYZ. For now we just squash from
    // LHS.
    auto zRange = shape.drop_back(2);
    workload[2] = std::accumulate(zRange.begin(), zRange.end(), 1,
                                  std::multiplies<int32_t>());
    workload[1] = shape[shape.size() - 2];
    workload[0] = shape.back();
  }
}

Value calculateWorkload(Operation *op, Value baseOperand) {
  OpBuilder builder(op);

  std::array<int32_t, 3> workload = {1, 1, 1};

  // TODO(b/139353314): lookup/calculate based on type/etc.
  auto resultType = baseOperand->getType();
  if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
    if (!shapedType.hasStaticShape()) {
      op->emitOpError() << "Dynamic shapes not yet supported";
      return nullptr;
    }
    auto shape = shapedType.getShape();
    if (auto conv = dyn_cast_or_null<xla_hlo::ConvOp>(op)) {
      workload[2] =
          shape[conv.dimension_numbers().output_batch_dimension().getInt()];
      int i = 0;
      for (const auto &dim : conv.dimension_numbers()
                                 .output_spatial_dimensions()
                                 .getIntValues()) {
        if (i > 1) {
          break;
        }
        workload[1 - i++] = shape[dim.getSExtValue()];
      }
    } else {
      calculateWorkload(shape, workload);
    }
  }

  // TODO(b/139353314): optimize workload layout.

  auto constantType = RankedTensorType::get({3}, builder.getIntegerType(32));
  return builder.create<ConstantOp>(
      op->getLoc(), constantType,
      DenseIntElementsAttr::get(constantType, workload));
}

bool isTriviallyDispatchable(FuncOp func) {
  if (func.empty()) return false;
  auto &block = func.front();
  if (block.getOperations().size() != 2) return false;
  auto &op0 = block.front();
  auto &op1 = block.back();
  auto regionOp = dyn_cast<IREE::DispatchRegionOp>(op0);
  auto returnOp = dyn_cast<ReturnOp>(op1);
  if (!regionOp || !returnOp ||
      regionOp.getNumResults() != returnOp.getNumOperands()) {
    return false;
  }
  for (int i = 0; i < regionOp.getNumResults(); ++i) {
    if (regionOp.getResult(i) != returnOp.getOperand(i)) return false;
  }
  return true;
}

namespace {

// Returns the set of values that must be captured for use by |ops| and the
// set of values defined by |ops| that are used outside of the set.
LogicalResult analyzeOpRangeValues(
    const llvm::SmallDenseSet<Operation *> &opSet,
    llvm::SetVector<Value> *capturedValues,
    llvm::SetVector<Value> *escapingValues) {
  for (auto *op : opSet) {
    for (auto value : op->getOperands()) {
      if (!llvm::is_contained(opSet, value->getDefiningOp())) {
        // Op is using a value not in the ops set, ensure we capture it.
        capturedValues->insert(value);
      }
    }
    for (auto value : op->getResults()) {
      for (auto &use : value->getUses()) {
        if (!llvm::is_contained(opSet, use.getOwner())) {
          // An op outside of the ops set is using the value, needs to escape.
          escapingValues->insert(value);
        }
      }
    }
  }
  return success();
}

}  // namespace

LogicalResult buildDispatchRegion(FuncOp func, Block *parentBlock,
                                  Value workload, ArrayRef<Operation *> ops) {
  // Fused location with all ops.
  SmallVector<Location, 16> opLocs;
  for (auto *op : ops) {
    opLocs.push_back(op->getLoc());
  }
  auto regionLoc = FusedLoc::get(opLocs, func.getContext());

  // Get a list of values that we need to capture and values that escape the
  // region and need to be returned.
  llvm::SmallDenseSet<Operation *> opSet;
  opSet.reserve(ops.size());
  opSet.insert(ops.begin(), ops.end());
  llvm::SetVector<Value> capturedValues;
  llvm::SetVector<Value> escapingValues;
  if (failed(analyzeOpRangeValues(opSet, &capturedValues, &escapingValues))) {
    return failure();
  }
  SmallVector<Type, 8> escapingTypes;
  for (auto value : escapingValues) escapingTypes.push_back(value->getType());

  // Build the region op and add it to the parent block.
  OpBuilder parentBuilder(parentBlock);
  parentBuilder.setInsertionPoint(ops.back());
  auto dispatchRegionOp = parentBuilder.create<IREE::DispatchRegionOp>(
      regionLoc, escapingTypes, workload, capturedValues.getArrayRef());

  // Create the block and setup the arg mapping for captured values.
  auto *regionBlock = new Block();
  dispatchRegionOp.getBody().push_back(regionBlock);
  OpBuilder regionBuilder(regionBlock);
  BlockAndValueMapping mapping;
  for (auto capturedValue : capturedValues) {
    auto blockArg = regionBlock->addArgument(capturedValue->getType());
    mapping.map(capturedValue, blockArg);
  }

  // Clone ops into the new region block.
  for (auto *op : ops) {
    // Note that this updates the mapping with the new values (so at the end
    // we have those new values).
    regionBuilder.clone(*op, mapping);
  }

  // Return results (as we need a terminator in our block).
  // These are all of the values that escape our region.
  SmallVector<Value, 8> resultValues;
  for (auto oldValue : escapingValues) {
    resultValues.push_back(mapping.lookupOrDefault(oldValue));
  }
  regionBuilder.create<IREE::ReturnOp>(opLocs.back(), resultValues);

  // Replace usage of values with the results of the region.
  for (int i = 0; i < escapingValues.size(); ++i) {
    escapingValues[i]->replaceAllUsesWith(dispatchRegionOp.getResult(i));
  }

  // Remove original ops from the parent region.
  for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    (*it)->erase();
  }

  return success();
}

namespace {

// Replaces |returnOp| with a clone including |newOperands| appended.
LogicalResult appendReturnOperands(IREE::ReturnOp returnOp,
                                   ArrayRef<Value> newOperands) {
  // Insert prior to the original return.
  OpBuilder builder(returnOp);

  // Clone with new args.
  SmallVector<Value, 8> operands;
  operands.reserve(returnOp.getNumOperands() + newOperands.size());
  operands.append(returnOp.operand_begin(), returnOp.operand_end());
  operands.append(newOperands.begin(), newOperands.end());
  builder.create<IREE::ReturnOp>(returnOp.getLoc(), operands);

  // Remove original.
  returnOp.erase();

  return success();
}

// Replaces |regionOp| with a clone including |newArgs| and |newResults|.
IREE::DispatchRegionOp appendRegionArgsAndResults(
    IREE::DispatchRegionOp &regionOp, ArrayRef<Value> newArgs,
    ArrayRef<Value> newResults, Location otherLoc) {
  // Insert prior to the original region.
  OpBuilder builder(regionOp);

  // Location is original region + new region location (both probably fused).
  SmallVector<Location, 2> fusedLocs = {regionOp.getLoc(), otherLoc};
  auto fusedLoc = FusedLoc::get(fusedLocs, regionOp.getContext());

  // Clone with new results.
  SmallVector<Value, 8> operands;
  operands.append(regionOp.getArgOperands().begin(),
                  regionOp.getArgOperands().end());
  operands.append(newArgs.begin(), newArgs.end());
  SmallVector<Type, 8> resultTypes;
  resultTypes.append(regionOp.result_type_begin(), regionOp.result_type_end());
  for (auto newResult : newResults) {
    resultTypes.push_back(newResult->getType());
  }
  auto newRegionOp = builder.create<IREE::DispatchRegionOp>(
      fusedLoc, resultTypes, regionOp.getWorkload(), operands,
      regionOp.getAttrs());
  newRegionOp.getBody().takeBody(regionOp.getBody());

  // Replace uses of original values with the new values.
  for (int i = 0; i < regionOp.getNumResults(); ++i) {
    regionOp.getResult(i)->replaceAllUsesWith(newRegionOp.getResult(i));
  }

  // Erase the original region.
  regionOp.erase();

  return newRegionOp;
}

// Removes results that are not used from the dispatch region.
// Returns the new operation. There may be unused ops in the region but DCE
// should take care of that later.
IREE::DispatchRegionOp removeUnusedResults(IREE::DispatchRegionOp regionOp) {
  // Find return value within the region.
  auto &regionBlock = regionOp.getBody().getBlocks().front();
  auto returnOp = dyn_cast<IREE::ReturnOp>(regionBlock.getTerminator());
  if (!returnOp) {
    regionBlock.getParent()->getParentOfType<FuncOp>().emitError()
        << "Block does not contain an iree.return op";
  }

  // Calculate new return values.
  SmallVector<Type, 8> newReturnTypes;
  SmallVector<Value, 8> newReturnValues;
  SmallVector<Value, 8> newRegionResults;
  for (int i = 0; i < returnOp.getNumOperands(); ++i) {
    auto resultValue = regionOp.getResult(i);
    if (!resultValue->use_empty()) {
      // Still has uses so we will preserve it.
      newReturnTypes.push_back(resultValue->getType());
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
  auto newRegionOp = builder.create<IREE::DispatchRegionOp>(
      regionOp.getLoc(), newReturnTypes, regionOp.getWorkload(),
      regionOp.getArgOperands(), regionOp.getAttrs());
  newRegionOp.getBody().takeBody(regionOp.getBody());

  // Replace uses of original values with the new values.
  for (int i = 0; i < newRegionResults.size(); ++i) {
    newRegionResults[i]->replaceAllUsesWith(newRegionOp.getResult(i));
  }

  // Erase the original region.
  regionOp.erase();

  return newRegionOp;
}

// Returns true if |lhs| and |rhs| have either an identical workload or one that
// is compatible.
bool areDispatchRegionWorkloadsCompatible(IREE::DispatchRegionOp &lhs,
                                          IREE::DispatchRegionOp &rhs) {
  // TODO(benvanik): more sophisticated checking; right now it's just identical.
  return lhs.getWorkload() == rhs.getWorkload();
}

// Returns true if |value| depends in any way on |op| through any path.
bool doesValueDependOnOperation(Value value, Operation *op) {
  if (!value->getDefiningOp()) {
    return false;
  } else if (value->getDefiningOp() == op) {
    return true;
  } else if (value->getDefiningOp()->getBlock() == op->getBlock() &&
             value->getDefiningOp()->isBeforeInBlock(op)) {
    // Can't depend on |op| as it is defined prior to it.
    return false;
  }
  for (auto operand : value->getDefiningOp()->getOperands()) {
    if (doesValueDependOnOperation(operand, op)) {
      return true;
    }
  }
  return true;
}

// Returns true if |rhs| transitively depends on any out of |lhs|.
// |rhs| may depend directly on the results of |lhs| but no other ops in the
// parent block will use the results prior to |rhs|.
bool areDispatchRegionsTransitivelyDependent(IREE::DispatchRegionOp &lhs,
                                             IREE::DispatchRegionOp &rhs) {
  for (auto arg : rhs.getArgOperands()) {
    if (arg->getDefiningOp() != lhs && doesValueDependOnOperation(arg, lhs)) {
      // Transitively dependent - boo - can't merge yet.
      return true;
    }
  }
  return false;
}

// Returns true if the dispatch region contains only a single block.
// This is because our merge isn't very smart and will not preserve the CFG
// right now. We can fix this when needed.
bool isDispatchRegionMergable(IREE::DispatchRegionOp &regionOp) {
  // Disallow merging of dispatch regions containing matmuls and other big ops.
  // We do this to allow backends to lower the big op as entirely isolated such
  // that substituting library calls is easier.
  for (auto &block : regionOp.getBody().getBlocks()) {
    for (auto &op : block) {
      if (isa<xla_hlo::DotOp>(op) || isa<xla_hlo::ConvOp>(op)) {
        return false;
      }
    }
  }
  return regionOp.getBody().getBlocks().size() == 1;
}

// Merges |rhs| into |lhs| and returns the new |lhs| op.
// Precondition: !areDispatchRegionsTransitivelyDependent
IREE::DispatchRegionOp mergeDispatchRegions(IREE::DispatchRegionOp &lhs,
                                            IREE::DispatchRegionOp &rhs) {
  auto &lhsBlock = lhs.getBody().front();
  auto &rhsBlock = rhs.getBody().front();

  // Find the values used as return values in the lhs.
  // We'll need to replace the uses in rhs with these.
  auto lhsReturnOp = cast<IREE::ReturnOp>(lhsBlock.getTerminator());
  SmallVector<Value, 8> lhsReturnValues;
  lhsReturnValues.reserve(lhsReturnOp.getNumOperands());
  lhsReturnValues.append(lhsReturnOp.operand_begin(),
                         lhsReturnOp.operand_end());

  // Find the values used as return values in the rhs.
  // We'll add these to the results of the lhs region.
  auto rhsReturnOp = cast<IREE::ReturnOp>(rhsBlock.getTerminator());
  SmallVector<Value, 8> rhsReturnValues;
  rhsReturnValues.reserve(rhsReturnOp.getNumOperands());
  rhsReturnValues.append(rhsReturnOp.operand_begin(),
                         rhsReturnOp.operand_end());

  // Compute new args.
  BlockAndValueMapping mapping;
  SmallVector<Value, 8> newArgs;
  for (int rhsOpIdx = 0; rhsOpIdx < rhs.getNumArgOperands(); ++rhsOpIdx) {
    bool didElide = false;
    // Find if the rhs arg already exists on the lhs and dedupe.
    for (int lhsOpIdx = 0; lhsOpIdx < lhs.getNumArgOperands(); ++lhsOpIdx) {
      if (rhs.getArgOperand(rhsOpIdx) == lhs.getArgOperand(lhsOpIdx)) {
        mapping.map(rhsBlock.getArgument(rhsOpIdx),
                    lhsBlock.getArgument(lhsOpIdx));
        didElide = true;
        break;
      }
    }
    // Find if the arg has a direct dependency on the results of the lhs.
    for (int lhsResultIdx = 0; lhsResultIdx < lhs.getNumResults();
         ++lhsResultIdx) {
      if (rhs.getArgOperand(rhsOpIdx) == lhs.getResult(lhsResultIdx)) {
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
      auto newArg = lhsBlock.addArgument(oldArg->getType());
      mapping.map(rhsBlock.getArgument(rhsOpIdx), newArg);
      newArgs.push_back(oldArg);
    }
  }

  OpBuilder regionBuilder(&lhsBlock);

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
    rhs.getResult(i)->replaceAllUsesWith(
        newRegionOp.getResult(lhsReturnValues.size() + i));
  }

  // Remove rhs region.
  rhs.erase();

  // Remove results from the lhs that aren't used anymore as they may have been
  // elided when we merged as only the rhs was using them.
  newRegionOp = removeUnusedResults(newRegionOp);

  return newRegionOp;
}

}  // namespace

LogicalResult mergeBlockDispatchRegions(FuncOp func, Block *parentBlock) {
  SmallVector<IREE::DispatchRegionOp, 8> mergableRegions;
  for (auto &op : *parentBlock) {
    if (auto regionOp = dyn_cast<IREE::DispatchRegionOp>(op)) {
      if (isDispatchRegionMergable(regionOp)) {
        mergableRegions.push_back(regionOp);
      } else {
        regionOp.emitRemark(
            "Unable to merge into following iree.dispatch_regions; "
            "contains non-trivial control flow");
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
        continue;
      }
      if (!isDispatchRegionMergable(rhs)) {
        // TODO(b/134675461): support non-trivial control flow.
        rhs.emitRemark(
            "Unable to merge into previous iree.dispatch_region; "
            "contains non-trivial control flow");
      }
      mergableRegions[i] = mergeDispatchRegions(lhs, rhs);
      if (!mergableRegions[i]) {
        return failure();
      }
      mergableRegions[j] = nullptr;
      --i;  // Try again to see if there are subsequent regions to merge.
      break;
    }
  }

  return success();
}

namespace {

// Recursively clones the given |sourceOp| and returns the newly cloned op.
Operation *recursivelyCloneOp(Operation *sourceOp, OpBuilder &builder,
                              BlockAndValueMapping *mapping) {
  // Note that we dedupe required operands in the case of multiple arguments
  // coming from the same source operation.
  SmallPtrSet<Operation *, 4> operandOps;
  for (auto operand : sourceOp->getOperands()) {
    operandOps.insert(operand->getDefiningOp());
  }
  for (auto *operandOp : operandOps) {
    recursivelyCloneOp(operandOp, builder, mapping);
  }
  return builder.clone(*sourceOp, *mapping);
}

// Clones the |sourceValue| op tree into |targetBlock|.
// |mapping| is used to lookup existing values that may be present in the block
// such as block arguments or already cloned ancestor ops. |mapping| will be
// updated as the tree is cloned.
Value cloneOpTreeIntoBlock(Value sourceValue, Block *targetBlock,
                           BlockAndValueMapping *mapping) {
  // If the op has already been cloned we can just reuse that.
  // This happens if multiple arguments reference the same trees.
  if (auto existingValue = mapping->lookupOrNull(sourceValue)) {
    return existingValue;
  }

  OpBuilder builder(targetBlock);
  builder.setInsertionPointToStart(targetBlock);
  auto *sourceOp = sourceValue->getDefiningOp();
  auto *clonedOp = recursivelyCloneOp(sourceOp, builder, mapping);

  // Return only the result matching our source value (in the case of multiple
  // results).
  int resultIndex = std::distance(
      sourceOp->result_begin(),
      std::find(sourceOp->result_begin(), sourceOp->result_end(), sourceValue));
  return clonedOp->getResult(resultIndex);
}

}  // namespace

LogicalResult inlineDispatchRegionOperandsUsingValue(
    IREE::DispatchRegionOp dispatchRegionOp, Value value) {
  // Find all args that are using this value.
  SmallVector<unsigned, 4> argIndices;
  for (auto arg : llvm::enumerate(dispatchRegionOp.getArgOperands())) {
    if (arg.value() == value) {
      argIndices.push_back(arg.index());
    }
  }
  if (argIndices.empty()) {
    // Not used? Wasteful call!
    return success();
  }

  // Clone the value (and the ops required to create it) into the entry block.
  auto &entryBlock = dispatchRegionOp.getBody().getBlocks().front();
  BlockAndValueMapping mapping;
  auto clonedValue = cloneOpTreeIntoBlock(value, &entryBlock, &mapping);

  // Replace all uses of the inner operand with the new value.
  for (unsigned argIndex : argIndices) {
    entryBlock.getArgument(argIndex)->replaceAllUsesWith(clonedValue);
  }

  // Remove the dispatch region args and the block args that have been
  // replaced.
  for (unsigned argIndex : llvm::reverse(argIndices)) {
    dispatchRegionOp.getOperation()->eraseOperand(
        dispatchRegionOp.mapArgOperandToOpOperand(argIndex));
    entryBlock.eraseArgument(argIndex);
  }

  return success();
}

namespace {

// Recursively finds all reachable functions from the given |rootFunc| and adds
// them to the |reachableFuncs| set.
//
// Note that indirect calls are not supported, however we don't allow those in
// dispatch regions anyway so they should not be present here.
LogicalResult findReachableFunctions(Operation *rootFunc,
                                     llvm::SetVector<FuncOp> &reachableFuncs) {
  bool allCallsValid = true;
  rootFunc->walk([&](CallOp op) {
    auto callee = rootFunc->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
        op.getCallee());
    if (!callee.getAttr("iree.dispatchable")) {
      allCallsValid = false;
      rootFunc->emitError() << callee.getName().str() << " is not dispatchable";
      return;
    }
    if (reachableFuncs.insert(callee)) {
      findReachableFunctions(callee, reachableFuncs);
    }
  });
  return success(allCallsValid);
}

}  // namespace

std::pair<IREE::MultiArchExecutableOp, FuncOp> createRegionExecutable(
    Operation *op, FunctionType functionType, StringRef symbolSuffix) {
  // Create the function and take the region body directly.
  // NOTE: this will get uniquified if we have multiple in the same block.
  auto parentFunc = op->getParentOfType<FuncOp>();
  std::string functionName =
      (parentFunc.getName().str() + "_rgn" + symbolSuffix).str();
  auto outlinedFunc = FuncOp::create(op->getLoc(), functionName, functionType);
  BlockAndValueMapping mapping;
  op->getRegion(0).cloneInto(&outlinedFunc.getBody(), mapping);

  // Gather all reachable functions.
  llvm::SetVector<FuncOp> reachableFuncs;
  findReachableFunctions(outlinedFunc, reachableFuncs);

  // Create the multi-arch executable that will contain the outlined region.
  // NOTE: this will get uniquified if we have multiple in the same block.
  auto parentModule = parentFunc.getParentOfType<ModuleOp>();
  OpBuilder parentModuleBuilder(parentModule);
  parentModuleBuilder.setInsertionPoint(parentFunc);
  std::string executableName =
      (parentFunc.getName().str() + "_ex" + symbolSuffix).str();
  auto multiArchExecutable =
      parentModuleBuilder.create<IREE::MultiArchExecutableOp>(
          outlinedFunc.getLoc(), executableName);

  // Create the executable op initially unspecified so that later
  // transformations can compile it to various formats.
  OpBuilder multiArchExecutableBuilder(multiArchExecutable);
  multiArchExecutableBuilder.setInsertionPointToStart(
      &multiArchExecutable.getBlock());
  auto executable = multiArchExecutableBuilder.create<IREE::ExecutableOp>(
      outlinedFunc.getLoc(), IREE::ExecutableFormat::Unspecified);

  // Create the inner ModuleOp that contains the original functions. We need
  // to provide this shim as some ops (like std.call) look for the
  // containing module to provide symbol resolution.
  OpBuilder executableBuilder(executable);
  executableBuilder.setInsertionPointToStart(&executable.getBlock());
  auto innerModule = executableBuilder.create<ModuleOp>(outlinedFunc.getLoc());

  // TODO(b/137674142): make an ExecutableEntryPointOp and convert the
  // entry thunk into that format.
  innerModule.push_back(outlinedFunc);

  // Copy all reachable functions into the executable.
  // Linker passes may dedupe these later on.
  for (auto reachableFunc : reachableFuncs) {
    auto clonedFunc = reachableFunc.clone();
    clonedFunc.removeAttr("iree.dispatchable");
    innerModule.push_back(clonedFunc);
  }

  return std::make_pair(multiArchExecutable, outlinedFunc);
}

Value insertDispatcherStore(Operation *op, Value value, OpBuilder &builder) {
  if (!value) {
    return nullptr;
  }

  // If the previous value was already a memref we don't need to change
  // anything.
  // TODO(benvanik): ensure indices make sense.
  if (value->getType().isa<MemRefType>()) {
    return value;
  } else if (value->getType().isa<TensorType>()) {
    auto castOp = builder.create<IREE::TensorToMemRefOp>(op->getLoc(), value);
    return castOp.getResult();
  }

  // Allocate the memref to store the value.
  auto newStorage = builder.create<AllocOp>(
      op->getLoc(), convertTypeToMemRef(value->getType()));

  // Insert the store we'll use to box the value.
  builder.create<StoreOp>(op->getLoc(), value, newStorage, ArrayRef<Value>{});

  return newStorage;
}

Value insertDispatcherLoad(Operation *op, Value originalValue,
                           Value allocatedValue, OpBuilder &builder) {
  // If old value was a memref we don't need to change anything.
  if (originalValue->getType().isa<MemRefType>()) {
    return allocatedValue;
  } else if (originalValue->getType().isa<TensorType>()) {
    auto castOp =
        builder.create<IREE::MemRefToTensorOp>(op->getLoc(), allocatedValue);
    originalValue->replaceAllUsesWith(castOp.getResult());
    return castOp.getResult();
  }

  // Insert the load we'll use to unbox the value.
  auto loadOp =
      builder.create<LoadOp>(op->getLoc(), allocatedValue, ArrayRef<Value>{});
  originalValue->replaceAllUsesWith(loadOp);
  return loadOp;
}

// TODO(benvanik): enough information to walk into dispatch region and compute
// shape when not static.
Value allocateDispatchOutputBuffer(Location loc, MemRefType type,
                                   OpBuilder &builder) {
  // TODO(benvanik): allocation algorithm:
  // - synthesize shape logic (magic) [[ for now assume fixed shapes ]]
  // - insert shape logic above region
  //   - rely on folding to merge multiple calculations together
  //   - unranked = death, need to be able to alloc shape outputs
  // - insert alloc
  SmallVector<Value, 4> dimPieces;
  return builder.create<AllocOp>(loc, type, dimPieces);
}

}  // namespace iree_compiler
}  // namespace mlir
