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

#include <algorithm>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// Returns true if the constant value is a splat constant and can be
// rematerialized in a dispatch region.
bool isSplatConstant(ConstantOp constantOp) {
  if (constantOp.getValue().isa<SplatElementsAttr>()) {
    // Splats are always small and can be much better handled by broadcasting
    // within the dispatch regions.
    return true;
  } else if (auto value = constantOp.getValue().dyn_cast<DenseElementsAttr>()) {
    return value.isSplat();
  } else if (constantOp.getType().isIntOrFloat()) {
    return true;
  }

  // Assume anything unshaped is small. This may not always be true in custom
  // dialects but is in std for now.
  return false;
}

// Returns true if the dispatch region is allowed to have constants inside.
// Certain regions that may get replaced or turned into kernel imports shouldn't
// have the constants moved into them as they'll just get lost.
bool canDispatchRegionContainConstants(DispatchRegionOp dispatchRegionOp) {
  for (auto &block : dispatchRegionOp.body()) {
    for (auto &op : block) {
      // TODO(b/144530470): replace with tablegen attributes/interfaces.
      if (isa<mhlo::DotOp>(&op) || isa<mhlo::ConvOp>(&op)) {
        // These two generally result in a lot of generated code so we try to
        // keep constants out such that can dedupe more. We may still want to
        // allow some parameters in (shapes/etc).
        return false;
      }
    }
  }
  return true;
}

// Recursively clones the given |sourceOp| and returns the newly cloned op.
Operation *recursivelyCloneOp(Operation *sourceOp, OpBuilder &builder,
                              BlockAndValueMapping *mapping) {
  // Note that we dedupe required operands in the case of multiple arguments
  // coming from the same source operation.
  SmallPtrSet<Operation *, 4> operandOps;
  for (auto operand : sourceOp->getOperands()) {
    operandOps.insert(operand.getDefiningOp());
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

  OpBuilder builder = OpBuilder::atBlockEnd(targetBlock);
  builder.setInsertionPointToStart(targetBlock);
  auto *sourceOp = sourceValue.getDefiningOp();
  auto *clonedOp = recursivelyCloneOp(sourceOp, builder, mapping);

  // Return only the result matching our source value (in the case of multiple
  // results).
  int resultIndex = std::distance(
      sourceOp->result_begin(),
      std::find(sourceOp->result_begin(), sourceOp->result_end(), sourceValue));
  return clonedOp->getResult(resultIndex);
}

// Modify the second operand of the SegmentSize attribute
// TODO(ataei): Remove this once we have flow.dispatch.workgroups only here.
template <typename DispatchOpType>
void dropOperands(DispatchOpType dispatchOp, ArrayRef<unsigned> deadIndices);

template <>
void dropOperands(DispatchRegionOp dispatchRegionOp,
                  ArrayRef<unsigned> deadIndices) {
  for (unsigned argIndex : llvm::reverse(deadIndices)) {
    dispatchRegionOp.argsMutable().erase(argIndex);
  }
}

template <>
void dropOperands(DispatchWorkgroupsOp dispatchWorkgroupsOp,
                  ArrayRef<unsigned> deadIndices) {
  for (unsigned argIndex : llvm::reverse(deadIndices)) {
    dispatchWorkgroupsOp.operandsMutable().erase(argIndex);
  }
}

// Inlines use of the given |value| from outside of a dispatch region to inside
// of it and removes the argument. Supports multiple arguments that reference
// |value| and will clone the entire value tree.
template <typename DispatchOpType>
LogicalResult inlineDispatchRegionOperandsUsingValue(DispatchOpType dispatchOp,
                                                     ValueRange args,
                                                     Value value) {
  // Find all args that are using this value.
  SmallVector<unsigned, 4> argIndices;
  for (auto arg : llvm::enumerate(args)) {
    if (arg.value() == value) {
      argIndices.push_back(arg.index());
    }
  }
  if (argIndices.empty()) {
    // Not used? Wasteful call!
    return success();
  }

  // Clone the value (and the ops required to create it) into the entry block.
  auto &entryBlock = dispatchOp.body().getBlocks().front();
  BlockAndValueMapping mapping;
  auto clonedValue = cloneOpTreeIntoBlock(value, &entryBlock, &mapping);

  // Replace all uses of the inner operand with the new value.
  for (unsigned argIndex : argIndices) {
    entryBlock.getArgument(argIndex).replaceAllUsesWith(clonedValue);
  }

  // Remove the dispatch region args and the block args that have been
  // replaced.
  for (unsigned argIndex : llvm::reverse(argIndices)) {
    entryBlock.eraseArgument(argIndex);
  }
  dropOperands(dispatchOp, argIndices);

  return success();
}

// Rematerializes a constant inside of all dispatch regions that use it.
// Afterward the constant is only removed if there are no other uses within
// the non-dispatch block (such as by sequencer ops).
LogicalResult rematerializeConstantInDispatchRegions(ConstantOp constantOp) {
  Value constantValue = constantOp.getResult();
  SmallVector<DispatchRegionOp, 4> usingRegionOps;
  for (auto *user : constantValue.getUsers()) {
    if (auto dispatchRegionOp = dyn_cast<DispatchRegionOp>(user)) {
      // Ensure this isn't just the workload and is used as an arg.
      if (std::find(dispatchRegionOp.args().begin(),
                    dispatchRegionOp.args().end(),
                    constantValue) != dispatchRegionOp.args().end()) {
        if (canDispatchRegionContainConstants(dispatchRegionOp)) {
          usingRegionOps.push_back(dispatchRegionOp);
        }
      }
    }
  }
  for (auto &dispatchRegionOp : usingRegionOps) {
    if (failed(inlineDispatchRegionOperandsUsingValue<DispatchRegionOp>(
            dispatchRegionOp, dispatchRegionOp.args(), constantValue))) {
      return failure();
    }
  }
  return success();
}

LogicalResult rematerializeConstantInDispatchWorkgroupsRegions(
    ConstantOp constantOp) {
  Value constantValue = constantOp.getResult();
  for (auto *user : constantValue.getUsers()) {
    if (auto dispatchWorkgroupsOp = dyn_cast<DispatchWorkgroupsOp>(user)) {
      if (failed(inlineDispatchRegionOperandsUsingValue<DispatchWorkgroupsOp>(
              dispatchWorkgroupsOp, dispatchWorkgroupsOp.operands(),
              constantValue))) {
        return failure();
      }
    }
  }
  return success();
}

}  // namespace

// Finds constant arguments to dispatch regions that are too small to be worth
// putting into constant pools. This prevents things like a CSE'd scalar
// constant of 0.0 being passed by reference to a bunch of regions. Later
// backend-specific passes running on the dispatch regions may also be able to
// improve their constant propagation chances by having the full constant value
// available.
//
// Note that this currently only operates at the block level. Constants that are
// pushed across branches are assumed to have been rematerialized within blocks
// already, but if that isn't the case then this pass can be extended to do
// that.
class RematerializeDispatchConstantsPass
    : public PassWrapper<RematerializeDispatchConstantsPass, FunctionPass> {
 public:
  void runOnFunction() override {
    for (auto &block : getFunction()) {
      SmallVector<ConstantOp, 8> smallConstantOps;
      for (auto constantOp : block.getOps<ConstantOp>()) {
        if (isSplatConstant(constantOp)) {
          smallConstantOps.push_back(constantOp);
        }
      }
      // Note: we iterate in reverse so that the rematerialized constants appear
      // in the same order they did originally (as insertion is at the top).
      for (auto constantOp : llvm::reverse(smallConstantOps)) {
        if (failed(rematerializeConstantInDispatchRegions(constantOp))) {
          return signalPassFailure();
        }
        if (failed(
                rematerializeConstantInDispatchWorkgroupsRegions(constantOp))) {
          return signalPassFailure();
        }
        // Remove if there are no other uses within the block.
        if (constantOp.use_empty()) {
          constantOp.erase();
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>>
createRematerializeDispatchConstantsPass() {
  return std::make_unique<RematerializeDispatchConstantsPass>();
}

static PassRegistration<RematerializeDispatchConstantsPass> pass(
    "iree-flow-rematerialize-dispatch-constants",
    "Rematerializes small previously-CSE'd constants into dispatch regions");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
