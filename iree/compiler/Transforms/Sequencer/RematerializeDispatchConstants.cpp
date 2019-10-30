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

#include "iree/compiler/IR/Ops.h"
#include "iree/compiler/Utils/DispatchUtils.h"
#include "third_party/llvm/llvm/include/llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/Ops.h"
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
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Chosen randomly for now. We can measure and see what makes sense.
constexpr int64_t kMaxRematerializedConstantSizeInBytes = 1 * 1024;

// Returns true if the constant value is under a certain threshold.
// This threshold is fixed for all backends as a value that is assumed small
// enough to be worth inlining possibly several times (at the cost of binary
// bloat).
bool isConstantSmall(ConstantOp constantOp) {
  if (auto shapedType = constantOp.getType().dyn_cast<ShapedType>()) {
    return shapedType.getSizeInBits() / 8 <=
           kMaxRematerializedConstantSizeInBytes;
  }

  // Assume anything unshaped is small. This may not always be true in custom
  // dialects but is in std for now.
  return true;
}

// Returns true if the dispatch region is allowed to have constants inside.
// Certain regions that may get replaced or turned into kernel imports shouldn't
// have the constants moved into them as they'll just get lost.
bool canDispatchRegionContainConstants(
    IREE::DispatchRegionOp dispatchRegionOp) {
  for (auto &block : dispatchRegionOp.getBody()) {
    for (auto &op : block) {
      if (isa<xla_hlo::DotOp>(&op)) {
        return false;
      }
    }
  }
  return true;
}

// Rematerializes a constant inside of all dispatch regions that use it.
// Afterward the constant is only removed if there are no other uses within the
// non-dispatch block (such as by sequencer ops).
LogicalResult rematerializeConstantInDispatchRegions(ConstantOp constantOp) {
  Value *constantValue = constantOp.getResult();
  SmallVector<IREE::DispatchRegionOp, 4> usingRegionOps;
  for (auto *user : constantValue->getUsers()) {
    if (auto dispatchRegionOp = dyn_cast<IREE::DispatchRegionOp>(user)) {
      // Ensure this isn't just the workload and is used as an arg.
      if (std::find(dispatchRegionOp.arg_operand_begin(),
                    dispatchRegionOp.arg_operand_end(),
                    constantValue) != dispatchRegionOp.arg_operand_end()) {
        if (canDispatchRegionContainConstants(dispatchRegionOp)) {
          usingRegionOps.push_back(dispatchRegionOp);
        }
      }
    }
  }
  for (auto &dispatchRegionOp : usingRegionOps) {
    if (failed(inlineDispatchRegionOperandsUsingValue(dispatchRegionOp,
                                                      constantValue))) {
      return failure();
    }
  }

  // Remove if there are no other uses within the block.
  if (constantOp.use_empty()) {
    constantOp.erase();
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
    : public FunctionPass<RematerializeDispatchConstantsPass> {
 public:
  void runOnFunction() override {
    for (auto &block : getFunction()) {
      SmallVector<ConstantOp, 8> smallConstantOps;
      for (auto constantOp : block.getOps<ConstantOp>()) {
        if (isConstantSmall(constantOp)) {
          smallConstantOps.push_back(constantOp);
        }
      }
      // Note: we iterate in reverse so that the rematerialized constants appear
      // in the same order they did originally (as insertion is at the top).
      for (auto constantOp : llvm::reverse(smallConstantOps)) {
        if (failed(rematerializeConstantInDispatchRegions(constantOp))) {
          return signalPassFailure();
        }
      }
    }
  }
};

std::unique_ptr<OpPassBase<FuncOp>> createRematerializeDispatchConstantsPass() {
  return std::make_unique<RematerializeDispatchConstantsPass>();
}

static PassRegistration<RematerializeDispatchConstantsPass> pass(
    "iree-rematerialize-dispatch-constants",
    "Rematerializes small previously-CSE'd constants into dispatch regions.");

}  // namespace iree_compiler
}  // namespace mlir
