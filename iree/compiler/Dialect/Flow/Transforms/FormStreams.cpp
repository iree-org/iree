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

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/Utils/TypeConversion.h"
#include "iree/compiler/Utils/GraphUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
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

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Returns true if the given op can be used within a stream.
static bool isStreamableOp(Operation *op) {
  if (auto streamableOp = dyn_cast<StreamableOpInterface>(op)) {
    return streamableOp.isUsableInStream();
  }
  if (llvm::isa<Shape::TieShapeOp>(op)) {
    return true;
  }
  return false;
}

static inline bool usefulStreamOp(Operation *op) {
  return op->getDialect()->getNamespace() !=
         ShapeDialect::getDialectNamespace();
}

static inline bool usefulStreamWork(ArrayRef<Operation *> currentStreamOps) {
  return llvm::any_of(currentStreamOps, usefulStreamOp);
}

// Expand any compound types to primitive types in the stream fragment.
static void expandFragmentToPrimitiveTypes(ExStreamFragmentOp fragmentOp) {
  auto loc = fragmentOp.getLoc();
  Block *entryBlock = &fragmentOp.body().front();
  auto &typeExpander = Shape::getShapeToPrimitiveTypeExpander();
  OpBuilder expandBuilder(fragmentOp.getContext());
  typeExpander.expandBlockSignature(loc, entryBlock, expandBuilder);
  SmallVector<Value, 4> origFragmentArgs(fragmentOp.args());
  SmallVector<Value, 4> newFragmentArgs;
  expandBuilder.setInsertionPoint(fragmentOp);
  typeExpander.expandSourceValuesToTarget(loc, origFragmentArgs,
                                          newFragmentArgs, expandBuilder);
  fragmentOp.getOperation()->setOperands(newFragmentArgs);
}

// Temporary hack to get the experimental stream ops constructed. In the future
// this will run an analysis to identify compatible dispatches across the entire
// function CFG, create the streams, and then thread the streams through the CFG
// to append additional stream work. For now, we just look at basic blocks and
// cluster adjacent dispatches and flow ops together.
class FormStreamsPass : public PassWrapper<FormStreamsPass, FunctionPass> {
 public:
  void runOnFunction() override {
    for (auto &block : getFunction()) {
      auto streams = findStreamsInBlock(block);
      for (auto &streamOps : streams) {
        formStreamFragmentInBlock(block, std::move(streamOps));
      }
    }
  }

  // Returns an ordered list of streams within the block.
  // Each stream contains one or more ops that are stream-compatible.
  SmallVector<SmallVector<Operation *, 8>, 8> findStreamsInBlock(Block &block) {
    SmallVector<Operation *, 8> currentStreamOps;
    SmallVector<SmallVector<Operation *, 8>, 8> streams;
    for (Operation &op : block) {
      if (isStreamableOp(&op)) {
        currentStreamOps.push_back(&op);
        continue;
      }
      if (usefulStreamWork(currentStreamOps)) {
        streams.push_back(currentStreamOps);
      }
      currentStreamOps = {};
    }
    if (usefulStreamWork(currentStreamOps)) {
      streams.push_back(currentStreamOps);
    }
    currentStreamOps = {};

    return streams;
  }

  // Forms a stream fragment containing the identified stream ops and removes
  // the originals from the parent block.
  void formStreamFragmentInBlock(Block &block,
                                 SmallVector<Operation *, 8> streamOps) {
    auto *context = block.getParent()->getContext();
    OpBuilder blockBuilder = OpBuilder::atBlockEnd(&block);
    blockBuilder.setInsertionPointAfter(streamOps.back());
    auto fragmentLoc = FusedLoc::get(
        llvm::to_vector<8>(llvm::map_range(
            streamOps, [](Operation *op) { return op->getLoc(); })),
        context);

    // Find all input operands and results that escape the fragment.
    llvm::SmallSetVector<Operation *, 8> streamOpSet{streamOps.begin(),
                                                     streamOps.end()};
    SmallVector<Value, 8> fragmentOperands;
    SmallVector<Value, 8> fragmentResults;
    SmallVector<Type, 8> fragmentResultTypes;
    SmallVector<Operation *, 4> tieShapeOps;
    SmallVector<Value, 8> outsideTieShapeOperands;
    for (auto *op : streamOps) {
      for (auto operand : op->getOperands()) {
        if (std::find(fragmentOperands.begin(), fragmentOperands.end(),
                      operand) == fragmentOperands.end()) {
          if (!operand.getDefiningOp() ||
              !streamOpSet.count(operand.getDefiningOp())) {
            fragmentOperands.push_back(operand);

            auto operandDefiningOp = operand.getDefiningOp();
            if (operandDefiningOp &&
                llvm::isa<Shape::TieShapeOp>(operandDefiningOp)) {
              tieShapeOps.push_back(operand.getDefiningOp());
              auto definingOp =
                  dyn_cast<Shape::TieShapeOp>(operand.getDefiningOp());
              for (auto arg : definingOp.getOperands()) {
                outsideTieShapeOperands.push_back(arg);
              }
            }
          }
        }
      }

      for (auto result : op->getResults()) {
        bool onlyStreamUses = true;
        for (auto &use : result.getUses()) {
          if (!streamOpSet.count(use.getOwner())) {
            onlyStreamUses = false;
            break;
          }
        }
        if (!onlyStreamUses) {
          fragmentResults.push_back(result);
          fragmentResultTypes.push_back(result.getType());
        }
      }
    }

    // TODO(Tao Peng): pass args(operand and shape) which need by outside
    // tie_shape into fragment body, and ignore the tie_shape arg passed into
    // the fragment, it will not be used, and will be deleted by canonicalizer
    // later.
    outsideTieShapeOperands.append(fragmentOperands.begin(),
                                   fragmentOperands.end());
    fragmentOperands = outsideTieShapeOperands;

    // Create the fragment and clone in all of the ops.
    auto fragmentOp = blockBuilder.create<ExStreamFragmentOp>(
        fragmentLoc, fragmentResultTypes, fragmentOperands);
    auto *entryBlock = new Block();
    fragmentOp.body().getBlocks().push_back(entryBlock);
    entryBlock->addArguments(llvm::to_vector<8>(fragmentOp.getOperandTypes()));
    BlockAndValueMapping mapping;
    for (auto arg : entryBlock->getArguments()) {
      mapping.map(fragmentOperands[arg.getArgNumber()], arg);
    }
    OpBuilder fragmentBuilder = OpBuilder::atBlockEnd(entryBlock);
    for (auto *op : tieShapeOps) {
      fragmentBuilder.clone(*op, mapping);
    }
    for (auto *op : streamOps) {
      fragmentBuilder.clone(*op, mapping);
    }
    fragmentBuilder.create<IREE::Flow::ReturnOp>(
        UnknownLoc::get(context),
        llvm::to_vector<8>(llvm::map_range(fragmentResults, [&](Value value) {
          return mapping.lookup(value);
        })));
    for (auto resultOldNew :
         llvm::zip(fragmentResults, fragmentOp.getResults())) {
      auto oldValue = std::get<0>(resultOldNew);
      auto newValue = std::get<1>(resultOldNew);
      oldValue.replaceAllUsesWith(newValue);
    }

    // Erase the ops from the block now that we've cloned them.
    for (auto *op : llvm::reverse(streamOps)) {
      op->erase();
    }

    // Expand any shape types to corresponding primitives.
    expandFragmentToPrimitiveTypes(fragmentOp);
  }
};

std::unique_ptr<OperationPass<FuncOp>> createFormStreamsPass() {
  return std::make_unique<FormStreamsPass>();
}

static PassRegistration<FormStreamsPass> pass(
    "iree-flow-form-streams",
    "Identifies dispatches that can be grouped into streams within functions");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
