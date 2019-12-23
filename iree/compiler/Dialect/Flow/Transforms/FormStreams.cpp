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
  return false;
}

// Temporary hack to get the experimental stream ops constructed. In the future
// this will run an analysis to identify compatible dispatches across the entire
// function CFG, create the streams, and then thread the streams through the CFG
// to append additional stream work. For now, we just look at basic blocks and
// cluster all of the dispatches and flow ops together.
class FormStreamsPass : public FunctionPass<FormStreamsPass> {
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
    // Identify all ops with side-effects so that we can quickly identify when
    // we need to bail.
    SmallVector<Operation *, 8> opsWithSideEffects;
    for (auto &op : block) {
      if (!op.hasNoSideEffect() || op.hasTrait<OpTrait::IREE::YieldPoint>()) {
        opsWithSideEffects.push_back(&op);
      }
    }

    llvm::SmallSetVector<Operation *, 8> processedOps;
    SmallVector<Operation *, 8> currentStreamOps;
    llvm::SmallSetVector<Operation *, 8> currentOutsideOps;
    SmallVector<SmallVector<Operation *, 8>, 8> streams;
    llvm::SmallSetVector<Operation *, 8> scanList;

    auto resetCurrentStream = [&]() {
      if (!currentStreamOps.empty()) {
        streams.push_back(std::move(currentStreamOps));
        currentStreamOps = {};
      }
      currentOutsideOps.clear();
      scanList.clear();
    };

    auto markOpDAGOutside = [&](Operation *op) {
      llvm::SmallSetVector<Operation *, 8> markList;
      markList.insert(op);
      while (!markList.empty()) {
        auto *nextOp = markList.pop_back_val();
        if (!currentOutsideOps.insert(nextOp)) continue;
        for (auto operand : nextOp->getOperands()) {
          if (operand->getDefiningOp()) {
            markList.insert(operand->getDefiningOp());
          }
        }
      }
    };

    // Returns true if the stream can continue growing and false if the stream
    // should be split at the current op.
    auto scanAndFormStream = [&](Operation *op, Operation *blockerOp) {
      if (processedOps.count(op)) {
        // Op has already been added to a stream and can be skipped.
        return;
      }
      processedOps.insert(op);

      if (!op->hasNoSideEffect()) {
        // Op has side-effects and should split the current stream.
        resetCurrentStream();
        return;
      } else if (!isStreamableOp(op)) {
        // Op is not from the flow dialect and should be ignored.
        markOpDAGOutside(op);
        return;
      }

      if (currentOutsideOps.count(op)) {
        resetCurrentStream();
      }
      // Add the flow op to the current stream.
      currentStreamOps.push_back(op);

      // Recursively work through the inputs of the op to pull in any
      // dependencies that we are able to (are flow ops, have no side-effects).
      for (auto operand : op->getOperands()) {
        auto *depOp = operand->getDefiningOp();
        if (!depOp) {
          // Op is a block arg.
          continue;
        } else if (depOp->getBlock() != op->getBlock()) {
          // Op is from another block; we only are interested in ops defined
          // within this block as future optimizations will deal with
          // cross-block dependencies.
          markOpDAGOutside(depOp);
          continue;
        } else if (!depOp->hasNoSideEffect() || !isStreamableOp(depOp)) {
          // Source op has side effects or isn't streamable meaning that we
          // can't fold it into the stream region.
          markOpDAGOutside(depOp);
          continue;
        } else if (blockerOp && depOp->isBeforeInBlock(blockerOp)) {
          // The dep reaches across an op with side effects (even though it
          // doesn't have any data flowing through it). We are strict here and
          // even if the dep doesn't have side-effects we assume that all
          // side-effecting ops act as barriers.
          markOpDAGOutside(depOp);
          continue;
        }
        if (!currentOutsideOps.count(depOp)) {
          scanList.insert(depOp);
        }
      }
    };

    for (auto &op : llvm::reverse(block.getOperations())) {
      // Find the op prior to |op| that has side-effects.
      // We use this to block formation so that we don't try to pull in ops
      // across any op with side-effects.
      // TODO(benvanik): maybe make this less strict? Maybe just YieldPoint?
      while (!opsWithSideEffects.empty() &&
             op.isBeforeInBlock(opsWithSideEffects.back())) {
        opsWithSideEffects.pop_back();
      }
      Operation *blockerOp =
          opsWithSideEffects.empty() ? nullptr : opsWithSideEffects.back();

      // Scan all ops and try to pull them into a stream.
      if (isStreamableOp(&op)) {
        scanList.insert(&op);
        while (!scanList.empty()) {
          scanAndFormStream(scanList.pop_back_val(), blockerOp);
        }
      }
    }
    resetCurrentStream();

    // Reverse the streams as we iterated in reverse order.
    // We need to sort all of the ops within each stream as they may have been
    // inserted out of order during the scan.
    for (auto &stream : streams) {
      stream = sortOpsTopologically<8>(stream);
    }
    std::reverse(streams.begin(), streams.end());
    return streams;
  }

  // Forms a stream fragment containing the identified stream ops and removes
  // the originals from the parent block.
  void formStreamFragmentInBlock(Block &block,
                                 SmallVector<Operation *, 8> streamOps) {
    auto *context = block.getParent()->getContext();
    OpBuilder blockBuilder(&block);
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
    for (auto *op : streamOps) {
      for (auto operand : op->getOperands()) {
        if (std::find(fragmentOperands.begin(), fragmentOperands.end(),
                      operand) == fragmentOperands.end()) {
          if (!operand->getDefiningOp() ||
              !streamOpSet.count(operand->getDefiningOp())) {
            fragmentOperands.push_back(operand);
          }
        }
      }
      for (auto result : op->getResults()) {
        bool onlyStreamUses = true;
        for (auto &use : result->getUses()) {
          if (!streamOpSet.count(use.getOwner())) {
            onlyStreamUses = false;
            break;
          }
        }
        if (!onlyStreamUses) {
          fragmentResults.push_back(result);
          fragmentResultTypes.push_back(result->getType());
        }
      }
    }

    // Create the fragment and clone in all of the ops.
    auto fragmentOp = blockBuilder.create<ExStreamFragmentOp>(
        fragmentLoc, fragmentResultTypes, fragmentOperands);
    auto *entryBlock = new Block();
    fragmentOp.body().getBlocks().push_back(entryBlock);
    entryBlock->addArguments(llvm::to_vector<8>(fragmentOp.getOperandTypes()));
    BlockAndValueMapping mapping;
    for (auto arg : entryBlock->getArguments()) {
      mapping.map(fragmentOperands[arg->getArgNumber()], arg);
    }
    OpBuilder fragmentBuilder(entryBlock);
    for (auto *op : streamOps) {
      fragmentBuilder.clone(*op, mapping);
    }
    fragmentBuilder.create<ReturnOp>(
        UnknownLoc::get(context),
        llvm::to_vector<8>(llvm::map_range(fragmentResults, [&](Value value) {
          return mapping.lookup(value);
        })));
    for (auto resultOldNew :
         llvm::zip(fragmentResults, fragmentOp.getResults())) {
      auto oldValue = std::get<0>(resultOldNew);
      auto newValue = std::get<1>(resultOldNew);
      oldValue->replaceAllUsesWith(newValue);
    }

    // Erase the ops from the block now that we've cloned them.
    for (auto *op : llvm::reverse(streamOps)) {
      op->erase();
    }
  }
};

std::unique_ptr<OpPassBase<FuncOp>> createFormStreamsPass() {
  return std::make_unique<FormStreamsPass>();
}

static PassRegistration<FormStreamsPass> pass(
    "iree-flow-form-streams",
    "Identifies dispatches that can be grouped into streams within functions");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
