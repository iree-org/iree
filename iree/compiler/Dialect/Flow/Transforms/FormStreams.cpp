// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/IR/Builders.h"
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

// Returns true if the given op can be used within a stream.
static bool isStreamableOp(Operation *op) {
  if (isa<StreamableOpInterface>(op)) {
    return true;
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

// Temporary hack to get the experimental stream ops constructed. In the future
// this will run an analysis to identify compatible dispatches across the entire
// function CFG, create the streams, and then thread the streams through the CFG
// to append additional stream work. For now, we just look at basic blocks and
// cluster adjacent dispatches and flow ops together.
//
// TODO(#7277): remove when switched to streams (happens there now).
class FormStreamsPass : public FormStreamsBase<FormStreamsPass> {
 public:
  void runOnOperation() override {
    for (auto &block : getOperation()) {
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
        context, llvm::to_vector<8>(llvm::map_range(
                     streamOps, [](Operation *op) { return op->getLoc(); })));

    // Find all input operands and results that escape the fragment.
    llvm::SmallSetVector<Operation *, 8> streamOpSet{streamOps.begin(),
                                                     streamOps.end()};
    SmallVector<Value, 8> fragmentOperands;
    SmallVector<Value, 8> fragmentOperandDims;
    SmallVector<Value, 8> fragmentResults;
    SmallVector<Value, 8> fragmentResultDims;
    SmallVector<Type, 8> fragmentResultTypes;
    SmallVector<int64_t, 8> fragmentTiedOperands;
    for (auto *op : streamOps) {
      for (auto operand : op->getOperands()) {
        if (std::find(fragmentOperands.begin(), fragmentOperands.end(),
                      operand) == fragmentOperands.end()) {
          if (!operand.getDefiningOp() ||
              !streamOpSet.count(operand.getDefiningOp())) {
            fragmentOperands.push_back(operand);
            if (operand.getType().isa<ShapedType>()) {
              auto dynamicDims = Shape::buildOrFindDynamicDimsForValue(
                  fragmentLoc, operand, blockBuilder);
              fragmentOperandDims.append(dynamicDims);
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
          if (result.getType().isa<ShapedType>()) {
            auto dynamicDims = Shape::buildOrFindDynamicDimsForValue(
                fragmentLoc, result, blockBuilder);
            fragmentResultDims.append(dynamicDims);
          }
        }
      }
    }

    // Create the fragment and clone in all of the ops.
    auto fragmentOp = blockBuilder.create<ExStreamFragmentOp>(
        fragmentLoc, fragmentResultTypes, fragmentResultDims, fragmentOperands,
        fragmentOperandDims, fragmentTiedOperands);
    auto *entryBlock = new Block();
    fragmentOp.body().getBlocks().push_back(entryBlock);
    entryBlock->addArguments(TypeRange(fragmentOp.operands()));
    BlockAndValueMapping mapping;
    for (unsigned i = 0; i < fragmentOperands.size(); ++i) {
      auto arg = entryBlock->getArgument(i);
      mapping.map(fragmentOperands[arg.getArgNumber()], arg);
    }
    OpBuilder fragmentBuilder = OpBuilder::atBlockEnd(entryBlock);
    for (auto *op : streamOps) {
      fragmentBuilder.clone(*op, mapping);
    }
    fragmentBuilder.create<IREE::Flow::ReturnOp>(
        UnknownLoc::get(context),
        llvm::to_vector<8>(llvm::map_range(fragmentResults, [&](Value value) {
          return mapping.lookup(value);
        })));
    for (auto resultOldNew : llvm::zip(fragmentResults, fragmentOp.results())) {
      auto oldValue = std::get<0>(resultOldNew);
      auto newValue = std::get<1>(resultOldNew);
      oldValue.replaceAllUsesWith(newValue);
    }

    // Erase the ops from the block now that we've cloned them.
    // Note the backwards order as the ops may have dependencies on each other
    // and we have to erase the consumers before the producers.
    for (auto *op : llvm::reverse(streamOps)) {
      op->erase();
    }
  }
};

std::unique_ptr<OperationPass<mlir::FuncOp>> createFormStreamsPass() {
  return std::make_unique<FormStreamsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
