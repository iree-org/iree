// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <tuple>

#include "iree/split_mlir/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree {
namespace split_mlir {

#define GEN_PASS_DEF_OUTLINEFUNCTIONS
#include "iree/split_mlir/Passes.h.inc"  // IWYU pragma: export

namespace {

// Collect all operation ranges that are marked for outlining.
// The begining of a range is marked with the outline_range_first attribute.
// The last operation of a range is marked with the outline_range_last attribue.
// Example:
// %0 = arith.addi %arg0, %arg1 {outline_range_first} : i32
// %1 = arith.addi %arg2, %arg3 : i32
// %2 = arith.muli %arg3, %arg4 {outline_range_last} : i32
// The outline range will consist of the 3 operations.
LogicalResult getOutlineOpRanges(
    Block& block, SmallVector<iterator_range<Block::iterator>, 4>& res) {
  bool isInOutliningRange = false;
  Block::iterator rangeBegin;
  for (Block::iterator opIt = block.begin(); opIt != block.end(); ++opIt) {
    if (opIt->hasAttr("outline_range_first")) {
      if (isInOutliningRange) {
        return LogicalResult::failure();
      }
      isInOutliningRange = true;
      rangeBegin = opIt;
    }

    if (opIt->hasAttr("outline_range_last")) {
      if (!isInOutliningRange) {
        return LogicalResult::failure();
      }
      isInOutliningRange = false;
      res.emplace_back(rangeBegin, std::next(opIt));
    }
  }
  if (isInOutliningRange) {
    // No matching closing marker outline_range_last.
    return LogicalResult::failure();
  }

  return LogicalResult::success();
}

// Return all values that are an operand of some of the given ops that are
// produced by other ops. Also return all values that are a result of some of
// the given ops and have uses outside the ops range.
std::pair<SmallVector<Value, 64>, SmallVector<OpResult, 64>>
getOperandsAndResultsForIsolation(iterator_range<Block::iterator> opRange,
                                  const SmallPtrSet<Operation*, 4>& opsSet) {
  SmallVector<Value, 64> operands;
  SmallVector<OpResult, 64> results;
  SmallPtrSet<Value, 16> operandsSet;
  SmallPtrSet<OpResult, 16> resultsSet;
  for (Operation& op : opRange) {
    for (Value operand : op.getOperands()) {
      if (!opsSet.contains(operand.getDefiningOp())) {
        auto insertionResult = operandsSet.insert(operand);
        if (insertionResult.second) {
          operands.push_back(operand);
        }
      }
    }
    for (OpResult result : op.getResults()) {
      for (OpOperand operand : result.getUsers()) {
        if (!opsSet.contains(operand.getOwner())) {
          auto insertionResult = resultsSet.insert(result);
          if (insertionResult.second) {
            results.push_back(result);
          }
          break;
        }
      }
    }
  }
  return {operands, results};
}

template <typename ValueIt>
void replaceValueUsesWithNewBlockArguments(ValueIt valuesBegin,
                                           ValueIt valuesEnd, Block& block) {
  for (ValueIt valIt = valuesBegin; valIt != valuesEnd; ++valIt) {
    block.addArgument(valIt->getType(), valIt->getLoc());
    BlockArgument& blockArg = block.getArguments().back();
    valIt->replaceUsesWithIf(blockArg, [&block](OpOperand& operand) {
      return operand.getOwner()->getBlock() == &block;
    });
  }
}

void addBlockReturn(Block& block, ValueRange operands, OpBuilder& builder) {
  func::ReturnOp returnOp =
      builder.create<func::ReturnOp>(builder.getUnknownLoc(), operands);
  block.push_back(returnOp);
}

void moveOpsIntoBlock(iterator_range<Block::iterator> opRange, Block& block) {
  // Put ops into another container because opRange will be invalidated during
  // removal.
  SmallVector<Operation*, 64> ops;
  std::transform(opRange.begin(), opRange.end(), std::back_inserter(ops),
                 [](Operation& op) { return &op; });
  for (Operation* op : ops) {
    op->moveBefore(&block, block.end());
  }
}

void moveBlock(Region& srcRegion, Region& destRegion,
               Region::iterator srcBlockIt, Region::iterator destBlockIt) {
  Block* block = srcRegion.getBlocks().remove(srcBlockIt);
  destRegion.getBlocks().insert(destBlockIt, block);
}

bool isAncestorOfBlock(Operation* op, Block* block) {
  // Walk up the operation hierarchy and check each block.
  while (op != nullptr) {
    if (op->getBlock() == block) {
      return true;
    }
    op = op->getParentOp();
  }
  return false;
}

template <typename OriginalOpResultsIt, typename NewOpResultsIt>
void substititeUses(OriginalOpResultsIt originalBegin,
                    OriginalOpResultsIt originalEnd, NewOpResultsIt newBegin,
                    NewOpResultsIt newEnd, Block& excludedBlock) {
  assert(std::distance(originalBegin, originalEnd) ==
         std::distance(newBegin, newEnd));
  auto newIt = newBegin;
  for (auto originalIt = originalBegin; originalIt != originalEnd;
       ++originalIt, ++newIt) {
    originalIt->replaceUsesWithIf(*newIt, [&excludedBlock](OpOperand& operand) {
      return !isAncestorOfBlock(operand.getOwner(), &excludedBlock);
    });
  }
}

// All operations in the range `opRange` are moved into a new function with name
// `name`. The resulting function is put inside `moduleOp` and is properly
// isolated from above. This does not insert a call to the new function in place
// of the moved operations.
func::FuncOp createFunctionFromOps(iterator_range<Block::iterator> opRange,
                                   StringRef name, ModuleOp moduleOp,
                                   SmallVector<Value, 64>& rangeOperands,
                                   SmallVector<OpResult, 64>& rangeResults,
                                   OpBuilder& builder) {
  Region& region = *opRange.begin()->getParentRegion();
  Block& dstBlock = region.emplaceBlock();
  moveOpsIntoBlock(opRange, dstBlock);
  replaceValueUsesWithNewBlockArguments(rangeOperands.begin(),
                                        rangeOperands.end(), dstBlock);
  addBlockReturn(dstBlock,
                 ArrayRef<Value>(rangeResults.begin(), rangeResults.end()),
                 builder);
  func::FuncOp funcOp = builder.create<func::FuncOp>(
      builder.getUnknownLoc(), name,
      FunctionType::get(builder.getContext(), dstBlock.getArgumentTypes(),
                        dstBlock.back().getOperandTypes()));
  moduleOp.getBodyRegion().getBlocks().front().push_back(funcOp);
  moveBlock(region, funcOp.getBody(), std::prev(region.end()),
            funcOp.getBody().end());

  return funcOp;
}

void createCall(func::FuncOp funcOp, Block& block, Block::iterator pos,
                SmallVector<Value, 64>& rangeOperands,
                SmallVector<OpResult, 64>& rangeResults, OpBuilder& builder) {
  func::CallOp callOp = builder.create<func::CallOp>(
      builder.getUnknownLoc(), funcOp,
      ArrayRef<Value>(rangeOperands.begin(), rangeOperands.end()));
  block.getOperations().insert(pos, callOp);
  substititeUses(rangeResults.begin(), rangeResults.end(),
                 callOp.getResults().begin(), callOp.getResults().end(),
                 funcOp.getBody().back());
}

std::optional<func::FuncOp> outlineOpRange(
    iterator_range<Block::iterator> opRange, StringRef name, ModuleOp moduleOp,
    OpBuilder& builder) {
  if (opRange.empty()) {
    return std::nullopt;
  }

  SmallPtrSet<Operation*, 4> opsSet;
  for (Operation& op : opRange) {
    opsSet.insert(&op);
  }
  SmallVector<Value, 64> rangeOperands;
  SmallVector<OpResult, 64> rangeResults;
  std::tie(rangeOperands, rangeResults) =
      getOperandsAndResultsForIsolation(opRange, opsSet);
  Block& srcBlock = *opRange.begin()->getBlock();

  func::FuncOp funcOp = createFunctionFromOps(
      opRange, name, moduleOp, rangeOperands, rangeResults, builder);
  createCall(funcOp, srcBlock, opRange.end(), rangeOperands, rangeResults,
             builder);

  return funcOp;
}

std::string getOutlinedFuncName(StringRef prefix, int blockIndex,
                                int outlineRangeIndex) {
  return (Twine(prefix) + "_outline_" + Twine(blockIndex) + "_" +
          Twine(outlineRangeIndex))
      .str();
}

void removeOutlineMarkers(iterator_range<Block::iterator> opRange) {
  if (opRange.empty()) {
    return;
  }
  opRange.begin()->removeAttr("outline_range_first");
  std::prev(opRange.end())->removeAttr("outline_range_last");
}

// Each marked operation range in `funcOp` is outlined into a new function.
// A call to the new function is inserted in place of the outlined operations.
LogicalResult outlineOpRanges(func::FuncOp funcOp, ModuleOp moduleOp,
                              OpBuilder& builder) {
  Region& funcBody = funcOp.getFunctionBody();
  SmallVector<iterator_range<Block::iterator>, 4> outlineRanges;
  for (auto blockIt : llvm::enumerate(funcBody.getBlocks())) {
    outlineRanges.clear();
    if (failed(getOutlineOpRanges(blockIt.value(), outlineRanges))) {
      return LogicalResult::failure();
    }
    for (auto rangeIt : llvm::enumerate(outlineRanges)) {
      removeOutlineMarkers(rangeIt.value());
      std::string name = getOutlinedFuncName(funcOp.getSymName(),
                                             blockIt.index(), rangeIt.index());
      outlineOpRange(rangeIt.value(), name, moduleOp, builder);
    }
  }

  return LogicalResult::success();
}

struct OutlineFunctionsPass
    : public impl::OutlineFunctionsBase<OutlineFunctionsPass> {
  using OutlineFunctionsBase::OutlineFunctionsBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    Block& moduleBlock = *moduleOp.getBody();
    OpBuilder builder(&getContext());
    // Get all functions since we are going to insert new ones
    // that we don't want to iterate over.
    SmallVector<func::FuncOp, 4> funcOps(
        moduleBlock.getOps<func::FuncOp>().begin(),
        moduleBlock.getOps<func::FuncOp>().end());
    for (func::FuncOp op : funcOps) {
      if (failed(outlineOpRanges(op, moduleOp, builder))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createOutlineFunctionsPass() {
  return std::make_unique<OutlineFunctionsPass>();
}

}  // namespace split_mlir
}  // namespace iree
}  // namespace mlir
