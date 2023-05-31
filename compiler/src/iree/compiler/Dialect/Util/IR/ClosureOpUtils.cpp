// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/ClosureOpUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

//------------------------------------------------------------------------------
// Closure optimization
//------------------------------------------------------------------------------

void excludeClosureOperandsAndResults(
    SmallVector<Value, 4> &operandValues,
    ArrayRef<unsigned> excludedOperandIndices,
    SmallVector<Type, 4> &resultTypes,
    ArrayRef<unsigned> excludedResultIndices) {
  SmallVector<Value, 4> oldOperandValues = operandValues;
  operandValues.clear();
  for (auto it : llvm::enumerate(oldOperandValues)) {
    if (!llvm::count(excludedOperandIndices, it.index())) {
      operandValues.push_back(it.value());
    }
  }
  SmallVector<Type, 4> oldResultTypes = resultTypes;
  resultTypes.clear();
  for (auto it : llvm::enumerate(oldResultTypes)) {
    if (!llvm::count(excludedResultIndices, it.index())) {
      resultTypes.push_back(it.value());
    }
  }
}

void excludeClosureOperandsAndResults(
    SmallVector<Value, 4> &operandValues, SmallVector<Value, 4> &operandDims,
    ArrayRef<unsigned> excludedOperandIndices,
    SmallVector<Type, 4> &resultTypes, SmallVector<Value, 4> &resultDims,
    ArrayRef<unsigned> excludedResultIndices) {
  SmallVector<Value, 4> oldOperandValues = operandValues;
  SmallVector<Value, 4> oldOperandDims = operandDims;
  operandValues.clear();
  operandDims.clear();
  auto remainingOperandDims = llvm::ArrayRef(oldOperandDims);
  for (auto it : llvm::enumerate(oldOperandValues)) {
    unsigned numDynamicDims = 0;
    auto type = it.value().getType();
    if (auto shapedType = llvm::dyn_cast<ShapedType>(type)) {
      numDynamicDims = shapedType.getNumDynamicDims();
    } else if (llvm::isa<IREE::Util::SizeAwareTypeInterface>(type)) {
      numDynamicDims = 1;
    }
    if (!llvm::count(excludedOperandIndices, it.index())) {
      operandValues.push_back(it.value());
      for (auto dim : remainingOperandDims.take_front(numDynamicDims)) {
        operandDims.push_back(dim);
      }
    }
    remainingOperandDims = remainingOperandDims.drop_front(numDynamicDims);
  }

  SmallVector<Type, 4> oldResultTypes = resultTypes;
  SmallVector<Value, 4> oldResultDims = resultDims;
  resultTypes.clear();
  resultDims.clear();
  auto remainingResultDims = llvm::ArrayRef(oldResultDims);
  for (auto it : llvm::enumerate(oldResultTypes)) {
    unsigned numDynamicDims = 0;
    auto type = it.value();
    if (auto shapedType = llvm::dyn_cast<ShapedType>(type)) {
      numDynamicDims = shapedType.getNumDynamicDims();
    } else if (llvm::isa<IREE::Util::SizeAwareTypeInterface>(type)) {
      numDynamicDims = 1;
    }
    if (!llvm::count(excludedResultIndices, it.index())) {
      resultTypes.push_back(type);
      for (auto dim : remainingResultDims.take_front(numDynamicDims)) {
        resultDims.push_back(dim);
      }
    }
    remainingResultDims = remainingResultDims.drop_front(numDynamicDims);
  }
  assert(remainingResultDims.empty());
}

void eraseRegionResults(Region &region,
                        ArrayRef<unsigned> excludedResultIndices) {
  for (auto &block : region.getBlocks()) {
    auto *terminatorOp = block.getTerminator();
    if (terminatorOp && terminatorOp->hasTrait<OpTrait::ReturnLike>()) {
      llvm::SmallVector<Value, 4> newReturns;
      for (auto it : llvm::enumerate(terminatorOp->getOperands())) {
        if (!llvm::count(excludedResultIndices, it.index())) {
          newReturns.push_back(it.value());
        }
      }
      terminatorOp->setOperands(newReturns);
    }
  }
}

// Returns true if |constantOp| represents a (logically) small constant value
// that can be inlined into a closure.
//
// "Small" is relative and there's a risk that we'll bloat the closures by
// duplicating a bunch of constants however what we are able to save by not
// doing that usually wins. Think of the number of bytes used on instructions to
// allocate/place/copy, setup function call arguments, compute the address,
// dereference the memory, etc vs. a constant immediate value of 16 bytes -
// afterall, there are single x64 instructions that approach 15 bytes :)
//
// This is also still at a fairly high level (flow dialect): once the closures
// are expanded out in lower dialects things like CSE have a chance to once
// again get at the constants and dedupe them if they survive.
static bool isConstantInlinable(const ClosureOptimizationOptions &options,
                                arith::ConstantOp constantOp) {
  int64_t maxInlinedConstantBytes =
      options.maxInlinedConstantBytes.value_or(INT64_MAX);
  if (maxInlinedConstantBytes == 0) {
    // Inlining of constants disabled.
    return false;
  }

  auto constantValueAttr = constantOp.getValue();
  auto constantType = constantOp.getType();
  if (llvm::isa<SplatElementsAttr>(constantValueAttr)) {
    // Splats are always small and can often have special handling when we
    // know they are a splat - which is why it's so important we inline them
    // here so we know when they are used that's the case.
    return true;
  } else if (auto denseAttr =
                 llvm::dyn_cast<DenseElementsAttr>(constantValueAttr)) {
    // Smallish constants are worth moving inside.
    auto shapedType = llvm::cast<ShapedType>(constantType);
    uint64_t estimatedByteLength =
        shapedType.getNumElements() *
        getRoundedElementByteWidth(shapedType.getElementType());
    return denseAttr.isSplat() ||
           estimatedByteLength <= maxInlinedConstantBytes;
  } else if (constantType.isIntOrIndexOrFloat()) {
    // Primitives can always go in.
    return true;
  }

  return false;
}

// Returns true if the given value should be inlined into the closure region.
// This is non-recursive and only holds for this value. Recursively cloning
// trees is hard and it'd be better to model that differently such as by having
// a wrapper region for immutable blobs that can be inlined that this then
// returns true for.
static bool shouldInlineIntoClosure(const ClosureOptimizationOptions &options,
                                    Value value) {
  auto definingOp = value.getDefiningOp();
  if (auto constantOp = dyn_cast<arith::ConstantOp>(definingOp)) {
    // Constants are perfect!
    return isConstantInlinable(options, constantOp);
  }
  return false;
}

// Inlines operands of the closure into the entry block as appropriate.
// The closure operands and block arguments will remain untouched but all uses
// will be replaced with the newly cloned values.
//
// Note that if multiple operands reference the same value it will get cloned
// multiple times. That's fine, as anything we can inline here is something we
// should also be able to CSE and that happens later on anyway.
static void inlineClosureOperands(const ClosureOptimizationOptions &options,
                                  ClosureOpInterface &closureOp,
                                  Block &entryBlock,
                                  PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(&entryBlock);
  for (auto opArg : llvm::enumerate(closureOp.getClosureOperands())) {
    auto outerValue = opArg.value();
    auto *sourceOp = outerValue.getDefiningOp();
    if (!sourceOp) continue;  // can't clone block arguments into closures

    // We cannot just simply inline and replace all users if this is an
    // argument that can be written; for example, the region might perform
    // work after loading a initial constant from the argument and then
    // write back.
    if (!closureOp.getOperandAccess(opArg.index()).isReadOnly()) continue;

    if (closureOp.canClosureContainOp(sourceOp) &&
        shouldInlineIntoClosure(options, outerValue)) {
      // Clone the op (with regions).
      auto *clonedOp = rewriter.clone(*sourceOp);

      // Ensure we are using the right result in the case of ops with multiple
      // results. If we only end up using a single result then canonicalization
      // should take care of removing the unneeded ones.
      int resultIndex =
          std::distance(sourceOp->result_begin(),
                        std::find(sourceOp->result_begin(),
                                  sourceOp->result_end(), outerValue));
      auto newValue = clonedOp->getResult(resultIndex);

      // Replace all of the uses inside of the closure.
      BlockArgument blockArg = entryBlock.getArgument(opArg.index());
      rewriter.replaceAllUsesWith(blockArg, newValue);
    }
  }
}

LogicalResult optimizeClosureLikeOp(const ClosureOptimizationOptions &options,
                                    ClosureOpInterface closureOp,
                                    PatternRewriter &rewriter) {
  // NOTE: the block is transferred to the new op; we can update it in place.
  Block &entryBlock = closureOp.getClosureBodyRegion().front();

  // Find constants/metadata/etc that we can clone into the closure.
  // By doing this first we potentially create some dead operands that we can
  // then elide below. When we do inline things the operands will be changed
  // such that the following work is guaranteed to happen and thus our op will
  // be rebuilt.
  inlineClosureOperands(options, closureOp, entryBlock, rewriter);

  // Build data structure for unused operand elision.
  SmallVector<unsigned, 4> elidedOperands;
  llvm::SmallMapVector<Value, BlockArgument, 8> argToBlockMap;
  SmallVector<std::optional<BlockArgument>, 8> blockArgReplacements(
      entryBlock.getNumArguments());
  for (auto opArg : llvm::enumerate(closureOp.getClosureOperands())) {
    auto blockArg = entryBlock.getArgument(opArg.index());
    if (blockArg.use_empty()) {
      // Not used - Drop.
      elidedOperands.push_back(opArg.index());
      blockArgReplacements[opArg.index()] = BlockArgument();
      continue;
    }
    auto existingIt = argToBlockMap.find(opArg.value());
    if (existingIt == argToBlockMap.end()) {
      // Not found - Record for deduping.
      argToBlockMap.insert(std::make_pair(opArg.value(), blockArg));
    } else {
      // Found - Replace.
      elidedOperands.push_back(opArg.index());
      blockArgReplacements[opArg.index()] = existingIt->second;
    }
  }

  // Check for unused results.
  SmallVector<Value, 4> preservedResults;
  SmallVector<unsigned, 4> elidedResults;
  for (auto result : llvm::enumerate(closureOp.getClosureResults())) {
    // You can drop a result if the use is empty and not read via a tie.
    auto access = closureOp.getResultAccess(result.index());
    if (result.value().use_empty() && !access.isRead) {
      elidedResults.push_back(result.index());
    } else {
      preservedResults.push_back(result.value());
    }
  }

  if (elidedOperands.empty() && elidedResults.empty()) {
    // No optimization required.
    return failure();
  }

  if (elidedResults.size() == closureOp.getClosureResults().size() &&
      closureOp.getClosureResults().size() == closureOp->getNumResults()) {
    // The op is completely unused - delete it.
    rewriter.eraseOp(closureOp);
    return success();
  }

  // Replace duplicate block arguments.
  for (auto replacement : llvm::enumerate(blockArgReplacements)) {
    if (!replacement.value()) {
      // No change.
    } else if (!replacement.value().value()) {
      // Dropped.
    } else {
      // Replaced.
      rewriter.replaceAllUsesWith(entryBlock.getArgument(replacement.index()),
                                  *replacement.value());
    }
  }

  // Clone the op with the elidable operands and results removed.
  auto newOp = closureOp.cloneReplacementExcludingOperandsAndResults(
      elidedOperands, elidedResults, rewriter);

  // Fixup any non-closure results to point to the new op.
  SetVector<Value> oldResults(closureOp->getResults().begin(),
                              closureOp->getResults().end());
  SetVector<Value> oldClosureResults(closureOp.getClosureResults().begin(),
                                     closureOp.getClosureResults().end());
  SetVector<Value> newResults(newOp->getResults().begin(),
                              newOp->getResults().end());
  SetVector<Value> newClosureResults(newOp.getClosureResults().begin(),
                                     newOp.getClosureResults().end());
  oldResults.set_subtract(oldClosureResults);
  newResults.set_subtract(newClosureResults);
  assert(oldResults.size() == newResults.size() &&
         "expected non-closure results to match");
  for (auto [oldResult, newResult] : llvm::zip_equal(oldResults, newResults)) {
    rewriter.replaceAllUsesWith(oldResult, newResult);
  }

  // Replace original uses of the closure results.
  for (auto [oldResult, newResult] :
       llvm::zip_equal(preservedResults, newOp.getClosureResults())) {
    rewriter.replaceAllUsesWith(oldResult, newResult);
  }

  // Erase the original op.
  rewriter.eraseOp(closureOp);

  return success();
}

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
