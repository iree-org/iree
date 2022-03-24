// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/PyDM/Utils/TypeInference.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"

using namespace mlir;
namespace PYDM = mlir::iree_compiler::IREE::PYDM;
using namespace PYDM;

PermutedTypePropagator::PermutedBlockInfo *
PermutedTypePropagator::addPermutedBlockToParent(ParentBlockInfo *parentInfo,
                                                 Block *block) {
  auto *permutedInfo = new (allocator) PermutedBlockInfo();
  permutedInfo->permutedBlock = block;
  permutedInfo->parentInfo = parentInfo;
  permutedInfo->signature =
      FunctionType::get(context, block->getArgumentTypes(), TypeRange{});
  permutedInfo->next = parentInfo->permutationHead;
  parentInfo->permutationHead = permutedInfo;
  parentInfo->size += 1;
  return permutedInfo;
}

PermutedTypePropagator::ParentBlockInfo *
PermutedTypePropagator::lookupParentBlock(Block *forBlock) {
  auto it = permutedBlocks.find(forBlock);
  if (it == permutedBlocks.end()) {
    // Unaccounted for blocks are assumed to be parents.
    auto *parentInfo = allocator.Allocate<ParentBlockInfo>();
    new (parentInfo) ParentBlockInfo();

    parentInfo->parentBlock = forBlock;
    // The parent is also considered a permutation.
    auto *permutedInfo = addPermutedBlockToParent(parentInfo, forBlock);
    permutedBlocks.insert(std::make_pair(forBlock, permutedInfo));
    return parentInfo;
  }

  return it->second->parentInfo;
}

Block *PermutedTypePropagator::findBlockPermutation(ParentBlockInfo *parentInfo,
                                                    FunctionType signature) {
  for (PermutedBlockInfo *info = parentInfo->permutationHead; info;
       info = info->next) {
    if (info->signature == signature)
      return info->permutedBlock;
  }
  return nullptr;
}

static bool checkAllBlockArgsMapped(Block *block,
                                    BlockAndValueMapping &mapping) {
  for (Value arg : block->getArguments()) {
    if (!mapping.contains(arg))
      return false;
  }
  return true;
}

Block *PermutedTypePropagator::createBlockPermutation(
    Location loc, ParentBlockInfo *parentInfo, TypeRange newArgumentTypes,
    BlockPermuteCallback initializeCallback) {
  Block *parentBlock = parentInfo->parentBlock;
  Block *newBlock = new Block();
  for (Type newArgumentType : newArgumentTypes) {
    newBlock->addArgument(newArgumentType, loc);
  }
  newBlock->insertBefore(parentBlock);

  BlockAndValueMapping mapping;
  mapping.map(parentBlock, newBlock);
  initializeCallback(newBlock, parentBlock, mapping);
  assert(checkAllBlockArgsMapped(parentBlock, mapping) &&
         "permuted block initializer did not map all block arguments");

  // Inline.
  for (auto &op : *parentBlock) {
    newBlock->push_back(op.clone(mapping));
  }

  addPermutedBlockToParent(parentInfo, newBlock);
  return newBlock;
}

SmallVector<PermutedTypePropagator::BlockPredecessor>
PermutedTypePropagator::findMismatchedBlockPredecessors(Block *block) {
  SmallVector<BlockPredecessor> results;
  for (Block *predecessor : block->getPredecessors()) {
    Operation *terminator = predecessor->getTerminator();
    auto branchOp = llvm::cast<BranchOpInterface>(terminator);
    unsigned successorIndex = 0;
    for (Block *successor : terminator->getSuccessors()) {
      if (successor == block)
        break;
      successorIndex += 1;
    }
    auto successorOperands = branchOp.getSuccessorOperands(successorIndex);
    assert(successorOperands && "expected branch with explicit operands");
    TypeRange operandTypes(*successorOperands);
    if (block->getArgumentTypes() != operandTypes) {
      results.push_back(BlockPredecessor{
          branchOp, successorIndex,
          FunctionType::get(context, operandTypes, TypeRange{})});
    }
  }
  return results;
}
