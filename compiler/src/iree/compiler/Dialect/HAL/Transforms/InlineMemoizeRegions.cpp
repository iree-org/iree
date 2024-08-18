// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_INLINEMEMOIZEREGIONSPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-inline-memoize-regions
//===----------------------------------------------------------------------===//

struct InlineMemoizeRegionsPass
    : public IREE::HAL::impl::InlineMemoizeRegionsPassBase<
          InlineMemoizeRegionsPass> {
  void runOnOperation() override {
    auto parentOp = getOperation();
    SmallVector<IREE::HAL::DeviceMemoizeOp> memoizeOps;
    parentOp->walk([&](IREE::HAL::DeviceMemoizeOp memoizeOp) {
      memoizeOps.push_back(memoizeOp);
    });
    for (auto memoizeOp : memoizeOps) {
      SmallVector<IREE::HAL::ReturnOp> returnOps;
      SmallVector<SmallVector<Location>> resultLocs;
      resultLocs.resize(memoizeOp.getNumResults());
      for (auto returnOp : memoizeOp.getOps<IREE::HAL::ReturnOp>()) {
        returnOps.push_back(returnOp);
        for (auto result : llvm::enumerate(returnOp.getOperands())) {
          resultLocs[result.index()].push_back(result.value().getLoc());
        }
      }
      auto fusedResultLocs =
          llvm::map_to_vector(resultLocs, [&](auto resultLocs) {
            return FusedLoc::get(memoizeOp.getContext(), resultLocs);
          });

      // Update any command buffer creation flags to have the one-shot flag.
      memoizeOp.walk([&](IREE::HAL::CommandBufferCreateOp createOp) {
        createOp.setModes(createOp.getModes() |
                          IREE::HAL::CommandBufferModeBitfield::OneShot);
      });

      // ^initialBlock:
      //   <existing>
      //   cf.br ^entryBlock
      // ^entryBlock:
      //   <memoize op region>
      //   cf.br ^continueBlock(<memoize op results>)
      // ^continueBlock(<memoize op results>):
      //   <existing using block arg results>
      IRRewriter rewriter(memoizeOp);
      auto *initialBlock = memoizeOp->getBlock();
      auto *entryBlock = &memoizeOp.getBody().front();
      auto *continueBlock = rewriter.splitBlock(memoizeOp->getBlock(),
                                                Block::iterator(memoizeOp));
      memoizeOp.replaceAllUsesWith(continueBlock->addArguments(
          memoizeOp.getResultTypes(), fusedResultLocs));
      rewriter.inlineRegionBefore(memoizeOp.getBody(), continueBlock);
      rewriter.setInsertionPointToEnd(initialBlock);
      rewriter.create<mlir::cf::BranchOp>(memoizeOp.getLoc(), entryBlock);
      for (auto returnOp : returnOps) {
        rewriter.setInsertionPoint(returnOp);
        rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(returnOp, continueBlock,
                                                        returnOp.getOperands());
      }
      rewriter.eraseOp(memoizeOp);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
