// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/PartitionableLoopsInterface.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

static const char kAttributeName[] = "__test_interface__";

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

/// For ops that implement the `PartitionableLoopsInterface` that have the
/// `__test_interface__` attribute set generates a `util.unfoldable_constant`
/// with a value of type `tensor<axindex>`, where `a` is the number of loops and
/// the value has zeros for non-partitionable loops and 1 for partitionable
/// loops.
struct TestPartitionableLoopsInterfacePattern
    : public OpInterfaceRewritePattern<PartitionableLoopsInterface> {
  using OpInterfaceRewritePattern<
      PartitionableLoopsInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(PartitionableLoopsInterface interfaceOp,
                                PatternRewriter &rewriter) const {
    if (!interfaceOp->hasAttr(kAttributeName)) {
      return failure();
    }
    unsigned numLoops = interfaceOp.getNumLoops();
    SmallVector<unsigned> partitionableLoops =
        interfaceOp.getPartitionableLoops(3);
    SmallVector<int64_t> loopInfo(numLoops, 0);
    for (auto partitionableLoop : partitionableLoops) {
      loopInfo[partitionableLoop] = 1;
    }
    auto type = RankedTensorType::get(numLoops, rewriter.getIndexType());
    auto constantAttr = DenseIntElementsAttr::get(type, loopInfo);
    rewriter.create<Util::UnfoldableConstantOp>(interfaceOp.getLoc(),
                                                constantAttr);
    rewriter.updateRootInPlace(
        interfaceOp, [&] { interfaceOp->removeAttr(kAttributeName); });
    return success();
  }
};

struct TestPartitionableLoopsInterfacePass
    : public PassWrapper<TestPartitionableLoopsInterfacePass,
                         OperationPass<void>> {
  StringRef getArgument() const override {
    return "iree-flow-test-partitionable-loops-interface";
  }

  StringRef getDescription() const override {
    return "Test the PartitionableLoopsInterface using operations that "
           "implement that interface.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<FlowDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TestPartitionableLoopsInterfacePattern>(patterns.getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<void>>
createTestPartitionableLoopsInterfacePass() {
  return std::make_unique<TestPartitionableLoopsInterfacePass>();
}

static PassRegistration<TestPartitionableLoopsInterfacePass> pass;

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
