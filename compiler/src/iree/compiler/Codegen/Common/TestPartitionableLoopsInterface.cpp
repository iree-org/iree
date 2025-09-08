// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

static const char kAttributeName[] = "__test_interface__";

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TESTPARTITIONABLELOOPSINTERFACEPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

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
    SmallVector<unsigned> partitionableLoops =
        interfaceOp.getPartitionableLoops(3);
    auto type =
        RankedTensorType::get(partitionableLoops.size(), rewriter.getI32Type());
    auto constantAttr = DenseIntElementsAttr::get(type, partitionableLoops);
    IREE::Util::UnfoldableConstantOp::create(rewriter, interfaceOp.getLoc(),
                                             constantAttr);
    rewriter.modifyOpInPlace(interfaceOp,
                             [&] { interfaceOp->removeAttr(kAttributeName); });
    return success();
  }
};

struct TestPartitionableLoopsInterfacePass final
    : impl::TestPartitionableLoopsInterfacePassBase<
          TestPartitionableLoopsInterfacePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TestPartitionableLoopsInterfacePattern>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
