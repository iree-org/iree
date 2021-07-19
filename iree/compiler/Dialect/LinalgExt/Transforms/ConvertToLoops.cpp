// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace linalg_ext {
namespace {

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

struct BubbleSortConversion : public OpRewritePattern<linalg_ext::SortOp> {
  using OpRewritePattern<linalg_ext::SortOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg_ext::SortOp op,
                                PatternRewriter& rewriter) const final {
    if (!op.hasBufferSemantics()) return failure();

    auto arg0 = op.getOutputOperand(0);
    Location loc = op.getLoc();
    SmallVector<Value, 4> lbs, ubs, steps;
    for (auto en : llvm::enumerate(op.getShape(arg0))) {
      if (ShapedType::isDynamic(en.value())) {
        ubs.push_back(
            rewriter.create<memref::DimOp>(loc, arg0->get(), en.index()));
      } else {
        ubs.push_back(rewriter.create<ConstantIndexOp>(loc, en.value()));
      }
    }
    Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<ConstantIndexOp>(loc, 1);
    lbs.append(op.getRank(arg0), zero);
    steps.append(op.getRank(arg0), one);

    bool fail = false;
    mlir::scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps, ValueRange{},
        [&](OpBuilder& b, Location loc, ValueRange ivs, ValueRange iters) {
          if (failed(op.generateScalarImplementation(b, loc, ivs))) {
            fail = true;
          }
          b.create<scf::YieldOp>(loc);
          return scf::ValueVector();
        });
    if (fail) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

struct ScatterConversion : public OpRewritePattern<linalg_ext::ScatterOp> {
  using OpRewritePattern<linalg_ext::ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg_ext::ScatterOp op,
                                PatternRewriter& rewriter) const final {
    if (!op.hasBufferSemantics()) return failure();
    auto updates = op.updates();
    Location loc = op.getLoc();
    Value zero = rewriter.create<ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<ConstantIndexOp>(loc, 1);
    Value ub = rewriter.createOrFold<memref::DimOp>(loc, updates, zero);
    bool fail = false;
    rewriter.create<scf::ForOp>(
        loc, zero, ub, one, ValueRange{},
        [&](OpBuilder& b, Location loc, Value iv, ValueRange iters) {
          if (failed(op.generateScalarImplementation(b, loc, iv))) {
            fail = true;
          }
          b.create<scf::YieldOp>(loc);
        });
    if (fail) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LinalgExtToLoopsPass
    : public LinalgExtToLoopsBase<LinalgExtToLoopsPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<linalg::LinalgDialect, StandardOpsDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext* context = &getContext();

    OwningRewritePatternList patterns(context);
    patterns.insert<BubbleSortConversion, ScatterConversion>(context);

    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, memref::MemRefDialect,
                           StandardOpsDialect, scf::SCFDialect>();
    target.addIllegalDialect<linalg_ext::LinalgExtDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLinalgExtToLoopsPass() {
  return std::make_unique<LinalgExtToLoopsPass>();
}

}  // namespace linalg_ext
}  // namespace iree_compiler
}  // namespace mlir
