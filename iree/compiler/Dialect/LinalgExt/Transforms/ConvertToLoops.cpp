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

    uint64_t sortDim = 0;
    if (op.dimensionAttr()) sortDim = op.dimension().getValue();
    Value ub = rewriter.create<SubIOp>(loc, ubs[sortDim], one);
    mlir::scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps, ValueRange{},
        [&](OpBuilder& b, Location loc, ValueRange ivs, ValueRange iters) {
          SmallVector<Value> indices, sortBlkArgs;
          indices.append(ivs.begin(), ivs.end());
          // Bubble sort innermost loop.
          auto scfFor = b.create<scf::ForOp>(
              loc, zero, ub, one, iters,
              [&](OpBuilder& b, Location loc, Value iv, ValueRange iters) {
                SmallVector<Value> indices(ivs);
                Value ivPlusOne = b.create<AddIOp>(loc, iv, one);
                for (auto output : op.getOutputOperands()) {
                  indices[sortDim] = iv;
                  sortBlkArgs.push_back(
                      b.create<memref::LoadOp>(loc, output->get(), indices));
                  indices[sortDim] = ivPlusOne;
                  sortBlkArgs.push_back(
                      b.create<memref::LoadOp>(loc, output->get(), indices));
                }
                // A block must end with a terminator. This op will be erased
                // later.
                b.create<scf::YieldOp>(loc);
              });

          Region& region = scfFor.region();
          rewriter.mergeBlockBefore(&op.region().front(),
                                    region.front().getTerminator(),
                                    sortBlkArgs);
          rewriter.eraseOp(region.front().getTerminator());

          // The erasion of an op will happen later, so we can not use
          // .getTerminator() method here.
          auto linalgExtYieldOp = llvm::to_vector<4>(
              region.front().getOps<linalg_ext::YieldOp>())[0];
          Value cond = linalgExtYieldOp.getOperand(0);
          rewriter.replaceOp(linalgExtYieldOp, {});

          b.setInsertionPointToEnd(&region.front());
          b.create<scf::IfOp>(
              loc, TypeRange{}, cond,
              [&](OpBuilder& b, Location loc) {
                // Do not swap the pairs if true.
                b.create<scf::YieldOp>(loc);
              },
              [&](OpBuilder& b, Location loc) {
                // Swap the pairs if false.
                SmallVector<Value> indices(ivs.begin(), ivs.end());
                Value ivPlusOne =
                    b.create<AddIOp>(loc, scfFor.getInductionVar(), one);
                for (int i = 0, e = op.getNumOutputs(); i < e; ++i) {
                  Value v1 = sortBlkArgs[i * 2];
                  Value v2 = sortBlkArgs[i * 2 + 1];
                  indices[sortDim] = scfFor.getInductionVar();
                  b.create<memref::StoreOp>(
                      loc, v2, op.getOutputOperand(i)->get(), indices);
                  indices[sortDim] = ivPlusOne;
                  b.create<memref::StoreOp>(
                      loc, v1, op.getOutputOperand(i)->get(), indices);
                }
                b.create<scf::YieldOp>(loc);
              });

          b.create<scf::YieldOp>(loc);
          return scf::ValueVector();
        });
    rewriter.eraseOp(op);
    return success();
  }
};

struct LinalgExtToLoopsPass
    : public LinalgExtToLoopsBase<LinalgExtToLoopsPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry
        .insert<StandardOpsDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext* context = &getContext();

    OwningRewritePatternList patterns(context);
    patterns.insert<BubbleSortConversion>(context);

    ConversionTarget target(getContext());
    target.addLegalDialect<memref::MemRefDialect, StandardOpsDialect,
                           scf::SCFDialect>();
    target.addIllegalOp<linalg_ext::SortOp>();

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
