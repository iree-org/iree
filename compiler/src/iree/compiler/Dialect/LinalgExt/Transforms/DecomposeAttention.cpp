// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <limits>

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_DECOMPOSEATTENTIONPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

struct DecomposeAttentionPass final
    : impl::DecomposeAttentionPassBase<DecomposeAttentionPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect, arith::ArithDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }
  void runOnOperation() override;
};
} // namespace

/// Returns an i1 that is true when the mask leaves some position visible.
static Value anyVisible(OpBuilder &b, Location loc, Value mask) {
  int64_t rank = cast<RankedTensorType>(mask.getType()).getRank();
  Value none = arith::ConstantIntOp::create(b, loc, 0, 1);
  Value init =
      tensor::EmptyOp::create(b, loc, ArrayRef<int64_t>{}, b.getI1Type());
  init = linalg::FillOp::create(b, loc, none, init).getResult(0);
  SmallVector<AffineMap> maps = {
      b.getMultiDimIdentityMap(rank),
      AffineMap::get(rank, /*symbolCount=*/0, b.getContext())};
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::reduction);
  auto genericOp = linalg::GenericOp::create(
      b, loc, init.getType(), mask, init, maps, iterators,
      [](OpBuilder &nb, Location nl, ValueRange args) {
        Value visible = args[0];
        if (auto floatType = dyn_cast<FloatType>(visible.getType())) {
          // A NaN entry is no exclusion marker.
          Value negInf = arith::ConstantOp::create(
              nb, nl,
              nb.getFloatAttr(floatType,
                              -std::numeric_limits<double>::infinity()));
          visible = arith::CmpFOp::create(nb, nl, arith::CmpFPredicate::UNE,
                                          visible, negInf);
        } else if (!visible.getType().isInteger(1)) {
          visible = arith::TruncIOp::create(nb, nl, nb.getI1Type(), visible);
        }
        Value seen = arith::OrIOp::create(nb, nl, visible, args[1]);
        linalg::YieldOp::create(nb, nl, seen);
      });
  return tensor::ExtractOp::create(b, loc, genericOp.getResult(0),
                                   ValueRange{});
}

void DecomposeAttentionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  getOperation().walk([&](OnlineAttentionOp onlineAtt) {
    rewriter.setInsertionPoint(onlineAtt);

    NamedAttrList decompositionConfig(onlineAtt.getDecompositionConfigAttr());
    decompositionConfig.set("use_exp2", rewriter.getBoolAttr(useExp2));
    onlineAtt.setDecompositionConfigAttr(
        decompositionConfig.getDictionary(context));

    Value mask = onlineAtt.getMask();
    scf::IfOp ifOp;
    if (skipMaskedTiles && mask) {
      Location loc = onlineAtt.getLoc();
      Value visible = anyVisible(rewriter, loc, mask);
      ifOp = scf::IfOp::create(rewriter, loc, onlineAtt->getResultTypes(),
                               visible, /*addThenBlock=*/true,
                               /*addElseBlock=*/true);
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      scf::YieldOp::create(rewriter, loc,
                           ValueRange{onlineAtt.getOutput(), onlineAtt.getMax(),
                                      onlineAtt.getSum()});
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    }

    FailureOr<SmallVector<Value>> results =
        onlineAtt.decomposeOperation(rewriter);
    if (failed(results)) {
      onlineAtt->emitOpError("Could not decompose online attention");
      return signalPassFailure();
    }
    if (!ifOp) {
      rewriter.replaceOp(onlineAtt, results.value());
      return;
    }
    scf::YieldOp::create(rewriter, onlineAtt.getLoc(), results.value());
    rewriter.replaceOp(onlineAtt, ifOp.getResults());
  });
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
