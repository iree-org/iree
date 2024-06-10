// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

namespace {
struct DecomposeAttentionPass
    : public DecomposeAttentionBase<DecomposeAttentionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
        linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
  }
  DecomposeAttentionPass() = default;
  void runOnOperation() override;
};

struct ConvertAttentionToOnlineAttentionPass final
    : ConvertAttentionToOnlineAttentionBase<
          ConvertAttentionToOnlineAttentionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
                linalg::LinalgDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override;
};

} // namespace

void convertToOnlineAttention(IREE::LinalgExt::AttentionOp attnOp,
                              SmallVectorImpl<Operation *> &ops,
                              RewriterBase &rewriter) {
  rewriter.setInsertionPoint(attnOp);

  Location loc = attnOp.getLoc();
  MLIRContext *ctx = attnOp.getContext();

  FailureOr<AttentionOpDetail> maybeOpInfo =
      AttentionOpDetail::get(attnOp.getIndexingMapsArray());
  assert(succeeded(maybeOpInfo) && "Invalid attention indexing maps");
  AttentionOpDetail opInfo = maybeOpInfo.value();

  // Create standard maps for max and sum: (batch, m)
  int64_t rank = opInfo.getDomainRank();
  AffineMap maxMap = AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/0, ctx);
  for (auto dim :
       llvm::concat<const int64_t>(opInfo.getBatchDims(), opInfo.getMDims())) {
    maxMap = maxMap.insertResult(rewriter.getAffineDimExpr(dim),
                                 maxMap.getNumResults());
  }
  AffineMap sumMap = maxMap;

  SmallVector<Range> sizes = attnOp.getIterationDomain(rewriter);

  // Create fill for acc, max and sum.
  // TODO: Acc should not need a fill. The attention op should get a filled
  // input instead of an empty input.
  Value zeroAcc = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(attnOp.getOutput().getType().getElementType()));
  Value accFill =
      rewriter
          .create<linalg::FillOp>(loc, ValueRange{zeroAcc}, attnOp.getOutput())
          .getResult(0);

  SmallVector<OpFoldResult> rowRedSize =
      llvm::map_to_vector(sizes, [](Range x) { return x.size; });
  rowRedSize = applyPermutationMap<OpFoldResult>(maxMap, rowRedSize);

  Type f32Type = rewriter.getF32Type();
  Value rowRedEmpty =
      rewriter.create<tensor::EmptyOp>(loc, rowRedSize, f32Type);

  Value maxInit =
      arith::getIdentityValue(arith::AtomicRMWKind::maximumf, f32Type, rewriter,
                              loc, /*useOnlyFiniteValue=*/true);
  Value sumInit = arith::getIdentityValue(arith::AtomicRMWKind::addf, f32Type,
                                          rewriter, loc);

  Value maxFill =
      rewriter.create<linalg::FillOp>(loc, ValueRange{maxInit}, rowRedEmpty)
          .getResult(0);
  Value sumFill =
      rewriter.create<linalg::FillOp>(loc, ValueRange{sumInit}, rowRedEmpty)
          .getResult(0);

  // Create online attention op.
  SmallVector<AffineMap> indexingMaps = attnOp.getIndexingMapsArray();
  indexingMaps.push_back(maxMap);
  indexingMaps.push_back(sumMap);
  OnlineAttentionOp onlineAttn = rewriter.create<OnlineAttentionOp>(
      loc, TypeRange{accFill.getType(), maxFill.getType(), sumFill.getType()},
      attnOp.getQuery(), attnOp.getKey(), attnOp.getValue(), attnOp.getScale(),
      accFill, maxFill, sumFill, rewriter.getAffineMapArrayAttr(indexingMaps));
  onlineAttn->setDiscardableAttrs(attnOp->getDiscardableAttrDictionary());
  ops.push_back(onlineAttn);

  Value x = onlineAttn.getResult(0);
  Value sum = onlineAttn.getResult(2);

  // Merge the outputs of online attention:
  //  x = (1 / sum) * x

  // Compress the indexing maps.
  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{sumMap, attnOp.getOutputMap()});

  SmallVector<utils::IteratorType> iteratorTypes(compressedMaps[0].getNumDims(),
                                                 utils::IteratorType::parallel);

  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, x.getType(), sum, x, compressedMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value one = b.create<arith::ConstantOp>(
            loc, b.getFloatAttr(args[0].getType(), 1.0));
        Value reciprocal = b.create<arith::DivFOp>(loc, one, args[0]);
        // Convert sum to the same datatype as x.
        reciprocal = convertScalarToDtype(b, loc, reciprocal, args[1].getType(),
                                          /*isUnsignedCast=*/false);
        Value result = b.create<arith::MulFOp>(loc, reciprocal, args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);

  rewriter.replaceOp(attnOp, genericOp);
}

void DecomposeAttentionPass::runOnOperation() {
  IRRewriter rewriter(&getContext());
  getOperation().walk([&](OnlineAttentionOp onlineAtt) {
    rewriter.setInsertionPoint(onlineAtt);
    FailureOr<SmallVector<Value>> results =
        onlineAtt.decomposeOperation(rewriter);
    if (failed(results)) {
      onlineAtt->emitOpError("Could not decompose online attention");
      return signalPassFailure();
    }
    rewriter.replaceOp(onlineAtt, results.value());
  });
}

void ConvertAttentionToOnlineAttentionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  getOperation().walk([&](AttentionOp attnOp) {
    SmallVector<Operation *> ops;
    convertToOnlineAttention(attnOp, ops, rewriter);
  });
}

std::unique_ptr<Pass> createDecomposeAttentionPass() {
  return std::make_unique<DecomposeAttentionPass>();
}

std::unique_ptr<Pass> createConvertAttentionToOnlineAttentionPass() {
  return std::make_unique<ConvertAttentionToOnlineAttentionPass>();
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
