// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_CONVERTATTENTIONTOONLINEATTENTIONPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

struct ConvertAttentionToOnlineAttentionPass final
    : impl::ConvertAttentionToOnlineAttentionPassBase<
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

  AffineMap accMap = attnOp.getOutputMap();

  SmallVector<Range> domain = attnOp.getIterationDomain(rewriter);

  // Create fill for acc, max and sum.
  // TODO: Acc should not need a fill. The attention op should get a filled
  // input instead of an empty input.

  SmallVector<OpFoldResult> sizes =
      llvm::map_to_vector(domain, [](Range x) { return x.size; });
  SmallVector<OpFoldResult> accSize =
      applyPermutationMap<OpFoldResult>(accMap, sizes);
  SmallVector<OpFoldResult> rowRedSize =
      applyPermutationMap<OpFoldResult>(maxMap, sizes);

  Type f32Type = rewriter.getF32Type();
  Value acc = rewriter.create<tensor::EmptyOp>(loc, accSize, f32Type);
  Value rowRedEmpty =
      rewriter.create<tensor::EmptyOp>(loc, rowRedSize, f32Type);

  Value accInit =
      arith::getIdentityValue(arith::AtomicRMWKind::addf, f32Type, rewriter,
                              loc, /*useOnlyFiniteValue=*/true);
  Value maxInit =
      arith::getIdentityValue(arith::AtomicRMWKind::maximumf, f32Type, rewriter,
                              loc, /*useOnlyFiniteValue=*/true);
  Value sumInit = arith::getIdentityValue(arith::AtomicRMWKind::addf, f32Type,
                                          rewriter, loc);

  Value accFill = rewriter.create<linalg::FillOp>(loc, ValueRange{accInit}, acc)
                      .getResult(0);
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

  Value mask = attnOp.getMask() ? attnOp.getMask() : Value();

  OnlineAttentionOp onlineAttn = rewriter.create<OnlineAttentionOp>(
      loc, TypeRange{accFill.getType(), maxFill.getType(), sumFill.getType()},
      attnOp.getQuery(), attnOp.getKey(), attnOp.getValue(), attnOp.getScale(),
      mask, accFill, maxFill, sumFill,
      rewriter.getAffineMapArrayAttr(indexingMaps),
      attnOp.getDecompositionConfigAttr());

  rewriter.cloneRegionBefore(attnOp.getRegion(), onlineAttn.getRegion(),
                             onlineAttn.getRegion().begin());
  onlineAttn->setDiscardableAttrs(attnOp->getDiscardableAttrDictionary());
  ops.push_back(onlineAttn);

  Value x = onlineAttn.getResult(0);
  Value sum = onlineAttn.getResult(2);

  // Merge the outputs of online attention:
  //  x = (1 / sum) * x

  // Compress the indexing maps.
  SmallVector<AffineMap> compressedMaps =
      compressUnusedDims(SmallVector<AffineMap>{sumMap, accMap, accMap});

  SmallVector<utils::IteratorType> iteratorTypes(compressedMaps[0].getNumDims(),
                                                 utils::IteratorType::parallel);

  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, attnOp.getOutput().getType(), ValueRange{sum, x}, attnOp.getOutput(),
      compressedMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value one = b.create<arith::ConstantOp>(
            loc, b.getFloatAttr(args[0].getType(), 1.0));
        Value reciprocal = b.create<arith::DivFOp>(loc, one, args[0]);
        // Both sum and x are in fp32, as created earlier, so we only need
        // to cast after the mul.
        Value result = b.create<arith::MulFOp>(loc, reciprocal, args[1]);
        // Cast result to the required type by attention output.
        result = convertScalarToDtype(b, loc, result, args[2].getType(),
                                      /*isUnsignedCast=*/false);
        b.create<linalg::YieldOp>(loc, result);
      });
  ops.push_back(genericOp);

  rewriter.replaceOp(attnOp, genericOp);
}

void ConvertAttentionToOnlineAttentionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  getOperation()->walk([&](AttentionOp attnOp) {
    SmallVector<Operation *> ops;
    convertToOnlineAttention(attnOp, ops, rewriter);
  });
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
