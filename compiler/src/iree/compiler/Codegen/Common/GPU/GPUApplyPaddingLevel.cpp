// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-apply-padding-level"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUAPPLYPADDINGLEVELPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUApplyPaddingLevelPass final
    : impl::GPUApplyPaddingLevelPassBase<GPUApplyPaddingLevelPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static llvm::SmallDenseSet<TilingInterface>
getTiledOps(Operation *funcOp, IREE::GPU::TilingLevel tilingLevel) {
  llvm::SmallDenseSet<TilingInterface> targets;
  unsigned opaqueLevel = llvm::to_underlying(tilingLevel);
  funcOp->walk([&](TilingInterface target) {
    // TODO: This would probably be easier with a lowering config interface
    // method that checks whether a particular level is tiled.
    if (IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
            getLoweringConfig(target)) {
      if (loweringConfig.hasTilingLevel(opaqueLevel)) {
        targets.insert(target);
      }
    }
  });
  return targets;
}

static TypedValue<ShapedType> setDefaultOnlineAttentionMask(
    RewriterBase &rewriter,
    IREE::LinalgExt::OnlineAttentionOp onlineAttentionOp) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(onlineAttentionOp);

  Location loc = onlineAttentionOp->getLoc();
  SmallVector<int64_t> maskStatic;
  SmallVector<Value> maskDyn;
  auto pushNext = [&](TypedValue<ShapedType> v, ShapedType t, int64_t idx) {
    maskStatic.push_back(t.getDimSize(idx));
    if (maskStatic.back() == ShapedType::kDynamic)
      maskDyn.push_back(rewriter.create<tensor::DimOp>(loc, v, idx));
  };

  TypedValue<ShapedType> query = onlineAttentionOp.getQuery();
  ShapedType queryTy = query.getType();
  for (int i = 0, s = queryTy.getRank() - 1; i < s; ++i)
    pushNext(query, queryTy, i);

  TypedValue<ShapedType> key = onlineAttentionOp.getQuery();
  ShapedType keyTy = key.getType();
  pushNext(key, keyTy, keyTy.getRank() - 2);

  Type maskType = getElementTypeOrSelf(queryTy);
  Value emptyMask =
      rewriter.create<tensor::EmptyOp>(loc, maskStatic, maskType, maskDyn);

  Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getFloatAttr(getElementTypeOrSelf(maskType), 0.0));
  Value mask =
      rewriter.create<linalg::FillOp>(loc, zero, emptyMask).getResult(0);

  rewriter.startOpModification(onlineAttentionOp);
  onlineAttentionOp.getMaskMutable().assign(mask);
  MLIRContext *ctx = onlineAttentionOp->getContext();
  int64_t opRank = onlineAttentionOp.getQueryMap().getNumDims();
  AffineMap mMap;
  if (opRank == 4) {
    AffineExpr m, n, k1, k2;
    bindDims(ctx, m, n, k1, k2);
    mMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {m, k2}, ctx);
  } else if (opRank == 5) {
    AffineExpr b, m, n, k1, k2;
    bindDims(ctx, b, m, n, k1, k2);
    mMap = AffineMap::get(/*dimCount=*/5, /*symbolCount=*/0, {b, m, k2}, ctx);
  } else {
    llvm_unreachable("unsupported rank");
  }
  SmallVector<AffineMap> maps = onlineAttentionOp.getIndexingMapsArray();
  int64_t maskIdx = onlineAttentionOp.getMaskOperandIndex().value();
  maps.insert(maps.begin() + maskIdx, mMap);
  onlineAttentionOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
  rewriter.finalizeOpModification(onlineAttentionOp);

  return cast<TypedValue<ShapedType>>(mask);
}

static LogicalResult applyPaddingLevel(RewriterBase &rewriter,
                                       TilingInterface tilingInterfaceOp,
                                       IREE::GPU::TilingLevel tilingLevel) {
  // 1.a. Get padding values.
  SmallVector<Attribute> paddingValues;
  for (Value operand : tilingInterfaceOp.getOperation()->getOperands()) {
    paddingValues.push_back(
        rewriter.getZeroAttr(getElementTypeOrSelf(operand.getType())));
  }

  // 1.b. Special adjustment for OnlineAttention mask padding that needs to be
  // mindful of softmax and pad to -inf.
  // TODO: Extract into an upstream PaddingOpInterface.
  if (auto onlineAttentionOp = dyn_cast<IREE::LinalgExt::OnlineAttentionOp>(
          tilingInterfaceOp.getOperation())) {
    TypedValue<ShapedType> mask = onlineAttentionOp.getMask();
    if (!mask) {
      mask = setDefaultOnlineAttentionMask(rewriter, onlineAttentionOp);
      int64_t maskIdx = onlineAttentionOp.getMaskOperandIndex().value();
      paddingValues.insert(
          paddingValues.begin() + maskIdx,
          rewriter.getZeroAttr(getElementTypeOrSelf(mask.getType())));
    }
    Type maskEltType = getElementTypeOrSelf(mask.getType());
    if (!llvm::isa<FloatType>(maskEltType)) {
      tilingInterfaceOp.emitRemark(
          "failed to pad op: -inf requires a float type");
      return failure();
    }
    int64_t maskIdx = onlineAttentionOp.getMaskOperandIndex().value();
    const auto &fltSemantics = cast<FloatType>(maskEltType).getFloatSemantics();
    paddingValues[maskIdx] = rewriter.getFloatAttr(
        maskEltType, APFloat::getInf(fltSemantics, /*Negative=*/true));
  }

  // 2. Get padding sizes from tileSizes.
  SmallVector<int64_t> tileSizes =
      getLoweringConfig(tilingInterfaceOp)
          .getStaticTilingLevelSizes(llvm::to_underlying(tilingLevel),
                                     tilingInterfaceOp);
  SmallVector<OpFoldResult> padSizes =
      getAsIndexOpFoldResult(rewriter.getContext(), tileSizes);

  // 3. Set options.
  auto options = linalg::PadTilingInterfaceOptions()
                     .setPaddingSizes(padSizes)
                     .setPaddingValues(paddingValues)
                     .setPadToMultipleOf(true);

  LLVM_DEBUG(DBGS() << "Start padding " << *tilingInterfaceOp << "\n";
             DBGS() << "--with tile sizes: "
                    << llvm::interleaved_array(options.paddingSizes) << "\n";
             DBGS() << "--with padding values: "
                    << llvm::interleaved_array(options.paddingValues) << "\n";
             DBGS() << "--with padToMultipleOf: " << options.padToMultipleOf
                    << "\n");

  // 4. Pad.
  SmallVector<tensor::PadOp> padOps;
  FailureOr<TilingInterface> maybePaddedOp =
      linalg::rewriteAsPaddedOp(rewriter, tilingInterfaceOp, options, padOps);
  if (failed(maybePaddedOp)) {
    tilingInterfaceOp.emitWarning("failed to pad op");
    return failure();
  }

  // 5. For each PadOp, create a linalg::CopyOp to allow dim propagations.
  TilingInterface paddedOp = *maybePaddedOp;
  for (auto padOp : padOps) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(padOp);

    // Record users for RAUW before creating new users.
    llvm::SmallDenseSet<Operation *> users(padOp.getResult().getUsers().begin(),
                                           padOp.getResult().getUsers().end());

    RankedTensorType tensorTy = padOp.getResultType();
    int64_t rank = tensorTy.getRank();
    SmallVector<OpFoldResult> sizes(rank, OpFoldResult());
    for (int64_t i = 0; i < rank; ++i) {
      sizes[i] = rewriter.createOrFold<tensor::DimOp>(paddedOp->getLoc(),
                                                      padOp.getResult(), i);
      if (auto v = dyn_cast<Value>(sizes[i]))
        sizes[i] = getAsOpFoldResult(v);
    }

    // padOp.getResultType(),
    Value out = rewriter.create<tensor::EmptyOp>(
        paddedOp.getLoc(), sizes, getElementTypeOrSelf(tensorTy));
    auto copied = rewriter.create<linalg::CopyOp>(paddedOp.getLoc(),
                                                  padOp.getResult(), out);
    rewriter.replaceUsesWithIf(padOp.getResult(), copied.getResult(0),
                               [&](OpOperand &opOperand) {
                                 return users.contains(opOperand.getOwner());
                               });
  }

  return success();
}

void GPUApplyPaddingLevelPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  llvm::SmallDenseSet<TilingInterface> targetOps =
      getTiledOps(funcOp, tilingLevel);

  IRRewriter rewriter(funcOp);
  for (TilingInterface op : targetOps) {
    // If some op does not get padded, that is fine for now.
    (void)applyPaddingLevel(rewriter, op, tilingLevel);
  }

  // TODO: in the pad then tile case (technically not applicable right now), we
  // will need some cleanup stuff to make things fold properly to static shapes.
}

} // namespace mlir::iree_compiler
