// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/WinogradConstants.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-tile-and-decompose-winograd-input"

namespace mlir {
namespace iree_compiler {

namespace {

class ReifyWinogradInputTransform final
    : public OpRewritePattern<IREE::LinalgExt::WinogradInputTransformOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  static void computeLoopParams(SmallVectorImpl<Value> &lbs,
                                SmallVectorImpl<Value> &ubs,
                                SmallVectorImpl<Value> &steps,
                                ArrayRef<int64_t> tensorShape, Location loc,
                                PatternRewriter &rewriter) {
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    for (auto shape : llvm::enumerate(tensorShape)) {
      if (shape.index() < 2) continue;
      lbs.push_back(zero);
      ubs.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, shape.value()));
      steps.push_back(one);
    }
  }

  LogicalResult matchAndRewrite(
      IREE::LinalgExt::WinogradInputTransformOp inputOp,
      PatternRewriter &rewriter) const override {
    auto loc = inputOp.getLoc();
    auto funcOp = inputOp->getParentOfType<func::FuncOp>();
    if (!funcOp) return failure();

    const float *BT{nullptr};
    auto inputTileSize = inputOp.getInputTileSize();
    auto outputTileSize = inputOp.getOutputTileSize();
    switch (outputTileSize) {
      case 6:
        BT = IREE::LinalgExt::Winograd::BT_6x6_3x3;
        break;
      default:
        return failure();
    }
    Value BTV = IREE::LinalgExt::createValueFrom2DConstant(
        BT, inputTileSize, inputTileSize, false, loc, rewriter);
    Value BV = IREE::LinalgExt::createValueFrom2DConstant(
        BT, inputTileSize, inputTileSize, true, loc, rewriter);

    auto input = inputOp.input();
    auto output = inputOp.output();
    auto outputType = output.getType().cast<ShapedType>();
    auto inputType = input.getType().cast<ShapedType>();
    auto outputShape = outputType.getShape();
    auto inputShape = inputType.getShape();
    auto elementType = outputType.getElementType();
    SmallVector<int64_t> inputTileSquare(2, inputTileSize);

    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    auto zeroF32 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    auto scratch =
        rewriter.create<tensor::EmptyOp>(loc, inputTileSquare, elementType);

    rewriter.setInsertionPoint(inputOp);
    SmallVector<Value> lbs, ubs, steps;
    computeLoopParams(lbs, ubs, steps, outputShape, loc, rewriter);
    // Construct loops
    auto loopNest = scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps, ValueRange({output}),
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange outputIvs,
            ValueRange iterArgs) -> scf::ValueVector { return {iterArgs[0]}; });

    // Add spir-v attributes to loops (H, W, C)
    const char *attrName = "iree.spirv.distribute_dim";
    for (int i = loopNest.loops.size() - 1, dim = 0; i > 0; --i) {
      loopNest.loops[i]->setAttr(attrName, rewriter.getIndexAttr(dim++));
    }

    // Extract input slice
    auto one = rewriter.getIndexAttr(1);
    auto zero = rewriter.getIndexAttr(0);
    auto inputTileSizeAttr = rewriter.getIndexAttr(inputTileSize);
    SmallVector<OpFoldResult> strides(inputOp.getInputOperandRank(), one);
    SmallVector<OpFoldResult> sizes(inputOp.getInputOperandRank(), one);
    SmallVector<OpFoldResult> offsets(inputOp.getInputOperandRank(), zero);
    SmallVector<Value> ivs;
    for (auto loop : loopNest.loops) {
      ivs.push_back(loop.getInductionVar());
    }
    for (int i = 0; i < inputShape.size(); i++) {
      if ((i == 0) || (i == 3)) {
        offsets[i] = ivs[i];
      } else {
        rewriter.setInsertionPointToStart(loopNest.loops[i].getBody());
        AffineExpr dim0;
        auto it = rewriter.getAffineConstantExpr(inputTileSize);
        auto ot = rewriter.getAffineConstantExpr(outputTileSize);
        auto delta = rewriter.getAffineConstantExpr(inputShape[i]);
        bindDims(rewriter.getContext(), dim0);
        AffineMap scaleMap =
            AffineMap::get(1, 0, {dim0 * ot}, rewriter.getContext());
        offsets[i] = rewriter.createOrFold<AffineApplyOp>(loc, scaleMap,
                                                          ValueRange{ivs[i]});
        AffineMap minMap =
            AffineMap::get(1, 0, {-dim0 + delta, it}, rewriter.getContext());
        sizes[i] = rewriter.createOrFold<AffineMinOp>(
            loc, minMap,
            ValueRange{
                getValueOrCreateConstantIndexOp(rewriter, loc, offsets[i])});
      }
    }
    rewriter.setInsertionPointToStart(loopNest.loops.back().getBody());
    auto tensorType = RankedTensorType::get(
        SmallVector<int64_t>(2, ShapedType::kDynamicSize), elementType);
    auto dynamicSlice = rewriter
                            .create<tensor::ExtractSliceOp>(
                                loc, tensorType, input, offsets, sizes, strides)
                            .getResult();

    // Copy input slice into zeroed padded scratch space
    strides = SmallVector<OpFoldResult>(2, one);
    offsets = SmallVector<OpFoldResult>(2, zero);
    sizes = SmallVector<OpFoldResult>{sizes[1], sizes[2]};
    Value zeroScratch = rewriter
                            .create<linalg::FillOp>(loc, ValueRange{zeroF32},
                                                    ValueRange{scratch})
                            .result();
    auto inputSlice =
        rewriter
            .create<tensor::InsertSliceOp>(loc, dynamicSlice, zeroScratch,
                                           offsets, sizes, strides)
            .getResult();

    // Extract output slice
    strides = SmallVector<OpFoldResult>(inputOp.getOutputOperandRank(), one);
    offsets = SmallVector<OpFoldResult>(2, zero);
    offsets.append(ivs.begin(), ivs.end());
    sizes = SmallVector<OpFoldResult>(inputOp.getOutputOperandRank(), one);
    sizes[0] = sizes[1] = inputTileSizeAttr;
    tensorType = RankedTensorType::get(inputTileSquare, elementType);
    Value iterArg = loopNest.loops.back().getRegionIterArg(0);
    auto outputSlice =
        rewriter
            .create<tensor::ExtractSliceOp>(loc, tensorType, iterArg, offsets,
                                            sizes, strides)
            .getResult();

    // Create computation
    Value accumulator, result, AMatrix, BMatrix;
    linalg::MatmulOp matmulOp;
    for (int i = 0; i < 2; i++) {
      accumulator = rewriter
                        .create<linalg::FillOp>(loc, ValueRange{zeroF32},
                                                ValueRange{outputSlice})
                        .result();
      if (i == 0) {
        AMatrix = inputSlice;
        BMatrix = BV;
      } else {
        AMatrix = BTV;
        BMatrix = result;
      }
      matmulOp = rewriter.create<linalg::MatmulOp>(
          loc, tensorType, ValueRange{AMatrix, BMatrix}, accumulator);
      if (i == 0) {
        matmulOp->setAttr(IREE::LinalgExt::Winograd::getWinogradAttrName(),
                          rewriter.getStringAttr("I x B"));
      } else {
        matmulOp->setAttr(IREE::LinalgExt::Winograd::getWinogradAttrName(),
                          rewriter.getStringAttr("B' x I x B"));
      }
      result = matmulOp.getResult(0);
    }

    // Insert results into output slice
    auto updatedOutput = rewriter
                             .create<tensor::InsertSliceOp>(
                                 loc, result, iterArg, offsets, sizes, strides)
                             .getResult();

    // Replace returned value
    if (scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(
            loopNest.loops.back().getBody()->getTerminator())) {
      rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, updatedOutput);
    }
    inputOp.getResults()[0].replaceAllUsesWith(loopNest.getResults()[0]);
    return success();
  }
};

}  // namespace

namespace {
struct TileAndDecomposeWinogradInputTransformPass
    : public TileAndDecomposeWinogradInputTransformBase<
          TileAndDecomposeWinogradInputTransformPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void TileAndDecomposeWinogradInputTransformPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(&getContext());
  patterns.insert<ReifyWinogradInputTransform>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createTileAndDecomposeWinogradInputTransformPass() {
  return std::make_unique<TileAndDecomposeWinogradInputTransformPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
