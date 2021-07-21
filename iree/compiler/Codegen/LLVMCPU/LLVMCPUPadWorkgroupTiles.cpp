// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-pad-workgroup-tiles"

namespace mlir {
namespace iree_compiler {

namespace {

// Creates linalg.pad_tensor op with constant padding.
static linalg::PadTensorOp createPadTensorOpWithStaticPadding(
    PatternRewriter &rewriter, mlir::Location loc, Type resultType, Value input,
    Value padding, ArrayRef<int64_t> lowPadding,
    ArrayRef<int64_t> highPadding) {
  auto padTensorOp = rewriter.create<linalg::PadTensorOp>(
      loc, resultType, input, ArrayRef<Value>{}, ArrayRef<Value>{},
      rewriter.getI64ArrayAttr(lowPadding),
      rewriter.getI64ArrayAttr(highPadding));

  int rank = padTensorOp.getResultType().getRank();
  SmallVector<Type, 4> blockArgTypes;
  blockArgTypes.assign(rank, rewriter.getIndexType());
  auto &region = padTensorOp.region();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.createBlock(&region, region.end(), blockArgTypes);
  rewriter.create<linalg::YieldOp>(loc, padding);
  return padTensorOp;
}

// Returns padding for dim to next integer multiple of the vector size.
static std::pair<int, int> getVectorPaddingSize(int dim, int vecSize) {
  if (dim < vecSize) {
    int size = vecSize;
    int padValue = size - dim;
    return {vecSize, padValue};
  } else {
    int size = ceil((float)dim / (float)vecSize) * vecSize;
    int padValue = size - dim;
    return {size, padValue};
  }
}

// Returns padding for dim to next integer multiple of the workgroup size.
// Note: KernelDispatch gurantess workgroup size is the largest integer
// multiple of the vector size.
std::pair<int, int> getWorkgroupPaddedTileSize(int dim, int workgoupSize,
                                               int vecSize) {
  if (dim > workgoupSize) {
    int size = ceil((float)dim / (float)workgoupSize) * workgoupSize;
    int padValue = size - dim;
    return {workgoupSize, padValue};
  } else {
    return getVectorPaddingSize(dim, vecSize);
  }
}

// Creates linalg.matmul with operands padded to the next integer multiple of
// the workgroup size.
class MatmulWorkgroupTilesPadding : public OpRewritePattern<linalg::MatmulOp> {
 public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!hasLoweringConfig(matmulOp)) return failure();
    auto loc = matmulOp.getLoc();
    auto lhs = matmulOp.getInputOperand(0)->get();
    auto rhs = matmulOp.getInputOperand(1)->get();
    auto result = matmulOp.getOutputOperand(0)->get();

    if (lhs.getDefiningOp<linalg::PadTensorOp>() ||
        rhs.getDefiningOp<linalg::PadTensorOp>())
      return failure();

    auto workgroupTileSizes = getTileSizes(
        matmulOp, static_cast<unsigned>(TilingLevel::WorkGroupTiles));
    auto vectorTileSizes =
        getTileSizes(matmulOp, static_cast<unsigned>(TilingLevel::Level2Tiles));
    if (workgroupTileSizes.empty() || vectorTileSizes.empty()) return failure();

    auto lhsFullSize = getUntiledShape(lhs);
    auto rhsFullSize = getUntiledShape(rhs);
    if (lhsFullSize.empty() || rhsFullSize.empty()) return failure();

    int problemSizeM = lhsFullSize[0];
    int problemSizeN = rhsFullSize[1];
    int problemSizeK = lhsFullSize[1];

    int paddedMSize, paddedNSize, paddedKSize;
    int paddingForM, paddingForN, paddingForK;
    std::tie(paddedMSize, paddingForM) = getWorkgroupPaddedTileSize(
        problemSizeM, workgroupTileSizes[0], vectorTileSizes[0]);
    std::tie(paddedNSize, paddingForN) = getWorkgroupPaddedTileSize(
        problemSizeN, workgroupTileSizes[1], vectorTileSizes[1]);
    std::tie(paddedKSize, paddingForK) =
        getVectorPaddingSize(problemSizeK, vectorTileSizes[2]);

    // No padding.
    if (paddingForM == 0 && paddingForN == 0 && paddingForK == 0)
      return failure();

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      auto l1TileSizes = getTileSizes(
          matmulOp, static_cast<unsigned>(TilingLevel::Level1Tiles));
      llvm::dbgs() << "Problem-size: "
                   << "[" << problemSizeM << "," << problemSizeK << "]"
                   << ", "
                   << "[" << problemSizeK << "," << problemSizeN << "]\n";
      llvm::dbgs() << "Workgroup-tile-sizes:"
                   << "[" << workgroupTileSizes[0] << ", "
                   << workgroupTileSizes[1] << "]\n";
      llvm::dbgs() << "L1-tile-sizes:"
                   << "[" << l1TileSizes[0] << ", " << l1TileSizes[1] << ","
                   << l1TileSizes[2] << "]\n";
      llvm::dbgs() << "Vector-tile-sizes:"
                   << "[" << vectorTileSizes[0] << ", " << vectorTileSizes[1]
                   << ", " << vectorTileSizes[2] << "]\n";
      auto lhsStackSize = paddedMSize * paddedKSize * 4;
      auto rhsStackSize = paddedKSize * paddedNSize * 4;
      auto outputStackSize = paddedMSize * paddedNSize * 4;
      llvm::dbgs() << "LHS after padding:"
                   << "[" << paddedMSize << "," << paddedKSize
                   << "], size_in_bytes = " << lhsStackSize << "\n";
      llvm::dbgs() << "RHS after padding:"
                   << "[" << paddedKSize << "," << paddedNSize
                   << "], size_in_bytes = " << rhsStackSize << "\n";
      llvm::dbgs() << "Result after padding:"
                   << "[" << paddedMSize << "," << paddedNSize
                   << "], size_in_bytes = " << outputStackSize << "\n";
    });

    auto getPaddedOperand = [&](Value operand, ArrayRef<int64_t> shape,
                                ArrayRef<int64_t> highPadding) -> Value {
      if (llvm::all_of(highPadding,
                       [](int64_t val) -> bool { return val == 0; })) {
        return operand;
      }
      auto elementType =
          operand.getType().cast<RankedTensorType>().getElementType();
      auto paddedType = RankedTensorType::get(shape, elementType);
      auto paddingValue =
          rewriter.create<ConstantOp>(loc, rewriter.getZeroAttr(elementType));
      auto paddedOperand =
          createPadTensorOpWithStaticPadding(rewriter, loc, paddedType, operand,
                                             paddingValue, {0, 0}, highPadding);
      return paddedOperand;
    };

    auto paddedLhs = getPaddedOperand(lhs, {paddedMSize, paddedKSize},
                                      {paddingForM, paddingForK});

    auto paddedrhs = getPaddedOperand(rhs, {paddedKSize, paddedNSize},
                                      {paddingForK, paddingForN});

    auto resultType = RankedTensorType::get(
        {paddedMSize, paddedNSize},
        result.getType().cast<RankedTensorType>().getElementType());

    // Padding for K-dim only result doesn't change result size.
    if (paddingForM == 0 && paddingForN == 0) {
      auto paddedMatmulOp =
          cast<linalg::LinalgOp>(matmulOp.getOperation())
              .clone(rewriter, loc, {resultType},
                     ArrayRef<Value>{paddedLhs, paddedrhs, result});
      rewriter.replaceOp(matmulOp, paddedMatmulOp->getResults());
    } else {
      // Padding eather M or N requires changing the result size.
      auto getActualSizes = [](Value operand) {
        auto defOp = operand.getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
        return defOp.sizes();
      };
      // Get the actual output tile size (before padding).
      auto lhsSizes = getActualSizes(lhs);
      auto rhsSizes = getActualSizes(rhs);
      SmallVector<OpFoldResult> sizes;
      if (lhsSizes.empty()) {
        sizes.push_back(rewriter.getIndexAttr(paddedMSize));
      } else {
        sizes.push_back(lhsSizes.front());
      }
      if (rhsSizes.empty()) {
        sizes.push_back(rewriter.getIndexAttr(paddedNSize));
      } else {
        sizes.push_back(rhsSizes.back());
      }
      auto elementType = matmulOp.getResults()[0]
                             .getType()
                             .cast<ShapedType>()
                             .getElementType();
      auto staticResult = rewriter.create<linalg::InitTensorOp>(
          loc, ArrayRef<int64_t>{paddedMSize, paddedNSize}, elementType);

      Value zero =
          rewriter.create<ConstantOp>(loc, rewriter.getZeroAttr(elementType));
      auto filledStaticResult =
          rewriter.create<linalg::FillOp>(loc, zero, staticResult);
      auto paddedMatmulOp =
          cast<linalg::LinalgOp>(matmulOp.getOperation())
              .clone(rewriter, loc, {resultType},
                     ArrayRef<Value>{paddedLhs, paddedrhs,
                                     filledStaticResult.result()});
      SmallVector<OpFoldResult> offsets(2, rewriter.getI64IntegerAttr(0));
      SmallVector<OpFoldResult> strides(2, rewriter.getI64IntegerAttr(1));
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          matmulOp, paddedMatmulOp->getResults()[0], offsets, sizes, strides);
    }
    return success();
  }
};

struct LLVMCPUPadWorkgroupTilesPass
    : LLVMCPUPadWorkgroupTilesBase<LLVMCPUPadWorkgroupTilesPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<MatmulWorkgroupTilesPadding>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLLVMCPUPadWorkgroupTilesPass() {
  return std::make_unique<LLVMCPUPadWorkgroupTilesPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
