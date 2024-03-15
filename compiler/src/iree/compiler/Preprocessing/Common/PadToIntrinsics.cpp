// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::Preprocessing {

namespace {
/// A pattern to pad statically shaped matmul operands to the next integer
/// multiple of padSize.
class PadConvOpFilter : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  PadConvOpFilter(MLIRContext *context, SmallVector<GPUMatmulShapeType> &intr,
                  PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern(context, benefit), intrinsics(intr) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    // Operation *op = linalgOp.getOperation();
    if (!isa<linalg::ConvolutionOpInterface>(*linalgOp)) {
      return failure();
    }
    // Check that conv has met conditions to go down mfma.
    SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
    FailureOr<mlir::linalg::ConvolutionDimensions> convolutionDims =
        mlir::linalg::inferConvolutionDims(linalgOp);
    assert(succeeded(convolutionDims) && "Could not infer contraction dims");

    if (convolutionDims->outputChannel.size() != 1 ||
        convolutionDims->inputChannel.size() != 1 ||
        convolutionDims->filterLoop.size() < 1 ||
        convolutionDims->outputImage.size() < 1 ||
        convolutionDims->depth.size() != 0) {
      return failure();
    }

    auto isAllOnesList = [](ArrayRef<int64_t> list) {
      return llvm::all_of(list, [](int64_t i) { return i == 1; });
    };

    // TODO: Support non-unit strides/dilations.
    if (!isAllOnesList(convolutionDims->strides) ||
        !isAllOnesList(convolutionDims->dilations)) {
      return failure();
    }

    int64_t mDim = convolutionDims->outputImage.back();
    int64_t nDim = convolutionDims->outputChannel.front();
    // TODO: Support NCHW convolutions. This is just a matmul_transpose_a,
    // however the distribution patterns currently do not support that variant.
    if (mDim > nDim) {
      return failure();
    }
    int64_t kDim = convolutionDims->inputChannel.front();
    int64_t mSize = bounds[mDim];
    int64_t nSize = bounds[nDim];
    int64_t kSize = bounds[kDim];

    // TODO: Generalize to other dimensions.
    // Try to search for pad value and check only filter dimension is blocked.
    int64_t targetPadSize = std::numeric_limits<int64_t>::max();
    for (auto &intrinsic : intrinsics) {
      if (mSize % intrinsic.mSize == 0 && nSize % intrinsic.nSize == 0 &&
          kSize % intrinsic.kSize == 0) {
        return failure();
      }

      if (mSize % intrinsic.mSize == 0 && nSize % intrinsic.nSize != 0 &&
          kSize % intrinsic.kSize == 0) {
        int64_t candidatetargetPadSize =
            std::ceil(float(nSize) / intrinsic.nSize) * intrinsic.nSize;
        if (candidatetargetPadSize < targetPadSize)
          targetPadSize = candidatetargetPadSize;
      }
    }
    if (targetPadSize == std::numeric_limits<int64_t>::max())
      return failure();
    // TODO: Handle other variants.
    if (!isa<linalg::Conv2DNhwcHwcfOp>(linalgOp))
      return failure();

    auto createPadding = [&](ArrayRef<int64_t> padding) {
      SmallVector<OpFoldResult> result;
      for (auto pad : padding) {
        result.push_back(rewriter.getI64IntegerAttr(pad));
      }
      return result;
    };

    // Pad filter
    // TODO: Generalize padding hi,lo creation for different forms.
    const int kFDim = 3;
    auto loc = linalgOp.getLoc();
    Value filter = linalgOp.getDpsInputOperand(1)->get();
    auto filterType = llvm::dyn_cast<RankedTensorType>(filter.getType());
    if (!filterType)
      return failure();
    SmallVector<int64_t> filterPaddedShape(filterType.getShape());
    filterPaddedShape[kFDim] = targetPadSize;
    auto filterPaddedType =
        RankedTensorType::get(filterPaddedShape, filterType.getElementType());
    Value filterPaddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(filterType.getElementType()));
    Value filterPadOp = rewriter.create<tensor::PadOp>(
        loc, filterPaddedType, filter, createPadding({0, 0, 0, 0}),
        createPadding({0, 0, 0, targetPadSize - nSize}), filterPaddingValue);

    // Create conv2d with padded filter
    Value input = linalgOp.getDpsInputOperand(0)->get();
    Value result = linalgOp.getDpsInitOperand(0)->get();
    auto resultType = llvm::dyn_cast<RankedTensorType>(result.getType());
    if (!resultType)
      return failure();
    SmallVector<int64_t> newResultShape(resultType.getShape());
    newResultShape[kFDim] = targetPadSize;

    auto newResultType =
        RankedTensorType::get(newResultShape, resultType.getElementType());
    Value resultPaddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultType.getElementType()));
    Value paddedResult = rewriter.create<tensor::PadOp>(
        loc, newResultType, result, createPadding({0, 0, 0, 0}),
        createPadding({0, 0, 0, targetPadSize - nSize}), resultPaddingValue);
    auto paddedConv2dOp =
        mlir::clone(rewriter, linalgOp, {newResultType},
                    ArrayRef<Value>{input, filterPadOp, paddedResult});
    // extract slice.
    auto zero = rewriter.getI64IntegerAttr(0);
    auto one = rewriter.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> offsets, strides, sizes;
    offsets.assign(4, zero);
    strides.assign(4, one);
    auto resultShape = resultType.getShape();
    sizes = {rewriter.getIndexAttr(resultShape[0]),
             rewriter.getIndexAttr(resultShape[1]),
             rewriter.getIndexAttr(resultShape[2]),
             rewriter.getIndexAttr(resultShape[3])};
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        linalgOp, paddedConv2dOp->getResults()[0], offsets, sizes, strides);
    return success();
  }

private:
  SmallVector<GPUMatmulShapeType> intrinsics;
};

class PadBatchMatmulOp : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  PadBatchMatmulOp(MLIRContext *context, SmallVector<GPUMatmulShapeType> &intr,
                   PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern(context, benefit), intrinsics(intr) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    // Operation *op = linalgOp.getOperation();
    if (!isa<linalg::BatchMatmulOp>(*linalgOp)) {
      return failure();
    }
    // Check that conv has met conditions to go down mfma.
    SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
    FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
        mlir::linalg::inferContractionDims(linalgOp);
    assert(succeeded(contractionDims) && "Could not infer contraction dims");

    if (contractionDims->m.size() != 1 || contractionDims->k.size() != 1 ||
        contractionDims->n.size() != 1 || contractionDims->batch.size() > 1) {
      return failure();
    }

    int64_t mDim = contractionDims->m.back();
    int64_t nDim = contractionDims->n.front();
    int64_t kDim = contractionDims->k.front();
    int64_t mSize = bounds[mDim];
    int64_t nSize = bounds[nDim];
    int64_t kSize = bounds[kDim];

    // TODO: Generalize to other dimensions.
    // Try to search for pad value and check only filter dimension is blocked.
    int64_t targetPadSize = __INT64_MAX__;
    for (auto &intrinsic : intrinsics) {
      if (mSize % intrinsic.mSize == 0 && nSize % intrinsic.nSize == 0 &&
          kSize % intrinsic.kSize == 0) {
        return failure();
      }

      if (mSize % intrinsic.mSize != 0 && nSize % intrinsic.nSize == 0 &&
          kSize % intrinsic.kSize == 0) {
        int64_t candidatetargetPadSize =
            std::ceil(float(mSize) / intrinsic.mSize) * intrinsic.mSize;
        if (candidatetargetPadSize < targetPadSize)
          targetPadSize = candidatetargetPadSize;
      }
    }
    if (targetPadSize == __INT64_MAX__) {
      return failure();
    }

    auto createPadding = [&](ArrayRef<int64_t> padding) {
      SmallVector<OpFoldResult> result;
      for (auto pad : padding) {
        result.push_back(rewriter.getI64IntegerAttr(pad));
      }
      return result;
    };

    // Pad inputs
    const int mLhsDim = 1;
    auto loc = linalgOp.getLoc();
    auto lhs = linalgOp->getOperand(0);
    auto lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
    if (!lhsType) {
      return failure();
    }
    SmallVector<int64_t> paddedShape(lhsType.getShape());
    paddedShape[mLhsDim] = targetPadSize;
    auto paddedType =
        RankedTensorType::get(paddedShape, lhsType.getElementType());
    Value paddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(lhsType.getElementType()));
    Value paddedLhs = rewriter.create<tensor::PadOp>(
        loc, paddedType, lhs, createPadding({0, 0, 0}),
        createPadding({0, targetPadSize - mSize, 0}), paddingValue);

    // Create conv2d with padded filter
    Value rhs = linalgOp.getDpsInputOperand(1)->get();
    Value result = linalgOp.getDpsInitOperand(0)->get();
    auto resultType = llvm::dyn_cast<RankedTensorType>(result.getType());
    if (!resultType) {
      return failure();
    }
    SmallVector<int64_t> newResultShape(resultType.getShape());
    newResultShape[mLhsDim] = targetPadSize;

    auto newResultType =
        RankedTensorType::get(newResultShape, resultType.getElementType());
    Value resultPaddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultType.getElementType()));
    Value paddedResult = rewriter.create<tensor::PadOp>(
        loc, newResultType, result, createPadding({0, 0, 0}),
        createPadding({0, targetPadSize - mSize, 0}), resultPaddingValue);
    auto paddedBatchMatmulOp =
        mlir::clone(rewriter, linalgOp, {newResultType},
                    ArrayRef<Value>{paddedLhs, rhs, paddedResult});
    // extract slice.
    auto zero = rewriter.getI64IntegerAttr(0);
    auto one = rewriter.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> offsets, strides, sizes;
    offsets.assign(3, zero);
    strides.assign(3, one);
    auto resultShape = resultType.getShape();
    sizes = {rewriter.getIndexAttr(resultShape[0]),
             rewriter.getIndexAttr(resultShape[1]),
             rewriter.getIndexAttr(resultShape[2])};
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        linalgOp, paddedBatchMatmulOp->getResults()[0], offsets, sizes,
        strides);
    return success();
  }

private:
  SmallVector<GPUMatmulShapeType> intrinsics;
};

class PadToIntrinsicsPass : public PadToIntrinsicsBase<PadToIntrinsicsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    auto funcOp = getOperation();
    ArrayAttr mmaKinds = nullptr;
    for (auto targetAttr :
         IREE::HAL::DeviceTargetAttr::lookupExecutableTargets(funcOp)) {
      FailureOr<ArrayAttr> candidateMmaKinds =
          getSupportedMmaTypes(targetAttr.getConfiguration());
      if (succeeded(candidateMmaKinds)) {
        mmaKinds = *candidateMmaKinds;
        break;
      }
    }
    if (!mmaKinds)
      return;

    auto mmaAttrs = llvm::to_vector(mmaKinds.getAsRange<IREE::GPU::MmaAttr>());
    SmallVector<GPUMatmulShapeType> intrinsics;
    intrinsics.reserve(mmaKinds.size());
    for (auto mma : mmaAttrs) {
      auto [mSize, nSize, kSize] = mma.getMNKShape();
      auto [aType, bType, cType] = mma.getABCElementTypes();
      intrinsics.emplace_back(mSize, nSize, kSize, aType, bType, cType);
    }
    patterns.insert<PadConvOpFilter>(context, intrinsics);
    patterns.insert<PadBatchMatmulOp>(context, intrinsics);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createPadToIntrinsicsPass() {
  return std::make_unique<PadToIntrinsicsPass>();
}

} // namespace mlir::iree_compiler::Preprocessing
