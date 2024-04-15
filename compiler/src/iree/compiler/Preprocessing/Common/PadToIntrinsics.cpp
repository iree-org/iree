// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <limits>
#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::Preprocessing {

namespace {
/// A pattern to pad statically shaped matmul operands to the next integer
/// multiple of `padSize`.
struct PadConvOpFilter final : OpInterfaceRewritePattern<linalg::LinalgOp> {
  PadConvOpFilter(MLIRContext *context, ArrayRef<GPUMatmulShapeType> intr,
                  PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern(context, benefit), intrinsics(intr) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    // Operation *op = linalgOp.getOperation();
    if (!isa<linalg::ConvolutionOpInterface>(*linalgOp)) {
      return failure();
    }
    // TODO: Handle other variants.
    if (!isa<linalg::Conv2DNhwcHwcfOp>(linalgOp))
      return failure();

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
    static constexpr int64_t kI64Max = std::numeric_limits<int64_t>::max();
    int64_t targetPadSize = kI64Max;
    for (const GPUMatmulShapeType &intrinsic : intrinsics) {
      if (mSize % intrinsic.mSize == 0 && nSize % intrinsic.nSize == 0 &&
          kSize % intrinsic.kSize == 0) {
        return failure();
      }

      if ((mSize % intrinsic.mSize == 0) && (nSize % intrinsic.nSize != 0) &&
          (kSize % intrinsic.kSize == 0)) {
        int64_t candidatetargetPadSize =
            llvm::divideCeil(nSize, intrinsic.nSize) * intrinsic.nSize;
        targetPadSize = std::min(targetPadSize, candidatetargetPadSize);
      }
    }
    if (targetPadSize == kI64Max)
      return failure();

    auto createPadding = [&](ArrayRef<int64_t> padding) {
      SmallVector<OpFoldResult> result;
      for (int64_t pad : padding) {
        result.push_back(rewriter.getI64IntegerAttr(pad));
      }
      return result;
    };

    // Pad filter.
    // TODO: Generalize padding hi,lo creation for different forms.
    const int kFDim = 3;
    Location loc = linalgOp.getLoc();
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
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
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
    linalg::LinalgOp paddedConv2dOp =
        mlir::clone(rewriter, linalgOp, {newResultType},
                    ArrayRef<Value>{input, filterPadOp, paddedResult});

    // Extract slice.
    IntegerAttr zero = rewriter.getI64IntegerAttr(0);
    IntegerAttr one = rewriter.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> offsets(4, zero);
    SmallVector<OpFoldResult> strides(4, one);
    ArrayRef<int64_t> resultShape = resultType.getShape();
    SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(resultShape[0]),
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

struct PadToIntrinsicsPass final : PadToIntrinsicsBase<PadToIntrinsicsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    auto funcOp = getOperation();
    ArrayAttr mmaKinds = nullptr;
    for (IREE::HAL::ExecutableTargetAttr targetAttr :
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

    auto intrinsics = llvm::map_to_vector(
        mmaKinds.getAsRange<IREE::GPU::MMAAttr>(), [](IREE::GPU::MMAAttr mma) {
          auto [mSize, nSize, kSize] = mma.getMNKShape();
          auto [aType, bType, cType] = mma.getABCElementTypes();
          return GPUMatmulShapeType{mSize, nSize, kSize, aType, bType, cType};
        });
    patterns.add<PadConvOpFilter>(context, intrinsics);
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
