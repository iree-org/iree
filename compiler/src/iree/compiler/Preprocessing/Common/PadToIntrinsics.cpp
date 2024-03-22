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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::Preprocessing {

static Value getPaddedValue(RewriterBase &rewriter, Location loc,
                            Value padSource, ArrayRef<int64_t> padding) {
  auto sourceType = padSource.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> sourceShape = sourceType.getShape();
  auto paddedShape =
      llvm::map_to_vector(llvm::zip_equal(sourceShape, padding), [](auto it) {
        return std::get<0>(it) + std::get<1>(it);
      });
  auto paddedResultType =
      RankedTensorType::get(paddedShape, sourceType.getElementType());
  Value paddingValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(sourceType.getElementType()));
  SmallVector<OpFoldResult> low(padding.size(), rewriter.getIndexAttr(0));
  auto high = llvm::map_to_vector(padding, [&](int64_t v) -> OpFoldResult {
    return rewriter.getIndexAttr(v);
  });
  Value paddedResult = rewriter.create<tensor::PadOp>(
      loc, paddedResultType, padSource, low, high, paddingValue);
  return paddedResult;
}

static void padConvOp(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                      ArrayRef<GPUMatmulShapeType> intrinsics) {
  // Operation *op = linalgOp.getOperation();
  if (!isa<linalg::ConvolutionOpInterface>(*linalgOp)) {
    return;
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
    return;
  }

  auto isAllOnesList = [](ArrayRef<int64_t> list) {
    return llvm::all_of(list, [](int64_t i) { return i == 1; });
  };

  // TODO: Support non-unit strides/dilations.
  if (!isAllOnesList(convolutionDims->strides) ||
      !isAllOnesList(convolutionDims->dilations)) {
    return;
  }

  int64_t mDim = convolutionDims->outputImage.back();
  int64_t nDim = convolutionDims->outputChannel.front();
  // TODO: Support NCHW convolutions. This is just a matmul_transpose_a,
  // however the distribution patterns currently do not support that variant.
  if (mDim > nDim) {
    return;
  }
  int64_t kDim = convolutionDims->inputChannel.front();
  int64_t mSize = bounds[mDim];
  int64_t nSize = bounds[nDim];
  int64_t kSize = bounds[kDim];

  // TODO: Generalize to other dimensions.
  // Try to search for pad value and check only filter dimension is blocked.
  SmallVector<std::array<int64_t, 3>> mnkPaddingCandidates;
  for (auto &intrinsic : intrinsics) {
    std::optional<int64_t> mPadding, nPadding, kPadding;
    auto getPadding = [](int64_t value, int64_t padTo) {
      return ((value + padTo - 1) / padTo) * padTo - value;
    };

    if (mSize % intrinsic.mSize != 0) {
      mPadding = getPadding(mSize, intrinsic.mSize);
    }

    if (nSize % intrinsic.nSize != 0) {
      nPadding = getPadding(nSize, intrinsic.nSize);
    }

    if (kSize % intrinsic.kSize != 0) {
      kPadding = getPadding(kSize, intrinsic.kSize);
    }

    if (!mPadding && !nPadding && !kPadding) {
      // Some intrinsic matches. Nothing to do.
      return;
    }
    mnkPaddingCandidates.push_back(
        {mPadding.value_or(0), nPadding.value_or(0), kPadding.value_or(0)});
  }
  if (mnkPaddingCandidates.empty()) {
    return;
  }

  std::array<int64_t, 3> mnkPadding = mnkPaddingCandidates.front();
  // TODO: Handle other variants.
  if (!isa<linalg::Conv2DNhwcHwcfOp>(linalgOp))
    return;

  Value newInput = linalgOp.getDpsInputOperand(0)->get();
  Value newFilter = linalgOp.getDpsInputOperand(1)->get();
  Value newOuts = linalgOp.getDpsInitOperand(0)->get();

  Location loc = linalgOp.getLoc();
  int64_t mPadding = mnkPadding[0];
  int64_t nPadding = mnkPadding[1];
  int64_t kPadding = mnkPadding[2];
  if (mPadding != 0 || kPadding != 0) {
    // For NHWC, the m-padding is for W and k-padding is for C
    newInput =
        getPaddedValue(rewriter, loc, newInput, {0, 0, mPadding, kPadding});
  }
  if (nPadding != 0 || kPadding != 0) {
    // For HWCF, the n-padding is for F and k-padding is for C
    newFilter =
        getPaddedValue(rewriter, loc, newFilter, {0, 0, kPadding, nPadding});
  }
  if (mPadding != 0 || nPadding != 0) {
    // For output, the m-padding is for W and k-padding is for F
    newOuts =
        getPaddedValue(rewriter, loc, newOuts, {0, 0, mPadding, nPadding});
  }

  auto paddedConv2dOp =
      mlir::clone(rewriter, linalgOp, {newOuts.getType()},
                  ArrayRef<Value>{newInput, newFilter, newOuts});
  // extract slice.
  auto zero = rewriter.getI64IntegerAttr(0);
  auto one = rewriter.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> offsets(4, zero), strides(4, one), sizes;
  auto resultType = linalgOp->getResult(0).getType().cast<RankedTensorType>();
  auto resultShape = resultType.getShape();
  sizes = {rewriter.getIndexAttr(resultShape[0]),
           rewriter.getIndexAttr(resultShape[1]),
           rewriter.getIndexAttr(resultShape[2]),
           rewriter.getIndexAttr(resultShape[3])};
  rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
      linalgOp, paddedConv2dOp->getResults()[0], offsets, sizes, strides);
}

static void padBatchGemmOp(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                           ArrayRef<GPUMatmulShapeType> intrinsics) {
  if (!isa<linalg::BatchMatmulOp>(linalgOp)) {
    return;
  }

  // Try to search for pad value.
  SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
  int64_t mSize = bounds[1];
  if (ShapedType::isDynamic(mSize)) {
    return;
  }
  SmallVector<std::array<int64_t, 3>> mnkPaddingCandidates;
  for (auto &intrinsic : intrinsics) {
    std::optional<int64_t> mPadding, nPadding, kPadding;
    auto getPadding = [](int64_t value, int64_t padTo) {
      return ((value + padTo - 1) / padTo) * padTo - value;
    };

    if (mSize % intrinsic.mSize != 0) {
      mPadding = getPadding(mSize, intrinsic.mSize);
    }

    if (!mPadding && !nPadding && !kPadding) {
      // Some intrinsic matches. Nothing to do.
      return;
    }
    mnkPaddingCandidates.push_back(
        {mPadding.value_or(0), nPadding.value_or(0), kPadding.value_or(0)});
  }
  if (mnkPaddingCandidates.empty()) {
    return;
  }
  std::array<int64_t, 3> mnkPadding = mnkPaddingCandidates.front();

  Value newLhs = linalgOp.getDpsInputOperand(0)->get();
  Value newRhs = linalgOp.getDpsInputOperand(1)->get();
  Value newOuts = linalgOp.getDpsInitOperand(0)->get();

  Location loc = linalgOp.getLoc();
  int64_t mPadding = mnkPadding[0];
  assert(mPadding != 0);
  newLhs = getPaddedValue(rewriter, loc, newLhs, {0, mPadding, 0});
  newOuts = getPaddedValue(rewriter, loc, newOuts, {0, mPadding, 0});
  auto paddedMatmulOp = mlir::clone(rewriter, linalgOp, {newOuts.getType()},
                                    ArrayRef<Value>{newLhs, newRhs, newOuts});

  // extract slice.
  auto zero = rewriter.getI64IntegerAttr(0);
  auto one = rewriter.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> offsets(3, zero), strides(3, one), sizes;
  auto resultType = linalgOp->getResult(0).getType().cast<RankedTensorType>();
  auto resultShape = resultType.getShape();
  sizes = {rewriter.getIndexAttr(resultShape[0]),
           rewriter.getIndexAttr(resultShape[1]),
           rewriter.getIndexAttr(resultShape[2])};
  rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
      linalgOp, paddedMatmulOp->getResults()[0], offsets, sizes, strides);
}

namespace {

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

    SmallVector<linalg::LinalgOp> targetOps;
    funcOp->walk([&](linalg::LinalgOp linalgOp) {
      if (isa<linalg::Conv2DNhwcHwcfOp, linalg::BatchMatmulOp>(
              linalgOp.getOperation()))
        targetOps.push_back(linalgOp);
    });

    IRRewriter rewriter(context);
    for (auto linalgOp : llvm::make_early_inc_range(targetOps)) {
      rewriter.setInsertionPoint(linalgOp);
      TypeSwitch<Operation *, void>(linalgOp.getOperation())
          .Case<linalg::Conv2DNhwcHwcfOp>(
              [&](auto convOp) { padConvOp(rewriter, linalgOp, intrinsics); })
          //.Case<linalg::BatchMatmulOp>([&](auto matmulOp) {
            //padBatchGemmOp(rewriter, linalgOp, intrinsics);
          //})
          .Default([&](Operation *op) {});
    }
  }
};

} // namespace

std::unique_ptr<Pass> createPadToIntrinsicsPass() {
  return std::make_unique<PadToIntrinsicsPass>();
}

} // namespace mlir::iree_compiler::Preprocessing
