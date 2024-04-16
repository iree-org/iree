// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <limits>
#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_PADTOINTRINSICS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: export

namespace {

static Value getPaddedValue(RewriterBase &rewriter, Location loc,
                            Value padSource, ArrayRef<int64_t> padding) {
  auto sourceType = cast<RankedTensorType>(padSource.getType());
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
  if (!isa<linalg::ConvolutionOpInterface>(*linalgOp)) {
    return;
  }
  // TODO: Handle other variants.
  if (!isa<linalg::Conv2DNhwcHwcfOp>(linalgOp))
    return;

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
  for (const GPUMatmulShapeType &intrinsic : intrinsics) {
    std::optional<int64_t> mPadding, nPadding, kPadding;
    auto getPadding = [](int64_t value, int64_t padTo) {
      return llvm::divideCeil(value, padTo) * padTo - value;
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

  linalg::LinalgOp paddedConv2dOp =
      mlir::clone(rewriter, linalgOp, {newOuts.getType()},
                  ArrayRef<Value>{newInput, newFilter, newOuts});
  // Extract slice.
  IntegerAttr zero = rewriter.getI64IntegerAttr(0);
  IntegerAttr one = rewriter.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> offsets(4, zero);
  SmallVector<OpFoldResult> strides(4, one);
  auto resultType = cast<RankedTensorType>(linalgOp->getResult(0).getType());
  ArrayRef<int64_t> resultShape = resultType.getShape();
  SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(resultShape[0]),
                                     rewriter.getIndexAttr(resultShape[1]),
                                     rewriter.getIndexAttr(resultShape[2]),
                                     rewriter.getIndexAttr(resultShape[3])};
  Value extracted = rewriter.createOrFold<tensor::ExtractSliceOp>(
      loc, paddedConv2dOp->getResults()[0], offsets, sizes, strides);
  rewriter.replaceOp(linalgOp, extracted);
}

struct PadToIntrinsicsPass final
    : impl::PadToIntrinsicsBase<PadToIntrinsicsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    Operation *rootOp = getOperation();
    ArrayAttr mmaKinds = nullptr;
    for (IREE::HAL::ExecutableTargetAttr targetAttr :
         IREE::HAL::DeviceTargetAttr::lookupExecutableTargets(rootOp)) {
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

    SmallVector<linalg::Conv2DNhwcHwcfOp> targetOps;
    rootOp->walk(
        [&](linalg::Conv2DNhwcHwcfOp convOp) { targetOps.push_back(convOp); });

    IRRewriter rewriter(context);
    for (linalg::Conv2DNhwcHwcfOp convOp :
         llvm::make_early_inc_range(targetOps)) {
      rewriter.setInsertionPoint(convOp);
      padConvOp(rewriter, convOp, intrinsics);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::Preprocessing
