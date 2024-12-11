// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen--materialize-encoding"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_MATERIALIZEDEVICEENCODINGPASS
#define GEN_PASS_DEF_MATERIALIZEHOSTENCODINGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

using namespace IREE::Encoding;
using IREE::Codegen::MaterializeEncodingInfo;
using IREE::Codegen::TileSwizzle;

namespace {

static SmallVector<ReassociationIndices>
getReassociationIndices(int outerDims,
                        const TileSwizzle::ExpandShapeType &expandShape) {
  SmallVector<ReassociationIndices> result;
  int expandedIdx = 0;
  for (int i = 0; i < outerDims; ++i) {
    result.push_back({expandedIdx++});
  }
  for (auto expandShapeDim : expandShape) {
    result.push_back({});
    for (int i = 0, e = expandShapeDim.size(); i < e; ++i) {
      result.back().push_back(expandedIdx++);
    }
  }
  return result;
}

/// Convert iree_linalg_ext.set_encoding op to pack + tile swizzling ops. We use
/// expand_shape + linalg.transpose to represent a tile swizzling op.
struct SetEncodingOpLoweringConversion
    : public OpMaterializeEncodingPattern<IREE::Encoding::SetEncodingOp> {
  using OpMaterializeEncodingPattern<
      IREE::Encoding::SetEncodingOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());
    auto packedValue = lowerSetEncodingOpToPackOp(
        rewriter, encodingOp, adaptor.getSource(), *converter,
        this->materializeEncodingValueFn);
    if (failed(packedValue)) {
      Type targetType =
          getTypeConverter()->convertType(encodingOp.getResultType());
      Value result = rewriter.createOrFold<tensor::CastOp>(
          encodingOp.getLoc(), targetType, adaptor.getSource());
      rewriter.replaceOp(encodingOp, result);
      return success();
    }

    MaterializeEncodingInfo encodingInfo =
        converter->getEncodingInfo(encodingOp.getResultType());
    if (!encodingInfo.swizzle) {
      rewriter.replaceOp(encodingOp, packedValue.value());
      return success();
    }

    Location loc = encodingOp.getLoc();

    // Create expand_shape op to tile the innermost two dimensions.
    int origRank = encodingOp.getSourceType().getRank();
    SmallVector<int64_t> expandShapeShape(
        cast<ShapedType>(packedValue->getType())
            .getShape()
            .take_front(origRank));
    expandShapeShape.append(
        getExpandedTileShape(encodingInfo.swizzle->expandShape));
    RankedTensorType expandShapeType =
        encodingOp.getSourceType().clone(expandShapeShape);

    SmallVector<ReassociationIndices> reassociation =
        getReassociationIndices(origRank, encodingInfo.swizzle->expandShape);
    auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandShapeType, packedValue.value(), reassociation);

    SmallVector<int64_t> transposePerm =
        llvm::to_vector(llvm::seq<int64_t>(0, origRank));
    for (auto perm : encodingInfo.swizzle->permutation) {
      transposePerm.push_back(origRank + perm);
    }
    SmallVector<OpFoldResult> transposeResultDims =
        tensor::getMixedSizes(rewriter, loc, expandShapeOp.getResult());
    applyPermutationToVector(transposeResultDims, transposePerm);

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, transposeResultDims, encodingOp.getSourceType().getElementType());
    auto transposeOp = rewriter.create<linalg::TransposeOp>(
        loc, expandShapeOp, emptyTensor, transposePerm);
    rewriter.replaceOp(encodingOp, transposeOp->getResult(0));

    return success();
  }
};

struct UnsetEncodingOpLoweringConversion
    : public OpMaterializeEncodingPattern<IREE::Encoding::UnsetEncodingOp> {
  using OpMaterializeEncodingPattern<
      IREE::Encoding::UnsetEncodingOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Encoding::UnsetEncodingOp unsetEncodingOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());

    MaterializeEncodingInfo encodingInfo =
        converter->getEncodingInfo(unsetEncodingOp.getSource().getType());
    if (IREE::Codegen::isIdentityLayout(encodingInfo)) {
      Type targetType =
          getTypeConverter()->convertType(unsetEncodingOp.getSourceType());
      Value result = rewriter.createOrFold<tensor::CastOp>(
          unsetEncodingOp.getLoc(), targetType, adaptor.getSource());
      rewriter.replaceOp(unsetEncodingOp, result);
      return success();
    }

    Location loc = unsetEncodingOp.getLoc();
    Value unpackSrc = adaptor.getSource();
    if (encodingInfo.swizzle) {
      int targetRank = unsetEncodingOp.getResultType().getRank();
      auto srcConvertedType =
          cast<RankedTensorType>(adaptor.getSource().getType());
      SmallVector<OpFoldResult> emptyShape =
          tensor::getMixedSizes(rewriter, loc, adaptor.getSource());
      emptyShape.resize(targetRank);
      for (auto i : getExpandedTileShape(encodingInfo.swizzle->expandShape)) {
        emptyShape.push_back(rewriter.getIndexAttr(i));
      }
      auto emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, emptyShape, unsetEncodingOp.getSourceType().getElementType());

      SmallVector<int64_t> transposePerm =
          llvm::to_vector(llvm::seq<int64_t>(0, targetRank));
      for (auto perm : encodingInfo.swizzle->permutation) {
        transposePerm.push_back(targetRank + perm);
      }
      auto invertedTransposePerm = invertPermutationVector(transposePerm);
      auto transposeOp = rewriter.create<linalg::TransposeOp>(
          loc, adaptor.getSource(), emptyTensor, invertedTransposePerm);

      SmallVector<ReassociationIndices> reassociation = getReassociationIndices(
          targetRank, encodingInfo.swizzle->expandShape);
      SmallVector<int64_t> unpackSrcShape(
          srcConvertedType.getShape().take_front(targetRank));
      unpackSrcShape.append(encodingInfo.innerTileSizes.begin(),
                            encodingInfo.innerTileSizes.end());
      RankedTensorType unpackSrcType =
          unsetEncodingOp.getResultType().clone(unpackSrcShape);
      unpackSrc = rewriter.create<tensor::CollapseShapeOp>(
          loc, unpackSrcType, transposeOp->getResult(0), reassociation);
    }

    auto unpackedValue = lowerUnsetEncodingToUnpackOp(
        rewriter, unsetEncodingOp, unpackSrc, *converter,
        this->materializeEncodingValueFn);
    if (failed(unpackedValue)) {
      Type targetType =
          getTypeConverter()->convertType(unsetEncodingOp.getResultType());
      Value result = rewriter.createOrFold<tensor::CastOp>(loc, targetType,
                                                           adaptor.getSource());
      rewriter.replaceOp(unsetEncodingOp, result);
      return success();
    }
    rewriter.replaceOp(unsetEncodingOp, unpackedValue.value());
    return success();
  }
};

/// Pattern to convert contraction operations.
class MaterializeContractionOp
    : public OpInterfaceConversionPattern<linalg::LinalgOp> {
public:
  MaterializeContractionOp(
      MLIRContext *context,
      const MaterializeEncodingTypeConverter &typeConverter,
      MaterializeEncodingValueFn materializeEncodingValueFn = {},
      PatternBenefit benefit = 1)
      : OpInterfaceConversionPattern<linalg::LinalgOp>(typeConverter, context,
                                                       benefit),
        materializeEncodingValueFn(materializeEncodingValueFn) {}

  LogicalResult
  matchAndRewrite(linalg::LinalgOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(op)) {
      return rewriter.notifyMatchFailure(
          op, "does not implement ContractionOpInterface");
    }

    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        this->getTypeConverter());

    IREE::Codegen::LayoutAttrInterface layoutAttr = converter->getLayoutAttr();
    SmallVector<Type> convertedResTypes;
    for (auto init : op.getDpsInits()) {
      convertedResTypes.push_back(converter->convertType(init.getType()));
    }
    Operation *newOp =
        layoutAttr.lowerOp(rewriter, op, convertedResTypes, operands);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

protected:
  const MaterializeEncodingValueFn materializeEncodingValueFn;
};

static FailureOr<MaterializeEncodingValueInfo>
chooseDynamicEncodingInfoVMVXMicrokernels(RankedTensorType tensorType,
                                          OpBuilder &builder, Location loc) {
  SmallVector<Type> resultTypes(tensorType.getRank(), builder.getIndexType());
  auto op = builder.create<IREE::Codegen::QueryTileSizesOp>(
      loc, resultTypes, TypeAttr::get(tensorType));
  MaterializeEncodingValueInfo result;
  result.innerTileSizes = op.getResults();
  return result;
}

static MaterializeEncodingValueFn
getMaterializeEncodingValueFn(IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (isVMVXBackend(targetAttr) && hasUkernel(targetAttr)) {
    return chooseDynamicEncodingInfoVMVXMicrokernels;
  }
  return {};
}

static LogicalResult
materializeFuncOpEncodings(FunctionOpInterface funcOp,
                           IREE::HAL::ExecutableTargetAttr targetAttr,
                           bool testCLGPUTarget = false) {
  MLIRContext *ctx = funcOp.getContext();
  {
    RewritePatternSet patterns(ctx);
    IREE::Codegen::LayoutAttrInterface layoutAttr;
    if (isVMVXBackend(targetAttr)) {
      LDBG("Select VMVXEncodingLayoutAttr attribute as the layout attribute.");
      layoutAttr = cast<IREE::Codegen::LayoutAttrInterface>(
          IREE::CPU::VMVXEncodingLayoutAttr::get(
              ctx, targetAttr.getConfiguration()));
    } else if (isLLVMCPUBackend(targetAttr)) {
      LDBG("Select CPUEncodingLayoutAttr attribute as the layout attribute.");
      layoutAttr = cast<IREE::Codegen::LayoutAttrInterface>(
          IREE::CPU::CPUEncodingLayoutAttr::get(ctx,
                                                targetAttr.getConfiguration()));
    } else if (isROCMBackend(targetAttr)) {
      LDBG("Select GPUEncodingLayoutAttr attribute as the layout attribute.");
      layoutAttr = cast<IREE::Codegen::LayoutAttrInterface>(
          IREE::GPU::GPUEncodingLayoutAttr::get(ctx,
                                                getGPUTargetAttr(targetAttr)));
    } else if (testCLGPUTarget) {
      LDBG("Select GPUEncodingLayoutAttr attribute as the layout attribute. "
           "(testCLGPUTarget)");
      layoutAttr = cast<IREE::Codegen::LayoutAttrInterface>(
          IREE::GPU::GPUEncodingLayoutAttr::get(ctx, getCLGPUTarget(ctx)));
    } else {
      LDBG("Select EncodingNopLayoutAttr attribute as the layout attribute.");
      layoutAttr = IREE::Codegen::EncodingNopLayoutAttr::get(ctx);
    }
    MaterializeEncodingTypeConverter typeConverter(layoutAttr);
    MaterializeEncodingConversionTarget target(*ctx);
    auto materializeEncodingValueFn = getMaterializeEncodingValueFn(targetAttr);
    patterns.insert<MaterializeContractionOp, SetEncodingOpLoweringConversion,
                    UnsetEncodingOpLoweringConversion>(
        ctx, typeConverter, materializeEncodingValueFn);
    populateShapeIndependentMaterializeEncodingPatterns(
        patterns, target, typeConverter, materializeEncodingValueFn);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      funcOp.emitOpError("materialization failed");
      return failure();
    }
  }

  // Add patterns to fold pack/unpack ops with pad/extract_slice ops and
  // resolve dims ops.
  {
    RewritePatternSet patterns(ctx);
    tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::populateFoldIntoPackAndUnpackPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitOpError("folding patterns failed");
      return failure();
    }
  }

  return success();
}

// Returns the executable targets used within |funcOp|.
//
// TODO(multi-device): delete this pass and rely on tensor-based analysis to
// materialize encodings based on where tensors are used. This pass is not able
// to handle that.
static std::optional<SetVector<IREE::HAL::ExecutableTargetAttr>>
getFuncExecutableTargetAttrs(FunctionOpInterface funcOp,
                             IREE::Stream::AffinityAnalysis &affinityAnalysis,
                             IREE::HAL::DeviceAnalysis &deviceAnalysis) {
  // Get a set of all unique affinities used by resources within the function.
  SetVector<IREE::Stream::AffinityAttr> uniqueAffinityAttrs;
  SmallVector<IREE::Stream::AffinityAttr> lookupAffinityAttrs;
  funcOp.walk([&](Operation *op) {
    if (affinityAnalysis.tryLookupExecutionAffinity(op, lookupAffinityAttrs)) {
      uniqueAffinityAttrs.insert(lookupAffinityAttrs.begin(),
                                 lookupAffinityAttrs.end());
    }
    lookupAffinityAttrs.clear();
  });

  // Resolve affinities to executable targets.
  SetVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
  for (auto affinityAttr : uniqueAffinityAttrs) {
    deviceAnalysis.gatherRequiredExecutableTargets(affinityAttr, funcOp,
                                                   executableTargetAttrs);
  }
  return executableTargetAttrs;
}

struct MaterializeHostEncodingPass
    : public impl::MaterializeHostEncodingPassBase<
          MaterializeHostEncodingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect,
                    IREE::Codegen::IREECodegenDialect,
                    IREE::CPU::IREECPUDialect, IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Run required analysis passes.
    IREE::Stream::AffinityAnalysis affinityAnalysis(moduleOp);
    if (failed(affinityAnalysis.run())) {
      return signalPassFailure();
    }
    IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run())) {
      return signalPassFailure();
    }

    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      // Gather the required executable targets for the function. Note that it's
      // possible there are more required for ops nested within the function but
      // this pass is a hack and can't handle that :shrug:.
      auto executableTargets = getFuncExecutableTargetAttrs(
          funcOp, affinityAnalysis, deviceAnalysis);
      if (!executableTargets) {
        funcOp.emitOpError()
            << "could not determine executable targets for the function";
        return signalPassFailure();
      } else if (executableTargets->empty()) {
        // Probably no tensors.
        continue;
      }

      // HACK: this pass is run on the host _but shouldn't be_. Because it's
      // run on the host and IREE is a compiler capable of multi-targeting there
      // may be multiple executable targets at any point in the host program.
      // This pass can't handle that and assumes it's been checked earlier by
      // spooky action at a distance. This needs to be fixed.
      if (executableTargets->size() != 1) {
        funcOp.emitOpError() << "has multiple executable targets and CPU data "
                                "tiling isn't built to support that";
        return signalPassFailure();
      }

      // Materialize encodings within the function.
      if (failed(
              materializeFuncOpEncodings(funcOp, executableTargets->front()))) {
        return signalPassFailure();
      }
    }
  }
};

// NOTE: this runs on host modules and executables and has two paths to handle
// that. It should _not_ be running on both - target-specific codegen passes
// are not allowed on host programs and it's a big violation of layering that
// this exists.
struct MaterializeDeviceEncodingPass
    : public impl::MaterializeDeviceEncodingPassBase<
          MaterializeDeviceEncodingPass> {
  using impl::MaterializeDeviceEncodingPassBase<
      MaterializeDeviceEncodingPass>::MaterializeDeviceEncodingPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect,
                    IREE::Codegen::IREECodegenDialect,
                    IREE::CPU::IREECPUDialect, IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto executableTargetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (failed(materializeFuncOpEncodings(funcOp, executableTargetAttr,
                                          testCLGPUTarget))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler
