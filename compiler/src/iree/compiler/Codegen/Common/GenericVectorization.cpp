// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Interfaces/VectorizableOpInterface.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/Im2colUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-generic-vectorization"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GENERICVECTORIZATIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

// Returns the vector sizes from the local lowering config, materialized
// tile size attributes, or tries to infer them from the tensor shapes and
// tiled loops in the IR.
static std::optional<SizesAndScalableFlags>
getVectorSizes(Operation *op, bool useConfiguredVectorSizes) {
  // Get vector sizes from the lowering config, if available in the op itself.
  IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
      getLoweringConfig(op);
  if (useConfiguredVectorSizes && loweringConfig) {
    LDBG() << "Use configured vector sizes from lowering config";
    std::optional<SmallVector<int64_t>> vectorSizes =
        loweringConfig.getVectorSizes();
    SmallVector<bool> scalableFlags = loweringConfig.getVectorScalableFlags();
    if (vectorSizes) {
      if (scalableFlags.empty()) {
        scalableFlags.assign(vectorSizes->size(), false);
      }
      if (auto unpackOp = dyn_cast<linalg::UnPackOp>(op)) {
        std::optional<SizesAndScalableFlags> maybeInputVectorSizes =
            getVectorInputSizesFromDestTiles(unpackOp, *vectorSizes,
                                             scalableFlags);
        if (maybeInputVectorSizes) {
          std::tie(vectorSizes, scalableFlags) = maybeInputVectorSizes.value();
        } else {
          LDBG() << "Failed to get input vector sizes for unpack op";
          return std::nullopt;
        }
      }
      if (auto packOp = dyn_cast<linalg::PackOp>(op)) {
        std::optional<SizesAndScalableFlags> maybeInputVectorSizes =
            getVectorInputSizesFromUnpackedDomain(packOp, *vectorSizes,
                                                  scalableFlags);
        if (maybeInputVectorSizes) {
          std::tie(vectorSizes, scalableFlags) = maybeInputVectorSizes.value();
        } else {
          LDBG() << "Failed to get input vector sizes for pack op";
          return std::nullopt;
        }
      }
      // Zero vector sizes are invalid (VectorType requires positive dims).
      // Bail out to the IR inference path which derives correct sizes from
      // the tensor shapes after tiling.
      if (llvm::is_contained(*vectorSizes, 0)) {
        LDBG() << "Vector sizes contain zeros, fall back to inference";
      } else {
        return std::make_pair(*vectorSizes, scalableFlags);
      }
    }
    LDBG() << "Failed to get configured vector sizes, fall back to inference";
  }

  // Try to get vector sizes from materialized tile size attribute.
  if (auto tileSizesAttr =
          op->getAttrOfType<DenseI64ArrayAttr>(kVectorTileSizesAttrName)) {
    LDBG() << "Use vector sizes from materialized tile size attribute";
    SmallVector<int64_t> vectorSizes(tileSizesAttr.asArrayRef());
    SmallVector<bool> scalableFlags(vectorSizes.size(), false);
    return std::make_pair(vectorSizes, scalableFlags);
  }

  // Try to infer the vector sizes from the IR.
  std::optional<SmallVector<int64_t>> vectorSizes;
  SmallVector<bool> scalableFlags;
  TypeSwitch<Operation *, void>(op)
      .Case([&](linalg::LinalgOp linalgOp) {
        std::optional<VectorizationTileSizes> result =
            inferSizesFromIR(linalgOp, /*opResult=*/std::nullopt);
        if (result) {
          vectorSizes = result->vectorSizes;
          scalableFlags = result->vectorScalableFlags;
        }
      })
      .Case<linalg::PackOp, linalg::UnPackOp>([&](auto op) {
        std::optional<VectorizationTileSizes> result = inferSizesFromIR(op);
        if (result) {
          vectorSizes = result->vectorSizes;
        }
      })
      .Case([&](tensor::PadOp padOp) {
        auto ty = padOp.getResultType();
        // TODO(hanchung): Infer the vector sizes for pad op after
        // maskedVectorize method allows dynamic result shapes.
        if (!ty.hasStaticShape()) {
          return;
        }
        vectorSizes = SmallVector<int64_t>(ty.getShape());
      })
      .Case([&](IREE::LinalgExt::GatherOp gatherOp) {
        std::optional<VectorizationTileSizes> result =
            inferSizesFromIR(gatherOp.getOutput());
        if (result) {
          vectorSizes = result->vectorSizes;
        }
      })
      .Case([&](IREE::LinalgExt::ArgCompareOp argCompareOp) {
        // Infer from the input operand because it contains the full iteration
        // space, including the reduction dimension, for vectorization.
        std::optional<VectorizationTileSizes> result =
            inferSizesFromIR(argCompareOp.getInputValue());
        if (result) {
          vectorSizes = result->vectorSizes;
        }
      })
      .Case([&](IREE::LinalgExt::Im2colOp im2colOp) {
        OpBuilder builder(op);
        vectorSizes =
            IREE::LinalgExt::computeIm2colVectorTileSizes(builder, im2colOp);
      })
      .Default([&](Operation *) {});

  if (vectorSizes) {
    scalableFlags.resize(vectorSizes->size(), false);
    return std::make_pair(vectorSizes.value(), scalableFlags);
  }
  return std::nullopt;
}

static LogicalResult isWithinVectorSizeLimit(linalg::LinalgOp linalgOp,
                                             int64_t maxVectorSize) {
  int64_t maxFlatVecSize = 1;
  for (OpOperand &operand : linalgOp->getOpOperands()) {
    auto type = dyn_cast<ShapedType>(operand.get().getType());
    if (!type) {
      continue;
    }
    if (!type.hasStaticShape()) {
      return failure();
    }
    maxFlatVecSize = std::max(maxFlatVecSize, type.getNumElements());
  }
  return success(maxFlatVecSize < maxVectorSize);
}

class GenericVectorizationPass final
    : public impl::GenericVectorizationPassBase<GenericVectorizationPass> {
public:
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<tensor::TensorDialect, linalg::LinalgDialect,
                vector::VectorDialect, IREE::VectorExt::IREEVectorExtDialect>();
  }
  void runOnOperation() override;
};

void GenericVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  IRRewriter rewriter(context);

  // Build DictionaryAttr options from pass options. These are forwarded to
  // upstream linalg::vectorize().
  SmallVector<NamedAttribute, 2> linalgOptionsList;
  linalgOptionsList.push_back(
      rewriter.getNamedAttr("vectorizeNDExtract", rewriter.getBoolAttr(true)));
  if (vectorizeToTransferGather) {
    linalgOptionsList.push_back(rewriter.getNamedAttr(
        "vectorizeToTransferGather", rewriter.getBoolAttr(true)));
  }
  auto linalgOptions = DictionaryAttr::get(context, linalgOptionsList);

  SmallVector<VectorizableOpInterface> candidates;
  funcOp.walk([&](VectorizableOpInterface op) {
    Operation *operation = op.getOperation();
    // Filter out CopyOp based on pass option.
    if (isa<linalg::CopyOp>(operation) && !vectorizeCopies) {
      return;
    }
    // Filter out PadOp/PackOp/UnPackOp when masking is disabled.
    // TODO(hanchung): Enable the vectorization without masking. This is mostly
    // legacy code because it used to not working without masking.
    if (!enableVectorMasking &&
        isa<tensor::PadOp, linalg::PackOp, linalg::UnPackOp>(operation)) {
      return;
    }
    if (!vectorizeMapStore && isa<IREE::LinalgExt::MapStoreOp>(operation)) {
      return;
    }
    if (!vectorizeArgCompare && isa<IREE::LinalgExt::ArgCompareOp>(operation)) {
      return;
    }
    candidates.push_back(op);
  });

  // The vector input sizes inference needs to use producers, so we apply
  // vectorization from bottom to top.
  std::reverse(candidates.begin(), candidates.end());
  for (VectorizableOpInterface vectorizableOp : candidates) {
    Operation *op = vectorizableOp.getOperation();
    SmallVector<int64_t> vectorSizes;
    SmallVector<bool> scalableVecDims;
    if (enableVectorMasking) {
      std::optional<SizesAndScalableFlags> vectorSizesAndScalableDims =
          getVectorSizes(op, useConfiguredVectorSizes);
      if (vectorSizesAndScalableDims) {
        std::tie(vectorSizes, scalableVecDims) = *vectorSizesAndScalableDims;
      }
    }

    // Driver-level vector size limit check for linalg ops.
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // Do not vectorize the op if the vector size is greater than or equal
      // to limit.
      if (enableVectorMasking) {
        if (llvm::product_of(vectorSizes) >= maxVectorSize) {
          continue;
        }
      } else {
        if (failed(isWithinVectorSizeLimit(linalgOp, maxVectorSize))) {
          continue;
        }
      }
    }
    // Pad scalable dims with `false` to match the vector sizes.
    scalableVecDims.resize(vectorSizes.size());

    if (!vectorizableOp.isVectorizable(vectorSizes, scalableVecDims,
                                       linalgOptions)) {
      continue;
    }

    FailureOr<SmallVector<Value>> result = vectorizableOp.vectorize(
        rewriter, vectorSizes, scalableVecDims, linalgOptions);
    if (failed(result)) {
      LDBG() << "vectorize() failed after isVectorizable() returned true: "
             << *op;
      continue;
    }
    rewriter.replaceOp(op, *result);
  }

  {
    // Eliminate (all-true) vector masks as early as possible (to avoid missing
    // optimizations/folds). This is particularly beneficial for scalable
    // vectors that use dynamic tensor shapes.
    auto targetAttr =
        iree_compiler::IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    auto vscaleRange = iree_compiler::getDefaultVscaleRange(targetAttr);
    vector::eliminateVectorMasks(rewriter, funcOp, vscaleRange);
  }

  {
    // Canonicalize mask related ops before we lower them.
    RewritePatternSet maskCanonPatterns(funcOp.getContext());
    memref::populateResolveRankedShapedTypeResultDimsPatterns(
        maskCanonPatterns);
    tensor::DimOp::getCanonicalizationPatterns(maskCanonPatterns, context);
    vector::CreateMaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                      funcOp.getContext());
    vector::ConstantMaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                        funcOp.getContext());
    vector::MaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                funcOp.getContext());
    if (failed(applyPatternsGreedily(funcOp, std::move(maskCanonPatterns)))) {
      return signalPassFailure();
    }
  }

  // TODO: Move this down the pipeline once we have the ODM-based masking
  // representation.
  RewritePatternSet vectorizationPatterns(funcOp.getContext());
  if (generateContract) {
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorizationPatterns);
    vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
    vector::populateSinkVectorOpsPatterns(vectorizationPatterns);
  }
  if (foldCastIntoContract) {
    vector::populateFoldArithExtensionPatterns(vectorizationPatterns);
  }
  if (enableVectorMasking) {
    vector::populateVectorMaskLoweringPatternsForSideEffectingOps(
        vectorizationPatterns);
    vectorizationPatterns.add<linalg::LinalgCopyVTRForwardingPattern,
                              linalg::LinalgCopyVTWForwardingPattern>(
        funcOp.getContext(), /*benefit=*/2);
  }

  if (enableCleanup) {
    vector::TransferReadOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                        funcOp.getContext());
    vector::TransferWriteOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                         funcOp.getContext());
  }
  (void)applyPatternsGreedily(funcOp, std::move(vectorizationPatterns));
}

} // namespace
} // namespace mlir::iree_compiler
