// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/Transforms/Transforms.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-materialize-encoding-into-padding"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_MATERIALIZEENCODINGINTOPADDINGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

using namespace IREE::Encoding;

namespace {
struct MaterializeEncodingIntoPaddingPass final
    : impl::MaterializeEncodingIntoPaddingPassBase<
          MaterializeEncodingIntoPaddingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect, IREE::Codegen::IREECodegenDialect,
                    IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface operation = getOperation();

    // Retrieve the config from executable target attribute, if any. Otherwise,
    // retrieve the config from CLI GPU target and construct a virtual
    // configuration.
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(operation);
    DictionaryAttr targetConfig;
    if (targetAttr) {
      targetConfig = targetAttr.getConfiguration();
    } else {
      IREE::GPU::TargetAttr gpuTargetAttr = getCLGPUTarget(context);
      SmallVector<NamedAttribute> items;
      items.emplace_back(
          IREE::Encoding::kEncodingResolverAttrName,
          IREE::GPU::getHIPTargetEncodingLayoutAttr(gpuTargetAttr, "pad"));
      targetConfig = DictionaryAttr::get(context, items);
    }

    // The layoutAttr should come in without any target info attached to it,
    // so we need to clone the layout attrs with the configuration so it can
    // access the target info during materialization.
    //
    // Otherwise, fall back to the nop layout.
    IREE::Encoding::LayoutMaterializerAttr layoutAttr;
    if (targetConfig &&
        targetConfig.contains(IREE::Encoding::kEncodingResolverAttrName)) {
      layoutAttr = targetConfig.getAs<IREE::Encoding::LayoutMaterializerAttr>(
          IREE::Encoding::kEncodingResolverAttrName);
      auto resolverAttr = cast<IREE::Encoding::LayoutResolverAttr>(layoutAttr);
      layoutAttr = cast<IREE::Encoding::LayoutMaterializerAttr>(
          resolverAttr.cloneWithSimplifiedConfig(targetConfig));
    } else {
      layoutAttr = cast<IREE::Encoding::LayoutMaterializerAttr>(
          IREE::Encoding::IdentityResolverAttr::get(context));
    }

    RewritePatternSet materializeEncodingPattern(context);
    MaterializeEncodingTypeConverter typeConverter(layoutAttr);
    MaterializeEncodingConversionTarget target(*context);
    populateMaterializeEncodingPatterns(materializeEncodingPattern, target,
                                        typeConverter);
    if (failed(applyPartialConversion(operation, target,
                                      std::move(materializeEncodingPattern)))) {
      operation.emitOpError("materialization failed");
      return signalPassFailure();
    }

    // Add patterns to resolve dims ops and cleanups.
    {
      RewritePatternSet patterns(context);
      memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
      context->getOrLoadDialect<tensor::TensorDialect>()
          ->getCanonicalizationPatterns(patterns);
      // TODO: Drop these when we deprecate partial loads/stores.
      IREE::TensorExt::populateTensorSliceOpWithDispatchTensorOpFoldingPatterns(
          patterns, context);
      if (failed(applyPatternsGreedily(operation, std::move(patterns)))) {
        operation.emitOpError("folding patterns failed");
        return signalPassFailure();
      }
    }
  }
};
} // namespace

void addEncodingToPaddingPasses(FunctionLikeNest &passManager) {
  passManager.addPass(createMaterializeEncodingIntoPaddingPass)
      .addPass(createBufferizeCopyOnlyDispatchesPass)
      .addPass(createCanonicalizerPass);
}

} // namespace mlir::iree_compiler
