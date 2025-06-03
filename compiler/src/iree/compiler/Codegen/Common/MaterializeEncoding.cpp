// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-materialize-encoding"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_MATERIALIZEDEVICEENCODINGPASS
#define GEN_PASS_DEF_MATERIALIZEHOSTENCODINGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

using namespace IREE::Encoding;

namespace {

static LogicalResult
materializeFuncOpEncodings(FunctionOpInterface funcOp,
                           IREE::HAL::ExecutableTargetAttr targetAttr,
                           bool testCLGPUTarget = false) {
  MLIRContext *ctx = funcOp.getContext();
  {
    RewritePatternSet patterns(ctx);
    DictionaryAttr targetConfig =
        targetAttr ? targetAttr.getConfiguration() : nullptr;
    // Check if the encoding resolver is a GPUPadLayoutAttr. For padding
    // encoding materialization, we use a separate pass, so skip materialization
    // here.
    // TODO(#20160): Support GPUPadLayoutAttr materialization through this
    // pass, and remove the ad-hoc materialization pass for padding.
    if (targetConfig && targetConfig.getAs<IREE::GPU::GPUPadLayoutAttr>(
                            IREE::Encoding::kEncodingResolverAttrName)) {
      LDBG("Found GPUPadLayoutAttr encoding resolver. Materialization will "
           "be handled later.");
      return success();
    }

    auto getTestTargetOrNopLayout =
        [&]() -> IREE::Encoding::LayoutMaterializerAttr {
      if (testCLGPUTarget) {
        LDBG("Select GPUEncodingLayoutAttr attribute as the layout attribute. "
             "(testCLGPUTarget)");
        return cast<IREE::Encoding::LayoutMaterializerAttr>(
            IREE::GPU::GPUEncodingLayoutAttr::get(
                ctx,
                DictionaryAttr::get(ctx, NamedAttribute(kGPUTargetAttrName,
                                                        getCLGPUTarget(ctx)))));
      }
      LDBG("Select EncodingNopLayoutAttr attribute as the layout "
           "attribute (Encoding resolver unknown or unsupported).");
      return cast<IREE::Encoding::LayoutMaterializerAttr>(
          IREE::Codegen::EncodingNopLayoutAttr::get(ctx));
    };

    // The layoutAttr should come in without any target info attached to it,
    // so we need to clone the layout attrs with the targetAttr configuration
    // so it can access the target info during materialization.
    //
    // If the layoutAttr was not found, or if it does not implement the layout
    // resolver interface, fall back to the resolver for getCLGPUTarget. If
    // there is also no test target set, fall back to the nop layout.
    IREE::Encoding::LayoutMaterializerAttr layoutAttr =
        targetConfig
            ? targetConfig.getAs<IREE::Encoding::LayoutMaterializerAttr>(
                  IREE::Encoding::kEncodingResolverAttrName)
            : nullptr;
    auto resolverAttr =
        llvm::dyn_cast_or_null<IREE::Encoding::LayoutResolverAttr>(layoutAttr);

    IREE::Encoding::LayoutMaterializerAttr layoutAttrWithTargetInfo =
        layoutAttr && resolverAttr
            ? cast<IREE::Encoding::LayoutMaterializerAttr>(
                  resolverAttr.cloneWithSimplifiedConfig(targetConfig))
            : getTestTargetOrNopLayout();

    LDBG("Selected Encoding::LayoutMaterializerAttr with target configuration: "
         << layoutAttrWithTargetInfo);

    MaterializeEncodingTypeConverter typeConverter(layoutAttrWithTargetInfo);
    MaterializeEncodingConversionTarget target(*ctx);
    populateMaterializeEncodingPatterns(patterns, target, typeConverter);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      funcOp.emitOpError("materialization failed");
      return failure();
    }
  }

  // Run patterns to fold pack/unpack ops with pad/extract_slice ops, resolve
  // dims ops, and eliminate common sub-expressions.
  {
    RewritePatternSet patterns(ctx);
    populateReshapeToInterfaceTensorPatterns(patterns);
    tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    linalg::FillOp::getCanonicalizationPatterns(patterns, ctx);
    linalg::PackOp::getCanonicalizationPatterns(patterns, ctx);
    linalg::UnPackOp::getCanonicalizationPatterns(patterns, ctx);
    linalg::populateFoldIntoPackAndUnpackPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitOpError("folding patterns failed");
      return failure();
    }

    IRRewriter rewriter(ctx);
    DominanceInfo domInfo;
    mlir::eliminateCommonSubExpressions(rewriter, domInfo, funcOp);
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

struct MaterializeHostEncodingPass final
    : impl::MaterializeHostEncodingPassBase<MaterializeHostEncodingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect,
                    IREE::Codegen::IREECodegenDialect,
                    IREE::CPU::IREECPUDialect, IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

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
struct MaterializeDeviceEncodingPass final
    : impl::MaterializeDeviceEncodingPassBase<MaterializeDeviceEncodingPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect,
                    IREE::Codegen::IREECodegenDialect,
                    IREE::CPU::IREECPUDialect, IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    auto executableTargetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (failed(materializeFuncOpEncodings(funcOp, executableTargetAttr,
                                          testCLGPUTarget))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler
