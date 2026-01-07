// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-materialize-encoding"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_MATERIALIZEDEVICEENCODINGPASS
#define GEN_PASS_DEF_MATERIALIZEHOSTENCODINGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static void
updateFuncSignature(FunctionOpInterface funcOp,
                    const MaterializeEncodingTypeConverter &typeConverter) {
  // Do not convert the type if the type converter does not understand the
  // conversion. E.g., `!hal.buffer_view` type.
  auto convertType = [&](Type t) {
    Type newType = typeConverter.convertType(t);
    return newType ? newType : t;
  };
  SmallVector<Type> newInputs =
      llvm::map_to_vector(funcOp.getArgumentTypes(), convertType);
  SmallVector<Type> newResults =
      llvm::map_to_vector(funcOp.getResultTypes(), convertType);
  funcOp.setType(FunctionType::get(funcOp.getContext(), newInputs, newResults));
  for (auto [arg, newType] : llvm::zip(funcOp.getArguments(), newInputs)) {
    arg.setType(newType);
  }
}

static LogicalResult
materializeFuncOpEncodings(FunctionOpInterface funcOp,
                           IREE::HAL::ExecutableTargetAttr targetAttr,
                           detail::TestingResolverKind resolverSource =
                               detail::TestingResolverKind::kNone) {
  MLIRContext *ctx = funcOp.getContext();
  {
    DictionaryAttr targetConfig =
        targetAttr ? targetAttr.getConfiguration() : nullptr;
    auto getTestTargetResolverOrIdentityResolver =
        [&]() -> IREE::Encoding::LayoutMaterializerAttr {
      SmallVector<NamedAttribute> configItems;
      IREE::GPU::TargetAttr targetAttr =
          getGPUTargetAttr(ctx, /*target=*/nullptr);
      if (targetAttr) {
        addConfigGPUTarget(ctx, targetAttr, configItems);
        switch (resolverSource) {
        case detail::TestingResolverKind::kGPUDataTiling: {
          LDBG() << "Select GPUEncodingResolverAttr attribute as the layout "
                    "attribute. (kGPUDataTiling)";
          return cast<IREE::Encoding::LayoutMaterializerAttr>(
              IREE::GPU::GPUEncodingResolverAttr::get(
                  ctx, DictionaryAttr::get(ctx, configItems)));
        }
        case detail::TestingResolverKind::kGPUPadding: {
          LDBG() << "Select GPUPaddingResolverAttr attribute as the layout "
                    "attribute. (kGPUPadding)";
          std::optional<IREE::GPU::L1CacheInfo> cache =
              IREE::GPU::getL1CacheInfo(targetAttr);
          return cast<IREE::Encoding::LayoutMaterializerAttr>(
              IREE::GPU::GPUPaddingResolverAttr::get(ctx, cache->cacheLineBytes,
                                                     cache->cacheSets));
        }
        case detail::TestingResolverKind::kNone:
          break;
        }
      }
      LDBG() << "Select IdentityResolverAttr attribute as the layout "
                "attribute (Encoding resolver unknown or unsupported).";
      return cast<IREE::Encoding::LayoutMaterializerAttr>(
          IREE::Encoding::IdentityResolverAttr::get(ctx));
    };

    // The layoutAttr should come in without any target info attached to it,
    // so we need to clone the layout attrs with the targetAttr configuration
    // so it can access the target info during materialization.
    //
    // If the layoutAttr was not found, or if it does not implement the layout
    // resolver interface, fall back to the resolver for TestingResolverKind
    // resolver. If there is also no test target set or it is kNone, fall back
    // to the identity resolver.
    IREE::Encoding::LayoutMaterializerAttr layoutAttr =
        targetConfig
            ? targetConfig.getAs<IREE::Encoding::LayoutMaterializerAttr>(
                  IREE::Encoding::kEncodingResolverAttrName)
            : nullptr;
    auto resolverAttr =
        dyn_cast_if_present<IREE::Encoding::LayoutResolverAttr>(layoutAttr);

    IREE::Encoding::LayoutMaterializerAttr layoutAttrWithTargetInfo =
        layoutAttr && resolverAttr
            ? cast<IREE::Encoding::LayoutMaterializerAttr>(
                  resolverAttr.cloneWithSimplifiedConfig(targetConfig))
            : getTestTargetResolverOrIdentityResolver();

    LDBG() << "Selected Encoding::LayoutMaterializerAttr with target "
              "configuration: "
           << layoutAttrWithTargetInfo;

    MaterializeEncodingTypeConverter typeConverter(layoutAttrWithTargetInfo);
    MaterializeEncodingConversionTarget target(*ctx);

    // Decompose mismatched encodings in load/store ops before actual encoding
    // materialization.
    // When a load/store op has mismatched source and result encodings that
    // materialize to different layouts, decompose it into a sequence of
    // load/store + unset_encoding + set_encoding ops. This phase uses greedy
    // pattern rewriting to handle these layout transfer cases before the main
    // dialect conversion. It is needed because no-fallback conversion driver
    // requires all the immediate ops to be legal, but the patterns can
    // introduce illegal load/store ops because they still have encodings.
    {
      RewritePatternSet patterns(ctx);
      populateDecomposeMismatchedLayoutLoadStoreOpsPatterns(patterns,
                                                            typeConverter);
      walkAndApplyPatterns(funcOp, std::move(patterns));
    }

    RewritePatternSet patterns(ctx);
    populateMaterializeEncodingPatterns(patterns, target, typeConverter);

    // Replace any unrealized conversions to tensor.cast ops if they come from
    // block arguments. The function signature is updated to match the converted
    // types after the partial conversion. This is used in testing, where
    // function arguments have encodings to reduce the amount of IR, but we do
    // not expect function arguments to have encodings in practice.
    auto castFnArguments = [](OpBuilder &builder, Type resultTy,
                              ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1) {
        return Value();
      }
      Value input = inputs[0];
      if (!isa<BlockArgument>(input) ||
          !isa<RankedTensorType>(input.getType())) {
        return Value();
      }
      return tensor::CastOp::create(builder, loc, resultTy, input);
    };
    typeConverter.addTargetMaterialization(castFnArguments);
    typeConverter.addSourceMaterialization(castFnArguments);

    mlir::ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns),
                                      config))) {
      return funcOp.emitOpError("materialization failed");
    }

    // Update the function signature to match the converted types.
    updateFuncSignature(funcOp, typeConverter);
  }

  // Run patterns to fold pack/unpack ops with pad/extract_slice ops, resolve
  // dims ops, and eliminate common sub-expressions.
  {
    RewritePatternSet patterns(ctx);
    // NOTE: These patterns are currently load-bearing for sub-byte floats.
    populateReshapeToInterfaceTensorPatterns(patterns);
    tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    linalg::FillOp::getCanonicalizationPatterns(patterns, ctx);
    linalg::PackOp::getCanonicalizationPatterns(patterns, ctx);
    linalg::UnPackOp::getCanonicalizationPatterns(patterns, ctx);
    linalg::populateFoldIntoPackAndUnpackPatterns(
        patterns, [](OpOperand *opOperand) {
          Operation *producer = opOperand->get().getDefiningOp();
          Operation *consumer = opOperand->getOwner();
          // If we have a pack/unpack consumer and a producer that has multiple
          // uses, this _probably_ means the producer won't get dce'd. If that
          // is the case, by folding the consumer pack/unpack, we break the
          // producer consumer chain between them and inhibit fusion later in
          // the pipeline.
          if (isa<linalg::PackOp, linalg::UnPackOp>(consumer) &&
              isa_and_nonnull<TilingInterface>(producer) &&
              !producer->hasOneUse())
            return false;
          return true;
        });
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return funcOp.emitOpError("folding patterns failed");
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
    registry.insert<amdgpu::AMDGPUDialect, arith::ArithDialect,
                    tensor::TensorDialect, vector::VectorDialect,
                    IREE::Encoding::IREEEncodingDialect,
                    IREE::Codegen::IREECodegenDialect,
                    IREE::CPU::IREECPUDialect, IREE::GPU::IREEGPUDialect>();
  }

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    auto executableTargetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    if (failed(materializeFuncOpEncodings(funcOp, executableTargetAttr,
                                          testGPUEncodingResolver))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler
