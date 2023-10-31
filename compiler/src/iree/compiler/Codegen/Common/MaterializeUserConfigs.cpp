// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/UserConfig.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-materialize-library-calls"

namespace mlir {
namespace iree_compiler {

llvm::cl::opt<std::string> clCodegenTransformDialectTestName(
    "iree-codegen-use-transform-dialect-strategy",
    llvm::cl::desc(
        "Broadcasts the given transform dialect strategy specification to all"
        "dispatches. Supports two modes; a path to the MLIR file containing a"
        "transform dialect specification to apply, and a symbol reference to"
        "load from a library of transform specs (@library_call)"),
    llvm::cl::init(""));

llvm::cl::opt<std::string> clCodegenTransformDialectLibraryFileName(
    "iree-codegen-transform-dialect-library",
    llvm::cl::desc(
        "File path to a module containing a library of transform dialect"
        "strategies"),
    llvm::cl::init(""));

namespace {

static const char kTranslationInfoAttrName[] = "translation_info";

/// Sets compilation configuration annotated in the incoming IR.
LogicalResult
setUserConfig(func::FuncOp entryPointFn, Operation *computeOp,
              IREE::Codegen::CompilationInfoAttr compilationInfo) {
  if (auto translationInfo = getTranslationInfo(entryPointFn)) {
    return computeOp->emitOpError(
        "multiple ops within dispatch trying to set the translation "
        "info");
  }

  auto info = compilationInfo.getTranslationInfo();
  if (failed(setTranslationInfo(entryPointFn, info)))
    return failure();

  SmallVector<int64_t> workgroupSize = compilationInfo.getWorkgroupSizeVals();
  std::optional<int64_t> subgroupSize = compilationInfo.getSubgroupSize();
  if (failed(setDispatchConfig(entryPointFn, workgroupSize, subgroupSize))) {
    return failure();
  }

  setLoweringConfig(computeOp, compilationInfo.getLoweringConfig());
  eraseCompilationInfo(computeOp);
  return success();
}

static void createEmptyTransformStrategy(ModuleOp innerModule) {
  Location loc = innerModule.getLoc();
  OpBuilder b = OpBuilder::atBlockEnd(innerModule.getBody());
  auto topLevelTransformModule = b.create<ModuleOp>(loc);
  Region &topLevelTransformRegion = topLevelTransformModule.getBodyRegion();
  b.setInsertionPointToStart(&topLevelTransformRegion.front());
  auto anyOpType = transform::AnyOpType::get(b.getContext());

  // Create the include for the named sequence with the expectation that the
  // external definition will be linked in later.
  auto sequence = b.create<transform::SequenceOp>(
      loc, TypeRange{}, transform::FailurePropagationMode::Propagate, anyOpType,
      [&](OpBuilder &b, Location loc, Value variantH) {
        b.create<transform::PrintOp>(loc, variantH);
        b.create<transform::YieldOp>(loc);
      });
  (void)sequence;
}

struct MaterializeUserConfigsPass
    : public MaterializeUserConfigsBase<MaterializeUserConfigsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registerTransformDialectTranslationDependentDialects(registry);
  }

  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variantOp = getOperation();
    ModuleOp moduleOp = variantOp.getInnerModule();
    llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
        getAllEntryPoints(moduleOp);
    MLIRContext *context = moduleOp.getContext();

    std::optional<ModuleOp> transformLibrary = std::nullopt;
    if (!clCodegenTransformDialectLibraryFileName.empty()) {
      auto dialect =
          context->getOrLoadDialect<IREE::Codegen::IREECodegenDialect>();
      auto maybeTransformLibrary = dialect->getOrLoadTransformLibraryModule(
          clCodegenTransformDialectLibraryFileName);
      if (failed(maybeTransformLibrary)) {
        return signalPassFailure();
      }
      transformLibrary = *maybeTransformLibrary;
    }

    IREE::Codegen::DispatchLoweringPassPipeline tdPipeline =
        IREE::Codegen::DispatchLoweringPassPipeline::TransformDialectCodegen;
    std::optional<IREE::Codegen::TranslationInfoAttr> clTranslationInfo;
    // Here we always set the pipeline strategy to transform dialect if the
    // flag is non-empty to ensure we pick the right lowering pipeline in the
    // event a file path is given.
    if (!clCodegenTransformDialectTestName.empty()) {
      clTranslationInfo = IREE::Codegen::TranslationInfoAttr::get(
          context, tdPipeline,
          /*softwarePipelineDepth=*/0,
          /*softwarePipelineStoreStage=*/1,
          /*codegenSpec=*/clCodegenTransformDialectTestName[0] == '@'
              ? SymbolRefAttr::get(
                    context, llvm::StringRef(
                                 clCodegenTransformDialectTestName.substr(1)))
              : SymbolRefAttr());
    }

    std::optional<IREE::Codegen::TranslationInfoAttr> translationInfo;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      auto exportOp = exportOps.lookup(funcOp.getName());
      if (!exportOp) {
        continue;
      }

      /// First, apply all user configs.
      auto res = funcOp.walk([&](Operation *op) {
        if (auto compilationInfo = getCompilationInfo(op)) {
          if (failed(setUserConfig(funcOp, op, compilationInfo))) {
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });

      if (res.wasInterrupted()) {
        moduleOp.emitOpError("error in setting user configuration");
        return signalPassFailure();
      }

      /// Let user configs take priority over the global strategy flag.
      if (IREE::Codegen::TranslationInfoAttr exportedTranslationInfo =
              getTranslationInfo(exportOp)) {
        if (translationInfo) {
          /// Currently codegen is rooted on the variant, meaning every entry
          /// must go through the same codegen pipeline. For multi-targeting we
          /// will want to have multiple functions per variant, as well as
          /// multple exports per variant, meaning eventually the nesting of
          /// the translation pipeline will need to change to the function, or
          /// we'll need another level of module op nesting.
          if (exportedTranslationInfo != translationInfo.value()) {
            moduleOp.emitOpError(
                "unhandled compilation of entry point functions with different "
                "translation info");
            return signalPassFailure();
          }
        } else {
          translationInfo = exportedTranslationInfo;
        }
      } else {
        if (translationInfo && translationInfo != clTranslationInfo) {
          moduleOp.emitOpError(
              "unhandled compilation of entry point functions with translation "
              "info optionality");
          return signalPassFailure();
        }
        if (clTranslationInfo) {
          translationInfo = clTranslationInfo;
          if (failed(setTranslationInfo(funcOp, translationInfo.value()))) {
            moduleOp.emitOpError("failed to set command line translation info");
            return signalPassFailure();
          }
        }
      }
    }

    /// We only need to resolve symbols for transform dialect based strategies.
    if (!translationInfo ||
        translationInfo.value().getDispatchLoweringPassPipeline() !=
            tdPipeline) {
      return;
    }

    std::optional<SymbolRefAttr> libraryFunc =
        translationInfo.value().getCodegenSpec();
    if (!libraryFunc || *libraryFunc == SymbolRefAttr()) {
      return;
    }

    /// If we have a symbol, verify the existence of the symbol within the
    /// transform library.
    if (!transformLibrary || !(*transformLibrary) ||
        !transform::detail::findTransformEntryPoint(
            variantOp, *transformLibrary, libraryFunc->getLeafReference())) {
      moduleOp.emitOpError("failed to find transform strategy symbol");
      return signalPassFailure();
    }

    // TODO: At this point we could allow the user to (optionally) return a
    // translation info attribute to use, however there currently isn't a way
    // upstream to retrieve the results of the named sequence.

    /// Attempt to execute the strategy.  symbol (from the flag or otherwise) at
    /// the same time. Because the strategy is rooted on the variant op, the
    /// strategy can change the translation info on the exports if needed, else
    /// back to default IREE codegen.
    StringRef entryPoint = libraryFunc->getLeafReference();
    Operation *transformRoot = transform::detail::findTransformEntryPoint(
        variantOp, *transformLibrary, entryPoint);
    if (!transformRoot) {
      return;
    }
    if (failed(transform::applyTransformNamedSequence(
            variantOp, transformRoot, *transformLibrary,
            options.enableExpensiveChecks(true)))) {
      return signalPassFailure();
    }

    // Re-retrieve the export ops and mark all exports with unchanged
    // translation info as un-translated.
    // TODO: Currently this is the only way to "fall back" to codegen. If the
    // user wants to do all of codegen themselves they can set a `None`
    // pipeline.
    exportOps = getAllEntryPoints(variantOp.getInnerModule());
    for (auto &it : exportOps) {
      auto exportOp = it.second;
      if (getTranslationInfo(exportOp) == translationInfo) {
        exportOp->removeAttr(kTranslationInfoAttrName);
      }
    }
  }

private:
  /// Transform interpreter options.
  transform::TransformOptions options;
};

} // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createMaterializeUserConfigsPass() {
  return std::make_unique<MaterializeUserConfigsPass>();
}

} // namespace iree_compiler
} // namespace mlir
