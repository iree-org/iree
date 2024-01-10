// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iterator>
#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/UserConfig.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-materialize-user-configs"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

llvm::cl::opt<std::string> clCodegenTransformDialectStrategyName(
    "iree-codegen-use-transform-dialect-strategy",
    llvm::cl::desc(
        "Broadcasts the given transform dialect strategy specification to all "
        "dispatches. The specification is a symbol reference to load from a"
        "library of transform specs (@library_call)"),
    llvm::cl::init(""));

llvm::cl::opt<std::string> clCodegenTransformDialectLibraryFileName(
    "iree-codegen-transform-dialect-library",
    llvm::cl::desc(
        "File path to a module containing a library of transform dialect"
        "strategies. Can be suffixed with the name of a transform sequence"
        "within the library to run as preprocessing per executable variant."
        "This is specified as <file-path>::<sequence-name>. If not specified,"
        "this will default to `__kernel_config`."),
    llvm::cl::init(""));

namespace {

static const char kTranslationInfoAttrName[] = "translation_info";

enum StrategyRunResult {
  Success = 0,
  NotFound = 1,
  Failed = 2,
};

static StrategyRunResult
runTransformConfigurationStrategy(Operation *payloadRoot,
                                  StringRef entryPointName,
                                  ModuleOp &transformLibrary) {
  /// If we have a symbol, verify the existence of the symbol within the
  /// transform library.
  Operation *entryPoint = transform::detail::findTransformEntryPoint(
      payloadRoot, transformLibrary, entryPointName);
  if (!entryPoint) {
    return StrategyRunResult::NotFound;
  }

  transform::TransformOptions options;
  if (failed(transform::applyTransformNamedSequence(
          payloadRoot, entryPoint, transformLibrary,
          options.enableExpensiveChecks(true)))) {
    return StrategyRunResult::Failed;
  }
  return StrategyRunResult::Success;
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

    // Parse the file path and kernel config strategy from flags. There are
    // three possible usage flows:
    //   1. Specify only a transform dialect library, e.g.
    //      --iree-codegen-transform-dialect-library=/path/to/library.mlir
    //      This will load the transform library and attempt to use two default
    //      strategies, `__kernel_config` before strategy configuration (i.e.
    //      now), and `__transform_main` for codegen. If `__kernel_config` is
    //      present in the library, it will run it and use the annotations set
    //      by it. If not present, `__transform_main` will be broadcasted to all
    //      dispatches.
    //
    //   2. Specify a library path with a strategy name instead of
    //      `__transform_main`. This is the same as (1) except it uses a
    //      different default strategy.
    //
    //   3. Specify a library path suffixed with a kernel config entry point.
    //      This will throw an error if the specified kernel config strategy is
    //      not found.
    SmallVector<StringRef, 2> parts;
    llvm::SplitString(llvm::StringRef(clCodegenTransformDialectLibraryFileName),
                      parts, "::");
    if (parts.size() > 2) {
      variantOp.emitError()
          << "Invalid transform library path and sequence name "
          << clCodegenTransformDialectLibraryFileName;
      return signalPassFailure();
    }
    bool hasTransformConfig = parts.size() == 2;
    bool hasTransformLibrary = parts.size() >= 1;
    bool hasTransformStrategy = !clCodegenTransformDialectStrategyName.empty();

    std::string libraryFileName;
    if (hasTransformConfig) {
      if (parts[0].empty()) {
        variantOp.emitError() << "Cannot specify an empty library path";
        return signalPassFailure();
      }
      libraryFileName = parts[0];
    }

    std::string entrySequenceName;
    if (hasTransformConfig) {
      if (parts[1].empty()) {
        variantOp.emitError() << "Cannot specify an empty sequence name";
        return signalPassFailure();
      }
      entrySequenceName = parts[1];
    }

    LDBG("MaterializeUserConfigsPass on variant: " << variantOp);
    std::optional<ModuleOp> transformLibrary = std::nullopt;
    if (hasTransformLibrary) {
      auto dialect =
          context->getOrLoadDialect<IREE::Codegen::IREECodegenDialect>();
      auto maybeTransformLibrary =
          dialect->getOrLoadTransformLibraryModule(libraryFileName);
      if (failed(maybeTransformLibrary)) {
        variantOp.emitError()
            << "failed to load transform library module: " << libraryFileName;
        return signalPassFailure();
      }
      transformLibrary = *maybeTransformLibrary;
      LDBG("--found transform library @" << libraryFileName);
    }

    // Run the user specified transform configuration, or attempt to run
    // `__kernel_config` if no sequence name is specified.
    if (hasTransformConfig) {
      // If we get here, the transform library must necessarily have been
      // loaded.
      assert(transformLibrary && *transformLibrary &&
             "Unexpected unloaded transform library");
      if (runTransformConfigurationStrategy(variantOp, entrySequenceName,
                                            *transformLibrary) !=
          StrategyRunResult::Success) {
        variantOp.emitError() << "transform kernel config strategy `"
                              << entrySequenceName << "` failed to apply";
        return signalPassFailure();
      }
    } else if (transformLibrary && (*transformLibrary)) {
      StrategyRunResult res = runTransformConfigurationStrategy(
          variantOp, "__kernel_config", *transformLibrary);
      if (res == StrategyRunResult::Failed) {
        variantOp.emitError()
            << "default transform __kernel_config strategy failed to apply";
        return signalPassFailure();
      }
      // If `__kernel_config` was found and ran successfully, indicate that
      // there was a config strategy.
      if (res == StrategyRunResult::Success) {
        hasTransformConfig = true;
      }
    }

    IREE::Codegen::DispatchLoweringPassPipeline tdPipeline =
        IREE::Codegen::DispatchLoweringPassPipeline::TransformDialectCodegen;
    std::optional<IREE::Codegen::TranslationInfoAttr> clTranslationInfo;
    // Here we always set the pipeline strategy to transform dialect if the
    // flag is non-empty to ensure we pick the right lowering pipeline in the
    // event a strategy symbol is defined.
    if ((hasTransformLibrary && !hasTransformConfig) || hasTransformStrategy) {
      StringRef strategyName =
          (!hasTransformStrategy)
              ? StringRef(
                    transform::TransformDialect::kTransformEntryPointSymbolName)
              : clCodegenTransformDialectStrategyName;
      clTranslationInfo = IREE::Codegen::TranslationInfoAttr::get(
          context, tdPipeline,
          /*softwarePipelineDepth=*/0,
          /*softwarePipelineStoreStage=*/1,
          /*codegenSpec=*/
          SymbolRefAttr::get(context, llvm::StringRef(strategyName)));
      LDBG("--clTranslationInfo: " << clTranslationInfo);
    }

    LDBG("--start iterating over: "
         << std::distance(moduleOp.getOps<func::FuncOp>().begin(),
                          moduleOp.getOps<func::FuncOp>().end())
         << " functions");
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
          /// multiple exports per variant, meaning eventually the nesting of
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

    LDBG("--guaranteed unique translationInfo: " << translationInfo);
    /// We only need to resolve symbols for transform dialect based strategies.
    if (!translationInfo ||
        translationInfo.value().getDispatchLoweringPassPipeline() !=
            tdPipeline) {
      return;
    }

    // From now on, we know we have a transform dialect strategy. We now need to
    // ensure it can resolve and apply in a subsequent interpreter pass or else
    // we need to fall back to codegen.
    bool failedToResolve = false;
    auto g = llvm::make_scope_exit([&]() {
      if (!failedToResolve)
        return;

      exportOps = getAllEntryPoints(variantOp.getInnerModule());
      for (auto &it : exportOps) {
        auto exportOp = it.second;
        if (getTranslationInfo(exportOp) == translationInfo) {
          exportOp->removeAttr(kTranslationInfoAttrName);
        }
      }
    });

    std::optional<SymbolRefAttr> strategyName =
        translationInfo.value().getCodegenSpec();
    if (!strategyName || *strategyName == SymbolRefAttr()) {
      failedToResolve = true;
      return;
    }

    /// If we have a symbol, verify the existence of the symbol within the
    /// transform library.
    StringRef entryPoint = strategyName->getLeafReference();
    if (!transformLibrary || !(*transformLibrary) ||
        !transform::detail::findTransformEntryPoint(
            variantOp, *transformLibrary, entryPoint)) {
      moduleOp.emitOpError("failed to find transform strategy symbol");
      failedToResolve = true;
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

} // namespace mlir::iree_compiler
