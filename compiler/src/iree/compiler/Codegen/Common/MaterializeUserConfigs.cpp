// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/UserConfig.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#define DEBUG_TYPE "iree-codegen-materialize-user-configs"

namespace mlir::iree_compiler {

llvm::cl::opt<std::string> clCodegenTransformDialectLibraryFileName(
    "iree-codegen-transform-dialect-library",
    llvm::cl::desc(
        "File path to a module containing a library of transform dialect"
        "strategies. Can be suffixed with the name of a transform sequence"
        "within the library to run as preprocessing per executable variant."
        "This is specified as <file-path>@<sequence-name>. If not specified,"
        "this will default to `__kernel_config`."),
    llvm::cl::init(""));

llvm::cl::opt<bool> clCodegenNotifyTransformDialectLibraryApplication(
    "iree-codegen-test-notify-transform-strategy-application",
    llvm::cl::desc(
        "Emit a remark when a transform configuration strategy successfully "
        "applies on a function. This is intended for testing/debuging."),
    llvm::cl::Hidden, llvm::cl::init(false));

#define GEN_PASS_DEF_MATERIALIZEUSERCONFIGSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

constexpr StringLiteral kTranslationInfoAttrName =
    IREE::Codegen::TranslationInfoAttr::name;

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

struct TransformLibraryWithEntrypoint {
  ModuleOp transformLibrary;
  std::string entrypointName;
};

static FailureOr<TransformLibraryWithEntrypoint>
getTransformLibraryFromPath(ModuleOp compiledModule, StringRef path) {
  SmallVector<StringRef, 2> parts;
  llvm::SplitString(path, parts, "@");
  if (parts.empty()) {
    return failure();
  }
  if (parts.size() > 2) {
    return compiledModule.emitError()
           << "Invalid transform library path and sequence name " << path;
  }
  StringRef libraryFileName = parts[0];
  StringRef entrySequenceName = kKernelConfigSpecName;
  if (parts.size() == 2) {
    entrySequenceName = parts[1];
  }

  // Validate both the file name and the spec name.
  if (libraryFileName.empty()) {
    return compiledModule.emitError() << "Cannot specify an empty library path";
  }
  if (entrySequenceName.empty()) {
    return compiledModule.emitError()
           << "Cannot specify an empty sequence name";
  }

  MLIRContext *ctx = compiledModule->getContext();
  auto dialect = ctx->getOrLoadDialect<IREE::Codegen::IREECodegenDialect>();
  auto maybeTransformLibrary =
      dialect->getOrLoadTransformLibraryModule(libraryFileName.str());
  if (failed(maybeTransformLibrary)) {
    return compiledModule.emitError()
           << "Failed to load transform library module: " << libraryFileName;
  }
  LDBG() << "--found transform library " << libraryFileName << "@"
         << entrySequenceName;
  return TransformLibraryWithEntrypoint{*maybeTransformLibrary,
                                        entrySequenceName.str()};
}

/// Look up the tuning spec in the given module or any of its parents.
static LogicalResult getModuleTuningSpec(ModuleOp compiledModule,
                                         OwningOpRef<ModuleOp> &tuningSpec) {
  IREE::Util::SerializableAttrInterface serializedTuningSpec;
  Operation *op = compiledModule;
  while (!serializedTuningSpec && op) {
    serializedTuningSpec =
        op->getAttrOfType<IREE::Util::SerializableAttrInterface>(
            kSerializedTuningSpecAttrName);
    op = op->getParentOp();
  }

  if (!serializedTuningSpec) {
    return failure();
  }

  SmallVector<char, 0> bytecode;
  if (failed(serializedTuningSpec.serializeToVector(
          compiledModule->getLoc(), llvm::endianness::native, bytecode))) {
    return compiledModule.emitError()
           << "Failed to read attribute " << kSerializedTuningSpecAttrName;
  }

  ParserConfig config(compiledModule.getContext());
  tuningSpec = parseSourceString<ModuleOp>(
      StringRef(bytecode.data(), bytecode.size()), config);
  if (!tuningSpec) {
    return compiledModule.emitError() << "Failed to parse tuning spec in "
                                      << kSerializedTuningSpecAttrName;
  }
  LDBG() << "--loaded tuning spec";
  return success();
}

struct MaterializeUserConfigsPass final
    : impl::MaterializeUserConfigsPassBase<MaterializeUserConfigsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registerTransformDialectTranslationDependentDialects(registry);
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Try to load the transform library from the user flag first. If none is
    // specified, fall back to using the module tuning spec.
    FailureOr<TransformLibraryWithEntrypoint> userTransformLibrary =
        getTransformLibraryFromPath(moduleOp,
                                    clCodegenTransformDialectLibraryFileName);
    OwningOpRef<ModuleOp> tuningSpec;
    if (failed(userTransformLibrary)) {
      if (succeeded(getModuleTuningSpec(moduleOp, tuningSpec))) {
        assert(tuningSpec);
        userTransformLibrary = TransformLibraryWithEntrypoint{
            tuningSpec.get(), kKernelConfigSpecName.str()};
      }
    }

    // Remove the tuning spec, if any, from the current module. If the tuning
    // spec is attached to some other parent op, we conservatively keep it
    // as-is, as we are not sure who the producer is and if they want it
    // removed.
    if (moduleOp->hasAttr(kSerializedTuningSpecAttrName)) {
      moduleOp->removeAttr(kSerializedTuningSpecAttrName);
      LDBG() << "--dropped the serialized tuning spec from the module";
    }

    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {

      // Parse the file path and kernel config strategy from flags. There are
      // two possible usage flows for transform dialect libraries.
      //   1. Use `__kernel_config` to match and annotate variants with the
      //      strategy to use. This could either be a transform dialect strategy
      //      or any other IREE codegen pipeline.
      //
      //   2. Use the configuration strategy to do codegen directly. At the end
      //      of the strategy, the variant needs to be annotated with:
      //      ```mlir
      //      "translation_info" =
      //        #iree_codegen.translation_info<pipeline = None>
      //      ```
      LDBG() << "MaterializeUserConfigsPass on function: " << funcOp;
      if (succeeded(userTransformLibrary)) {
        StringRef libraryModuleName =
            userTransformLibrary->transformLibrary.getSymName().value_or(
                "<unnamed>");
        StringRef entrySequenceName = userTransformLibrary->entrypointName;
        auto runResult = runTransformConfigurationStrategy(
            funcOp, entrySequenceName, userTransformLibrary->transformLibrary);
        if (runResult == StrategyRunResult::NotFound) {
          funcOp.emitError() << "transform kernel config strategy `"
                             << entrySequenceName << " not found";
          return signalPassFailure();
        }
        if (runResult == StrategyRunResult::Failed) {
          funcOp.emitError() << "transform kernel config strategy `"
                             << entrySequenceName << "` failed to apply";
          return signalPassFailure();
        }

        if (clCodegenNotifyTransformDialectLibraryApplication) {
          funcOp->emitRemark()
              << "Applied transform configuration strategy @"
              << libraryModuleName << "::@" << entrySequenceName;
        }
      }

      /// Nothing to do if the export already has a config.
      IREE::Codegen::TranslationInfoAttr translationInfo =
          getTranslationInfo(funcOp);
      if (translationInfo) {
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
        funcOp.emitOpError("error in setting user configuration");
        return signalPassFailure();
      }

      translationInfo = getTranslationInfo(funcOp);
      LDBG() << "--guaranteed unique translationInfo: " << translationInfo;
      /// We only need to resolve symbols for transform dialect based
      /// strategies.
      if (!translationInfo ||
          translationInfo.getDispatchLoweringPassPipeline() !=
              IREE::Codegen::DispatchLoweringPassPipeline::
                  TransformDialectCodegen) {
        continue;
      }

      std::optional<SymbolRefAttr> strategyName =
          translationInfo.getCodegenSpec();
      if (!strategyName || *strategyName == SymbolRefAttr()) {
        continue;
      }

      /// If we have a symbol, verify the existence of the symbol within the
      /// transform library.
      StringRef entryPoint = strategyName->getLeafReference();
      if (failed(userTransformLibrary) ||
          !transform::detail::findTransformEntryPoint(
              funcOp, userTransformLibrary->transformLibrary, entryPoint)) {
        funcOp.emitOpError("failed to find transform strategy symbol");
      }
    }
  }

private:
  /// Transform interpreter options.
  transform::TransformOptions options;
};

} // namespace
} // namespace mlir::iree_compiler
