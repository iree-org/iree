// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/EmbeddedDataDirectory.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/FileUtilities.h"

#define DEBUG_TYPE "iree-codegen-materialize-tuning-specs"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_MATERIALIZETUNINGSPECSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

llvm::cl::opt<std::string> clCodegenTuningSpecPath(
    "iree-codegen-tuning-spec-path",
    llvm::cl::desc("File path to a module containing a tuning spec (transform "
                   "dialect library)."),
    llvm::cl::init(""));

llvm::cl::opt<bool> clCodegenEnableDefaultTuningSpecs(
    "iree-codegen-enable-default-tuning-specs",
    llvm::cl::desc("Whether to enable default tuning spec transform libraries "
                   "shipped with the compiler"),
    llvm::cl::init(false));

llvm::cl::opt<std::string> clCodegenTuningSpecDumpDir(
    "iree-codegen-dump-tuning-specs-to",
    llvm::cl::desc(
        "Dump the final tuning spec modules to the specified directory. When "
        "set to '-', prints the tuning spec to stdout."),
    llvm::cl::init(""));

using mlir::transform::NamedSequenceOp;

static LogicalResult dumpFinalTuningSpecToDir(ModuleOp tuningSpec) {
  StringRef dir = clCodegenTuningSpecDumpDir;
  if (dir.empty()) {
    return success();
  }
  if (dir == "-") {
    tuningSpec->print(llvm::outs());
    return success();
  }

  llvm::sys::fs::create_directories(dir);
  llvm::SmallString<64> dumpPath;
  auto dumpFileEC = llvm::sys::fs::createUniqueFile(
      Twine(dir) + "/iree_tuning_spec_%%.mlir", dumpPath);
  if (dumpFileEC) {
    return tuningSpec->emitError()
           << "Failed to create a unique file in " << dir << "\n";
  }
  LDBG("Linked tuning spec file path: " << dumpPath);

  std::string error;
  auto file = mlir::openOutputFile(dumpPath, &error);
  if (!file) {
    return tuningSpec->emitError()
           << "Failed to open a tuning spec dump file " << dumpPath << "\n";
  }

  tuningSpec->print(file->os());
  file->keep();
  return success();
}

static FailureOr<ModuleOp>
getUserTuningSpec(ModuleOp module, IREE::Codegen::IREECodegenDialect &dialect) {
  if (clCodegenTuningSpecPath.empty()) {
    return failure();
  }

  FailureOr<ModuleOp> maybeTransformLibrary =
      dialect.getOrLoadTransformLibraryModule(clCodegenTuningSpecPath);
  if (failed(maybeTransformLibrary)) {
    return module->emitError()
           << "Failed to load tuning spec transform dialect library from "
           << clCodegenTuningSpecPath;
  }

  return *maybeTransformLibrary;
}

static std::optional<StringRef> fetchDefaultTuningSpec(StringRef identifier) {
  std::string tuningSpecName =
      llvm::formatv("iree_default_tuning_spec_{}.mlir", identifier);
  std::optional<StringRef> tuningSpecSource;

  EmbeddedDataDirectory::withGlobal([&](EmbeddedDataDirectory &dir) {
    tuningSpecSource = dir.getFile(tuningSpecName);
  });

  return tuningSpecSource;
}

static FailureOr<ModuleOp>
getDefaultTuningSpec(ModuleOp module,
                     IREE::Codegen::IREECodegenDialect &dialect) {
  if (!clCodegenEnableDefaultTuningSpecs) {
    return failure();
  }

  Operation *annotationSite = nullptr;
  auto target =
      IREE::HAL::ExecutableTargetAttr::lookup(module, &annotationSite);
  if (!target || !target.getConfiguration()) {
    return failure();
  }

  auto storageAttr = dyn_cast_if_present<IREE::Util::StoredModuleAttrInterface>(
      target.getConfiguration().get("iree_codegen.default_tuning_spec"));
  if (!storageAttr) {
    return failure();
  }

  FailureOr<ModuleOp> defaultTransformLibrary =
      storageAttr.getModule(annotationSite);

#ifndef NDEBUG
  if (succeeded(defaultTransformLibrary) &&
      failed(mlir::verify(*defaultTransformLibrary)))
    return (*defaultTransformLibrary).emitError()
           << "Default tuning spec from " << storageAttr << " failed to verify";
#endif

  return defaultTransformLibrary;
}

static FailureOr<DenseElementsAttr>
serializeTuningSpecToAttr(ModuleOp tuningSpec) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  if (failed(writeBytecodeToFile(tuningSpec, os))) {
    return failure();
  }

  auto bufferSize = static_cast<int64_t>(buffer.size());
  auto bufferShape = VectorType::get(
      bufferSize, IntegerType::get(tuningSpec->getContext(), 8));
  return DenseElementsAttr::getFromRawBuffer(
      bufferShape, ArrayRef(buffer.data(), buffer.data() + bufferSize));
}

struct MaterializeTuningSpecsPass final
    : impl::MaterializeTuningSpecsPassBase<MaterializeTuningSpecsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registerTransformDialectTranslationDependentDialects(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    auto dialect = ctx->getOrLoadDialect<IREE::Codegen::IREECodegenDialect>();
    assert(dialect);

    FailureOr<ModuleOp> userTuningSpec = getUserTuningSpec(module, *dialect);
    const bool hasUserTuningSpec = succeeded(userTuningSpec);
    if (!hasUserTuningSpec && !clCodegenTuningSpecPath.empty()) {
      // When a user spec is requested but fails to load, this is a hard
      // failure.
      return signalPassFailure();
    }

    FailureOr<ModuleOp> defaultTuningSpec =
        getDefaultTuningSpec(module, *dialect);
    const bool hasDefaultTuningSpec = succeeded(defaultTuningSpec);
    if (!hasUserTuningSpec && !hasDefaultTuningSpec) {
      // No specs available, nothing to do.
      return;
    }

    // When the user tuning spec and default spec are available, link all
    // available libraries into a single module. We insert the default tuning
    // spec last, so that any user-specified tuning configurations take
    // precedence.
    SmallVector<ModuleOp, 2> allSpecs;
    if (hasUserTuningSpec) {
      allSpecs.push_back(*userTuningSpec);
    }
    if (hasDefaultTuningSpec) {
      allSpecs.push_back(*defaultTuningSpec);
    }

    // Determine if the linking pass should be skipped.
    // Skip if there is only one tuning spec (either user-provided or default)
    // with the default attribute.
    if (allSpecs.size() == 1 &&
        allSpecs[0]->hasAttr(kTuningSpecDefaultEntrypointAttrName)) {
      // Use the appropriate tuning spec (user or default).
      ModuleOp tuningSpecWithDefaultAttr = allSpecs[0];
      if (failed(dumpFinalTuningSpecToDir(tuningSpecWithDefaultAttr))) {
        return signalPassFailure();
      }
      FailureOr<DenseElementsAttr> serializedSpec =
          serializeTuningSpecToAttr(tuningSpecWithDefaultAttr);
      if (failed(serializedSpec)) {
        module->emitError("Failed to serialize default tuning specs");
        return signalPassFailure();
      }
      module->setAttr(kSerializedTuningSpecAttrName, *serializedSpec);
      return;
    }

    Location loc =
        FusedLoc::get(ctx, llvm::map_to_vector<2>(allSpecs, [](ModuleOp m) {
                        return m.getLoc();
                      }));

    // This module will always be released at the end of the pass.
    OwningOpRef<ModuleOp> linkedTuningSpec(
        ModuleOp::create(loc, "iree_linked_tuning_spec"));
    linkedTuningSpec.get()->setAttr(
        transform::TransformDialect::kWithNamedSequenceAttrName,
        UnitAttr::get(ctx));
    for (auto [idx, spec] : llvm::enumerate(allSpecs)) {
      ModuleOp clonedSpec = spec.clone();
      // Make sure there are no symbol name collisions.
      clonedSpec.setSymName(
          llvm::formatv("{}_{}", clonedSpec.getSymName().value(), idx).str());
      linkedTuningSpec->push_back(clonedSpec);
    }

    // TODO(https://github.com/iree-org/iree/issues/19214): Add linked tuning
    // spec memoization to IREECodegenDialect. We should be able to provide a
    // list of input libraries that may have already been linked and ask the
    // dialect to return it to us, or invoke a callback that will insert it if
    // not found.
    FailureOr<transform::NamedSequenceOp> newEntrypoint =
        linkTuningSpecs(linkedTuningSpec.get());
    if (failed(newEntrypoint)) {
      module->emitError("Failed to link tuning specs");
      return signalPassFailure();
    }

    if (failed(dumpFinalTuningSpecToDir(linkedTuningSpec.get()))) {
      return signalPassFailure();
    }

    FailureOr<DenseElementsAttr> serializedSpec =
        serializeTuningSpecToAttr(linkedTuningSpec.get());
    if (failed(serializedSpec)) {
      module->emitError("Failed to serialize linked tuning specs");
      return signalPassFailure();
    }
    module->setAttr(kSerializedTuningSpecAttrName, *serializedSpec);
  }
};

} // namespace
} // namespace mlir::iree_compiler
