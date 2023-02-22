// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {

struct TransformOptions : public PassPipelineOptions<TransformOptions> {
  // TODO(benvanik): replace the global iree-hal-target-backends flag with this.
  // ListOption<std::string> targets{
  //     *this, "targets", llvm::cl::desc("One or more HAL devices to target."),
  //     llvm::cl::ZeroOrMore};
  Option<bool> serializeExecutables{
      *this, "serialize-executables",
      llvm::cl::desc("Whether to serialize hal.executable.variant ops to "
                     "hal.executable.binary ops."),
      llvm::cl::init(true)};
  Option<bool> linkExecutables{
      *this, "link-executables",
      llvm::cl::desc("Whether to link hal.executable ops together."),
      llvm::cl::init(true)};
};

static llvm::cl::opt<unsigned> clBenchmarkDispatchRepeatCount{
    "iree-hal-benchmark-dispatch-repeat-count",
    llvm::cl::desc(
        "The number of times to repeat each hal.command_buffer.dispatch op. "
        "This simply duplicates the dispatch op and inserts barriers. It's "
        "meant for command buffers having linear dispatch structures."),
    llvm::cl::init(1)};

static llvm::cl::list<std::string> clSubstituteExecutableSource{
    "iree-hal-substitute-executable-source",
    llvm::cl::desc(
        "A `executable_name=object_file.xxx` pair specifying a "
        "hal.executable symbol name that will be substituted with the source "
        "object file at the given path. Source object paths are relative to "
        "those specified on `--iree-hal-executable-object-search-path=`. If a "
        "`.mlir` or `.mlirbc` file is specified the entire executable will be "
        "replaced with an equivalently named hal.executable in the referenced "
        "file and otherwise the executable will be externalized and link the "
        "referenced file (`.ptx`/`.spv`/etc)."),
};

static llvm::cl::opt<std::string> clSubstituteExecutableSourcesFrom{
    "iree-hal-substitute-executable-sources-from",
    llvm::cl::desc(
        "Substitutes any hal.executable with a file in the given path with "
        "the same name ala `--iree-hal-substitute-executable-source=`."),
    llvm::cl::init(""),
};

static llvm::cl::list<std::string> clSubstituteExecutableObject{
    "iree-hal-substitute-executable-object",
    llvm::cl::desc(
        "A `executable_name=object_file.xxx` pair specifying a "
        "hal.executable symbol name that will be substituted with the object "
        "file at the given path. Object paths are relative to those "
        "specified on `--iree-hal-executable-object-search-path=`. If a "
        "`.mlir` or `.mlirbc` file is specified the entire executable will be "
        "replaced with an equivalently named hal.executable in the referenced "
        "file and otherwise the executable will be externalized and link the "
        "referenced file (`.ptx`/`.spv`/etc)."),
};

static llvm::cl::opt<std::string> clSubstituteExecutableObjectsFrom{
    "iree-hal-substitute-executable-objects-from",
    llvm::cl::desc(
        "Substitutes any hal.executable with a file in the given path with "
        "the same name ala `--iree-hal-substitute-executable-object=`."),
    llvm::cl::init(""),
};

static llvm::cl::list<std::string> clPreprocessExecutablesWith{
    "iree-hal-preprocess-executables-with",
    llvm::cl::desc(
        "Passes each hal.executable to the given command. Multiple "
        "commands may be specified and they will be "
        "executed in order. A command may either be a pass pipeline available "
        "within the IREE compiler specified as `builtin.module(...)` or a "
        "shell tool that consumes a hal.executable MLIR file on stdin and "
        "produces a modified hal.executable on stdout. Non-zero exit codes "
        "will fail compilation."),
};

}  // namespace

using FunctionLikeNest = MultiOpNest<func::FuncOp, IREE::Util::InitializerOp>;

static void addCleanupPatterns(OpPassManager &passManager) {
  // Standard MLIR cleanup.
  passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addPass(mlir::createCSEPass());

  // Simplify util.global accesses; this can help with data flow tracking as
  // redundant store-loads are removed.
  FunctionLikeNest(passManager)
      .addPass(IREE::Util::createSimplifyGlobalAccessesPass);

  // Cleanup and canonicalization of util.global (and other util ops).
  passManager.addPass(IREE::Util::createApplyPatternsPass());
  passManager.addPass(IREE::Util::createFoldGlobalsPass());
  passManager.addPass(IREE::Util::createFuseGlobalsPass());
}

void buildHALConfigurationPassPipeline(OpPassManager &passManager,
                                       const TargetOptions &targetOptions) {
  //----------------------------------------------------------------------------
  // Input cleanup and simplification
  //----------------------------------------------------------------------------

  // Perform cleanup upon entry so that our IR is in a good state for assignment
  // and initial interface analysis (we rely on CSE and such having been run).
  addCleanupPatterns(passManager);

  //----------------------------------------------------------------------------
  // Device assignment and interface materialization
  //----------------------------------------------------------------------------

  // The HAL must know its targets early on in the process. This pass discovers/
  // derives/specifies the target devices and annotates the module with that
  // information. This allows subsequent passes to lookup which devices they are
  // targeting.
  if (!targetOptions.targets.empty()) {
    // Today we just assign devices from parameters but we should instead be
    // performing analysis at the flow level and then doing magic device
    // database lookups here.
    passManager.addPass(createAssignTargetDevicesPass(targetOptions.targets));
  }
  passManager.addPass(createVerifyTargetEnvironmentPass());

  // Each executable needs a hal.interface to specify how the host and
  // device communicate across the ABI boundary.
  passManager.addPass(createMaterializeInterfacesPass());

  // Dump a source listing of each hal.executable and update the source
  // locations in the IR. This will allow us to easily inspect each executable
  // and give downstream tools that can display source information something
  // more useful and slim than the entire original source model.
  if (!targetOptions.sourceListingPath.empty()) {
    passManager.addPass(
        createDumpExecutableSourcesPass(targetOptions.sourceListingPath));
  }

  // Substitute hal.executables we've generated from earlier phases of
  // compilation with those specified on the command line. This developer
  // feature allows for splicing in hand-authored or hand-modified executables
  // in various forms without modifying the end-to-end compiler. Note that we do
  // this prior to dumping benchmarks in order to allow generating new
  // benchmarks using the substituted executables.
  if (!clSubstituteExecutableSourcesFrom.empty()) {
    passManager.addPass(createSubstituteExecutablesPass(
        clSubstituteExecutableSourcesFrom.getValue()));
  }
  if (!clSubstituteExecutableSource.empty()) {
    passManager.addPass(
        createSubstituteExecutablesPass(clSubstituteExecutableSource));
  }

  // Dump standalone hal.executable benchmark modules.
  // Today this only works for executables that have static dispatch parameters
  // and is only useful for basic microbenchmarking.
  if (!targetOptions.executableBenchmarksPath.empty()) {
    passManager.addPass(createDumpExecutableBenchmarksPass(
        targetOptions.executableBenchmarksPath));
  }
}

void buildHALTransformPassPipeline(OpPassManager &passManager,
                                   const TargetOptions &targetOptions,
                                   const TransformOptions &transformOptions) {
  //----------------------------------------------------------------------------
  // Device assignment and interface materialization
  //----------------------------------------------------------------------------

  buildHALConfigurationPassPipeline(passManager, targetOptions);

  //----------------------------------------------------------------------------
  // Executable translation
  //----------------------------------------------------------------------------

  // Preprocess executables using an external tool. The tool may mutate one or
  // more variants and even insert or remove variants.
  for (auto command : clPreprocessExecutablesWith) {
    passManager.addNestedPass<IREE::HAL::ExecutableOp>(
        createPreprocessExecutablesPass(command));
  }

  // TODO(benvanik): move translation after conversion; today translation
  // inserts the workgroup count logic we need to convert but we could instead
  // insert placeholder ops that are expanded after translation.
  //
  // Translate each executable variant to its target IR form.
  // It's extremely important this runs parallelized as it's where a large
  // majority of our compilation time lives (we invoke LLVM and lld and such).
  //
  // After this point the executables are opaque blobs and we cannot change
  // their interfaces.
  passManager.addNestedPass<IREE::HAL::ExecutableOp>(
      createTranslateExecutablesPass());

  // Substitute hal.executables we've translated with those specified on the
  // command line. This developer feature allows for splicing in hand-authored
  // or hand-modified executables in various forms without modifying the
  // end-to-end compiler. We support substituting prior to translation as well
  // but sometimes translation is required to produce the host code required for
  // specialization and workgroup counts and we need to perform the substitution
  // later.
  if (!clSubstituteExecutableObjectsFrom.empty()) {
    passManager.addPass(createSubstituteExecutablesPass(
        clSubstituteExecutableObjectsFrom.getValue()));
  }
  if (!clSubstituteExecutableObject.empty()) {
    passManager.addPass(
        createSubstituteExecutablesPass(clSubstituteExecutableObject));
  }

  //----------------------------------------------------------------------------
  // Host program conversion
  //----------------------------------------------------------------------------

  // Convert supported input dialects (std, stream, etc) into the HAL dialect.
  passManager.addPass(createConvertToHALPass());

  // If any devices require the legacy synchronous execution behavior then
  // make all async operations blocking.
  passManager.addPass(createFixupLegacySyncPass());

  addCleanupPatterns(passManager);

  //----------------------------------------------------------------------------
  // Executable packing and runtime loading
  //----------------------------------------------------------------------------

  // TODO(benvanik): move translation down to here.

  // After all executables are translated and before resolving export
  // ordinals, we allow the backends to link executables together. For example,
  // the LLVM AOT backend may combine all executable targets for the same
  // architecture into a single executable and link it as a shared library.
  if (transformOptions.linkExecutables) {
    passManager.addPass(createLinkExecutablesPass());
  }

  // Resolve export ordinals from nested symbol references prior to
  // serialization. As this pass creates lookup ops it should run before
  // MaterializeResourceCachesPass.
  passManager.addPass(createResolveExportOrdinalsPass());

  // Gather cacheable resources such as executables and descriptor sets and
  // cache them at initialization-time.
  passManager.addPass(createMaterializeResourceCachesPass(targetOptions));

  //----------------------------------------------------------------------------
  // Device management and specialization
  //----------------------------------------------------------------------------

  // Inline hal.device.switch ops and memoize their queries such that we can
  // better CSE/fold dispatch logic.
  FunctionLikeNest(passManager).addPass(createInlineDeviceSwitchesPass);

  // Memoize device queries such that we don't need to repeatedly ask the same
  // information at runtime.
  passManager.addPass(createMemoizeDeviceQueriesPass());

  // Big cleanup after all our conversion and materialization.
  addCleanupPatterns(passManager);

  // HACK: repeat dispatch ops for benchmarks.
  if (clBenchmarkDispatchRepeatCount != 1) {
    passManager.addNestedPass<mlir::func::FuncOp>(
        createBenchmarkBatchDispatchesPass(clBenchmarkDispatchRepeatCount));
  }

  // Elide redundant command buffer state ops created during conversion.
  FunctionLikeNest(passManager).addPass(createElideRedundantCommandsPass);

  // Fixup workgroup count calculations that may have used the affine dialect.
  // Kind of random here but can happen if the benchmarking code does things.
  passManager.addPass(mlir::createLowerAffinePass());

  // TODO(benvanik): remove the need for this; some cleanup passes such as
  // SimplifyGlobalAccesses are currently broken with scf present.
  FunctionLikeNest(passManager).addPass(mlir::createConvertSCFToCFPass);

  // Combine the initializers we emitted during resource cache materialization.
  passManager.addPass(IREE::Util::createCombineInitializersPass());

  //----------------------------------------------------------------------------
  // Executable serialization
  //----------------------------------------------------------------------------

  // Happens at the very end as IR is much more debuggable with the executable
  // contents not turned into a big base64 string.
  if (transformOptions.serializeExecutables) {
    passManager.addNestedPass<IREE::HAL::ExecutableOp>(
        createSerializeExecutablesPass(
            targetOptions.debugLevel, targetOptions.executableIntermediatesPath,
            targetOptions.executableBinariesPath));

    // NOTE: symbol DCE will destroy executable target contents, so only run it
    // if we serialized things.
    passManager.addPass(mlir::createSymbolDCEPass());
  }

  //----------------------------------------------------------------------------
  // Whole-program optimization
  //----------------------------------------------------------------------------

  {
    // We run these under a fixed-point iteration such that we can perform
    // inter-procedural, intra-procedural, and canonicalization as separably
    // verifiable/reusable passes. IPO will fold duplicate arguments/results and
    // inline constants to allow the local optimizations to work more
    // effectively.
    OpPassManager ipoPipeline(mlir::ModuleOp::getOperationName());

    // IPO and other cleanups.
    addCleanupPatterns(ipoPipeline);

    // Large IPO pass. Note that this can introduce a significant amount of
    // duplication/inlined constants and we'll want to ensure we're running
    // cleanup again after (this entire set of patterns is run in a fixed-point
    // iteration to do that).
    ipoPipeline.addPass(IREE::Util::createIPOPass());

    // Run fixed-point iteration on the IPO pipeline.
    passManager.addPass(
        IREE::Util::createFixedPointIteratorPass(std::move(ipoPipeline)));
  }
}

void buildHALTransformPassPipeline(OpPassManager &passManager,
                                   const TargetOptions &targetOptions) {
  TransformOptions transformOptions;
  buildHALTransformPassPipeline(passManager, targetOptions, transformOptions);
}

void registerHALConfigurationPassPipeline() {
  PassPipelineRegistration<>("iree-hal-configuration-pipeline",
                             "Runs the IREE HAL dialect configuration pipeline",
                             [](OpPassManager &passManager) {
                               buildHALConfigurationPassPipeline(
                                   passManager,
                                   TargetOptions::FromFlags::get());
                             });
}

void registerHALTransformPassPipeline() {
  PassPipelineRegistration<TransformOptions>(
      "iree-hal-transformation-pipeline",
      "Runs the full IREE HAL dialect transformation pipeline",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildHALTransformPassPipeline(
            passManager, TargetOptions::FromFlags::get(), transformOptions);
      });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
