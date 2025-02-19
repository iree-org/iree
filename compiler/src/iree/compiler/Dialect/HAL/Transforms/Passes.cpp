// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/Devices/LocalDevice.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/OptionUtils.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::HAL {

namespace {

struct TransformOptions : public PassPipelineOptions<TransformOptions> {
  Option<bool> serializeExecutables{
      *this,
      "serialize-executables",
      llvm::cl::desc("Whether to serialize hal.executable.variant ops to "
                     "hal.executable.binary ops."),
      llvm::cl::init(true),
  };
  Option<bool> linkExecutables{
      *this,
      "link-executables",
      llvm::cl::desc("Whether to link hal.executable ops together."),
      llvm::cl::init(true),
  };
};

static llvm::cl::opt<bool> clMemoization{
    "iree-hal-memoization",
    llvm::cl::desc(
        "Whether to memoize device resources such as command buffers."),
    llvm::cl::init(true),
};

static llvm::cl::opt<unsigned> clBenchmarkDispatchRepeatCount{
    "iree-hal-benchmark-dispatch-repeat-count",
    llvm::cl::desc(
        "The number of times to repeat each hal.command_buffer.dispatch op. "
        "This simply duplicates the dispatch op and inserts barriers. It's "
        "meant for command buffers having linear dispatch structures."),
    llvm::cl::init(1),
};

static llvm::cl::opt<llvm::cl::PowerOf2ByteSize> clInstrumentDispatchBufferSize{
    "iree-hal-instrument-dispatches",
    llvm::cl::desc("Enables dispatch instrumentation with a power-of-two byte "
                   "size used for storage (16mib, 64mib, 2gib, etc)."),
    llvm::cl::init(llvm::cl::PowerOf2ByteSize(0)),
};

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

static llvm::cl::list<std::string> clSubstituteExecutableConfiguration{
    "iree-hal-substitute-executable-configuration",
    llvm::cl::desc(
        "A `executable_name=object_file.xxx` pair specifying a hal.executable "
        "symbol name that will be substituted with the configured executable "
        "file at the given path. Configured execuable paths are relative to "
        "those specified on `--iree-hal-executable-object-search-path=`. If a "
        "`.mlir` or `.mlirbc` file is specified the entire executable will be "
        "replaced with an equivalently named hal.executable in the referenced "
        "file and otherwise the executable will be externalized and link the "
        "referenced file (`.ptx`/`.spv`/etc)."),
};

static llvm::cl::opt<std::string> clSubstituteExecutableConfigurationsFrom{
    "iree-hal-substitute-executable-configurations-from",
    llvm::cl::desc(
        "Substitutes any hal.executable with a file in the given path with "
        "the same name ala `--iree-hal-substitute-executable-configuration=`."),
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

static llvm::cl::opt<bool> clLinkExecutables{
    "iree-hal-link-executables",
    llvm::cl::desc(
        "Controls linking of executables. The default is to always link, "
        "however disabling linking allows inspecting serialization "
        "of each executable in isolation and will dump a single binary per "
        "executable when used in conjunction with "
        "`--iree-hal-dump-executable-binaries-to`."),
    llvm::cl::init(true),
};

} // namespace

using FunctionLikeNest =
    MultiOpNest<func::FuncOp, IREE::Util::InitializerOp, IREE::Util::FuncOp>;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static void addCleanupPatterns(OpPassManager &passManager) {

  FunctionLikeNest(passManager)
      // Standard MLIR cleanup.
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      // Simplify util.global accesses; this can help with data flow tracking as
      // redundant store-loads are removed.
      .addPass(IREE::Util::createSimplifyGlobalAccessesPass)

      // Aggressive cleanup.
      .addPass(IREE::Util::createApplyPatternsPass);

  // Cleanup and canonicalization of util.global (and other util ops).
  passManager.addPass(IREE::Util::createFoldGlobalsPass());
  passManager.addPass(IREE::Util::createFuseGlobalsPass());
}

static void addExecutableSubstitutionPasses(OpPassManager &passManager,
                                            ArrayRef<std::string> substitutions,
                                            StringRef fromPath) {
  if (!fromPath.empty()) {
    SubstituteExecutablesPassOptions substituteOptions;
    substituteOptions.searchPath = fromPath;
    passManager.addPass(
        IREE::HAL::createSubstituteExecutablesPass(substituteOptions));
  }
  if (!substitutions.empty()) {
    SubstituteExecutablesPassOptions substituteOptions;
    substituteOptions.substitutions.assign(substitutions.begin(),
                                           substitutions.end());
    passManager.addPass(
        IREE::HAL::createSubstituteExecutablesPass(substituteOptions));
  }
}

//===----------------------------------------------------------------------===//
// --iree-hal-device-assignment-pipeline
//===----------------------------------------------------------------------===//

void buildHALDeviceAssignmentPassPipeline(
    OpPassManager &passManager, const TargetRegistry &targetRegistry,
    const AssignmentOptions &assignmentOptions) {
  // The HAL must know its targets early on in the process. This pass discovers/
  // derives/specifies the target devices and annotates the module with that
  // information. This allows subsequent passes to lookup which devices they are
  // targeting.
  if (!assignmentOptions.legacyTargetBackends.empty()) {
    // Today we just assign devices from parameters but we should instead be
    // performing analysis at the flow level and then doing magic device
    // database lookups here.
    AssignLegacyTargetDevicesPassOptions options;
    options.targetRegistry = &targetRegistry;
    options.targetBackends.assign(
        assignmentOptions.legacyTargetBackends.begin(),
        assignmentOptions.legacyTargetBackends.end());
    passManager.addPass(
        IREE::HAL::createAssignLegacyTargetDevicesPass(options));
  }
  if (!assignmentOptions.targetDevices.empty()) {
    AssignTargetDevicesPassOptions options;
    options.targetDevices.assign(assignmentOptions.targetDevices.begin(),
                                 assignmentOptions.targetDevices.end());
    passManager.addPass(IREE::HAL::createAssignTargetDevicesPass(options));
  }

  // Create globals for each device (if needed).
  passManager.addPass(IREE::HAL::createMaterializeTargetDevicesPass(
      {assignmentOptions.defaultDevice}));

  // Resolve #hal.device.promise and #hal.device.alias attributes.
  passManager.addPass(IREE::HAL::createResolveDevicePromisesPass());
  passManager.addPass(
      IREE::HAL::createResolveDeviceAliasesPass({&targetRegistry}));

  // Verify devices are valid.
  passManager.addPass(IREE::HAL::createVerifyDevicesPass({&targetRegistry}));
}

//===----------------------------------------------------------------------===//
// --iree-hal-configuration-pipeline
//===----------------------------------------------------------------------===//

void buildHALConfigurationPassPipeline(OpPassManager &passManager,
                                       const TargetRegistry &targetRegistry,
                                       const TargetOptions &targetOptions,
                                       PipelineHooks hooks) {
  //----------------------------------------------------------------------------
  // Input cleanup and simplification
  //----------------------------------------------------------------------------

  // Perform cleanup upon entry so that our IR is in a good state for assignment
  // and initial interface analysis (we rely on CSE and such having been run).
  addCleanupPatterns(passManager);

  // Verify devices are valid.
  passManager.addPass(IREE::HAL::createVerifyDevicesPass({&targetRegistry}));

  //----------------------------------------------------------------------------
  // Device-specific interface materialization
  //----------------------------------------------------------------------------

  // Add dispatch instrumentation prior to materializing interfaces so we can
  // more easily mutate the stream dispatch ops and exports.
  if (auto bufferSize = clInstrumentDispatchBufferSize.getValue()) {
    passManager.addPass(IREE::HAL::createMaterializeDispatchInstrumentationPass(
        {bufferSize.value}));
  }

  // Each executable needs a hal.interface to specify how the host and
  // device communicate across the ABI boundary.
  passManager.addPass(IREE::HAL::createMaterializeInterfacesPass());

  // Prune unused executables and their contents.
  passManager.addPass(IREE::HAL::createPruneExecutablesPass());

  // Dump a source listing of each hal.executable and update the source
  // locations in the IR. This will allow us to easily inspect each executable
  // and give downstream tools that can display source information something
  // more useful and slim than the entire original source model.
  if (!targetOptions.executableSourcesPath.empty()) {
    passManager.addPass(IREE::HAL::createDumpExecutableSourcesPass(
        {targetOptions.executableSourcesPath}));
  }

  // Substitute hal.executables we've generated from earlier phases of
  // compilation with those specified on the command line. This developer
  // feature allows for splicing in hand-authored or hand-modified executables
  // in various forms without modifying the end-to-end compiler. Note that we do
  // this prior to dumping benchmarks in order to allow generating new
  // benchmarks using the substituted executables.
  addExecutableSubstitutionPasses(passManager, clSubstituteExecutableSource,
                                  clSubstituteExecutableSourcesFrom);

  // If debug information is requested capture the MLIR source text of each
  // executable variant and associate it with the entry points. This allows us
  // to preserve this information after translation and the original input IR
  // has been erased.
  if (targetOptions.debugLevel >= 3) {
    passManager.addPass(
        IREE::HAL::createCaptureExecutableSourcesPass({"0.source"}));
  }
}

//===----------------------------------------------------------------------===//
// --iree-hal-transformation-pipeline
//===----------------------------------------------------------------------===//

void buildHALTransformPassPipeline(OpPassManager &passManager,
                                   const TargetRegistry &targetRegistry,
                                   const TargetOptions &targetOptions,
                                   const TransformOptions &transformOptions,
                                   PipelineHooks hooks,
                                   PipelinePhase compileFrom,
                                   PipelinePhase compileTo) {
  //----------------------------------------------------------------------------
  // Device assignment and interface materialization
  //----------------------------------------------------------------------------

  if (hooks.beforePhase) {
    hooks.beforePhase(PipelinePhase::ExecutableSources, passManager);
  }

  if (compileFrom < PipelinePhase::ExecutableSources) {
    AssignmentOptions assignmentOptions;
    assignmentOptions.legacyTargetBackends = targetOptions.legacyTargetBackends;
    assignmentOptions.targetDevices = targetOptions.targetDevices;
    assignmentOptions.defaultDevice = targetOptions.defaultDevice;
    buildHALDeviceAssignmentPassPipeline(passManager, targetRegistry,
                                         assignmentOptions);
    buildHALConfigurationPassPipeline(passManager, targetRegistry,
                                      targetOptions, hooks);

    // Preprocess executables using an external tool. The tool may mutate one or
    // more variants and even insert or remove variants.
    for (auto command : clPreprocessExecutablesWith) {
      passManager.addNestedPass<IREE::HAL::ExecutableOp>(
          IREE::HAL::createPreprocessExecutablesPass(command));
    }
  }

  if (hooks.afterPhase) {
    hooks.afterPhase(PipelinePhase::ExecutableSources, passManager);
  }
  if (compileTo == PipelinePhase::ExecutableSources) {
    return;
  }

  //----------------------------------------------------------------------------
  // Executable translation
  //----------------------------------------------------------------------------

  if (hooks.beforePhase) {
    hooks.beforePhase(PipelinePhase::ExecutableConfigurations, passManager);
  }

  if (compileFrom < PipelinePhase::ExecutableConfigurations) {
    // Select a translation strategy for each hal.executable.variant and
    // generate the IR to condition on support for the variant. In the future,
    // this or neighboring passes can expand/contract variants based on the
    // selected translation strategies and the features each translation
    // strategy are known to require or not require.
    passManager.addNestedPass<IREE::HAL::ExecutableOp>(
        IREE::HAL::createConfigureExecutablesPass({targetRegistry}));

    // Dump a second listing of each hal.executable after preprocessing and
    // configuration of executables, as well as update locations in the IR.
    if (!targetOptions.executableConfigurationsPath.empty()) {
      passManager.addPass(IREE::HAL::createDumpExecutableSourcesPass(
          {targetOptions.executableConfigurationsPath, "configured"}));
    }

    // If debug information is requested capture the MLIR source text of each
    // configured executable variant and associate it with the entry points.
    if (targetOptions.debugLevel >= 3) {
      passManager.addPass(
          IREE::HAL::createCaptureExecutableSourcesPass({"1.configured"}));
    }

    // Substitute hal.executables we've configured with those specified on the
    // command line. This developer feature allows for hand editing the
    // configured executable with different lowering parameters.
    addExecutableSubstitutionPasses(passManager,
                                    clSubstituteExecutableConfiguration,
                                    clSubstituteExecutableConfigurationsFrom);

    // Dump standalone hal.executable benchmark modules.
    // Today this only works for executables that have static dispatch
    // parameters and is only useful for basic microbenchmarking. We do this
    // after configuration to make it easy to tweak configurations directly
    // from the benchmark.
    if (!targetOptions.executableBenchmarksPath.empty()) {
      passManager.addPass(IREE::HAL::createDumpExecutableBenchmarksPass(
          {targetOptions.executableBenchmarksPath}));
    }
  }

  if (hooks.afterPhase) {
    hooks.afterPhase(PipelinePhase::ExecutableConfigurations, passManager);
  }
  if (compileTo == PipelinePhase::ExecutableConfigurations) {
    return;
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

  if (hooks.beforePhase) {
    hooks.beforePhase(PipelinePhase::ExecutableTargets, passManager);
  }

  if (compileFrom < PipelinePhase::ExecutableTargets) {
    passManager.addNestedPass<IREE::HAL::ExecutableOp>(
        IREE::HAL::createTranslateAllExecutablesPass({targetRegistry}));
  }

  // If debug information is requested capture the translated MLIR source text
  // of each executable variant and associate it with the entry points. This
  // allows us to compare the input IR with the translated IR before
  // serialization (LLVM dialect, SPIR-V dialect, etc).
  if (targetOptions.debugLevel >= 3) {
    passManager.addPass(
        IREE::HAL::createCaptureExecutableSourcesPass({"2.translated"}));
  }

  if (hooks.afterPhase) {
    hooks.afterPhase(PipelinePhase::ExecutableTargets, passManager);
  }
  if (compileTo == PipelinePhase::ExecutableTargets) {
    return;
  }

  // Substitute hal.executables we've translated with those specified on the
  // command line. This developer feature allows for splicing in hand-authored
  // or hand-modified executables in various forms without modifying the
  // end-to-end compiler. We support substituting prior to translation as well
  // but sometimes translation is required to produce the host code required
  // for specialization and workgroup counts and we need to perform the
  // substitution later.
  addExecutableSubstitutionPasses(passManager, clSubstituteExecutableObject,
                                  clSubstituteExecutableObjectsFrom);

  //----------------------------------------------------------------------------
  // Host program conversion
  //----------------------------------------------------------------------------

  // Convert supported input dialects (std, stream, etc) into the HAL dialect.
  passManager.addPass(IREE::HAL::createConvertToHALPass());

  // If memoization is disabled then inline any regions that were created during
  // conversion.
  if (!clMemoization) {
    FunctionLikeNest(passManager)
        .addPass(IREE::HAL::createInlineMemoizeRegionsPass);
  } else {
    passManager.addPass(IREE::HAL::createOutlineMemoizeRegionsPass());
  }

  // Prune unused executables and their contents.
  passManager.addPass(IREE::HAL::createPruneExecutablesPass());

  addCleanupPatterns(passManager);

  //----------------------------------------------------------------------------
  // Executable packing and runtime loading
  //----------------------------------------------------------------------------

  // TODO(benvanik): move translation down to here.

  // After all executables are translated and before resolving export
  // ordinals we allow the backends to link executables together. For
  // example, the LLVM AOT backend may combine all executable targets for the
  // same architecture into a single executable and link it as a shared
  // library.
  if (transformOptions.linkExecutables && clLinkExecutables) {
    passManager.addPass(
        IREE::HAL::createLinkAllExecutablesPass({targetRegistry}));
  }

  // If any executable variants have external objects referenced within them
  // we hoist them up to the top-level variant. This is done after linking so
  // that we have the greatest chance of combining executables without different
  // object attrs preventing the merging.
  passManager.nest<IREE::HAL::ExecutableOp>()
      .addNestedPass<IREE::HAL::ExecutableVariantOp>(
          IREE::HAL::createHoistExecutableObjectsPass());

  // Resolve export ordinals from nested symbol references prior to
  // serialization. As this pass creates lookup ops it should run before
  // MaterializeResourceCachesPass.
  passManager.addPass(IREE::HAL::createResolveExportOrdinalsPass());

  // Gather cacheable resources such as executables and descriptor sets and
  // cache them at initialization-time.
  passManager.addPass(IREE::HAL::createMaterializeResourceCachesPass());

  //----------------------------------------------------------------------------
  // Device management and specialization
  //----------------------------------------------------------------------------

  // Memoize device queries such that we don't need to repeatedly ask the same
  // information at runtime.
  passManager.addPass(IREE::HAL::createMemoizeDeviceQueriesPass());

  // Big cleanup after all our conversion and materialization.
  addCleanupPatterns(passManager);

  // Benchmarking only: repeat dispatch ops a certain number of times.
  // This is guaranteed to invalidate program output and may introduce crashes
  // if there are in-place dispatches that expect specific input data.
  if (clBenchmarkDispatchRepeatCount != 1) {
    FunctionLikeNest(passManager).addPass([&]() {
      return IREE::HAL::createRepeatDispatchesPass(
          {clBenchmarkDispatchRepeatCount});
    });
  }

  // Elide redundant command buffer state ops created during conversion.
  FunctionLikeNest(passManager)
      .addPass(IREE::HAL::createElideRedundantCommandsPass);

  // Initialize device globals now that we've done the analysis that is easier
  // with them in their original target specification.
  passManager.addPass(IREE::HAL::createInitializeDevicesPass({targetRegistry}));

  // TODO: Maybe this should be a part of Affine lowering pass.
  // Remove if it is added there.
  // https://github.com/llvm/llvm-project/issues/78458
  passManager.addPass(affine::createAffineExpandIndexOpsPass());
  // Fixup workgroup count calculations that may have used the affine dialect.
  // Kind of random here but can happen if the benchmarking code does things.
  passManager.addPass(mlir::createLowerAffinePass());

  // TODO(benvanik): remove the need for this; some cleanup passes such as
  // SimplifyGlobalAccesses are currently broken with scf present.
  FunctionLikeNest(passManager).addPass(mlir::createSCFToControlFlowPass);

  //----------------------------------------------------------------------------
  // Executable serialization
  //----------------------------------------------------------------------------

  // Happens at the very end as IR is much more debuggable with the executable
  // contents not turned into a big base64 string.
  if (transformOptions.serializeExecutables) {
    passManager.addNestedPass<IREE::HAL::ExecutableOp>(
        IREE::HAL::createSerializeAllExecutablesPass(
            {&targetRegistry, targetOptions.debugLevel,
             targetOptions.executableIntermediatesPath,
             targetOptions.executableBinariesPath}));

    // NOTE: symbol DCE will destroy executable target contents, so only run
    // it if we serialized things.
    passManager.addPass(IREE::HAL::createPruneExecutablesPass());
    passManager.addPass(mlir::createSymbolDCEPass());
  }

  //----------------------------------------------------------------------------
  // Whole-program optimization
  //----------------------------------------------------------------------------

  {
    // We run these under a fixed-point iteration such that we can perform
    // inter-procedural, intra-procedural, and canonicalization as separably
    // verifiable/reusable passes. IPO will fold duplicate arguments/results
    // and inline constants to allow the local optimizations to work more
    // effectively.
    OpPassManager ipoPipeline(mlir::ModuleOp::getOperationName());

    // IPO and other cleanups.
    addCleanupPatterns(ipoPipeline);

    // Large IPO pass. Note that this can introduce a significant amount of
    // duplication/inlined constants and we'll want to ensure we're running
    // cleanup again after (this entire set of patterns is run in a
    // fixed-point iteration to do that).
    ipoPipeline.addPass(IREE::Util::createIPOPass());

    // Run fixed-point iteration on the IPO pipeline.
    passManager.addPass(
        IREE::Util::createFixedPointIteratorPass(std::move(ipoPipeline)));
  }
}

void buildHALTransformPassPipeline(OpPassManager &passManager,
                                   const TargetRegistry &targetRegistry,
                                   const TargetOptions &targetOptions,
                                   PipelineHooks hooks,
                                   PipelinePhase compileFrom,
                                   PipelinePhase compileTo) {
  TransformOptions transformOptions;
  buildHALTransformPassPipeline(passManager, targetRegistry, targetOptions,
                                transformOptions, hooks, compileFrom,
                                compileTo);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc" // IWYU pragma: export
} // namespace

void registerHALPasses() {
  // Force the flags to be bound.
  // TODO(benvanik): remove the global flags and only rely on pipeline flags.
  (void)IREE::HAL::TargetOptions::FromFlags::get();
  // TODO(multi-device): move the local device registration somewhere more
  // centralized. For now we piggy-back on the pass registration as that's where
  // the local device is used.
  (void)IREE::HAL::LocalDevice::Options::FromFlags::get();
  IREE::HAL::TargetDeviceList deviceList;
  deviceList.add("local", [=]() {
    return std::make_shared<LocalDevice>(
        IREE::HAL::LocalDevice::Options::FromFlags::get());
  });
  IREE::HAL::TargetRegistry::getMutableTargetRegistry().mergeFrom(deviceList);

  // Generated.
  registerPasses();

  // Pipelines.
  PassPipelineRegistration<AssignmentOptions>(
      "iree-hal-device-assignment-pipeline",
      "Runs HAL target device assignment pipeline.",
      [](OpPassManager &passManager,
         const AssignmentOptions &assignmentOptions) {
        buildHALDeviceAssignmentPassPipeline(
            passManager, TargetRegistry::getGlobal(), assignmentOptions);
      });
  PassPipelineRegistration<>("iree-hal-configuration-pipeline",
                             "Runs HAL target configuration pipeline.",
                             [](OpPassManager &passManager) {
                               buildHALConfigurationPassPipeline(
                                   passManager, TargetRegistry::getGlobal(),
                                   TargetOptions::FromFlags::get());
                             });
  PassPipelineRegistration<TransformOptions>(
      "iree-hal-transformation-pipeline",
      "Runs the full IREE HAL conversion/lowering pipeline.",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildHALTransformPassPipeline(passManager, TargetRegistry::getGlobal(),
                                      TargetOptions::FromFlags::get(),
                                      transformOptions, PipelineHooks{},
                                      PipelinePhase::Start, PipelinePhase::End);
      });
}

} // namespace mlir::iree_compiler::IREE::HAL
