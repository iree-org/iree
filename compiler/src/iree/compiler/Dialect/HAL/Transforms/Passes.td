// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_HAL_PASSES
#define IREE_DIALECT_HAL_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

def ConvertToHALPass :
    Pass<"iree-hal-conversion", "mlir::ModuleOp"> {
  let summary = "Converts from stream and other intermediate dialects into the hal dialect.";
  let description = [{
    Converts supported intermediate dialects (`stream`, `util`, and various
    upstream dialects like `cf`/`scf`) into the hal dialect. After conversion
    host code scheduling work and allocations will act on `!hal.device` queues
    and `!hal.buffer` (and other) resources.

    It's expected that executable interface materialization has been performed
    so that the information required to marshal buffers and operands to the
    device is available for conversion.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "IREE::HAL::HALDialect",
    "IREE::Stream::StreamDialect",
    "IREE::Util::UtilDialect",
    // TODO(benvanik): add a registration system for extra dialects?
    "IREE::IO::Parameters::IOParametersDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Device management
//===----------------------------------------------------------------------===//

def VerifyTargetEnvironmentPass :
    Pass<"iree-hal-verify-target-environment", "mlir::ModuleOp"> {
  let summary = "Verifies that the target execution environment is valid.";
  let description = [{
    Verifies that the target execution environment is valid.
    `#hal.device.target` and `#hal.executable.target` attribute placement and
    definition will be checked that they reference the available target backends
    and that they are structurally valid.
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetBackendRegistryRef", "",
      "Target backend registry containing the list of available backends."
    >,
  ];
}

def AssignTargetDevicesPass :
    Pass<"iree-hal-assign-target-devices", "mlir::ModuleOp"> {
  let summary = "Assigns the HAL devices the module will target to the given list of targets.";
  let description = [{
    DO NOT SUBMIT
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetBackendRegistryRef", "",
      "Target backend registry containing the list of available backends."
    >,
    ListOption<
      "targets", "targets",
      "std::string",
      "List of devices to assign as targets."
    >,
  ];
}

def FixupLegacySyncPass :
    Pass<"iree-hal-fixup-legacy-sync", "mlir::ModuleOp"> {
  let summary = "Applies fixups to the program for when using legacy HAL devices.";
  let description = [{
    Applies fixups to the program for when using legacy HAL devices that only
    support synchronous execution. Once all devices support async this will be
    removed.

    NOTE: this pass only exists for backwards compatibility with legacy HAL
    drivers. It will be removed once all have migrated to the modern async APIs.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "IREE::HAL::HALDialect",
    "IREE::Util::UtilDialect",
  ];
}

def MemoizeDeviceQueriesPass :
    Pass<"iree-hal-memoize-device-queries", "mlir::ModuleOp"> {
  let summary = "Finds hal.device.query ops and creates variables initialized on startup.";
  let description = [{
    Finds all `hal.device.query`-related ops that are hoistable and moves them
    into globals that are initialized on startup. This prevents repeated queries
    at runtime and allows for optimization as queries are CSEd across the entire
    program.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "IREE::HAL::HALDialect",
    "IREE::Util::UtilDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Executable translation
//===----------------------------------------------------------------------===//

def MaterializeInterfacesPass :
    Pass<"iree-hal-materialize-interfaces", "mlir::ModuleOp"> {
  let summary = "Defines hal.executable variants for stream.executable ops.";
  let description = [{
    Defines hal.executables and one hal.variant for each required target. The
    interfaces required to marshal buffers and operands across the host-device
    boundary are declared on the executables and annotated on the dispatch
    sites so that subsequent conversion can consume them.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "IREE::HAL::HALDialect",
  ];
}

def SubstituteExecutablesPass :
    Pass<"iree-hal-substitute-executables", "mlir::ModuleOp"> {
  let summary = "Substitutes hal.executable ops with files on disk.";
  let description = [{
    Substitutes hal.executable ops with externally referenced MLIR files or
    target-specific object files. When provided a .mlir/.mlirbc file with a
    top-level hal.executable the entire executable will be replaced including
    all variants contained with. All other files such as .o, .bc, and .spv will
    be set as external object files on the original executable variants and the
    original contents will be dropped.

    Substitutions can be specified by providing a file system path where there
    exists files matching the executable names in one of the supported formats
    or by specifying the file each executable name maps to directly.
  }];
  let options = [
    ListOption<
      "substitutions", "substitutions",
      "std::string",
      "Substitution `executable_name=file.xxx` key-value pairs."
    >,
    Option<
      "searchPath", "search-path",
      "std::string", "",
      "Path to source executable substitutions from."
    >,
  ];
  let dependentDialects = [
    // NOTE: we may need more for anything we load from MLIR files.
    "IREE::HAL::HALDialect",
  ];
}

def PreprocessExecutablesWithPipelinePass :
    Pass<"iree-hal-preprocess-executables-with-pipeline", "IREE::HAL::ExecutableOp"> {
  let summary = "Preprocess each executable with an MLIR pass pipeline.";
  let description = [{
    Runs the given MLIR pass pipeline as parsed by the `--pass-pipeline=` flag
    on each hal.executable in the program. The passes must be linked into the
    compiler to be discovered.
  }];
  let options = [
    Option<
      "pipeline", "pipeline",
      "std::string", "",
      "MLIR pass pipeline description to run on each executable."
    >,
  ];
}

def PreprocessExecutablesWithToolPass :
    Pass<"iree-hal-preprocess-executables-with-tool", "IREE::HAL::ExecutableOp"> {
  let summary = "Preprocess each executable with an external command line tool.";
  let description = [{
    Passes each hal.executable in the program to the given command line tool
    as stdin and parses the resulting MLIR from stdout to replace them. This
    is equivalent to `iree-hal-preprocess-executables-with-pipeline` but allows
    for an external `mlir-opt`/`iree-opt`-like tool to be used containing the
    pipelines instead of requiring the passes to be linked into the compiler.
  }];
  let options = [
    Option<
      "command", "command",
      "std::string", "",
      "stdin->stdout command to run on each hal.executable MLIR op."
    >,
  ];
}

def ConfigureExecutablesPass :
    Pass<"iree-hal-configure-executables", "IREE::HAL::ExecutableOp"> {
  let summary = "Configures hal.executable ops via a nested translation pipeline.";
  let description = [{
    DO NOT SUBMIT
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetBackendRegistryRef", "",
      "Target backend registry containing the list of available backends."
    >,
  ];
}

def ConfigureTargetExecutableVariantsPass :
    Pass<"iree-hal-configure-target-executable-variants", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Configures hal.executable.variant ops for the specified target backend.";
  let description = [{
    DO NOT SUBMIT
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetBackendRegistryRef", "",
      "Target backend registry containing the list of available backends."
    >,
    Option<
      "target", "target",
      "std::string", "",
      "Target backend name whose executable variants will be configured by this pass."
    >,
  ];
}

def TranslateExecutablesPass :
    Pass<"iree-hal-translate-executables", "IREE::HAL::ExecutableOp"> {
  let summary = "Translates hal.executable ops via a nested translation pipeline.";
  let description = [{
    DO NOT SUBMIT
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetBackendRegistryRef", "",
      "Target backend registry containing the list of available backends."
    >,
  ];
}

def TranslateTargetExecutableVariantsPass :
    Pass<"iree-hal-translate-target-executable-variants", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Translates hal.executable.variant ops for the specified target backend.";
  let description = [{
    DO NOT SUBMIT
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetBackendRegistryRef", "",
      "Target backend registry containing the list of available backends."
    >,
    Option<
      "target", "target",
      "std::string", "",
      "Target backend name whose executable variants will be translated by this pass."
    >,
  ];
}

def LinkExecutablesPass :
    Pass<"iree-hal-link-executables", "mlir::ModuleOp"> {
  let summary = "Links hal.executable ops into one or more hal.executable ops.";
  let description = [{
    Calls into each target backend to have it link multiple `hal.executable` ops
    together (if the backend desires).

    DO NOT SUBMIT
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetBackendRegistryRef", "",
      "Target backend registry containing the list of available backends."
    >,
  ];
}

def LinkTargetExecutablesPass :
    Pass<"iree-hal-link-target-executables", "mlir::ModuleOp"> {
  let summary = "Links executables for the specified target backend.";
  let description = [{
    DO NOT SUBMIT
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetBackendRegistryRef", "",
      "Target backend registry containing the list of available backends."
    >,
    Option<
      "target", "target",
      "std::string", "",
      "Target backend name whose executables will be linked by this pass."
    >,
  ];
}

def ResolveExportOrdinalsPass :
    Pass<"iree-hal-resolve-export-ordinals", "mlir::ModuleOp"> {
  let summary = "Resolves symbolic hal.executable.export references to ordinals.";
  let description = [{
    Severs symbolic references to hal.executable.export ops from dispatch sites
    by replacing them with the ordinal assigned to the exports. This allows for
    subsequent passes to collapse the executables into opaque blobs.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "IREE::HAL::HALDialect",
  ];
}

def SerializeExecutablesPass :
    Pass<"iree-hal-serialize-executables", "IREE::HAL::ExecutableOp"> {
  let summary = "Converts hal.executable.variants to one or more hal.executable.binary ops.";
  let description = [{
    DO NOT SUBMIT
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetBackendRegistryRef", "",
      "Target backend registry containing the list of available backends."
    >,
    Option<
      "debugLevel", "debug-level",
      "int", "2",
      "Debug level for serialization (0 (no information) to 3 (all information))."
    >,
    Option<
      "dumpIntermediatesPath", "dump-intermediates-path",
      "std::string", "",
      "Path to write translated executable intermediates (.bc, .o, etc) into for debugging."
    >,
    Option<
      "dumpBinariesPath", "dump-binaries-path",
      "std::string", "",
      "Path to write translated and serialized executable binaries into for debugging."
    >,
  ];
}

def SerializeTargetExecutablesPass :
    Pass<"iree-hal-serialize-target-executables", "IREE::HAL::ExecutableOp"> {
  let summary = "Serializes executables for the specified target backend.";
  let description = [{
    DO NOT SUBMIT
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetBackendRegistryRef", "",
      "Target backend registry containing the list of available backends."
    >,
    Option<
      "target", "target",
      "std::string", "",
      "Target backend name whose executables will be serialized by this pass."
    >,
    Option<
      "debugLevel", "debug-level",
      "int", "2",
      "Debug level for serialization (0 (no information) to 3 (all information))."
    >,
    Option<
      "dumpIntermediatesPath", "dump-intermediates-path",
      "std::string", "",
      "Path to write translated executable intermediates (.bc, .o, etc) into for debugging."
    >,
    Option<
      "dumpBinariesPath", "dump-binaries-path",
      "std::string", "",
      "Path to write translated and serialized executable binaries into for debugging."
    >,
  ];
}

//===----------------------------------------------------------------------===//
// Resource initialization, caching, and optimization
//===----------------------------------------------------------------------===//

def MaterializeDispatchInstrumentationPass :
    Pass<"iree-hal-materialize-dispatch-instrumentation", "mlir::ModuleOp"> {
  let summary = "Materializes host and device dispatch instrumentation resources on stream IR.";
  let description = [{
    Adds dispatch instrumentation for both host and device prior to
    materializing interfaces so that the higher-level stream dialect can be used
    to easily mutate the dispatch sites, executable exports, and resources used
    for instrumentation storage.
  }];
  let options = [
    Option<
      "bufferSize", "buffer-size",
      "llvm::cl::PowerOf2ByteSize", "llvm::cl::PowerOf2ByteSize(64 * 1024 * 1024)",
      "Power-of-two byte size of the instrumentation buffer."
    >,
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "IREE::HAL::HALDialect",
    "IREE::Stream::StreamDialect",
    "IREE::Util::UtilDialect",
  ];
}

def MaterializeResourceCachesPass :
    Pass<"iree-hal-materialize-resource-caches", "mlir::ModuleOp"> {
  let summary = "Materializes cached globals for device resources.";
  let description = [{
    Scans the program for resource lookups such as `hal.executable.lookup` and
    materializes globals initialized on startup. The original lookup ops are
    replaced with global loads of the cached resources.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "IREE::HAL::HALDialect",
    "IREE::Util::UtilDialect",
  ];
}

def ElideRedundantCommandsPass :
    Pass<"iree-hal-elide-redundant-commands", ""> {
  let summary = "Elides stateful command buffer ops that set redundant state.";
  let description = [{
    Identifies sequences of stateful command buffer operations such as
    `hal.command_buffer.push_descriptor_set` that set redundant state that arise
    from trivial conversion from the stateless stream dialect and removes them
    to reduce binary size and runtime overhead.
  }];
  let dependentDialects = [
    "IREE::HAL::HALDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Benchmarking and debugging utilities
//===----------------------------------------------------------------------===//

def DumpExecutableSourcesPass :
    Pass<"iree-hal-dump-executable-sources", "mlir::ModuleOp"> {
  let summary = "Dumps individual hal.executable source listings to the provided path.";
  let description = [{
    Dumps a source listing of each hal.executable and updates the source
    locations in the IR to point at the produced files. This allows for easy
    inspection of each executable prior to translation and gives downstream
    tools that can display source information (Tracy, perf, etc) something more
    useful than the entire original source program.
  }];
  let options = [
    Option<
      "path", "path",
      "std::string", "",
      "File system path to write each executable source MLIR file."
    >,
    Option<
      "prefix", "prefix",
      "std::string", "",
      "String to prefix the written file names with."
    >,
  ];
}

def DumpExecutableBenchmarksPass :
    Pass<"iree-hal-dump-executable-benchmarks", "mlir::ModuleOp"> {
  let summary = "Dumps standalone hal.executable benchmarks to the provided path.";
  let description = [{
    Dumps one MLIR file per hal.executable containing the executable contents
    and the host code required to dispatch them with fake buffers and operands.
    These benchmarks can be run with the `iree-benchmark-module` tool to
    microbenchmark individual dispatches outside of the whole program context.

    The pass can only be run after executable translation but before host code
    conversion as the original stream dialect ops are required to synthesize
    the benchmarks.

    There are many caveats with this approach and it will fail to generate
    benchmarks in many cases such as dynamic shapes, dynamic operands, or
    stateful data dependencies. Users should always prefer to build dedicated
    benchmarks in their origin framework that can be guaranteed to match their
    expectations and use appropriate test data. For example some dispatches may
    produce NaNs or out-of-bounds accesses with the fake data generated by this
    pass and either crash or result in unrepresentative performance.

    In other words: don't blindly expect this pass to do anything but act as a
    starting point for microbenchmarking. Verify the outputs, the benchmarking
    methodology for the particular dispatch, and prepare to do more work. Or
    just author proper benchmarks in the original framework!
  }];
  let options = [
    Option<
      "path", "path",
      "std::string", "",
      "File system path to write each executable benchmark MLIR file."
    >,
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "IREE::HAL::HALDialect",
    "IREE::Util::UtilDialect",
  ];
}

def StripExecutableContentsPass :
    Pass<"iree-hal-strip-executable-contents", "mlir::ModuleOp"> {
  let summary = "Strips executable module contents for reducing IR size during debugging.";
  let description = [{
    A debugging pass for stripping translated executable contents (LLVM dialect,
    SPIR-V dialect, etc) to reduce IR size and noise from the device-only code.
  }];
}

def RepeatDispatchesPass :
    Pass<"iree-hal-repeat-dispatches", ""> {
  let summary = "Repeats each hal.command_buffer.dispatch op one or more times.";
  let description = [{
    Finds all hal.command_buffer.dispatch ops and repeats them the specified
    number of times by cloning them and inserting a barrier. This is extremely
    unreliable and nearly always creates incorrect programs that have wildly
    incorrect end-to-end execution timings. It must only be used when trying to
    profile (via sampling or performance counters) specific dispatches in-situ
    with the additional caveat that cache behavior and dispatch overhead are
    invalid. Do not trust any numbers produced by this method of benchmarking
    without verifying via external tooling.

    This should rarely be used. Prefer instead to build real benchmarks in
    origin frameworks that, for example, use independent data and ensure correct
    execution results (as if you're benchmarking known-incorrect results, are
    you really benchmarking something useful?). Any benchmarking of memory-bound
    operations using this approach will be questionable (such as matmuls, which
    we use this for today... heh ;).
  }];
  let options = [
    Option<
      "repeatCount", "count",
      "unsigned", "1",
      "Number of times to repeat each dispatch (including the original)."
    >,
  ];
  let dependentDialects = [
    "IREE::HAL::HALDialect",
  ];
}

#endif  // IREE_DIALECT_HAL_PASSES
