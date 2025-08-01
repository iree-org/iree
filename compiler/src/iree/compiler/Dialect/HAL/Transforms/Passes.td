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

def AssignLegacyTargetDevicesPass :
    Pass<"iree-hal-assign-legacy-target-devices", "mlir::ModuleOp"> {
  let summary = "Assigns the HAL devices the module will target to the given list of targets.";
  let description = [{
    Assigns target HAL devices to the module based on the given list.
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetRegistryRef", "",
      "Target registry containing the list of available devices and backends."
    >,
    ListOption<
      "targetBackends", "targetBackends",
      "std::string",
      "List of target backends to assign as device targets."
    >,
  ];
  let dependentDialects = [
    "IREE::HAL::HALDialect",
  ];
}

def AssignTargetDevicesPass :
    Pass<"iree-hal-assign-target-devices", "mlir::ModuleOp"> {
  let summary = "Assigns the HAL devices the module will target to the given list of target specifications.";
  let description = [{
    Assigns target HAL devices to the module based on the given list of target
    specifications.

    Targets can be specified in several ways depending on whether there are
    multiple devices, named devices, or devices imported from external files.
    Human-friendly device aliases can be used as shorthand for
    `IREE::HAL::TargetDevice` implementations providing their own configuration.
    The aliases are identical to those used by `#hal.device.alias<>`.

    If multiple targets are specified they will be available as multiple
    distinct devices. A single device may select from one or more targets such
    that the first enumerated that matches at runtime will be selected. For
    example a `gpu` device may select between CUDA, HIP, or Vulkan at runtime
    based on what kind of device the user has and what HAL implementations were
    compiled into the runtime.

    Examples using the canonical flag:
    ```mlir
    // Two devices, one the local host device and the other a Vulkan device:
    --iree-hal-target-device=local
    --iree-hal-target-device=vulkan

    // One device selecting between Vulkan if available and otherwise use the
    // local host device:
    --iree-hal-target-device=vulkan,local

    // Two CUDA devices selected by runtime ordinal; at runtime two --device=
    // flags are required to configure both devices:
    --iree-hal-target-device=cuda[0]
    --iree-hal-target-device=cuda[1]

    // A fully-defined target specification:
    --iree-hal-target-device=#hal.device.target<"cuda", {...}, [#hal.executable.target<...>]>

    // Named device for defining a reference by #hal.device.promise<@some_name>:
    --iree-hal-target-device=some_name=vulkan
    ```
  }];
  let options = [
    ListOption<
      "targetDevices", "targetDevices",
      "std::string",
      "List of target device specifications."
    >,
  ];
  let dependentDialects = [
    "IREE::HAL::HALDialect",
  ];
}

def MaterializeTargetDevicesPass :
    Pass<"iree-hal-materialize-target-devices", "mlir::ModuleOp"> {
  let summary = "Materializes global device handles based on a `hal.device.targets` spec.";
  let description = [{
    Materializes global `!hal.device` ops for the devices specified by the
    `hal.device.targets` attribute on the module. An optional default device can
    be specified to assign to ops that do not have a default device specified.
  }];
  let options = [
    Option<
      "defaultDevice", "defaultDevice",
      "std::string", "",
      "Which device is considered the default when no device affinity is specified."
    >,
  ];
  let dependentDialects = [
    "IREE::HAL::HALDialect",
    "IREE::Util::UtilDialect",
  ];
}

def ResolveDevicePromisesPass :
    Pass<"iree-hal-resolve-device-promises", "mlir::ModuleOp"> {
  let summary = "Resolves `#hal.device.promise` attributes to their devices.";
  let description = [{
    Resolves promised device affinities to the materialized device globals that
    were promised. Verifies that all promises are resolved.
  }];
  let dependentDialects = [
    "IREE::HAL::HALDialect",
  ];
}

def ResolveDeviceAliasesPass :
    Pass<"iree-hal-resolve-device-aliases", "mlir::ModuleOp"> {
  let summary = "Resolves `#hal.device.alias` attributes to their expanded configurations.";
  let description = [{
    Resolves device aliases to the concrete targets using defaults, flags, and
    registered device configurations.
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetRegistryRef", "",
      "Target registry containing the list of available devices and backends."
    >,
  ];
  let dependentDialects = [
    "IREE::HAL::HALDialect",
  ];
}

def VerifyDevicesPass :
    Pass<"iree-hal-verify-devices", "mlir::ModuleOp"> {
  let summary = "Verifies that all devices can be targeted with the available compiler plugins.";
  let description = [{
    Verifies that `#hal.device.target` and `#hal.executable.target` attributes
    reference targets that are registered with the compiler.
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetRegistryRef", "",
      "Target registry containing the list of available devices and backends."
    >,
  ];
}

def InlineMemoizeRegionsPass :
    Pass<"iree-hal-inline-memoize-regions", ""> {
  let summary = "Inlines `hal.device.memoize` regions into their parent region.";
  let description = [{
    Inlines any `hal.device.memoize` ops into their parent region and removes
    the op. This prevents memoization and has the same behavior as having never
    formed the memoization regions.
  }];
  let dependentDialects = [
    "mlir::cf::ControlFlowDialect",
    "IREE::HAL::HALDialect",
  ];
}

def OutlineMemoizeRegionsPass :
    Pass<"iree-hal-outline-memoize-regions", "mlir::ModuleOp"> {
  let summary = "Outlines `hal.device.memoize` regions and creates global resources.";
  let description = [{
    Outlines any `hal.device.memoize` ops in the module by creating functions
    and per-device globals with initializers.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::scf::SCFDialect",
    "IREE::HAL::HALDialect",
    "IREE::Util::UtilDialect",
  ];
}

def MemoizeDeviceSelectionPass :
    Pass<"iree-hal-memoize-device-selection", "mlir::ModuleOp"> {
  let summary = "Finds !hal.device selection ops and creates variables initialized on startup.";
  let description = [{
    Finds all `hal.*.select`-related ops that are hoistable and moves them
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

def ResolveTopologyQueriesPass :
    Pass<"iree-hal-resolve-topology-queries", "mlir::ModuleOp"> {
  let summary = "Try to statically resolve ops using topology information.";
  let description = [{
    Attempts to statically resolve `hal.allocator.resolve_memory_properties` ops
    that have the `hal.device.optimal` attribute. This is either done by checking the
    topology information for transparent access between devices or by using the
    DeviceAnalysis to check if two affinities refer to the same device. Otherwise,
    the operation is left unmodified to be resolved at runtime.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "IREE::HAL::HALDialect",
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
    Runs a nested pipeline on each executable to attach target-specific
    configuration information to variants.
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetRegistryRef", "",
      "Target registry containing the list of available devices and backends."
    >,
  ];
}

def ConfigureTargetExecutableVariantsPass :
    Pass<"iree-hal-configure-target-executable-variants", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Configures hal.executable.variant ops for the specified target backend.";
  let description = [{
    Attaches target-specific configuration information to a variant controlling
    how code generation operates.
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetRegistryRef", "",
      "Target registry containing the list of available devices and backends."
    >,
    Option<
      "target", "target",
      "std::string", "",
      "Target backend name whose executable variants will be configured by this pass."
    >,
  ];
}

def TranslateAllExecutablesPass :
    Pass<"iree-hal-translate-all-executables", "IREE::HAL::ExecutableOp"> {
  let summary = "Translates hal.executable ops via a nested translation pipeline.";
  let description = [{
    Runs a nested pipeline on each executable to translate its variants from
    their generic MLIR dialects (such as `linalg`) to their target-specific
    dialects (`llvm`, `spirv`, etc).
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetRegistryRef", "",
      "Target registry containing the list of available devices and backends."
    >,
  ];
}

def TranslateTargetExecutableVariantsPass :
    Pass<"iree-hal-translate-target-executable-variants", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Translates hal.executable.variant ops for the specified target backend.";
  let description = [{
    Translates an executable variant for a specific target from its generic
    MLIR dialects (such as `linalg`) to the target-specific dialects (`llvm`,
    `spirv`, etc).
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetRegistryRef", "",
      "Target registry containing the list of available devices and backends."
    >,
    Option<
      "target", "target",
      "std::string", "",
      "Target backend name whose executable variants will be translated by this pass."
    >,
  ];
}

def HoistExecutableObjectsPass :
    Pass<"iree-hal-hoist-executable-objects", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Hoists local executable object annotations to the parent `hal.executable.variant`.";
  let description = [{
    Finds all `hal.executable.objects` attrs on all ops within an executable
    inner module and moves them to the parent `hal.executable.variant` op.
  }];
}

def PruneExecutablesPass :
    Pass<"iree-hal-prune-executables", "mlir::ModuleOp"> {
  let summary = "Prunes executable variants and exports that are not referenced.";
  let description = [{
    Prunes executable variants and exports that are not referenced in the
    module. This is intended to be run late in the pipeline where no new
    dispatches will be inserted that may require the variants or exports that it
    removes.
  }];
}

def LinkAllExecutablesPass :
    Pass<"iree-hal-link-all-executables", "mlir::ModuleOp"> {
  let summary = "Links hal.executable ops into one or more hal.executable ops.";
  let description = [{
    Runs a nested pipeline to link multiple `hal.executable` ops together if the
    target backend the executables are used with desires.
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetRegistryRef", "",
      "Target registry containing the list of available devices and backends."
    >,
  ];
}

def LinkTargetExecutablesPass :
    Pass<"iree-hal-link-target-executables", "mlir::ModuleOp"> {
  let summary = "Links executables for the specified target backend.";
  let description = [{
    Links together multiple `hal.executable` ops for the given target backend if
    desired. Linking allows for intra-module deduplication and amortization of
    startup time, code size, and runtime overheads that come from managing
    multiple hundreds/thousands of executables.
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetRegistryRef", "",
      "Target registry containing the list of available devices and backends."
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

def SerializeAllExecutablesPass :
    Pass<"iree-hal-serialize-all-executables", "IREE::HAL::ExecutableOp"> {
  let summary = "Converts hal.executable.variants to one or more hal.executable.binary ops.";
  let description = [{
    Runs a nested pipeline on each executable to serialize its variants from
    their low-level MLIR dialects (such as `llvm`, `spirv`, etc) to their
    target-specific object format (static/shared libraries, SPIR-V, etc).
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetRegistryRef", "",
      "Target registry containing the list of available devices and backends."
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
    Serializes variants for the target backend from their low-level MLIR
    dialects (such as `llvm`, `spirv`, etc) to their target-specific object
    format (static/shared libraries, SPIR-V, etc).
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetRegistryRef", "",
      "Target registry containing the list of available devices and backends."
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

def InitializeDevicesPass :
    Pass<"iree-hal-initialize-devices", "mlir::ModuleOp"> {
  let summary = "Initializes global device handles based on their specification.";
  let description = [{
    Initializes each global `!hal.device` based on the specification attribute
    by building initializers that enumerate and select the appropriate device.
  }];
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetRegistryRef", "",
      "Target registry containing the list of available devices and backends."
    >,
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "IREE::HAL::HALDialect",
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

def AnnotateTargetDevicesPass :
    Pass<"iree-hal-annotate-target-devices", "mlir::ModuleOp"> {
  let summary = "Annotates device values with their analyzed potential target devices.";
  let description = [{
    Uses data flow analysis to identify potential targets for `!hal.device`
    values. This can be used to diagnose device analysis issues and otherwise
    does not impact compilation.
  }];
}

def CaptureExecutableSourcesPass :
    Pass<"iree-hal-capture-executable-sources", "mlir::ModuleOp"> {
  let summary = "Captures individual hal.executable.variant source listings and embeds them in the IR.";
  let description = [{
    Captures a source listing of each hal.executable.variant and attaches the
    source to the variant embedded in the IR. Entry points are assigned
    locations in the IR relative to the captured source.
  }];
  let options = [
    Option<
      "stage", "stage",
      "std::string", "\"source\"",
      "Name used to indicate what stage of compilation is captured."
    >,
  ];
}

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
