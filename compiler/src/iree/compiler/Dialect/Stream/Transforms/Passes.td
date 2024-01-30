// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_STREAM_PASSES
#define IREE_DIALECT_STREAM_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Tensor lowering and resource management
//===----------------------------------------------------------------------===//

def ConvertToStreamPass :
    Pass<"iree-stream-conversion", "mlir::ModuleOp"> {
  let summary = "Converts from flow and other input dialects into the stream dialect.";
  let description = [{
    Converts supported input dialects (`flow`, `tensor`, `util`, and various
    upstream dialects like `cf`/`scf`) into the stream dialect and adds
    additional metadata. After conversion all supported operations will act on
    `!stream.resource<*>` types and track resource storage sizes symbolically.

    Though the conversion requires that the program be in an implicitly
    synchronized form (SSA use-def chains on immutable tensor-like objects)
    limited support is available for a subset of the `hal` dialect ops that are
    used on the program ABI boundary for interoperating with external buffers
    and fences. These ops, such as `hal.tensor.import` and `hal.tensor.barrier`,
    will be converted to their `stream` dialect form and preserve the implicit
    synchronization guaranteeds required for proper analysis.

    Dispatched executables are allowed to be in one of the supported input
    dialects (like `flow.executable`), already be lowered into
    `stream.executable` ops, or be the final `hal.executable` ops. The amount of
    analysis and optimization that can be performed on `hal.executable` ops is
    limited and no retargetability is available when directly providing them.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "IREE::Stream::StreamDialect",
    "IREE::Util::UtilDialect",
  ];
}

def EncodeHostTensorsPass :
    Pass<"iree-stream-encode-host-tensors", ""> {
  let summary = "Encodes tensors into storage formats based on affinity and target support.";
  let description = [{
    Encodes `stream.tensor.*` ops on tensor-like objects into encoding-erased
    asynchronous `stream.async.*` ops and resolves (if possible) symbolic
    encoding ops such as `stream.tensor.sizeof` into their final values.

    Dense tensors are trivially lowerable but other encodings may require
    additional transfer and dispatch operations. For example, computing the
    minimal fixed storage size of an unblocked sparse tensor may require the
    pass to insert a dispatch that traverses the index tables to discover how
    many elements are present while a blocked sparse tensor may be able to
    resolve to a simpler calculation based solely on the number of fixed-size
    blocks.

    Sub-byte tensor types or those with non-trivial packing/encoding are also
    resolved here such as by calculating that a `tensor<Nxi4>` requires `N*4/8`
    bytes of storage. Some operations like slicing subranges of elements without
    known alignment may also require additional transfer and dispatch operations
    to preserve behavior while lowering into the type-erased forms.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::complex::ComplexDialect",
    "IREE::Stream::StreamDialect",
  ];
}

// TODO(benvanik): remove this pass and instead specify the encoding with
// attributes such that codegen can deterministically match the host behavior.
// This pass only exists today because we don't have a way to ensure all codegen
// backends do the "right" (or at least consistent) thing beyond convention.
def EncodeDeviceTensorsPass :
    Pass<"iree-stream-encode-device-tensors", ""> {
  let summary = "Encodes tensors into binary formats based on affinity and target support.";
  let description = [{
    Encodes `stream.binding.*` ops on tensor-like objects while handling packing
    and encoding as with the `iree-stream-encode-host-tensors` pass but within
    executables.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::complex::ComplexDialect",
    "IREE::Stream::StreamDialect",
  ];
}

def MaterializeBuiltinsPass :
    Pass<"iree-stream-materialize-builtins", "mlir::ModuleOp"> {
  let summary = "Materialize dispatches to builtin executables where required.";
  let description = [{
    Materializes dispatches to builtin executables when operations are not
    supported by lower layers of the stack. For example, an `stream.async.fill`
    op with an i64 pattern will be converted to a `stream.async.dispatch` of
    `__builtin_fill_i64` and the `stream.executable` will be merged into the
    module.

    Though in many cases this kind of emulation happens more naturally
    during the global optimization phase of the compiler and is more efficient
    as there is opportunity for fusion into existing dispatches sometimes it's
    not possible to statically know at the time such phases operate whether the
    operations are required and this catches those cases.

    Since it's often less efficient to materialize a builtin dispatch instead of
    having fused it with others or to have been able to make use of a pure
    transfer operation the materialization is seen as a pessimization that
    should be avoided. Generally builtins are only added to ensure correct
    execution and are not used to try to optimize the program.
  }];
  let dependentDialects = [
    // We need to include all dialects that the builtin modules use.
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect",
    "IREE::Flow::FlowDialect",
    "IREE::Stream::StreamDialect",
    "IREE::Util::UtilDialect",
  ];
}

def MaterializeCopyOnWritePass :
    Pass<"iree-stream-materialize-copy-on-write", ""> {
  let summary = "Materializes copy-on-write (🐄) behavior as explicit ops.";
  let description = [{
    Materializes copy-on-write behavior in the program by analyzing usage of
    `!stream.resource<*>` types by stream ops. Prior to this pass resources are
    implicitly immutable and follow SSA semantics while after the pass any cases
    where such implicit behavior is assumed has been expanded into appropriate
    clones of the resources or rematerialization of source values.

    As an example attempting to update the same immutable tensor will result in
    the original tensor being cloned such that each update sees a unique copy:

    ```mlir
    %init = stream.async.splat %c0
    %fill0 = stream.async.fill %c123, %init[...] -> %init
    %fill1 = stream.async.fill %c456, %init[...] -> %init
    ->
    %init = stream.async.splat %c0
    %clone0 = stream.async.clone %init
    %fill0 = stream.async.fill %c123, %clone0[...] -> %clone0
    %clone1 = stream.async.clone %init
    %fill1 = stream.async.fill %c456, %clone1[...] -> %clone1
    ```

    A subsequently run `iree-stream-elide-async-copies` pass can often elide or
    simplify some of the copies such as above where splatting and then cloning
    the splat twice is not required. The passes are split to allow for simple
    local analysis here and for the elision pass to catch input that may already
    have contained unneeded copies.
  }];
  let dependentDialects = [
    "IREE::Stream::StreamDialect",
  ];
}

def ElideAsyncCopiesPass :
    Pass<"iree-stream-elide-async-copies", "mlir::ModuleOp"> {
  let summary = "Elides copies when they are not performing meaningful work.";
  let description = [{
    Performs whole-program analysis to identify copies that are not required for
    program correctness or enabling concurrency, such as clones of the last use
    of a value. This eliminates copies both from input programs and those
    materialized by the `iree-stream-materialize-copy-on-write` pass.
  }];
}

def EmplaceAllocationsPass :
    Pass<"iree-stream-emplace-allocations", ""> {
  let summary = "Emplaces transient tensor allocations to remove copies.";
  let description = [{
    Identifies opportunities for placing operation results directly into
    existing resources when analysis determines it is safe to do so. This is
    intended to run after copy-on-write materialization when such analysis can
    be performed local to the operations. The common case this helps with is
    insertions of produced results into larger resources such as performed by
    tensor concatenation.
  }];
  let dependentDialects = [
    "IREE::Stream::StreamDialect",
  ];
}

def RefineUsagePass :
    Pass<"iree-stream-refine-usage", "mlir::ModuleOp"> {
  let summary = "Refines resource usage bits and inserts transfers where appropriate.";
  let description = [{
    Performs whole-program analysis to assign lifetime and usage attributes to
    `!stream.resource<*>` types that have not yet been fixed. Resources are
    tracked across global loads/stores, function calls, control flow, and
    operations acting on them to determine how they are used (transfers, host
    staging, constants, etc). Upon completion all resources have a fixed
    lifetime and any new resources introduced into the program with an
    unspecified lifetime (`!stream.resource<*>`) will require the pass to be
    run again prior to continued lowering.
  }];
  let dependentDialects = [
    "IREE::Stream::StreamDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Stream formation and scheduling
//===----------------------------------------------------------------------===//

def ScheduleExecutionPass :
    InterfacePass<"iree-stream-schedule-execution", "mlir::CallableOpInterface"> {
  let summary = "Identifies and groups asynchronous operations into executable regions within function-like regions.";
  let description = [{
    Partitions `stream.async.*` operations into execution regions that are
    executed atomically on a single device. The partitioning algorithm uses the
    operations being performed and the affinity assigned to them (if any) to
    determine which are allowed to execute together and is allowed to produce
    any number of partitions to cover the workload. Original executing ordering
    is preserved by the resulting `stream.async.execute` operations using
    `!stream.timepoint` to maintain explicit SSA use-def-based wait-on and
    signal-to behavior. Scheduling may insert host waits on device work that can
    be later avoided by timepoint propagation and elision.
  }];
  let dependentDialects = [
    "IREE::Stream::StreamDialect",
  ];
}

def ScheduleConcurrencyPass :
    InterfacePass<"iree-stream-schedule-concurrency", "mlir::CallableOpInterface"> {
  let summary = "Identifies and groups asynchronous operations within executable regions that can run concurrently and groups them into streams.";
  let description = [{
    Partitions operations that can execute concurrently within
    `stream.async.execute` regions into a tree with `stream.async.concurrent`
    ops indicating two or more operations that are allowed to execute
    concurrently even if resources may alias.
  }];
  let dependentDialects = [
    "IREE::Stream::StreamDialect",
  ];
}

def PropagateTimepointsPass :
    Pass<"iree-stream-propagate-timepoints", "mlir::ModuleOp"> {
  let summary = "Materializes timepoints and sinks them to consumers throughout the whole program.";
  let description = [{
    Propagates `!stream.timepoint` values across the whole program in order to
    avoid host-device and device-device waits where possible without changing
    correct execution ordering. For example a host wait on a timepoint via a
    `stream.timepoint.await` op guarding a resource passed to a function call
    will be changed to pass the timepoint to the callee and have the wait occur
    in there thus allowing it to be chained with subsequent device operations
    that may consume the resource. Such propagation happens across global stores
    and loads, function calls, and control flow.
  }];
  let dependentDialects = [
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "IREE::Stream::StreamDialect",
    "IREE::Util::UtilDialect",
  ];
}

def ElideTimepointsPass :
    Pass<"iree-stream-elide-timepoints", "mlir::ModuleOp"> {
  let summary = "Elides timepoints that are known to be covered by dependent timepoints.";
  let description = [{
    Elides waits on timepoints that are known to be reached by a dependent
    timepoint. Errs on the side of preserving timepoints if analysis can't
    guarantee that a particular wait is covered.

    Example:
    ```mlir
      %timepoint0 = ...
      %timepoint1 = ... await(%timepoint0)
      %timepoint2 = stream.timepoint.join max(%timepoint0, %timepoint1)
    ->
      %timepoint0 = ...
      %timepoint1 = ... await(%timepoint0)
      %timepoint2 = stream.timepoint.join max(%timepoint1)
    -> (canonicalization) ->
      %timepoint0 = ...
      %timepoint1 = ... await(%timepoint0)
      %timepoint2 = %timepoint1
    ```
  }];
  let dependentDialects = [
    "IREE::Stream::StreamDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Allocation and command issuing
//===----------------------------------------------------------------------===//

def ScheduleAllocationPass :
    Pass<"iree-stream-schedule-allocation", "mlir::ModuleOp"> {
  let summary = "Allocates resources and converts to explicit stream commands.";
  let description = [{
    Schedules allocation of resources and converts the program from the implicit
    resource management scheme of the `stream.async.*` ops into the explicit
    resource management scheme of the `stream.cmd.*` ops. After conversion the
    program cannot be raised as aliasing is introduced and local liveness ranges
    are erased.

    Allocations are performed by asynchronous operations like
    `stream.resource.alloca` (and the matching `stream.resource.dealloca`) and
    sequenced in the device timeline by `!stream.timepoint` values.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "IREE::Stream::StreamDialect",
  ];
}

def PackConstantsPass :
    InterfacePass<"iree-stream-pack-constants", "mlir::CallableOpInterface"> {
  let summary = "Packs and allocates backing storage for fused constant resources.";
  let description = [{
    Packs slices of `stream.resource.constants` ops and materializes operations
    to initialize them based on their contents. Embedded constants are turned
    into inline host buffers with operations that try to map them into device
    memory or perform device-accelerated file I/O asynchronously with other
    initialization code. Parameters are expanded based on the the device memory
    model to be loads (which may allow mapping memory on devices with unified
    memory) or gathers (that require allocation and staging on devices with
    discrete memory).
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "IREE::Stream::StreamDialect",
    "IREE::Util::UtilDialect",
  ];
}

def LayoutSlicesPass :
    InterfacePass<"iree-stream-layout-slices", "mlir::CallableOpInterface"> {
  let summary = "Lays out packed slices and produces arithmetic required for all offsets.";
  let description = [{
    Performs target-aware layout of packed slices in `stream.resource.pack` ops.
    Alignment, padding, and static/dynamic offset calculation of the slices
    within larger allocated resources happens with awareness of both the
    resource slices being packed and where they will be consumed.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "IREE::Stream::StreamDialect",
    "IREE::Util::UtilDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Memoization
//===----------------------------------------------------------------------===//

// TODO(benvanik): outline streams (ala dispatch regions).
// TODO(benvanik): deduplicate outlined streams.

//===----------------------------------------------------------------------===//
// Dispatch optimization
//===----------------------------------------------------------------------===//

def FoldUniformOperandsPass :
    Pass<"iree-stream-fold-uniform-operands", "mlir::ModuleOp"> {
  let summary = "Folds redundant and uniformly constant dispatch operands.";
  let description = [{
    Performs whole-program analysis to find all dispatch sites to each dispatch
    and fold or inline operands that are uniformly passed. For example if
    multiple dispatch sites pass the same SSA value for two operands (even if
    dynamically computed) they will be folded into a single value, and if
    multiple dispatch sites pass the same constant value for the same operand
    the constant value will be inlined and the operand removed.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
  ];
}

def FuseDispatchBindingsPass :
    Pass<"iree-stream-fuse-dispatch-bindings", "mlir::ModuleOp"> {
  let summary = "Fuses bindings to the same underlying storage to reduce binding count.";
  let description = [{
    Erases dispatch binding subranges and attempts to fuse bindings that
    originate from the same resources across all dispatch sites.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "IREE::Stream::StreamDialect",
  ];
}

def SpecializeDispatchesPass :
    Pass<"iree-stream-specialize-dispatches", "mlir::ModuleOp"> {
  let summary = "Specializes executables by inlining/fusing operands based on dispatch sites.";
  let description = [{
    Reduces the number of operands passed to dispatches by identifying common
    patterns at dispatch sites across the program that can be compressed into
    unique dispatch site identifiers. For example, if a dispatch takes several
    operands that are [0, 1, ...] at one dispatch site and [10, 11, ...] at
    another the dispatch will be changed to take a single value indicating which
    set of operands to use and the operands themselves will be placed into a
    lookup table within the dispatch.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def AnnotateDispatchArgumentsPass :
    Pass<"iree-stream-annotate-dispatch-arguments", "mlir::ModuleOp"> {
  let summary = "Annotates dispatch arguments with potential values derived from dispatch sites.";
  let description = [{
    Uses data flow analysis to identify potential value sets and alignments
    (or divisibility) for dispatch operands and bindings. Upon successful
    analysis the dispatch executables are annotated such that further lowering
    in codegen has the analysis results locally without needing to inspect the
    entire program.

    Operands are annotated with `stream.values` and/or `stream.alignment`
    attributes indicating all known constant values at all dispatch sites and/or
    their divisibility. `stream.values` is only added when only statically-known
    values are passed and `stream.alignment` is added in cases where some
    minimum divisibility is identified even if the values are dynamic (such as
    all values passed in going through `util.align` or `arith.muli` prior).

    Bindings are annotated with `stream.alignment` attributes indicating their
    base alignment prior to the offset specified on the binding op itself. Note
    that just because the base alignment is some value does not mean the offset
    is always known to be aligned in the same way.
  }];
}

def PackDispatchOperandsPass :
    Pass<"iree-stream-pack-dispatch-operands", "mlir::ModuleOp"> {
  let summary = "Packs stream dispatch operands into i32 push constants.";
  let description = [{
    Packs dispatch operands (such as `i2`, `i64`, `complex<f32>`, etc) into the
    required `i32` values on the dispatch ABI. May optimize multiple wider
    bit-width operands with known ranges or alignments into or across fewer
    operands to reduce the total operand count.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::complex::ComplexDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

def DumpStatisticsPass :
    Pass<"iree-stream-dump-statistics", "mlir::ModuleOp"> {
  let summary = "Dumps stream dialect usage information to a file.";
  let options = [
    Option<
      "outputFormat", "output-format",
      "IREE::Stream::DumpOutputFormat",
      "IREE::Stream::DumpOutputFormat::Pretty",
      "Specifies the output format to produce.",
      [{::llvm::cl::values(
        clEnumValN(IREE::Stream::DumpOutputFormat::Pretty, "pretty", "Human-readable pretty printed output."),
        clEnumValN(IREE::Stream::DumpOutputFormat::Verbose, "verbose", "Pretty printed output with additional IR."),
        clEnumValN(IREE::Stream::DumpOutputFormat::CSV, "csv", "Comma separated values.")
      )}]
    >,
    Option<
      "outputFile", "output-file",
      "std::string",
      /*default=*/"std::string()",
      "File path to write to; or `` for stderr or `-` for stdout."
    >,
  ];
}

def VerifyInputPass :
    Pass<"iree-stream-verify-input", "mlir::ModuleOp"> {
  let summary = "Verifies that input dialects are supported by the streams dialect.";
}

def VerifyLoweringToTensorsPass :
    Pass<"iree-stream-verify-lowering-to-tensors", "mlir::ModuleOp"> {
  let summary = "Verifies that input dialects are converted to stream.tensor.* ops.";
}

def VerifyLoweringToAsyncResourcesPass :
    Pass<"iree-stream-verify-lowering-to-async-resources", "mlir::ModuleOp"> {
  let summary = "Verifies that all stream.tensor.* ops and types are fully lowered to streama.async.* resource ops.";
}

def VerifyLoweringToAsyncPass :
    Pass<"iree-stream-verify-lowering-to-async", "mlir::ModuleOp"> {
  let summary = "Verifies that all stream.tensor.* ops and types are fully lowered to stream.async.* ops and all resources have an assigned lifetime.";
}

def VerifyAsyncAccessRangesPass :
    Pass<"iree-stream-verify-async-access-ranges", "mlir::ModuleOp"> {
  let summary = "Verifies that stream.async.* access ranges are in bounds where possible.";
}

def VerifyLoweringToCmdPass :
    Pass<"iree-stream-verify-lowering-to-cmd", "mlir::ModuleOp"> {
  let summary = "Verifies that all stream.async.* ops and types are fully lowered to stream.cmd.* ops.";
}

#endif  // IREE_DIALECT_STREAM_PASSES
