// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_STREAM_BASE
#define IREE_DIALECT_STREAM_BASE

include "iree/compiler/Dialect/Stream/IR/StreamInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// IREE stream modeling dialect
//===----------------------------------------------------------------------===//

def Stream_Dialect : Dialect {
  let name = "stream";
  let cppNamespace = "::mlir::iree_compiler::IREE::Stream";

  let summary = [{
    A dialect designed to model execution partitioning and scheduling.
  }];
  let description = [{
    The stream dialect is designed to take tensor programs and convert them to
    explicitly scheduled asynchronous programs. This includes placing ops on
    specific targets, partitioning the work between the targets, scheduling the
    work for concurrency, and encoding tensors into target-specific resources.

    ```
    +--------+    +----------+    +-------+
    | flow.* | -> | stream.* | -> | hal.* |
    +--------+    +----------+    +-------+
    ```

    This sits in-between the `flow` and `hal` dialects.

    * `flow` models tensor programs by separating work into dispatchable
      functions in order to isolate the main host program data flow and the
      dense tensor compute operations.

    * `stream` models explicitly scheduled asynchronous programs by partitioning
      the dispatchable work, specifying target affinities, encoding tensors into
      target-specific forms, and scheduling the work to run concurrently.

    * `hal` models a low-level hardware abstraction layer used to manage
      buffers and issue asynchronous work across a variety of device types. The
      dialect is largely 1:1 with the IREE HAL C API.

    Transforms in the dialect lower tensor values into opaque resources with the
    goal of ensuring no tensors survive in the IR. At entry `stream.tensor.*`
    ops are used to capture the source tensor encoding information (data type,
    shapes, etc) and then lowered into `stream.async.*` ops that model the
    asynchronous workloads on the opaque resources. The asynchronous operations
    are then partitioned, allocated, and scheduled for execution using the
    `stream.cmd.*` ops.

    It's intended that after transformation through the stream dialect the
    program is ready for execution on an abstract machine. At this level of
    representation buffers have still not been allocated and devices are not
    yet resolved, however the information captured in the `stream` IR allows
    such operations to be done trivially. To this end all ops carry the symbolic
    size of the resources on which they operate as well as the lifetime of the
    resources they are acting upon. This manifests in the usage of the
    `!stream.resource` type:

    ```mlir
    // Unresolved lifetime (resolved during the iree-stream-refine-usage pass):
    !stream.resource<*>
    // An externally managed value (passed in via the program API).
    !stream.resource<external>
    // A staging buffer for uploads/downloads.
    !stream.resource<staging>
    // A short-lived value that is used across streams.
    !stream.resource<transient>
    // A long-lived value that persists across streams in globals.
    !stream.resource<variable>
    // An immutable value that persists for the duration of the program.
    !stream.resource<constant>
    ```

    Operations using resources carry the size of all operand result resources:

    ```mlir
    // %update (40 bytes) is being inserted into %target (296 bytes).
    // Can be dynamic values such as those originating from dynamic dimensions.
    %13 = stream.async.update %update, %target[%c256 to %c296] :
        !stream.resource<transient>{%c40} ->
        %target as !stream.resource<transient>{%c296}
    ```

    Once all `stream.async.*` work is moved into executable regions (such as
    `stream.async.execute`) `!stream.timepoint` values are used to sequence
    the execution. These timepoints represent some point in time where all
    execution up to that timepoint has completed and any results that were
    produced by the execution are available for use. Attempting to use the
    resources before their corresponding timepoint has been reached will lead
    to undefined behavior. The benefit of this is that after timepoints are
    established in the IR it's possible to induce aliasing of resources without
    breaking execution correctness.
  }];

  let dependentDialects = [
    "mlir::complex::ComplexDialect",
    "IREE::Util::UtilDialect",
  ];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base stream dialect op classes
//===----------------------------------------------------------------------===//

class Stream_Op<string mnemonic, list<Trait> traits = []> :
    Op<Stream_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Stream dialect types
//===----------------------------------------------------------------------===//

def Stream_PrimitiveType : AnyTypeOf<[Index, AnyInteger, AnyFloat, AnyComplex]>;
def Stream_FillPatternType : AnyTypeOf<[I8, I16, I32, I64]>;
def Stream_NativeFillPatternType : AnyTypeOf<[I8, I16, I32]>;

def Stream_Offset : TypeAlias<Index>;
def Stream_Size : TypeAlias<Index>;

def Stream_Tensor : TypeAlias<AnyRankedTensor>;
def Stream_Dim : TypeAlias<Index>;
def Stream_ShapeDynamicDims : Variadic<Stream_Dim>;

def Stream_IndexAttr : Util_IndexAttrBase<"index">;
def Stream_IndexArrayAttr : TypedArrayAttrBase<Stream_IndexAttr,
                                               "index array attribute"> {
  let constBuilderCall = "$_builder.getIndexArrayAttr($0)";
}

def Stream_ExecutableRefAttr : Util_AliasedSymbolRefAttr;
def Stream_GlobalRefAttr : Util_AliasedSymbolRefAttr;
def Stream_GlobalPtr : Util_AnyPtrOf<[Stream_Tensor, Stream_PrimitiveType]>;

//===----------------------------------------------------------------------===//
// Stream attributes
//===----------------------------------------------------------------------===//

// NOTE: the collective attributes here are replicated from the HAL layer as
// the HAL values are stable across the runtime ABI and a (potential) subset of
// what we may be able to model in the stream dialect for other targets.

def Stream_CollectiveKind_AllGather : I32EnumAttrCase<"AllGather", 0, "all_gather">;
def Stream_CollectiveKind_AllReduce : I32EnumAttrCase<"AllReduce", 1, "all_reduce">;
def Stream_CollectiveKind_AllToAll : I32EnumAttrCase<"AllToAll", 2, "all_to_all">;
def Stream_CollectiveKind_Broadcast : I32EnumAttrCase<"Broadcast", 3, "broadcast">;
def Stream_CollectiveKind_Reduce : I32EnumAttrCase<"Reduce", 4, "reduce">;
def Stream_CollectiveKind_ReduceScatter : I32EnumAttrCase<"ReduceScatter", 5, "reduce_scatter">;
def Stream_CollectiveKind_Send : I32EnumAttrCase<"Send", 6, "send">;
def Stream_CollectiveKind_Recv : I32EnumAttrCase<"Recv", 7, "recv">;
def Stream_CollectiveKind_SendRecv : I32EnumAttrCase<"SendRecv", 8, "send_recv">;
def Stream_CollectiveKindAttr :
    I32EnumAttr<"CollectiveKind", "valid CollectiveKind", [
      Stream_CollectiveKind_AllGather,
      Stream_CollectiveKind_AllReduce,
      Stream_CollectiveKind_AllToAll,
      Stream_CollectiveKind_Broadcast,
      Stream_CollectiveKind_Reduce,
      Stream_CollectiveKind_ReduceScatter,
      Stream_CollectiveKind_Send,
      Stream_CollectiveKind_Recv,
      Stream_CollectiveKind_SendRecv,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Stream";
}

def Stream_CollectiveReductionOp_None : I32EnumAttrCase<"None", 0, "none">;
def Stream_CollectiveReductionOp_ReductionSum : I32EnumAttrCase<"ReductionSum", 1, "sum">;
def Stream_CollectiveReductionOp_ReductionProduct : I32EnumAttrCase<"ReductionProduct", 2, "product">;
def Stream_CollectiveReductionOp_ReductionMinimum : I32EnumAttrCase<"ReductionMinimum", 3, "minimum">;
def Stream_CollectiveReductionOp_ReductionMaximum : I32EnumAttrCase<"ReductionMaximum", 4, "maximum">;
def Stream_CollectiveReductionOp_ReductionAverage : I32EnumAttrCase<"ReductionAverage", 5, "average">;
def Stream_CollectiveReductionOpAttr :
    I32EnumAttr<"CollectiveReductionOp", "valid CollectiveReductionOp", [
      Stream_CollectiveReductionOp_None,
      Stream_CollectiveReductionOp_ReductionSum,
      Stream_CollectiveReductionOp_ReductionProduct,
      Stream_CollectiveReductionOp_ReductionMinimum,
      Stream_CollectiveReductionOp_ReductionMaximum,
      Stream_CollectiveReductionOp_ReductionAverage,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::Stream";
}

def Stream_CollectiveElementType_Sint8 : I32EnumAttrCase<"Sint8", 0, "si8">;
def Stream_CollectiveElementType_Uint8 : I32EnumAttrCase<"Uint8", 1, "ui8">;
def Stream_CollectiveElementType_Sint16 : I32EnumAttrCase<"Sint16", 2, "si16">;
def Stream_CollectiveElementType_Uint16 : I32EnumAttrCase<"Uint16", 3, "ui16">;
def Stream_CollectiveElementType_Sint32 : I32EnumAttrCase<"Sint32", 4, "si32">;
def Stream_CollectiveElementType_Uint32 : I32EnumAttrCase<"Uint32", 5, "ui32">;
def Stream_CollectiveElementType_Sint64 : I32EnumAttrCase<"Sint64", 6, "si64">;
def Stream_CollectiveElementType_Uint64 : I32EnumAttrCase<"Uint64", 7, "ui64">;
def Stream_CollectiveElementType_Float16 : I32EnumAttrCase<"Float16", 8, "f16">;
def Stream_CollectiveElementType_Float32 : I32EnumAttrCase<"Float32", 9, "f32">;
def Stream_CollectiveElementType_Float64 : I32EnumAttrCase<"Float64", 10, "f64">;
def Stream_CollectiveElementType_BFloat16 : I32EnumAttrCase<"BFloat16", 11, "bf16">;
def Stream_CollectiveElementTypeAttr :
    I32EnumAttr<"CollectiveElementType", "valid CollectiveElementType", [
      Stream_CollectiveElementType_Sint8,
      Stream_CollectiveElementType_Uint8,
      Stream_CollectiveElementType_Sint16,
      Stream_CollectiveElementType_Uint16,
      Stream_CollectiveElementType_Sint32,
      Stream_CollectiveElementType_Uint32,
      Stream_CollectiveElementType_Sint64,
      Stream_CollectiveElementType_Uint64,
      Stream_CollectiveElementType_Float16,
      Stream_CollectiveElementType_Float32,
      Stream_CollectiveElementType_Float64,
      Stream_CollectiveElementType_BFloat16,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Stream";
}

def Stream_CollectiveAttr :
    AttrDef<Stream_Dialect, "Collective", []> {
  let mnemonic = "collective";
  let summary = [{collective operation and specification}];
  let description = [{
    Specifies the collective operation to perform and any mode bits required.
  }];
  let parameters = (ins
    AttrParameter<"CollectiveKind", "">:$kind,
    OptionalParameter<"std::optional<CollectiveReductionOp>">:$reduction,
    AttrParameter<"CollectiveElementType", "">:$element_type
  );
  let assemblyFormat = [{
    `<` $kind (`with` $reduction^)? `:` $element_type `>`
  }];
}

def Stream_Favor_Debug : I32EnumAttrCase<"Debug", 0, "debug">;
def Stream_Favor_MinPeakMemory : I32EnumAttrCase<"MinPeakMemory", 1, "min-peak-memory">;
def Stream_Favor_MaxConcurrency : I32EnumAttrCase<"MaxConcurrency", 2, "max-concurrency">;
def Stream_FavorAttr :
    I32EnumAttr<"Favor", "IREE partitioning bias", [
      Stream_Favor_Debug,
      Stream_Favor_MinPeakMemory,
      Stream_Favor_MaxConcurrency,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Stream";
}

def Stream_PartitioningConfigAttr :
    AttrDef<Stream_Dialect, "PartitioningConfig"> {
  let mnemonic = "partitioning_config";
  let summary = [{defines partitioning configuration}];
  let description = [{
    Configures the partitioning algorithm to use and its configuration.
    Partitioning is useful to adjust when scheduling behavior of targets is
    radically different - such as single-threaded vs. multi-threaded CPUs or
    bespoke ML accelerators vs. general purpose GPUs. This mechanism controls
    the amount of concurrency, parallelism, memory consumption, and latency.
  }];

  // TODO(benvanik): partitioning config.
  let parameters = (ins
    "IREE::Stream::FavorAttr":$favor
  );

  let valueType = NoneType;

  let builders = [
    AttrBuilderWithInferredContext<(ins "IREE::Stream::FavorAttr":$favor), [{
      return $_get(favor.getContext(), favor);
    }]>,
  ];

  let extraClassDeclaration = [{
    // Returns a partitioning config active for the given operation.
    // This will recursively walk parent operations until one with the
    // `stream.partitioning` attribute is found.
    static PartitioningConfigAttr lookup(Operation *op);
  }];

  let hasCustomAssemblyFormat = 1;
}

def Stream_MemoryModel_Unified : I32EnumAttrCase<"Unified", 0>;
def Stream_MemoryModel_Discrete : I32EnumAttrCase<"Discrete", 1>;
def Stream_MemoryModelAttr :
    I32EnumAttr<"MemoryModel", "stream resource memory model", [
      Stream_MemoryModel_Unified,
      Stream_MemoryModel_Discrete,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Stream";
}

def Stream_ResourceConfigAttr :
    AttrDef<Stream_Dialect, "ResourceConfig", []> {
  let mnemonic = "resource_config";

  let summary = [{defines resource constraints configuration}];
  let description = [{
    Defines resource storage constraints. These allow for packing and layout
    algorithms to ensure they are producing usable results on target devices.
  }];

  // TODO(benvanik): this was just copied from the HAL; probably good to make it
  // more generic such that we can classify entire device families instead of
  // specific values like this. As-is this is a phase-ordering violation.
  let parameters = (ins
    // The maximum size of a memory allocation that can be created, even if
    // there is more space available in the heap.
    "int64_t":$maxAllocationSize,
    // The minimum required alignment, in bytes, for offsets used in runtime
    // resource bindings. Offset values (both dynamic and static) must be an
    // integer multiple of this limit.
    "int64_t":$minBufferOffsetAlignment,
    // The maximum value that can be specified for size ranges of resource
    // bindings. The underlying allocation may be larger than this but only
    // up to this amount will be visible to kernels.
    "int64_t":$maxBufferRange,
    // The minimum required alignment, in bytes, for size ranges of resources
    // bindings.
    "int64_t":$minBufferRangeAlignment,
    // Number of bits in `index` values as passed across device boundaries.
    "int64_t":$indexBits,
    // Fuses bindings that are mutable instead of leaving them split.
    "bool":$aliasMutableBindings,
    // Memory model used for host-device memory access.
    "IREE::Stream::MemoryModel":$memoryModel
  );

  let valueType = NoneType;

  let extraClassDeclaration = [{
    // Returns the intersection (most conservative) constraints |lhs| ∩ |rhs|.
    static ResourceConfigAttr
    intersectBufferConstraints(ResourceConfigAttr lhs, ResourceConfigAttr rhs);

    // Returns a resource config compatible with the host.
    // These must only be used with resources when it is known that the device
    // is local or has unified memory.
    static ResourceConfigAttr
    getDefaultHostConstraints(MLIRContext *context);

    // Returns a resource config active for the given operation.
    // This will recursively walk parent operations until one with the
    // `stream.resources` attribute is found, an affinity specifies a
    // configuration, or as a fallback returns a conservative configuration.
    static ResourceConfigAttr lookup(Operation *op);
  }];

  let hasCustomAssemblyFormat = 1;
}

def Stream_ResourceAccess_None : I32BitEnumAttrCase<"None", 0x0000>;
def Stream_ResourceAccess_Read : I32BitEnumAttrCase<"Read", 0x0001>;
def Stream_ResourceAccess_Write : I32BitEnumAttrCase<"Write", 0x0002>;
def Stream_ResourceAccessBitfieldAttr :
    I32BitEnumAttr<"ResourceAccessBitfield", "valid ResourceAccess", [
      Stream_ResourceAccess_None,
      Stream_ResourceAccess_Read,
      Stream_ResourceAccess_Write,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::Stream";
}
def Stream_ResourceAccessArrayAttr :
    TypedArrayAttrBase<Stream_ResourceAccessBitfieldAttr,
                       "access array attribute"> {}

//===----------------------------------------------------------------------===//
// Parameter storage attributes
//===----------------------------------------------------------------------===//

def Stream_NamedParameterAttr :
    AttrDef<Stream_Dialect, "NamedParameter", [
      TypedAttrInterface,
      DeclareAttrInterfaceMethods<Util_SizedStorageAttr, [
        "getStorageSize",
      ]>,
    ]> {
  let mnemonic = "parameter.named";
  let summary = [{named parameter referenced an optional scope and key}];
  let description = [{
    Species an externally-defined parameter that can be referenced by an
    optional scope defining a set of parameters and a key uniquely identifying
    the parameter within its scope.
  }];
  let parameters = (ins
    AttributeSelfTypeParameter<"">:$type,
    OptionalParameter<"StringAttr">:$scope,
    AttrParameter<"StringAttr", "">:$key,
    OptionalParameter<"DictionaryAttr">:$config
  );
  let assemblyFormat = [{
    `<`
    custom<ParameterReference>($scope, $key)
    (`,` $config^)?
    `>`
  }];
}

//===----------------------------------------------------------------------===//
// Stream synchronization types
//===----------------------------------------------------------------------===//

def Stream_Timepoint : TypeDef<Stream_Dialect, "Timepoint", [
  Util_GlobalType,
]> {
  let mnemonic = "timepoint";

  let summary = [{a timepoint indicating execution availability}];
  let description = [{
    Represents a point in the execution timeline that when resolved indicates
    that all of the execution prior to this timepoint has completed and the
    results of the execution are available for use. This includes transitive
    dependencies as well; if timepoint B is dependent on timepoint A then when
    B is available so too must be A.
  }];

  // TODO(benvanik): track affinity so we know where timepoints come from.
  let parameters = (ins);
}

def Stream_TimepointAttr : AttrDef<Stream_Dialect, "Timepoint",
                             [TypedAttrInterface]> {
  let mnemonic = "timepoint";
  let summary = [{an immediately-resolved timepoint}];
  let description = [{}];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let valueType = Stream_Timepoint;
  let constBuilderCall = [{
    IREE::Stream::TimepointAttr::get(
        $_builder.getContext(),
        IREE::Stream::TimepointType::get($_builder.getContext()));
  }];
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Stream resource value types
//===----------------------------------------------------------------------===//

// Indicates a value whose lifetime is not yet analyzed.
def Stream_Lifetime_Unknown : I32EnumAttrCase<"Unknown", 0, "*">;
// An externally managed value.
def Stream_Lifetime_External : I32EnumAttrCase<"External", 1, "external">;
// A staging buffer for uploads/downloads.
def Stream_Lifetime_Staging : I32EnumAttrCase<"Staging", 2, "staging">;
// A short-lived value that is used across streams.
def Stream_Lifetime_Transient : I32EnumAttrCase<"Transient", 3, "transient">;
// A long-lived value that persists across streams.
def Stream_Lifetime_Variable : I32EnumAttrCase<"Variable", 4, "variable">;
// An immutable value that persist for the duration of the program.
def Stream_Lifetime_Constant : I32EnumAttrCase<"Constant", 5, "constant">;
def Stream_LifetimeAttr :
    I32EnumAttr<"Lifetime", "IREE Stream value lifetime", [
      Stream_Lifetime_Unknown,
      Stream_Lifetime_External,
      Stream_Lifetime_Staging,
      Stream_Lifetime_Transient,
      Stream_Lifetime_Variable,
      Stream_Lifetime_Constant,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Stream";
}

def Stream_AnyResource : Type<
    CPred<"$_self.isa<IREE::Stream::ResourceType>()">,
    "any stream-compatible type">;

// TODO(benvanik): support other types; the interface may be enough.
def Stream_TransferType : AnyTypeOf<[
  Stream_AnyResource,
  Stream_Tensor,
  Stream_PrimitiveType,
]>;

def Stream_Channel : TypeDef<Stream_Dialect, "Channel", [
  Util_ReferenceType
]> {
  let mnemonic = "channel";

  let summary = [{a collective communication channel}];
  let description = [{
    Represents a single participant in a collective clique. Multiple channels
    may exist within the same program to allow for partial operations or
    hierarchical operations.

    In programs that model SPMD behavior internally channels can be created or
    provided by hosting applications. For example, the program could expose a
    `@set_channels(!util.list<!stream.channel>)` method that stores the channels
    in globals for use throughout the program allowing for
    application-controlled channel configuration.
  }];
}

def Stream_Resource : TypeDef<Stream_Dialect, "Resource", [
  Util_ReferenceType,
  Util_SizeAwareType,
  DeclareTypeInterfaceMethods<Util_GlobalType, [
    "isAccessStorageCompatible",
  ]>,
  DeclareTypeInterfaceMethods<Util_InferTypeSize, [
    "inferSizeFromValue",
  ]>,
  DeclareTypeInterfaceMethods<Util_SubrangeType, [
    "createSubrangeOp",
  ]>,
]> {
  let mnemonic = "resource";

  let summary = [{a managed resource}];
  let description = [{
    Stream external values represent asynchronously-available and sequenced
    values that are owned and managed by external code - such as those passed in
    or out of the program entry points. Though external values are managed
    during an invocation the same as other stream values the visibility into
    them does not extend outside of the invocation they are provided to.

    Stream values are not usable directly outside of a stream execution or
    transfer operation. If the contents of the value are needed they must first
    be transferred via `stream.transfer` - which may incur a copy.
  }];

  let parameters = (ins
    "IREE::Stream::Lifetime":$lifetime
  );

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt, IREE::Stream::Lifetime::Unknown);
    }]>,
    TypeBuilderWithInferredContext<(ins
      "IREE::Stream::LifetimeAttr":$lifetime
    ), [{
      return $_get(lifetime.getContext(), lifetime.getValue());
    }]>,
  ];

  let hasCustomAssemblyFormat = 1;
}

def Stream_ResourceLifetimeUnknown : CPred<[{
  $_self.cast<IREE::Stream::ResourceType>().getLifetime() ==
      IREE::Stream::Lifetime::Unknown
}]>;
def Stream_ResourceLifetimeExternal : CPred<[{
  $_self.cast<IREE::Stream::ResourceType>().getLifetime() ==
      IREE::Stream::Lifetime::External
}]>;
def Stream_ResourceLifetimeStaging : CPred<[{
  $_self.cast<IREE::Stream::ResourceType>().getLifetime() ==
      IREE::Stream::Lifetime::Staging
}]>;
def Stream_ResourceLifetimeTransient : CPred<[{
  $_self.cast<IREE::Stream::ResourceType>().getLifetime() ==
      IREE::Stream::Lifetime::Transient
}]>;
def Stream_ResourceLifetimeVariable : CPred<[{
  $_self.cast<IREE::Stream::ResourceType>().getLifetime() ==
      IREE::Stream::Lifetime::Variable
}]>;
def Stream_ResourceLifetimeConstant : CPred<[{
  $_self.cast<IREE::Stream::ResourceType>().getLifetime() ==
      IREE::Stream::Lifetime::Constant
}]>;

def Stream_UnknownResource : DialectType<Stream_Dialect, And<[
  CPred<"$_self.isa<IREE::Stream::ResourceType>()">,
  Stream_ResourceLifetimeUnknown,
]>, "resource"> {
  let description = [{
    A stream resource that has not yet had its lifetime calculated.
  }];
}

def Stream_ExternalResource : DialectType<Stream_Dialect, And<[
  CPred<"$_self.isa<IREE::Stream::ResourceType>()">,
  Stream_ResourceLifetimeExternal,
]>, "external resource"> {
  let description = [{
    Stream external values represent asynchronously-available and sequenced
    values that are owned and managed by external code - such as those passed in
    or out of the program entry points. Though external values are managed
    during an invocation the same as other stream values the visibility into
    them does not extend outside of the invocation they are provided to.

    Stream values are not usable directly outside of a stream execution or
    transfer operation. If the contents of the value are needed they must first
    be transferred via `stream.transfer` - which may incur a copy.
  }];
}

def Stream_StagingResource : DialectType<Stream_Dialect, And<[
  CPred<"$_self.isa<IREE::Stream::ResourceType>()">,
  Stream_ResourceLifetimeStaging,
]>, "staging resource"> {
  let description = [{
    Stream upload/download staging resource. These are used outside of streams
    and then transferred to other stream resources such as variables or
    transients for use inside of streams. Dispatches and several other
    operations cannot directly operate on these resources.
  }];
}

def Stream_TransientResource : DialectType<Stream_Dialect, And<[
  CPred<"$_self.isa<IREE::Stream::ResourceType>()">,
  Stream_ResourceLifetimeTransient,
]>, "transient resource"> {
  let description = [{
    Stream transients represent asynchronously-available and sequenced values
    that have a short lifetime - often only passed between stream executions.
    It is expected that transient values are not stored in global state and have
    minimal lifetime as they may be heavily pooled or suballocated.

    Stream values are not usable directly outside of a stream execution or
    transfer operation. If the contents of the value are needed they must first
    be transferred via `stream.transfer` - which may incur a copy.
  }];
}

def Stream_VariableResource : DialectType<Stream_Dialect, And<[
  CPred<"$_self.isa<IREE::Stream::ResourceType>()">,
  Stream_ResourceLifetimeVariable,
]>, "variable resource"> {
  let description = [{
    Stream variables represent asynchronously-available and sequenced values
    that have a long lifetime relative to the work being performed on them.
    These variables are often stored in global state and may live for the entire
    duration of the program.

    Stream values are not usable directly outside of a stream execution or
    transfer operation. If the contents of the value are needed they must first
    be transferred via `stream.transfer` - which may incur a copy.
  }];
}

def Stream_ConstantResource : DialectType<Stream_Dialect, And<[
  CPred<"$_self.isa<IREE::Stream::ResourceType>()">,
  Stream_ResourceLifetimeConstant,
]>, "constant resource"> {
  let description = [{
    Stream constants are immutable values that are available for the lifetime of
    the program once initialized.
  }];
}

def Stream_AnyStreamResource : AnyTypeOf<[
  Stream_UnknownResource,
  Stream_ExternalResource,
  Stream_TransientResource,
  Stream_VariableResource,
  Stream_ConstantResource,
]>;

//===----------------------------------------------------------------------===//
// File resources
//===----------------------------------------------------------------------===//

def Stream_File : TypeDef<Stream_Dialect, "File", []> {
  let mnemonic = "file";

  let summary = [{a file handle used for I/O operations}];
  let description = [{
    A file handle that can be asynchronously read and written into/from
    stream resources.
  }];

  let parameters = (ins);

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// Executable bindings
//===----------------------------------------------------------------------===//

def Stream_Binding : TypeDef<Stream_Dialect, "Binding", []> {
  let mnemonic = "binding";

  let summary = [{a managed resource binding into an executable scope}];
  let description = [{
    A resource binding available within an executable dispatch function.
    The bindings map 1:1 with the resources bound during dispatch operations.
  }];

  // TODO(benvanik): carry lifetime like resources.
  let parameters = (ins);

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt);
    }]>,
  ];
}

def Stream_AnyBinding : AnyTypeOf<[
  Stream_Binding,
]>;

//===----------------------------------------------------------------------===//
// Stream op traits
//===----------------------------------------------------------------------===//

def Stream_TensorPhaseOp : NativeOpTrait<"IREE::Stream::TensorPhaseOp">;
def Stream_AsyncPhaseOp : NativeOpTrait<"IREE::Stream::AsyncPhaseOp">;
def Stream_CmdPhaseOp : NativeOpTrait<"IREE::Stream::CmdPhaseOp">;

#endif  // IREE_DIALECT_STREAM_BASE
