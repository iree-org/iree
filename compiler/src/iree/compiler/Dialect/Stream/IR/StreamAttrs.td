// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_STREAM_ATTRS
#define IREE_DIALECT_STREAM_ATTRS

include "iree/compiler/Dialect/Stream/IR/StreamBase.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

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
  let cppNamespace = "::mlir::iree_compiler::IREE::Stream";
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
    Stream_AttrDef<"Collective", []> {
  let mnemonic = "collective";
  let summary = [{Collective operation and specification.}];
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
    Stream_AttrDef<"PartitioningConfig"> {
  let mnemonic = "partitioning_config";
  let summary = [{Defines partitioning configuration.}];
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
    Stream_AttrDef<"ResourceConfig", []> {
  let mnemonic = "resource_config";

  let summary = [{Defines resource constraints configuration.}];
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

def Stream_NamedParameterAttr :
    Stream_AttrDef<"NamedParameter", [
      TypedAttrInterface,
      DeclareAttrInterfaceMethods<Util_SizedStorageAttr, [
        "getStorageSize",
      ]>,
    ]> {
  let mnemonic = "parameter.named";
  let summary = [{Named parameter referenced an optional scope and key.}];
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

#endif  // IREE_DIALECT_STREAM_ATTRS
