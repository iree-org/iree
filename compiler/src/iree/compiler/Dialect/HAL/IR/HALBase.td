// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_HAL_BASE
#define IREE_DIALECT_HAL_BASE

include "iree/compiler/Dialect/HAL/IR/HALDialect.td"
include "iree/compiler/Dialect/HAL/IR/HALInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// HAL enums
//===----------------------------------------------------------------------===//

def HAL_MemoryModel_Unified : I32EnumAttrCase<"Unified", 0>;
def HAL_MemoryModel_Discrete : I32EnumAttrCase<"Discrete", 1>;
def HAL_MemoryModelAttr :
    I32EnumAttr<"MemoryModel", "IREE HAL MemoryModel", [
      HAL_MemoryModel_Unified,
      HAL_MemoryModel_Discrete,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}

def HAL_MemoryType_None             : I32BitEnumAttrCase<"None",             0x0000>;  // ?
def HAL_MemoryType_Optimal          : I32BitEnumAttrCase<"Optimal",          0x0001>;  // !
def HAL_MemoryType_HostVisible      : I32BitEnumAttrCase<"HostVisible",      0x0002>;  // h
def HAL_MemoryType_HostCoherent     : I32BitEnumAttrCase<"HostCoherent",     0x0004>;  // c
def HAL_MemoryType_HostCached       : I32BitEnumAttrCase<"HostCached",       0x0008>;  // C
def HAL_MemoryType_HostLocal        : I32BitEnumAttrCase<"HostLocal",        0x0006>;  // H
def HAL_MemoryType_DeviceVisible    : I32BitEnumAttrCase<"DeviceVisible",    0x0010>;  // d
def HAL_MemoryType_DeviceLocal      : I32BitEnumAttrCase<"DeviceLocal",      0x0030>;  // D
def HAL_MemoryTypeBitfieldAttr :
    I32BitEnumAttr<"MemoryTypeBitfield", "valid MemoryType", [
      HAL_MemoryType_None,
      HAL_MemoryType_Optimal,
      HAL_MemoryType_HostVisible,
      HAL_MemoryType_HostCoherent,
      HAL_MemoryType_HostCached,
      HAL_MemoryType_HostLocal,
      HAL_MemoryType_DeviceVisible,
      HAL_MemoryType_DeviceLocal,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_BufferUsage_None                         : I32BitEnumAttrCase<"None",                         0x00000000>;
def HAL_BufferUsage_TransferSource               : I32BitEnumAttrCase<"TransferSource",               0x00000001>;
def HAL_BufferUsage_TransferTarget               : I32BitEnumAttrCase<"TransferTarget",               0x00000002>;
def HAL_BufferUsage_Transfer                     : I32BitEnumAttrCase<"Transfer",                     0x00000003>;
def HAL_BufferUsage_DispatchIndirectParams       : I32BitEnumAttrCase<"DispatchIndirectParams",       0x00000100>;
def HAL_BufferUsage_DispatchUniformRead          : I32BitEnumAttrCase<"DispatchUniformRead",          0x00000200>;
def HAL_BufferUsage_DispatchStorageRead          : I32BitEnumAttrCase<"DispatchStorageRead",          0x00000400>;
def HAL_BufferUsage_DispatchStorageWrite         : I32BitEnumAttrCase<"DispatchStorageWrite",         0x00000800>;
def HAL_BufferUsage_DispatchStorage              : I32BitEnumAttrCase<"DispatchStorage",              0x00000C00>;
def HAL_BufferUsage_DispatchImageRead            : I32BitEnumAttrCase<"DispatchImageRead",            0x00001000>;
def HAL_BufferUsage_DispatchImageWrite           : I32BitEnumAttrCase<"DispatchImageWrite",           0x00002000>;
def HAL_BufferUsage_DispatchImage                : I32BitEnumAttrCase<"DispatchImage",                0x00003000>;
def HAL_BufferUsage_SharingExport                : I32BitEnumAttrCase<"SharingExport",                0x00010000>;
def HAL_BufferUsage_SharingReplicate             : I32BitEnumAttrCase<"SharingReplicate",             0x00020000>;
def HAL_BufferUsage_SharingConcurrent            : I32BitEnumAttrCase<"SharingConcurrent",            0x00040000>;
def HAL_BufferUsage_SharingImmutable             : I32BitEnumAttrCase<"SharingImmutable",             0x00080000>;
def HAL_BufferUsage_MappingScoped                : I32BitEnumAttrCase<"MappingScoped",                0x01000000>;
def HAL_BufferUsage_MappingPersistent            : I32BitEnumAttrCase<"MappingPersistent",            0x02000000>;
def HAL_BufferUsage_MappingOptional              : I32BitEnumAttrCase<"MappingOptional",              0x04000000>;
def HAL_BufferUsage_MappingAccessRandom          : I32BitEnumAttrCase<"MappingAccessRandom",          0x08000000>;
def HAL_BufferUsage_MappingAccessSequentialWrite : I32BitEnumAttrCase<"MappingAccessSequentialWrite", 0x10000000>;
def HAL_BufferUsage_Mapping                      : I32BitEnumAttrCase<"Mapping",                      0x09000000>;
def HAL_BufferUsageBitfieldAttr :
    I32BitEnumAttr<"BufferUsageBitfield", "valid BufferUsage", [
      HAL_BufferUsage_None,
      HAL_BufferUsage_TransferSource,
      HAL_BufferUsage_TransferTarget,
      HAL_BufferUsage_Transfer,
      HAL_BufferUsage_DispatchIndirectParams,
      HAL_BufferUsage_DispatchUniformRead,
      HAL_BufferUsage_DispatchStorageRead,
      HAL_BufferUsage_DispatchStorageWrite,
      HAL_BufferUsage_DispatchStorage,
      HAL_BufferUsage_DispatchImageRead,
      HAL_BufferUsage_DispatchImageWrite,
      HAL_BufferUsage_DispatchImage,
      HAL_BufferUsage_SharingExport,
      HAL_BufferUsage_SharingReplicate,
      HAL_BufferUsage_SharingConcurrent,
      HAL_BufferUsage_SharingImmutable,
      HAL_BufferUsage_MappingScoped,
      HAL_BufferUsage_MappingPersistent,
      HAL_BufferUsage_MappingOptional,
      HAL_BufferUsage_MappingAccessRandom,
      HAL_BufferUsage_MappingAccessSequentialWrite,
      HAL_BufferUsage_Mapping,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_CommandBufferMode_None : I32BitEnumAttrCase<"None", 0x0000>;
def HAL_CommandBufferMode_OneShot : I32BitEnumAttrCase<"OneShot", 0x0001>;
def HAL_CommandBufferMode_AllowInlineExecution : I32BitEnumAttrCase<"AllowInlineExecution", 0x0010>;
def HAL_CommandBufferModeBitfieldAttr :
    I32BitEnumAttr<"CommandBufferModeBitfield", "valid CommandBufferMode", [
      HAL_CommandBufferMode_None,
      HAL_CommandBufferMode_OneShot,
      HAL_CommandBufferMode_AllowInlineExecution,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_CommandCategory_None : I32BitEnumAttrCase<"None", 0x0000>;
def HAL_CommandCategory_Transfer : I32BitEnumAttrCase<"Transfer", 0x0001>;
def HAL_CommandCategory_Dispatch : I32BitEnumAttrCase<"Dispatch", 0x0002>;
def HAL_CommandCategoryBitfieldAttr :
    I32BitEnumAttr<"CommandCategoryBitfield", "valid CommandCategory", [
      HAL_CommandCategory_None,
      HAL_CommandCategory_Transfer,
      HAL_CommandCategory_Dispatch,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_DescriptorType_UniformBuffer : I32EnumAttrCase<"UniformBuffer", 6, "uniform_buffer">;
def HAL_DescriptorType_StorageBuffer : I32EnumAttrCase<"StorageBuffer", 7, "storage_buffer">;
def HAL_DescriptorType_UniformBufferDynamic : I32EnumAttrCase<"UniformBufferDynamic", 8, "uniform_buffer_dynamic">;
def HAL_DescriptorType_StorageBufferDynamic : I32EnumAttrCase<"StorageBufferDynamic", 9, "storage_buffer_dynamic">;
def HAL_DescriptorTypeAttr :
    I32EnumAttr<"DescriptorType", "IREE HAL DescriptorType", [
      HAL_DescriptorType_UniformBuffer,
      HAL_DescriptorType_StorageBuffer,
      HAL_DescriptorType_UniformBufferDynamic,
      HAL_DescriptorType_StorageBufferDynamic,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}

def HAL_DescriptorSetLayoutUsageType_Immutable : I32EnumAttrCase<"Immutable", 0, "immutable">;
def HAL_DescriptorSetLayoutUsageType_PushOnly : I32EnumAttrCase<"PushOnly", 1, "push_only">;
def HAL_DescriptorSetLayoutUsageTypeAttr :
    I32EnumAttr<"DescriptorSetLayoutUsageType", "IREE HAL DescriptorSetLayoutType", [
      HAL_DescriptorSetLayoutUsageType_Immutable,
      HAL_DescriptorSetLayoutUsageType_PushOnly,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}

def HAL_ExecutionStage_None : I32BitEnumAttrCase<"None", 0x0000>;
def HAL_ExecutionStage_CommandIssue : I32BitEnumAttrCase<"CommandIssue", 0x0001>;
def HAL_ExecutionStage_CommandProcess : I32BitEnumAttrCase<"CommandProcess", 0x0002>;
def HAL_ExecutionStage_Dispatch : I32BitEnumAttrCase<"Dispatch", 0x0004>;
def HAL_ExecutionStage_Transfer : I32BitEnumAttrCase<"Transfer", 0x0008>;
def HAL_ExecutionStage_CommandRetire : I32BitEnumAttrCase<"CommandRetire", 0x0010>;
def HAL_ExecutionStage_Host : I32BitEnumAttrCase<"Host", 0x0020>;
def HAL_ExecutionStageBitfieldAttr :
    I32BitEnumAttr<"ExecutionStageBitfield", "valid ExecutionStage", [
      HAL_ExecutionStage_None,
      HAL_ExecutionStage_CommandIssue,
      HAL_ExecutionStage_CommandProcess,
      HAL_ExecutionStage_Dispatch,
      HAL_ExecutionStage_Transfer,
      HAL_ExecutionStage_CommandRetire,
      HAL_ExecutionStage_Host
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_ExecutionBarrierFlag_None : I32BitEnumAttrCase<"None", 0x0000>;
def HAL_ExecutionBarrierFlag_Reserved : I32BitEnumAttrCase<"Reserved", 0x0001>;
def HAL_ExecutionBarrierFlagBitfieldAttr :
    I32BitEnumAttr<"ExecutionBarrierFlagBitfield", "valid ExecutionBarrierFlag", [
      HAL_ExecutionBarrierFlag_None,
      HAL_ExecutionBarrierFlag_Reserved,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_AccessScope_None : I32BitEnumAttrCase<"None", 0x0000>;
def HAL_AccessScope_IndirectCommandRead : I32BitEnumAttrCase<"IndirectCommandRead", 0x0001>;
def HAL_AccessScope_ConstantRead : I32BitEnumAttrCase<"ConstantRead", 0x0002>;
def HAL_AccessScope_DispatchRead : I32BitEnumAttrCase<"DispatchRead", 0x0004>;
def HAL_AccessScope_DispatchWrite : I32BitEnumAttrCase<"DispatchWrite", 0x0008>;
def HAL_AccessScope_TransferRead : I32BitEnumAttrCase<"TransferRead", 0x0010>;
def HAL_AccessScope_TransferWrite : I32BitEnumAttrCase<"TransferWrite", 0x0020>;
def HAL_AccessScope_HostRead : I32BitEnumAttrCase<"HostRead", 0x0040>;
def HAL_AccessScope_HostWrite : I32BitEnumAttrCase<"HostWrite", 0x0080>;
def HAL_AccessScope_MemoryRead : I32BitEnumAttrCase<"MemoryRead", 0x0100>;
def HAL_AccessScope_MemoryWrite : I32BitEnumAttrCase<"MemoryWrite", 0x0200>;
def HAL_AccessScopeBitfieldAttr :
    I32BitEnumAttr<"AccessScopeBitfield", "valid AccessScope", [
      HAL_AccessScope_None,
      HAL_AccessScope_IndirectCommandRead,
      HAL_AccessScope_ConstantRead,
      HAL_AccessScope_DispatchRead,
      HAL_AccessScope_DispatchWrite,
      HAL_AccessScope_TransferRead,
      HAL_AccessScope_TransferWrite,
      HAL_AccessScope_HostRead,
      HAL_AccessScope_HostWrite,
      HAL_AccessScope_MemoryRead,
      HAL_AccessScope_MemoryWrite
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

//===----------------------------------------------------------------------===//
// HAL types
//===----------------------------------------------------------------------===//

def HAL_Allocator : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::AllocatorType>()">,
    "allocator"> {
  let description = [{
    Allocates buffers for a particular device memory space.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::AllocatorType>()";
}

def HAL_Buffer : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::BufferType>()">,
    "buffer"> {
  let description = [{
    A memory buffer with a specific memory_type that is used to describe the
    capabilities and behavior of the backing memory of the buffer. Buffers may
    be any mix of host-accessible, host-coherent, or device-accessible for
    various usages. Depending on these memory types the buffers may be mapped
    for access on the host as memory though certain restrictions may be imposed.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::BufferType>()";
}

def HAL_BufferView : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::BufferViewType>()">,
    "buffer_view"> {
  let description = [{
    A shaped and typed buffer reference. This just wraps an existing hal.buffer
    with its associated metadata to make it easier to pass across ABI
    boundaries. In most cases buffer views can be elided entirely by the
    compiler and they'll only be seen when calling external functions.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::BufferViewType>()";
}

def HAL_CommandBuffer : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::CommandBufferType>()">,
    "command_buffer"> {
  let description = [{
    Asynchronous command buffer recording interface. Commands are recorded by
    the implementation for later submission to command queues.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::CommandBufferType>()";
}

def HAL_DescriptorSet : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::DescriptorSetType>()">,
    "descriptor_set"> {
  let description = [{
    Descriptor set.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::DescriptorSetType>()";
}

def HAL_DescriptorSetLayout : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::DescriptorSetLayoutType>()">,
    "descriptor_set_layout"> {
  let description = [{
    Descriptor set layout.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::DescriptorSetLayoutType>()";
}

def HAL_Device : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::DeviceType>()">,
    "device"> {
  let description = [{
    Logical device instance.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::DeviceType>()";
}

def HAL_Event : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::EventType>()">,
    "event"> {
  let description = [{
    Events are used for defining synchronization scopes within CommandBuffers.
    An event only exists within a single CommandBuffer and must not be used
    across CommandBuffers from the same device or others.
  }];
}

def HAL_Executable : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::ExecutableType>()">,
    "executable"> {
  let description = [{
    A prepared and ready-to-dispatch executable.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::ExecutableType>()";
}

def HAL_ExecutableLayout : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::ExecutableLayoutType>()">,
    "executable_layout"> {
  let description = [{
    An executable layout describing the descriptor sets and push constants used.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::ExecutableLayoutType>()";
}

def HAL_Fence : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::FenceType>()">,
    "fence"> {
  let description = [{
    A set of semaphore timepoints defining a common point in time across
    multiple timelines.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::FenceType>()";
}

def HAL_RingBuffer : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::RingBufferType>()">,
    "ring_buffer"> {
  let description = [{
    Ringbuffer used for transient buffer allocation.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::RingBufferType>()";
}

def HAL_Semaphore : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::SemaphoreType>()">,
    "semaphore"> {
  let description = [{
    Synchronization mechanism for host->device, device->host, host->host,
    and device->device notification. Semaphores behave like Vulkan timeline
    semaphores (or D3D12 fences) and contain a monotonically increasing
    uint64_t payload. They may be waited on any number of times even if they
    have already been signaled for a particular value. They may also be waited
    on for a particular value prior to the signal for that value.
  }];
  let builderCall = "$_builder.getType<IREE::HAL::SemaphoreType>()";
}

def HAL_ObjectType : AnyTypeOf<[
  HAL_Allocator,
  HAL_Buffer,
  HAL_BufferView,
  HAL_CommandBuffer,
  HAL_DescriptorSet,
  HAL_DescriptorSetLayout,
  HAL_Device,
  HAL_Event,
  HAL_Executable,
  HAL_ExecutableLayout,
  HAL_RingBuffer,
  HAL_Semaphore,
]>;

def HAL_BufferType : AnyTypeOf<[
  HAL_Buffer,
]>;

def HAL_OrdinalAttr : Util_IndexAttrBase<"size_t">;

def HAL_ExecutableDataAttr : SignlessIntElementsAttr<8>;

def HAL_ElementType : TypeAlias<I32>;
def HAL_ElementTypeAttr : SignlessIntegerAttrBase<
  I32, "element type attribute">;

def HAL_EncodingType : TypeAlias<I32>;
def HAL_EncodingTypeAttr : SignlessIntegerAttrBase<
  I32, "encoding type attribute">;

def HAL_DeviceSize : TypeAlias<Index>;
def HAL_DeviceSizeAttr : Util_IndexAttrBase<"iree_device_size_t">;
def HAL_DeviceSizes : Variadic<HAL_DeviceSize>;

def HAL_HostSize : TypeAlias<Index>;
def HAL_HostSizeAttr : Util_IndexAttrBase<"size_t">;

def HAL_TimelineValue : TypeAlias<I64>;

def HAL_PrimitiveType : AnyTypeOf<[Index, AnySignlessInteger, AnyFloat]>;
def HAL_FillPatternType : AnyTypeOf<[I8, I16, I32]>;

def HAL_GlobalRefAttr : Util_AliasedSymbolRefAttr;
def HAL_GlobalType : AnyTypeOf<[HAL_PrimitiveType, AnyVector, HAL_ObjectType]>;
def HAL_GlobalPtr : Util_PtrOf<HAL_GlobalType>;

def HAL_IndexAttr : Util_IndexAttrBase<"index">;
def HAL_IndexArrayAttr : TypedArrayAttrBase<HAL_IndexAttr,
                                      "index array attribute"> {
  let constBuilderCall = "$_builder.getIndexArrayAttr($0)";
}

def HAL_Dim : TypeAlias<Index>;
def HAL_Dims : Variadic<HAL_Dim>;
def HAL_Shape : Variadic<HAL_Dim>;
def HAL_ShapeDynamicDims : Variadic<HAL_Dim>;

// TODO(benvanik): assert rank 3
def HAL_WorkgroupSizeAttr : TypedArrayAttrBase<
    Util_IndexAttrBase<"size_t">,
    "index array attribute"> {
  let constBuilderCall = "$_builder.getIndexArrayAttr($0)";
}

// A bitmask defining which queues an operation is allowed to execute on.
// The selection is wrapped to the total number of available queues, so 0b0101
// would enable queues 0 and 2 if there were four queues or queue 0 if there
// were two queues.
def HAL_DeviceQueueAffinity : TypeAlias<I64>;

def HAL_DeviceQueuePool : TypeAlias<I64>;

def HAL_DurationMillisAttr : SignlessIntElementsAttr<32> {
  // TODO(b/143184519): add typeDescription support to other things.
  // let description = [{
  //   A duration to wait in milliseconds. 0 indicates that the operation should
  //   return immediately without waiting and can be used as a way to poll handles.
  //   INT32_MAX will wait forever until the handle is signaled.
  // }];
}

//===----------------------------------------------------------------------===//
// HAL structs
//===----------------------------------------------------------------------===//

def HAL_DescriptorSetBindingAttr :
    AttrDef<HAL_Dialect, "DescriptorSetBinding", []> {
  let mnemonic = "descriptor_set.binding";
  let summary = [{descriptor set binding specification}];
  let description = [{
    Specifies a single binding within a descriptor set layout.
  }];
  let parameters = (ins
    AttrParameter<"int64_t", "">:$ordinal,
    AttrParameter<"DescriptorType", "">:$type
  );
  let assemblyFormat = [{
    `<` $ordinal `,` $type `>`
  }];
}

def HAL_DescriptorSetLayoutBindingArrayAttr :
    TypedArrayAttrBase<HAL_DescriptorSetBindingAttr,
                       "HAL descriptor set layout binding array attribute">;

def HAL_DescriptorSetLayoutAttr :
    AttrDef<HAL_Dialect, "DescriptorSetLayout", []> {
  let mnemonic = "descriptor_set.layout";
  let summary = [{descriptor set layout specification}];
  let description = [{
    Specifies the layout information of a single set of descriptors used within
    an executable layout. Multiple of these sets may be used by a single entry
    point to allow for bindings with similar update frequencies to be grouped.
  }];
  let parameters = (ins
    AttrParameter<"int64_t", "">:$ordinal,
    ArrayRefParameter<"DescriptorSetBindingAttr", "">:$bindings
  );
  let assemblyFormat = [{
    `<`
    $ordinal `,`
    `bindings` `=` `[` $bindings `]`
    `>`
  }];
}

def HAL_ExecutableLayoutAttr :
    AttrDef<HAL_Dialect, "ExecutableLayout", []> {
  let mnemonic = "executable.layout";
  let summary = [{executable entry point layout specification}];
  let description = [{
    Specifies the layout information used for interacting with executable
    functions. This allows host code to correctly map parameters to the
    lower-level target-specific argument passing behavior.
  }];
  let parameters = (ins
    AttrParameter<"int64_t", "">:$pushConstants,
    ArrayRefParameter<"DescriptorSetLayoutAttr", "">:$setLayouts
  );
  let assemblyFormat = [{
    `<`
    `push_constants` `=` $pushConstants `,`
    `sets` `=` `[` $setLayouts `]`
    `>`
  }];
}

def HAL_InterfaceBindingAttr :
    AttrDef<HAL_Dialect, "InterfaceBinding", []> {
  let mnemonic = "interface.binding";
  let summary = [{interface binding specification}];
  let description = [{
    Specifies the descriptor set and binding ordinal of a particular layout
    binding.

    Example:
    ```mlir
    #hal.interface.binding<0, 1>
    ```
  }];
  let parameters = (ins
    AttrParameter<"int64_t", "">:$set,
    AttrParameter<"int64_t", "">:$binding
  );
  let assemblyFormat = [{
    `<` $set `,` $binding `>`
  }];
}

//===----------------------------------------------------------------------===//
// Device and executable target specification
//===----------------------------------------------------------------------===//

def HAL_DeviceTargetAttr :
    AttrDef<HAL_Dialect, "DeviceTarget", []> {
  let mnemonic = "device.target";
  let summary = [{generic device target specification}];
  let description = [{
    Specifies the properties of a target runtime device.
    Target devices are specified with a canonical identifier matching those used
    by the runtime (such as `cpu`, `vulkan`, etc). Target devices may support
    several target executable formats specified with `#hal.executable.target`.
    An optional configuration dictionary allows for overriding backend defaults.

    Example:
    ```mlir
    #hal.device.target<"cpu", {
      executable_targets = [
        #hal.executable.target<"llvm", "embedded-elf-arm_32">,
        #hal.executable.target<"llvm", "embedded-elf-arm_64">,
      ]
    }>
    ```
  }];
  let parameters = (ins
    AttrParameter<"StringAttr", "">:$deviceID,
    AttrParameter<"DictionaryAttr", "">:$configuration
  );
  let builders = [
    AttrBuilder<(ins "StringRef":$deviceID)>,
  ];

  let extraClassDeclaration = [{
    // Returns a symbol-compatible name that pseudo-uniquely identifies this
    // target. Callers must perform deduplication when required.
    std::string getSymbolNameFragment();

    // Returns a hal.match.* expression tree that specifically matches this
    // target device.
    Attribute getMatchExpression();

    // Returns zero or more executable targets that this device supports.
    SmallVector<ExecutableTargetAttr, 4> getExecutableTargets();

    // Returns a list of target devices that may be active for the given
    // operation. This will recursively walk parent operations until one with
    // the `hal.device.targets` attribute is found.
    static SmallVector<DeviceTargetAttr, 4> lookup(Operation *op);

    // Returns a list of all target executable configurations that may be
    // required for the given operation.
    static SmallVector<ExecutableTargetAttr, 4>
    lookupExecutableTargets(Operation *op);
  }];
  let hasCustomAssemblyFormat = 1;
}

def HAL_ExecutableTargetAttr :
    AttrDef<HAL_Dialect, "ExecutableTarget", []> {
  let mnemonic = "executable.target";
  let summary = [{generic executable target specification}];
  let description = [{
    Specifies how to compile an executable for a specific target backend.
    A backend is used to translate and serialize the executable into the final
    form passed to the runtime. The format of the executable is a
    target-specific value indicating the required runtime support to load the
    deployed artifact. An optionally provided configuration dictionary overrides
    backend-specific defaults.

    Example:
    ```mlir
      // Produce a system-native ELF for x86-64 systems using the LLVM backend:
      #hal.executable.target<"llvm", "system-elf-x86_64", {
        triple = "x86_64-unknown-linux-elf",
        cpu = "host",
        cpu_features = "host",
        abi = "lp32",
        ...
      }>
    ```

    The same compilation backend may be used to translate executables for
    several different runtime devices. Likewise the same runtime device may use
    one of many different executable targets. Assume an N:M mapping between the
    two in all cases.
  }];
  let parameters = (ins
    AttrParameter<"StringAttr", "">:$backend,
    AttrParameter<"StringAttr", "">:$format,
    AttrParameter<"DictionaryAttr", "">:$configuration
  );
  let builders = [
    AttrBuilder<(ins "StringRef":$backend, "StringRef":$format)>,
  ];

  let extraClassDeclaration = [{
    // Returns a symbol-compatible name that pseudo-uniquely identifies this
    // target. Callers must perform deduplication when required.
    std::string getSymbolNameFragment();

    // Returns a hal.match.* expression tree that specifically matches a
    // device that can load an executable of this target.
    Attribute getMatchExpression();
  }];
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Expression matching attributes
//===----------------------------------------------------------------------===//

def HAL_DeviceQuery : NativeOpTrait<"IREE::HAL::DeviceQuery">;

def HAL_MatchAlwaysAttr :
    AttrDef<HAL_Dialect, "MatchAlways", [
      DeclareAttrInterfaceMethods<HAL_MatchAttrInterface>,
    ]> {
  let mnemonic = "match.always";
  let summary = [{always matches}];
  let description = [{
    Returns true (constant true).
  }];
}

def HAL_MatchAnyAttr :
    AttrDef<HAL_Dialect, "MatchAny", [
      DeclareAttrInterfaceMethods<HAL_MatchAttrInterface>,
    ]> {
  let mnemonic = "match.any";
  let summary = [{matches if any subexpression matches}];
  let description = [{
    Returns true if any subexpression matches (logical OR) and not empty.
  }];
  let parameters = (ins
    AttrParameter<"ArrayAttr", "">:$conditions
  );
  let builders = [
    AttrBuilder<(ins "ArrayRef<Attribute>":$conditions), [{
      return $_get(context, ArrayAttr::get(context, conditions));
    }]>,
  ];
  let hasCustomAssemblyFormat = 1;
}

def HAL_MatchAllAttr :
    AttrDef<HAL_Dialect, "MatchAll", [
      DeclareAttrInterfaceMethods<HAL_MatchAttrInterface>,
    ]> {
  let mnemonic = "match.all";
  let summary = [{matches if all subexpressions match}];
  let description = [{
    Returns true only if all subexpressions return true (logical AND) or empty.
  }];
  let parameters = (ins
    AttrParameter<"ArrayAttr", "">:$conditions
  );
  let builders = [
    AttrBuilder<(ins "ArrayRef<Attribute>":$conditions), [{
      return $_get(context, ArrayAttr::get(context, conditions));
    }]>,
  ];
  let hasCustomAssemblyFormat = 1;
}

def HAL_DeviceMatchIDAttr :
    AttrDef<HAL_Dialect, "DeviceMatchID", [
      DeclareAttrInterfaceMethods<HAL_MatchAttrInterface>,
    ]> {
  let mnemonic = "device.match.id";
  let summary = [{matches against a device ID pattern}];
  let description = [{
    Matches a device by its canonical compiler/runtime ID.
  }];
  let parameters = (ins
    AttrParameter<"StringAttr", "">:$pattern
  );
  let builders = [
    AttrBuilder<(ins "StringRef":$pattern), [{
      return $_get(context, StringAttr::get(context, pattern));
    }]>,
    AttrBuilderWithInferredContext<(ins "StringAttr":$pattern), [{
      return $_get(pattern.getContext(), pattern);
    }]>,
    AttrBuilderWithInferredContext<(ins "IREE::HAL::DeviceTargetAttr":$target), [{
      return $_get(target.getContext(), target.getDeviceID());
    }]>,
  ];
  let hasCustomAssemblyFormat = 1;
}

def HAL_DeviceMatchFeatureAttr :
    AttrDef<HAL_Dialect, "DeviceMatchFeature", [
      DeclareAttrInterfaceMethods<HAL_MatchAttrInterface>,
    ]> {
  let mnemonic = "device.match.feature";
  let summary = [{matches against a supported device feature pattern}];
  let description = [{
    Matches a device that supports the given feature. The format of the feature
    pattern is device-dependent.
  }];
  let parameters = (ins
    AttrParameter<"StringAttr", "">:$pattern
  );
  let builders = [
    AttrBuilder<(ins "StringRef":$pattern), [{
      return $_get(context, StringAttr::get(context, pattern));
    }]>,
    AttrBuilderWithInferredContext<(ins "StringAttr":$pattern), [{
      return $_get(pattern.getContext(), pattern);
    }]>,
  ];
  let hasCustomAssemblyFormat = 1;
}

def HAL_DeviceMatchArchitectureAttr :
    AttrDef<HAL_Dialect, "DeviceMatchArchitecture", [
      DeclareAttrInterfaceMethods<HAL_MatchAttrInterface>,
    ]> {
  let mnemonic = "device.match.architecture";
  let summary = [{matches against a device architecture pattern}];
  let description = [{
    Matches a device by its runtime architecture. The format of the architecture
    pattern is device-dependent.
  }];
  let parameters = (ins
    AttrParameter<"StringAttr", "">:$pattern
  );
  let builders = [
    AttrBuilder<(ins "StringRef":$pattern), [{
      return $_get(context, StringAttr::get(context, pattern));
    }]>,
    AttrBuilderWithInferredContext<(ins "StringAttr":$pattern), [{
      return $_get(pattern.getContext(), pattern);
    }]>,
  ];
  let hasCustomAssemblyFormat = 1;
}

def HAL_DeviceMatchExecutableFormatAttr :
    AttrDef<HAL_Dialect, "DeviceMatchExecutableFormat", [
      DeclareAttrInterfaceMethods<HAL_MatchAttrInterface>,
    ]> {
  let mnemonic = "device.match.executable.format";
  let summary = [{matches when a device supports the given executable format}];
  let description = [{
    Matches a device only if it claims to support the given executable format
    pattern. It's still possible that the executable cannot be loaded such as if
    it uses unavailable device features. This is used for queries such as
    "can you load ELF libraries?" to quickly get to a set of executables to
    attempt without needing to try dozens that definitely cannot be loaded.

    Note that different devices may share the same executable formats: for
    example a local synchronous CPU executor and a remote asynchronous CPU
    executor can both load ELF libraries. It's also possible for the same device
    to support multiple formats such as being able to load both
    platform-agnostic ELF libraries and platform-specific DLL/MachO/etc
    libraries.
  }];
  let parameters = (ins
    AttrParameter<"StringAttr", "">:$pattern
  );
  let builders = [
    AttrBuilder<(ins "StringRef":$pattern), [{
      return $_get(context, StringAttr::get(context, pattern));
    }]>,
    AttrBuilderWithInferredContext<(ins "StringAttr":$pattern), [{
      return $_get(pattern.getContext(), pattern);
    }]>,
    AttrBuilderWithInferredContext<(ins "IREE::HAL::ExecutableTargetAttr":$target), [{
      return $_get(target.getContext(), target.getFormat());
    }]>,
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Base HAL op classes
//===----------------------------------------------------------------------===//

class HAL_Op<string mnemonic, list<Trait> traits = []> :
    Op<HAL_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;
}

#endif  // IREE_DIALECT_HAL_BASE
