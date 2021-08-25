// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_HAL_BASE
#define IREE_DIALECT_HAL_BASE

include "iree/compiler/Dialect/HAL/IR/HALDialect.td"
include "iree/compiler/Dialect/HAL/IR/HALInterfaces.td"

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

def HAL_MemoryType_None           : BitEnumAttrCase<"None",           0x0000>;  // ?
def HAL_MemoryType_Transient      : BitEnumAttrCase<"Transient",      0x0001>;  // T
def HAL_MemoryType_HostVisible    : BitEnumAttrCase<"HostVisible",    0x0002>;  // h
def HAL_MemoryType_HostCoherent   : BitEnumAttrCase<"HostCoherent",   0x0004>;  // c
def HAL_MemoryType_HostCached     : BitEnumAttrCase<"HostCached",     0x0008>;  // C
def HAL_MemoryType_HostLocal      : BitEnumAttrCase<"HostLocal",      0x0006>;  // H
def HAL_MemoryType_DeviceVisible  : BitEnumAttrCase<"DeviceVisible",  0x0010>;  // d
def HAL_MemoryType_DeviceLocal    : BitEnumAttrCase<"DeviceLocal",    0x0030>;  // D
def HAL_MemoryTypeBitfieldAttr :
    BitEnumAttr<"MemoryTypeBitfield", "valid MemoryType", [
      HAL_MemoryType_None,
      HAL_MemoryType_Transient,
      HAL_MemoryType_HostVisible,
      HAL_MemoryType_HostCoherent,
      HAL_MemoryType_HostCached,
      HAL_MemoryType_HostLocal,
      HAL_MemoryType_DeviceVisible,
      HAL_MemoryType_DeviceLocal,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_MemoryAccess_None         : BitEnumAttrCase<"None",         0x0000>;  // ?
def HAL_MemoryAccess_Read         : BitEnumAttrCase<"Read",         0x0001>;  // R
def HAL_MemoryAccess_Write        : BitEnumAttrCase<"Write",        0x0002>;  // W
def HAL_MemoryAccess_Discard      : BitEnumAttrCase<"Discard",      0x0004>;  // D
def HAL_MemoryAccess_DiscardWrite : BitEnumAttrCase<"DiscardWrite", 0x0006>;
def HAL_MemoryAccess_MayAlias     : BitEnumAttrCase<"MayAlias",     0x0008>;  // A
def HAL_MemoryAccess_All          : BitEnumAttrCase<"All",          0x0007>;
def HAL_MemoryAccessBitfieldAttr :
    BitEnumAttr<"MemoryAccessBitfield", "valid MemoryAccess", [
      HAL_MemoryAccess_None,
      HAL_MemoryAccess_Read,
      HAL_MemoryAccess_Write,
      HAL_MemoryAccess_Discard,
      HAL_MemoryAccess_DiscardWrite,
      HAL_MemoryAccess_MayAlias,
      HAL_MemoryAccess_All,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_BufferUsage_None      : BitEnumAttrCase<"None",     0x0000>;  // ?
def HAL_BufferUsage_Constant  : BitEnumAttrCase<"Constant", 0x0001>;  // C
def HAL_BufferUsage_Transfer  : BitEnumAttrCase<"Transfer", 0x0002>;  // T
def HAL_BufferUsage_Mapping   : BitEnumAttrCase<"Mapping",  0x0004>;  // M
def HAL_BufferUsage_Dispatch  : BitEnumAttrCase<"Dispatch", 0x0008>;  // D
def HAL_BufferUsage_All       : BitEnumAttrCase<"All",      0x000E>;
def HAL_BufferUsageBitfieldAttr :
    BitEnumAttr<"BufferUsageBitfield", "valid BufferUsage", [
      HAL_BufferUsage_None,
      HAL_BufferUsage_Constant,
      HAL_BufferUsage_Transfer,
      HAL_BufferUsage_Mapping,
      HAL_BufferUsage_Dispatch,
      HAL_BufferUsage_All,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_CommandBufferMode_None : BitEnumAttrCase<"None", 0x0000>;
def HAL_CommandBufferMode_OneShot : BitEnumAttrCase<"OneShot", 0x0001>;
def HAL_CommandBufferMode_AllowInlineExecution : BitEnumAttrCase<"AllowInlineExecution", 0x0010>;
def HAL_CommandBufferModeBitfieldAttr :
    BitEnumAttr<"CommandBufferModeBitfield", "valid CommandBufferMode", [
      HAL_CommandBufferMode_None,
      HAL_CommandBufferMode_OneShot,
      HAL_CommandBufferMode_AllowInlineExecution,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_CommandCategory_None : BitEnumAttrCase<"None", 0x0000>;
def HAL_CommandCategory_Transfer : BitEnumAttrCase<"Transfer", 0x0001>;
def HAL_CommandCategory_Dispatch : BitEnumAttrCase<"Dispatch", 0x0002>;
def HAL_CommandCategoryBitfieldAttr :
    BitEnumAttr<"CommandCategoryBitfield", "valid CommandCategory", [
      HAL_CommandCategory_None,
      HAL_CommandCategory_Transfer,
      HAL_CommandCategory_Dispatch,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_DescriptorType_UniformBuffer : I32EnumAttrCase<"UniformBuffer", 6>;
def HAL_DescriptorType_StorageBuffer : I32EnumAttrCase<"StorageBuffer", 7>;
def HAL_DescriptorType_UniformBufferDynamic : I32EnumAttrCase<"UniformBufferDynamic", 8>;
def HAL_DescriptorType_StorageBufferDynamic : I32EnumAttrCase<"StorageBufferDynamic", 9>;
def HAL_DescriptorTypeAttr :
    I32EnumAttr<"DescriptorType", "IREE HAL DescriptorType", [
      HAL_DescriptorType_UniformBuffer,
      HAL_DescriptorType_StorageBuffer,
      HAL_DescriptorType_UniformBufferDynamic,
      HAL_DescriptorType_StorageBufferDynamic,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}

def HAL_DescriptorSetLayoutUsageType_Immutable : I32EnumAttrCase<"Immutable", 0>;
def HAL_DescriptorSetLayoutUsageType_PushOnly : I32EnumAttrCase<"PushOnly", 1>;
def HAL_DescriptorSetLayoutUsageTypeAttr :
    I32EnumAttr<"DescriptorSetLayoutUsageType", "IREE HAL DescriptorSetLayoutType", [
      HAL_DescriptorSetLayoutUsageType_Immutable,
      HAL_DescriptorSetLayoutUsageType_PushOnly,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}

def HAL_ExecutionStage_None : BitEnumAttrCase<"None", 0x0000>;
def HAL_ExecutionStage_CommandIssue : BitEnumAttrCase<"CommandIssue", 0x0001>;
def HAL_ExecutionStage_CommandProcess : BitEnumAttrCase<"CommandProcess", 0x0002>;
def HAL_ExecutionStage_Dispatch : BitEnumAttrCase<"Dispatch", 0x0004>;
def HAL_ExecutionStage_Transfer : BitEnumAttrCase<"Transfer", 0x0008>;
def HAL_ExecutionStage_CommandRetire : BitEnumAttrCase<"CommandRetire", 0x0010>;
def HAL_ExecutionStage_Host : BitEnumAttrCase<"Host", 0x0020>;
def HAL_ExecutionStageBitfieldAttr :
    BitEnumAttr<"ExecutionStageBitfield", "valid ExecutionStage", [
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

def HAL_ExecutionBarrierFlag_None : BitEnumAttrCase<"None", 0x0000>;
def HAL_ExecutionBarrierFlag_Reserved : BitEnumAttrCase<"Reserved", 0x0001>;
def HAL_ExecutionBarrierFlagBitfieldAttr :
    BitEnumAttr<"ExecutionBarrierFlagBitfield", "valid ExecutionBarrierFlag", [
      HAL_ExecutionBarrierFlag_None,
      HAL_ExecutionBarrierFlag_Reserved,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_AccessScope_None : BitEnumAttrCase<"None", 0x0000>;
def HAL_AccessScope_IndirectCommandRead : BitEnumAttrCase<"IndirectCommandRead", 0x0001>;
def HAL_AccessScope_ConstantRead : BitEnumAttrCase<"ConstantRead", 0x0002>;
def HAL_AccessScope_DispatchRead : BitEnumAttrCase<"DispatchRead", 0x0004>;
def HAL_AccessScope_DispatchWrite : BitEnumAttrCase<"DispatchWrite", 0x0008>;
def HAL_AccessScope_TransferRead : BitEnumAttrCase<"TransferRead", 0x0010>;
def HAL_AccessScope_TransferWrite : BitEnumAttrCase<"TransferWrite", 0x0020>;
def HAL_AccessScope_HostRead : BitEnumAttrCase<"HostRead", 0x0040>;
def HAL_AccessScope_HostWrite : BitEnumAttrCase<"HostWrite", 0x0080>;
def HAL_AccessScope_MemoryRead : BitEnumAttrCase<"MemoryRead", 0x0100>;
def HAL_AccessScope_MemoryWrite : BitEnumAttrCase<"MemoryWrite", 0x0200>;
def HAL_AccessScopeBitfieldAttr :
    BitEnumAttr<"AccessScopeBitfield", "valid AccessScope", [
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

def HAL_TimelineValue : TypeAlias<Index>;

def HAL_PrimitiveType : AnyTypeOf<[Index, AnySignlessInteger, AnyFloat]>;

def HAL_GlobalRefAttr : AliasedSymbolRefAttr;
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

def HAL_CommandQueueAffinityAttr : SignlessIntElementsAttr<32> {
  // TODO(b/143184519): add typeDescription support to other things.
  // let description = [{
  //   A bitmask defining which queues an operation is allowed to execute on.
  //   The selection is wrapped to the total number of available queues, so 0b0101
  //   would enable queues 0 and 2 if there were four queues or queue 0 if there
  //   were two queues.
  // }];
}

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

def HAL_BufferConstraintsAttr :
  Util_StructAttr<"buffer_constraints", "BufferConstraintsAttr", HAL_Dialect, [
    // The maximum size of a memory allocation that can be created, even if
    // there is more space available in the heap.
    Util_StructFieldAttr<"max_allocation_size", HAL_DeviceSizeAttr>,
    // The minimum required alignment, in bytes, for offsets used in runtime
    // buffer bindings for target backends. Offset values (both dynamic and
    // static) must be an integer multiple of this limit.
    Util_StructFieldAttr<"min_buffer_offset_alignment", HAL_DeviceSizeAttr>,
    // The maximum value that can be specified for size ranges of buffer
    // bindings. The underlying allocation may be larger than this but only
    // up to this amount will be visible to kernels.
    Util_StructFieldAttr<"max_buffer_range", HAL_DeviceSizeAttr>,
    // The minimum required alignment, in bytes, for size ranges of buffer
    // bindings.
    Util_StructFieldAttr<"min_buffer_range_alignment", HAL_DeviceSizeAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_DescriptorSetLayoutBindingAttr :
  Util_StructAttr<"descriptor_set_layout_binding",
                  "DescriptorSetLayoutBindingAttr",
                  HAL_Dialect, [
    Util_StructFieldAttr<"binding", I32Attr>,
    Util_StructFieldAttr<"type", HAL_DescriptorTypeAttr>,
    Util_StructFieldAttr<"access", HAL_MemoryAccessBitfieldAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_DescriptorSetLayoutBindingArrayAttr : TypedArrayAttrBase<
    HAL_DescriptorSetLayoutBindingAttr,
    "HAL descriptor set layout binding array attribute">;

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

    // Returns the constraints defining buffer allocation and usage allowed on
    // the device or the host defaults if none are defined.
    BufferConstraintsAttr getBufferConstraints();

    // Returns zero or more executable targets that this device supports.
    SmallVector<ExecutableTargetAttr, 4> getExecutableTargets();

    // Returns a list of target devices that may be active for the given
    // operation. This will recursively walk parent operations until one with
    // the `hal.device.targets` attribute is found.
    static SmallVector<DeviceTargetAttr, 4> lookup(Operation *op);

    // Returns a set of buffer constraints compatible with the host.
    // These must only be used with device buffers when it is known that the
    // device is local.
    static BufferConstraintsAttr
    getDefaultHostBufferConstraints(MLIRContext *context);

    // Tries to find the min/max constraints on buffers across all target
    // devices applicable to the given operation. If no target device specifies
    // constraints then a generally acceptable default set with minimums will be
    // returned.
    //
    // This can result in non-optimal constraints when multi-targeting wildly
    // different device types and should only be used as a fallback.
    static BufferConstraintsAttr
    lookupConservativeBufferConstraints(Operation *op);

    // Returns a list of all target executable configurations that may be
    // required for the given operation.
    static SmallVector<ExecutableTargetAttr, 4>
    lookupExecutableTargets(Operation *op);
  }];
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
}

//===----------------------------------------------------------------------===//
// Experimental binding metadata
//===----------------------------------------------------------------------===//
// TODO(benvanik): drop this when we have a HAL dispatch abstraction.

def HAL_ExConstantStorageAttr :
  Util_StructAttr<"ex.constant_storage",
                  "ExConstantStorageAttr",
                  HAL_Dialect, [
    Util_StructFieldAttr<"binding", StrAttr>,
    Util_StructFieldAttr<"storage", StrAttr>,
    Util_StructFieldAttr<"offset", IndexAttr>,
    Util_StructFieldAttr<"length", IndexAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}
def HAL_ExPushConstantAttr :
  Util_StructAttr<"ex.push_constant",
                  "ExPushConstantAttr",
                  HAL_Dialect, [
    Util_StructFieldAttr<"ordinal", IndexAttr>,
    Util_StructFieldAttr<"operand", IndexAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}
def HAL_ExOperandBufferAttr :
  Util_StructAttr<"ex.operand_buffer",
                  "ExOperandBufferAttr",
                  HAL_Dialect, [
    Util_StructFieldAttr<"binding", StrAttr>,
    Util_StructFieldAttr<"operand", IndexAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}
def HAL_ExResultBufferAttr :
  Util_StructAttr<"ex.result_buffer",
                  "ExResultBufferAttr",
                  HAL_Dialect, [
    Util_StructFieldAttr<"binding", StrAttr>,
    Util_StructFieldAttr<"result", IndexAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
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
}

//===----------------------------------------------------------------------===//
// Base HAL op classes
//===----------------------------------------------------------------------===//

class HAL_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<HAL_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, &result); }];
  let printer = [{ return print$cppClass(p, *this); }];
}

#endif  // IREE_DIALECT_HAL_BASE
