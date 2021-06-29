// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_HAL_BASE
#define IREE_DIALECT_HAL_BASE

include "iree/compiler/Dialect/HAL/IR/HALDialect.td"

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

def HAL_OrdinalAttr : IREE_IndexAttrBase<"size_t">;

def HAL_ExecutableDataAttr : SignlessIntElementsAttr<8>;

def HAL_ElementType : TypeAlias<I32>;
def HAL_ElementTypeAttr : SignlessIntegerAttrBase<
  I32, "element type attribute">;

def HAL_DeviceSize : TypeAlias<Index>;
def HAL_DeviceSizeAttr : IREE_IndexAttrBase<"iree_device_size_t">;
def HAL_DeviceSizes : Variadic<HAL_DeviceSize>;

def HAL_HostSize : TypeAlias<Index>;
def HAL_HostSizeAttr : IREE_IndexAttrBase<"size_t">;

def HAL_TimelineValue : TypeAlias<Index>;

def HAL_PrimitiveType : AnyTypeOf<[Index, AnySignlessInteger, AnyFloat]>;

def HAL_VariableRefAttr : AliasedSymbolRefAttr;
def HAL_VariableType : AnyTypeOf<[HAL_PrimitiveType, AnyVector, HAL_ObjectType]>;
def HAL_VariablePtr : PtrOf<HAL_VariableType>;

def HAL_IndexAttr : IREE_IndexAttrBase<"index">;
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
    IREE_IndexAttrBase<"size_t">,
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
  IREE_StructAttr<"buffer_constraints", "BufferConstraintsAttr", HAL_Dialect, [
    // The maximum size of a memory allocation that can be created, even if
    // there is more space available in the heap.
    IREE_StructFieldAttr<"max_allocation_size", HAL_DeviceSizeAttr>,
    // The minimum required alignment, in bytes, for offsets used in runtime
    // buffer bindings for target backends. Offset values (both dynamic and
    // static) must be an integer multiple of this limit.
    IREE_StructFieldAttr<"min_buffer_offset_alignment", HAL_DeviceSizeAttr>,
    // The maximum value that can be specified for size ranges of buffer
    // bindings. The underlying allocation may be larger than this but only
    // up to this amount will be visible to kernels.
    IREE_StructFieldAttr<"max_buffer_range", HAL_DeviceSizeAttr>,
    // The minimum required alignment, in bytes, for size ranges of buffer
    // bindings.
    IREE_StructFieldAttr<"min_buffer_range_alignment", HAL_DeviceSizeAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_ByteRangeAttr :
  IREE_StructAttr<"byte_range", "ByteRangeAttr", HAL_Dialect, [
    IREE_StructFieldAttr<"offset", HAL_DeviceSizeAttr>,
    IREE_StructFieldAttr<"length", HAL_DeviceSizeAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_DescriptorSetLayoutBindingAttr :
  IREE_StructAttr<"descriptor_set_layout_binding",
                  "DescriptorSetLayoutBindingAttr",
                  HAL_Dialect, [
    IREE_StructFieldAttr<"binding", I32Attr>,
    IREE_StructFieldAttr<"type", HAL_DescriptorTypeAttr>,
    IREE_StructFieldAttr<"access", HAL_MemoryAccessBitfieldAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_DescriptorSetLayoutBindingArrayAttr : TypedArrayAttrBase<
    HAL_DescriptorSetLayoutBindingAttr,
    "HAL descriptor set layout binding array attribute">;

// TODO(benvanik): drop this when we have a HAL dispatch abstraction.
def HAL_ExConstantStorageAttr :
  IREE_StructAttr<"ex.constant_storage",
                  "ExConstantStorageAttr",
                  HAL_Dialect, [
    IREE_StructFieldAttr<"binding", StrAttr>,
    IREE_StructFieldAttr<"storage", StrAttr>,
    IREE_StructFieldAttr<"offset", IndexAttr>,
    IREE_StructFieldAttr<"length", IndexAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}
def HAL_ExPushConstantAttr :
  IREE_StructAttr<"ex.push_constant",
                  "ExPushConstantAttr",
                  HAL_Dialect, [
    IREE_StructFieldAttr<"ordinal", IndexAttr>,
    IREE_StructFieldAttr<"operand", IndexAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}
def HAL_ExOperandBufferAttr :
  IREE_StructAttr<"ex.operand_buffer",
                  "ExOperandBufferAttr",
                  HAL_Dialect, [
    IREE_StructFieldAttr<"binding", StrAttr>,
    IREE_StructFieldAttr<"operand", IndexAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}
def HAL_ExResultBufferAttr :
  IREE_StructAttr<"ex.result_buffer",
                  "ExResultBufferAttr",
                  HAL_Dialect, [
    IREE_StructFieldAttr<"binding", StrAttr>,
    IREE_StructFieldAttr<"result", IndexAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

//===----------------------------------------------------------------------===//
// Expression matching attributes
//===----------------------------------------------------------------------===//

def HAL_DeviceQuery : NativeOpTrait<"IREE::HAL::DeviceQuery">;

def HAL_MatchAlwaysAttr :
  IREE_StructAttr<"match.always", "MatchAlwaysAttr", HAL_Dialect, []> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_MatchAnyAttr :
  IREE_StructAttr<"match.any", "MatchAnyAttr", HAL_Dialect, [
    IREE_StructFieldAttr<"conditions", AnyAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_MatchAllAttr :
  IREE_StructAttr<"match.all", "MatchAllAttr", HAL_Dialect, [
    IREE_StructFieldAttr<"conditions", AnyAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_DeviceMatchIDAttr :
  IREE_StructAttr<"device.match.id", "DeviceMatchIDAttr", HAL_Dialect, [
    IREE_StructFieldAttr<"pattern", StrAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_DeviceMatchMemoryModelAttr : IREE_StructAttr<
    "device.match.memory_model", "DeviceMatchMemoryModelAttr", HAL_Dialect, [
    IREE_StructFieldAttr<"memory_model", HAL_MemoryModelAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_DeviceMatchFeatureAttr :
  IREE_StructAttr<"device.match.feature",
                  "DeviceMatchFeatureAttr",
                  HAL_Dialect, [
    IREE_StructFieldAttr<"pattern", StrAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_DeviceMatchArchitectureAttr :
  IREE_StructAttr<"device.match.architecture",
                  "DeviceMatchArchitectureAttr",
                  HAL_Dialect, [
    IREE_StructFieldAttr<"pattern", StrAttr>,
  ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
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
