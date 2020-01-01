// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_DIALECT_HAL_BASE
#define IREE_DIALECT_HAL_BASE

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

//===----------------------------------------------------------------------===//
// IREE HAL (Hardware Abstraction Layer) dialect
//===----------------------------------------------------------------------===//

def HAL_Dialect : Dialect {
  let name = "hal";
  let cppNamespace = "IREE::HAL";

  let summary = [{
    A dialect representing operations against the IREE HAL.
  }];
  let description = [{
    This can be thought of as a Vulkan-like model with all of the graphics bits
    chopped out.

    The type set is limited to those that can be represented in the IREE HAL
    design: buffers and views, synchronization primitives like semaphores, and
    and command buffers. The intent is that if a device could implement the HAL
    interface the sequencer ops could run on that device, such as being able to
    run on a GPU via indirect command buffers.

    Though this is mostly a 1:1 mapping to the iree::hal API there are some
    methods omitted as they are not likely to be needed in IR. It's assumed that
    either sequencer interfaces will encapsulate the logic (such as device
    resolution) or that certain features are unsafe to expose to user-defined
    input.
  }];
}

//===----------------------------------------------------------------------===//
// HAL enums
//===----------------------------------------------------------------------===//

def HAL_MemoryType_None : BitEnumAttrCase<"None", 0x0000>;
def HAL_MemoryType_Transient : BitEnumAttrCase<"Transient", 0x0001>;
def HAL_MemoryType_HostVisible : BitEnumAttrCase<"HostVisible", 0x0002>;
def HAL_MemoryType_HostCoherent : BitEnumAttrCase<"HostCoherent", 0x0004>;
def HAL_MemoryType_HostCached : BitEnumAttrCase<"HostCached", 0x0008>;
def HAL_MemoryType_HostLocal : BitEnumAttrCase<"HostLocal", 0x0006>;
def HAL_MemoryType_DeviceVisible : BitEnumAttrCase<"DeviceVisible", 0x0010>;
def HAL_MemoryType_DeviceLocal : BitEnumAttrCase<"DeviceLocal", 0x0030>;
def HAL_MemoryTypeBitfieldAttr :
    BitEnumAttr<"MemoryTypeBitfield", "valid MemoryType", [
      HAL_MemoryType_None,
      HAL_MemoryType_Transient,
      HAL_MemoryType_HostVisible,
      HAL_MemoryType_HostCoherent,
      HAL_MemoryType_HostCached,
      HAL_MemoryType_HostLocal,
      HAL_MemoryType_DeviceVisible,
      HAL_MemoryType_DeviceLocal
    ]> {
  let returnType = "mlir::iree_compiler::IREE::HAL::MemoryTypeBitfield";
  let convertFromStorage = "static_cast<mlir::iree_compiler::IREE::HAL::MemoryTypeBitfield>($_self.getInt())";
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_MemoryAccess_None : BitEnumAttrCase<"None", 0x0000>;
def HAL_MemoryAccess_Read : BitEnumAttrCase<"Read", 0x0001>;
def HAL_MemoryAccess_Write : BitEnumAttrCase<"Write", 0x0002>;
def HAL_MemoryAccess_Discard : BitEnumAttrCase<"Discard", 0x0004>;
def HAL_MemoryAccess_DiscardWrite : BitEnumAttrCase<"DiscardWrite", 0x0006>;
def HAL_MemoryAccess_All : BitEnumAttrCase<"All", 0x0007>;
def HAL_MemoryAccessBitfieldAttr :
    BitEnumAttr<"MemoryAccessBitfield", "valid MemoryAccess", [
      HAL_MemoryAccess_None,
      HAL_MemoryAccess_Read,
      HAL_MemoryAccess_Write,
      HAL_MemoryAccess_Discard,
      HAL_MemoryAccess_DiscardWrite,
      HAL_MemoryAccess_All
    ]> {
  let returnType = "mlir::iree_compiler::IREE::HAL::MemoryAccessBitfield";
  let convertFromStorage = "static_cast<mlir::iree_compiler::IREE::HAL::MemoryAccessBitfield>($_self.getInt())";
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_BufferUsage_None : BitEnumAttrCase<"None", 0x0000>;
def HAL_BufferUsage_Constant : BitEnumAttrCase<"Constant", 0x0001>;
def HAL_BufferUsage_Transfer : BitEnumAttrCase<"Transfer", 0x0002>;
def HAL_BufferUsage_Mapping : BitEnumAttrCase<"Mapping", 0x0004>;
def HAL_BufferUsage_Dispatch : BitEnumAttrCase<"Dispatch", 0x0008>;
def HAL_BufferUsage_All : BitEnumAttrCase<"All", 0x000F>;
def HAL_BufferUsageBitfieldAttr :
    BitEnumAttr<"BufferUsageBitfield", "valid BufferUsage", [
      HAL_BufferUsage_None,
      HAL_BufferUsage_Constant,
      HAL_BufferUsage_Transfer,
      HAL_BufferUsage_Mapping,
      HAL_BufferUsage_Dispatch,
      HAL_BufferUsage_All
    ]> {
  let returnType = "mlir::iree_compiler::IREE::HAL::BufferUsageBitfield";
  let convertFromStorage = "static_cast<mlir::iree_compiler::IREE::HAL::BufferUsageBitfield>($_self.getInt())";
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_CommandBufferMode_None : BitEnumAttrCase<"None", 0x0000>;
def HAL_CommandBufferMode_OneShot : BitEnumAttrCase<"OneShot", 0x0001>;
def HAL_CommandBufferModeBitfieldAttr :
    BitEnumAttr<"CommandBufferModeBitfield", "valid CommandBufferMode", [
      HAL_CommandBufferMode_None,
      HAL_CommandBufferMode_OneShot
    ]> {
  let returnType = "mlir::iree_compiler::IREE::HAL::CommandBufferModeBitfield";
  let convertFromStorage = "static_cast<mlir::iree_compiler::IREE::HAL::CommandBufferModeBitfield>($_self.getInt())";
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

def HAL_CommandCategory_None : BitEnumAttrCase<"None", 0x0000>;
def HAL_CommandCategory_Transfer : BitEnumAttrCase<"Transfer", 0x0001>;
def HAL_CommandCategory_Dispatch : BitEnumAttrCase<"Dispatch", 0x0002>;
def HAL_CommandCategoryBitfieldAttr :
    BitEnumAttr<"CommandCategoryBitfield", "valid CommandCategory", [
      HAL_CommandCategory_None,
      HAL_CommandCategory_Transfer,
      HAL_CommandCategory_Dispatch
    ]> {
  let returnType = "mlir::iree_compiler::IREE::HAL::CommandCategoryBitfield";
  let convertFromStorage = "static_cast<mlir::iree_compiler::IREE::HAL::CommandCategoryBitfield>($_self.getInt())";
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
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
  let returnType = "mlir::iree_compiler::IREE::HAL::ExecutionStageBitfield";
  let convertFromStorage = "static_cast<mlir::iree_compiler::IREE::HAL::ExecutionStageBitfield>($_self.getInt())";
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
  let returnType = "mlir::iree_compiler::IREE::HAL::AccessScopeBitfield";
  let convertFromStorage = "static_cast<mlir::iree_compiler::IREE::HAL::AccessScopeBitfield>($_self.getInt())";
  let cppNamespace = "mlir::iree_compiler::IREE::HAL";
}

//===----------------------------------------------------------------------===//
// HAL types
//===----------------------------------------------------------------------===//

def HAL_Allocator : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::AllocatorType>()">,
    "allocator"> {
  let typeDescription = [{
    Allocates buffers for a particular device memory space.
  }];
}

def HAL_Buffer : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::BufferType>()">,
    "buffer"> {
  let typeDescription = [{
    A memory buffer with a specific memory_type that is used to describe the
    capabilities and behavior of the backing memory of the buffer. Buffers may
    be any mix of host-accessible, host-coherent, or device-accessible for
    various usages. Depending on these memory types the buffers may be mapped
    for access on the host as memory though certain restrictions may be imposed.
  }];
}

def HAL_CommandBuffer : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::CommandBufferType>()">,
    "command_buffer"> {
  let typeDescription = [{
    Asynchronous command buffer recording interface. Commands are recorded by
    the implementation for later submission to command queues.
  }];
}

def HAL_DescriptorSet : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::DescriptorSetType>()">,
    "descriptor_set"> {
  let typeDescription = [{
    Descriptor set.
  }];
}

def HAL_DescriptorSetLayout : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::DescriptorSetLayoutType>()">,
    "descriptor_set_layout"> {
  let typeDescription = [{
    Descriptor set layout.
  }];
}

def HAL_Device : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::DeviceType>()">,
    "device"> {
  let typeDescription = [{
    Logical device instance.
  }];
}

def HAL_Event : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::EventType>()">,
    "event"> {
  let typeDescription = [{
    Events are used for defining synchronization scopes within CommandBuffers.
    An event only exists within a single CommandBuffer and must not be used
    across CommandBuffers from the same device or others.
  }];
}

def HAL_Executable : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::ExecutableType>()">,
    "executable"> {
  let typeDescription = [{
    A prepared and ready-to-dispatch executable.
  }];
}

def HAL_ExecutableCache : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::ExecutableCacheType>()">,
    "executable_cache"> {
  let typeDescription = [{
    A cache of prepared executables for a particular device.
    Caches may be shared across multiple devices from the same driver or
    specific to individual devices. Caches may persist prepared executables
    across process launches or reprepare them each run. Callers should assume
    that the cache is a no-op and the returned Executables only live for as long
    as the cache does.
  }];
}

def HAL_Fence : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::FenceType>()">,
    "fence"> {
  let typeDescription = [{
    Synchronization mechanism for device->host notification.
    Fences behave like timeline semaphores and contain a monotonically
    increasing uint64_t payload. They may be waited on any number of times -
    even if they have already been signaled.
  }];
}

def HAL_RingBuffer : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::RingBufferType>()">,
    "ring_buffer"> {
  let typeDescription = [{
    Ringbuffer used for transient buffer allocation.
  }];
}

def HAL_Semaphore : DialectType<
    HAL_Dialect,
    CPred<"$_self.isa<IREE::HAL::SemaphoreType>()">,
    "semaphore"> {
  let typeDescription = [{
    A synchronization primitive used to indicate submission dependencies.
    Semaphores are either of type binary (signaled or unsignaled) or timeline
    (uint64_t payload with >= semantics).
  }];
}

def HAL_OrdinalAttr : IntegerAttrBase<I32, "32-bit integer ordinal attribute">;

// Values are FourCC codes.
// See https://github.com/google/iree/tree/master/iree/iree/hal/executable_format.h for more info.
// https://ubershmekel.github.io/fourcc-to-text/ can be used to generate the
// integer values for inclusion here.
//
// LINT.IfChange(executable_format)
def HAL_EF_Unspecified : I32EnumAttrCase<"Unspecified", 538976288>;
def HAL_EF_MlirText : I32EnumAttrCase<"MlirText", 1296845138>;
def HAL_EF_IreeBytecode : I32EnumAttrCase<"IreeBytecode", 1230128453>;
def HAL_EF_SpirV : I32EnumAttrCase<"SpirV", 1397773893>;
def HAL_ExecutableFormatAttr :
    I32EnumAttr<"ExecutableFormat", "IREE HAL Executable format", [
      HAL_EF_Unspecified,
      HAL_EF_MlirText,
      HAL_EF_IreeBytecode,
      HAL_EF_SpirV,
    ]> {
  let returnType = "uint32_t";
  let convertFromStorage = "static_cast<uint32_t>($_self.getInt())";
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}
// LINT.ThenChange(//iree/iree/hal/executable_format.h:fourcc)

def HAL_ExecutableDataAttr : IntElementsAttr<8>;

def HAL_DeviceSize : TypeAlias<I32>;
def HAL_DeviceSizeAttr : IntegerAttrBase<I32, "device_size_t">;

def HAL_HostSize : TypeAlias<I32>;
def HAL_HostSizeAttr : IntegerAttrBase<I32, "size_t">;

def HAL_TimelineValue : TypeAlias<I32>;

def HAL_Status : TypeAlias<I32>;

def HAL_PrimitiveType : AnyTypeOf<[Index, AnyInteger, AnyFloat]>;

def HAL_VariableRefAttr : AliasedSymbolRefAttr;
def HAL_VariableType : AnyTypeOf<[HAL_PrimitiveType, AnyVector, AnyRefPtr]>;

def HAL_Dim : I<32>;
def HAL_Dims : Variadic<HAL_Dim>;
def HAL_Shape : Variadic<HAL_Dim>;

// TODO(cl/277443143): moving into OpBase.td.
// A `width`-bit integer elements attribute. The attribute should be
// ranked and has a shape as specified in `dims`.
class HAL_RankedIntElementsAttr<int width, list<int> dims> : ElementsAttrBase<
  CPred<"$_self.isa<DenseIntElementsAttr>() &&"
      "$_self.cast<DenseIntElementsAttr>().getType()."
      "getElementType().isInteger(" # width # ") && "
      // Check that this is ranked and has the specified shape.
      "$_self.cast<DenseIntElementsAttr>().getType().hasRank() && "
      "$_self.cast<DenseIntElementsAttr>().getType().getShape() == "
      "llvm::ArrayRef<int64_t>({" # StrJoinInt<dims>.result # "})">,
  width # "-bit integer elements attribute of shape [" #
  StrJoinInt<dims>.result # "]"> {

  let storageType = [{ DenseIntElementsAttr }];
  let returnType = [{ DenseIntElementsAttr }];

  let constBuilderCall = "DenseElementsAttr::get("
    "VectorType::get({" # StrJoinInt<dims>.result #
    "}, $_builder.getIntegerType(" # width # ")), "
    "llvm::makeArrayRef($0)).cast<DenseIntElementsAttr>()";
  let convertFromStorage = "$_self";
}

def HAL_WorkgroupSizeAttr : HAL_RankedIntElementsAttr<32, [3]>;

def HAL_HostBufferRef : AnyTypeOf<[
  RefPtrOf<ByteBufferType>,
  RefPtrOf<MutableByteBufferType>,
]>;

def HAL_CommandQueueAffinityAttr : IntElementsAttr<32> {
  // TODO(b/143184519): add typeDescription support to other things.
  // let typeDescription = [{
  //   A bitmask defining which queues an operation is allowed to execute on.
  //   The selection is wrapped to the total number of available queues, so 0b0101
  //   would enable queues 0 and 2 if there were four queues or queue 0 if there
  //   were two queues.
  // }];
}

def HAL_DurationMillisAttr : IntElementsAttr<32> {
  // TODO(b/143184519): add typeDescription support to other things.
  // let typeDescription = [{
  //   A duration to wait in milliseconds. 0 indicates that the operation should
  //   return immediately without waiting and can be used as a way to poll handles.
  //   INT32_MAX will wait forever until the handle is signaled.
  // }];
}

//===----------------------------------------------------------------------===//
// HAL structs
//===----------------------------------------------------------------------===//

def HAL_MemoryBarrier : NamedTupleOf<[
    NamedTupleElement<0, "source_scope", I32>,
    NamedTupleElement<1, "target_scope", I32>
  ], "MemoryBarrier"> {
  let typeDescription = [{
    MemoryBarrier struct that can be passed to the command buffer barrier
    operations.
  }];
}
def HAL_MemoryBarrierList : TupleOf<[HAL_MemoryBarrier]>;

def HAL_BufferBarrier : NamedTupleOf<[
    NamedTupleElement<0, "source_scope", I32>,
    NamedTupleElement<1, "target_scope", I32>,
    NamedTupleElement<2, "buffer", RefPtrOf<HAL_Buffer>>,
    NamedTupleElement<3, "offset", HAL_DeviceSize>,
    NamedTupleElement<4, "length", HAL_DeviceSize>
  ], "BufferBarrier"> {
  let typeDescription = [{
    BufferBarrier struct that can be passed to the command buffer barrier
    operations.
  }];
}
def HAL_BufferBarrierList : TupleOf<[HAL_BufferBarrier]>;

def HAL_DescriptorSetBinding : NamedTupleOf<[
    NamedTupleElement<0, "binding", I32>,
    NamedTupleElement<1, "buffer", RefPtrOf<HAL_Buffer>>,
    NamedTupleElement<2, "buffer_offset", HAL_DeviceSize>,
    NamedTupleElement<3, "buffer_length", HAL_DeviceSize>,
    NamedTupleElement<4, "access", I32>,
  ], "DescriptorSetBinding"> {
  let typeDescription = [{
    DescriptorSetBinding struct that can be passed to the
    hal.descriptor_set.update operation. Constructed via the
    hal.descriptor_set.make_binding operation.
  }];
}

def HAL_FenceValue : NamedTupleOf<[
    NamedTupleElement<0, "fence", RefPtrOf<HAL_Fence>>,
    NamedTupleElement<1, "value", I64>
  ], "FenceValue"> {
  let typeDescription = [{
    A reference to a fence and associated payload value.
  }];
}

def HAL_SemaphoreValue : NamedTupleOf<[
    NamedTupleElement<0, "semaphore", RefPtrOf<HAL_Semaphore>>,
    NamedTupleElement<1, "value", I64>
  ], "SemaphoreValue"> {
  let typeDescription = [{
    A reference to a timline semaphore and associated payload value.
  }];
}

//===----------------------------------------------------------------------===//
// Base HAL op classes
//===----------------------------------------------------------------------===//

def HAL_OpInterface : OpInterface<"HALOp"> {
  let description = [{
    Interface for HAL ops.
  }];
}

class HAL_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<HAL_Dialect, mnemonic, !listconcat(traits, [HAL_OpInterface])> {
  let parser = [{ return parse$cppClass(parser, &result); }];
  let printer = [{ return print$cppClass(p, *this); }];
}

class HAL_PureOp<string mnemonic, list<OpTrait> traits = []> :
    HAL_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

class HAL_MakeTupleOp<string mnemonic, list<OpTrait> traits = []> :
    HAL_PureOp<mnemonic, traits>;

#endif  // IREE_DIALECT_HAL_BASE
