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

include "iree/compiler/Dialect/CommonBase.td"

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
def HAL_EventList : TupleOf<[RefPtrOf<HAL_Event>]>;

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

def HAL_ExecutableFormatAttr : IntegerAttrBase<I32, "uint32_t">;
def HAL_ExecutableDataAttr : IntElementsAttr<8>;

def HAL_DeviceSize : TypeAlias<I32>;
def HAL_DeviceSizeAttr : IntegerAttrBase<I32, "device_size_t">;

def HAL_HostSize : TypeAlias<I32>;
def HAL_HostSizeAttr : IntegerAttrBase<I32, "size_t">;

def HAL_TimelineValue : TypeAlias<I32>;

def HAL_Status : TypeAlias<I32>;

def HAL_Dim : I<32>;
def HAL_Dims : VectorOf<[HAL_Dim]>;
def HAL_Shape : TypeAlias<HAL_Dims>;

def HAL_Workgroups : VectorOfLengthAndType<[3], [I32]> {
  let typeDescription = [{
    An (X, Y, Z) workgroup count for a dispatch. These values will be multiplied
    by the workgroup size of the executable being dispatched to compute the
    total invocation count.
  }];
}

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
