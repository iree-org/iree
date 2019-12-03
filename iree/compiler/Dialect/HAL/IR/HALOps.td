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

#ifndef IREE_DIALECT_HAL_OPS
#define IREE_DIALECT_HAL_OPS

include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// Magic temporary hacks
//===----------------------------------------------------------------------===//
// TODO(benvanik): remove these as the sequencer/other HAL ops are added.

def HAL_ExSharedDeviceOp : HAL_PureOp<"ex.shared_device", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let results = (outs
    RefPtrOf<HAL_Device>:$result
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state
    }], [{
      state.addTypes({RefPtrType::get(DeviceType::get(builder->getContext()))});
    }]>,
  ];
}

def HAL_ExCacheExecutableOp : HAL_PureOp<"ex.cache_executable", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let arguments = (ins
    RefPtrOf<HAL_Device>:$device,
    FlatSymbolRefAttr:$executable
  );
  let results = (outs
    RefPtrOf<HAL_Executable>:$result
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state,
      Value *device, StringRef executable
    }], [{
      state.addOperands({device});
      state.addAttribute("executable", builder->getSymbolRefAttr(executable));
      state.addTypes({RefPtrType::get(ExecutableType::get(builder->getContext()))});
    }]>,
  ];
}

def HAL_ExExecutableDescriptorSetLayoutOp :
  HAL_PureOp<"ex.executable_descriptor_set_layout", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let arguments = (ins
    RefPtrOf<HAL_Executable>:$executable,
    I32Attr:$set
  );
  let results = (outs
    RefPtrOf<HAL_DescriptorSetLayout>:$result
  );
}

// TODO(benvanik): replace with resource sets.
def HAL_ExDeferReleaseOp : HAL_Op<"ex.defer_release"> {
  let arguments = (ins
    AnyRefPtr:$operand
  );
}

def HAL_ExSubmitAndWaitOp : HAL_Op<"ex.submit_and_wait", [YieldPoint]> {
  let arguments = (ins
    RefPtrOf<HAL_Device>:$device,
    RefPtrOf<HAL_CommandBuffer>:$command_buffer
  );
}

//===----------------------------------------------------------------------===//
// HAL struct definition ops
//===----------------------------------------------------------------------===//

def HAL_MakeMemoryBarrierOp : HAL_MakeTupleOp<"make_memory_barrier", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{temporary memory barrier allocation operation}];
  let description = [{
    Allocates a temporary MemoryBarrier struct that can be passed to the
    command buffer barrier operations.
  }];

  let arguments = (ins
    HAL_AccessScopeBitfieldAttr:$source_scope,
    HAL_AccessScopeBitfieldAttr:$target_scope
  );
  let results = (outs
    HAL_MemoryBarrier:$result
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state,
      IREE::HAL::AccessScopeBitfield sourceScope,
      IREE::HAL::AccessScopeBitfield targetScope
    }]>,
  ];
}

def HAL_MakeBufferBarrierOp : HAL_MakeTupleOp<"make_buffer_barrier", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{temporary buffer barrier allocation operation}];
  let description = [{
    Allocates a temporary BufferBarrier struct that can be passed to the
    command buffer barrier operations.
  }];

  let arguments = (ins
    HAL_AccessScopeBitfieldAttr:$source_scope,
    HAL_AccessScopeBitfieldAttr:$target_scope,
    RefPtrOf<HAL_Buffer>:$buffer,
    HAL_DeviceSize:$offset,
    HAL_DeviceSize:$length
  );
  let results = (outs
    HAL_BufferBarrier:$result
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state,
      IREE::HAL::AccessScopeBitfield sourceScope,
      IREE::HAL::AccessScopeBitfield targetScope,
      Value *buffer, Value *offset, Value *length
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// iree::hal::Allocator
//===----------------------------------------------------------------------===//

def HAL_AllocatorComputeSizeOp : HAL_PureOp<"allocator.compute_size", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{buffer allocation size computation operation}];
  let description = [{
    Computes the byte size required for a buffer of the given shape and type.
  }];

  let arguments = (ins
    RefPtrOf<HAL_Allocator>:$allocator,
    HAL_MemoryTypeBitfieldAttr:$memory_types,
    HAL_BufferUsageBitfieldAttr:$buffer_usage,
    HAL_Shape:$shape,
    I32Attr:$element_size
  );
  let results = (outs
    HAL_DeviceSize:$result
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value *allocator,
      IREE::HAL::MemoryTypeBitfield memoryTypes,
      IREE::HAL::BufferUsageBitfield bufferUsage,
      ArrayRef<Value *> shape, int32_t elementSize
    }]>,
  ];
}

def HAL_AllocatorAllocateOp : HAL_Op<"allocator.allocate", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{empty buffer allocation operation}];
  let description = [{
    Allocates a buffer from the allocator.
  }];

  let arguments = (ins
    RefPtrOf<HAL_Allocator>:$allocator,
    HAL_MemoryTypeBitfieldAttr:$memory_types,
    HAL_BufferUsageBitfieldAttr:$buffer_usage,
    HAL_DeviceSize:$allocation_size
  );
  let results = (outs
    RefPtrOf<HAL_Buffer>:$result
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value *allocator,
      IREE::HAL::MemoryTypeBitfield memoryTypes,
      IREE::HAL::BufferUsageBitfield bufferUsage,
      Value *allocationSize
    }]>,
  ];

  let hasCanonicalizer = 1;
}

def HAL_AllocatorAllocateShapedOp : HAL_Op<"allocator.allocate_shaped", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{empty buffer allocation operation}];
  let description = [{
    Allocates a buffer from the allocator.
    Acts as a fusion of hal.allocator.compute_size and hal.allocator.allocate
    for when only a single buffer is being allocated.
  }];

  let arguments = (ins
    RefPtrOf<HAL_Allocator>:$allocator,
    HAL_MemoryTypeBitfieldAttr:$memory_types,
    HAL_BufferUsageBitfieldAttr:$buffer_usage,
    HAL_Shape:$shape,
    I32Attr:$element_size
  );
  let results = (outs
    RefPtrOf<HAL_Buffer>:$result
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value *allocator,
      IREE::HAL::MemoryTypeBitfield memoryTypes,
      IREE::HAL::BufferUsageBitfield bufferUsage,
      ArrayRef<Value *> shape, int32_t elementSize
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// iree::hal::Buffer
//===----------------------------------------------------------------------===//

// TODO(benvanik): clone buffer op.

def HAL_BufferSubspanOp : HAL_PureOp<"buffer.subspan", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{buffer subspan operation}];
  let description = [{
    Returns a reference to a subspan of the buffer.
  }];

  let arguments = (ins
    RefPtrOf<HAL_Buffer>:$source_buffer,
    HAL_DeviceSize:$source_offset,
    HAL_DeviceSize:$length
  );
  let results = (outs
    RefPtrOf<HAL_Buffer>:$result
  );
}

def HAL_BufferFillOp : HAL_Op<"buffer.fill"> {
  let summary = [{buffer fill operation}];
  let description = [{
    Fills the target buffer with the given repeating value.
  }];

  let arguments = (ins
    RefPtrOf<HAL_Buffer>:$target_buffer,
    HAL_DeviceSize:$target_offset,
    HAL_DeviceSize:$length,
    I32:$pattern
  );
}

def HAL_BufferReadDataOp : HAL_Op<"buffer.read_data"> {
  let summary = [{buffer-to-heap read operation}];
  let description = [{
    Reads a block of byte data from the resource at the given offset.
  }];

  let arguments = (ins
    RefPtrOf<HAL_Buffer>:$source_buffer,
    HAL_DeviceSize:$source_offset,
    RefPtrOf<MutableByteBufferType>:$target_buffer,
    HAL_DeviceSize:$target_offset,
    HAL_DeviceSize:$length
  );
}

def HAL_BufferWriteDataOp : HAL_Op<"buffer.write_data"> {
  let summary = [{heap-to-buffer write operation}];
  let description = [{
    Writes a block of byte data into the resource at the given offset.
  }];

  let arguments = (ins
    HAL_HostBufferRef:$source_buffer,
    HAL_DeviceSize:$source_offset,
    RefPtrOf<HAL_Buffer>:$target_buffer,
    HAL_DeviceSize:$target_offset,
    HAL_DeviceSize:$length
  );
}

def HAL_BufferCopyDataOp : HAL_Op<"buffer.copy_data"> {
  let summary = [{buffer-to-buffer copy operation}];
  let description = [{
    Copies data from the provided source_buffer into the buffer.
  }];

  let arguments = (ins
    RefPtrOf<HAL_Buffer>:$source_buffer,
    HAL_DeviceSize:$source_offset,
    RefPtrOf<HAL_Buffer>:$target_buffer,
    HAL_DeviceSize:$target_offset,
    HAL_DeviceSize:$length
  );
}

//===----------------------------------------------------------------------===//
// iree::hal::BufferView
//===----------------------------------------------------------------------===//

def HAL_BufferViewComputeOffsetOp : HAL_PureOp<"buffer_view.compute_offset", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    SameVariadicOperandSize,
  ]> {
  let summary = [{buffer view indices to byte offset computation operation}];
  let description = [{
    Computes an element byte offset within a buffer.
  }];

  let arguments = (ins
    RefPtrOf<HAL_Buffer>:$buffer,
    HAL_Shape:$shape,
    HAL_Dims:$indices
  );

  let results = (outs
    HAL_DeviceSize:$offset
  );
}

def HAL_BufferViewComputeRangeOp : HAL_PureOp<"buffer_view.compute_range", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    SameVariadicOperandSize,
  ]> {
  let summary = [{buffer view byte range computation operation}];
  let description = [{
    Computes a byte range within a buffer for one or more elements.
  }];

  let arguments = (ins
    RefPtrOf<HAL_Buffer>:$buffer,
    HAL_Shape:$shape,
    HAL_Dims:$indices,
    HAL_Dims:$lengths
  );
  let results = (outs
    // TODO(benvanik): return a strides tuple instead, or one per dim.
    HAL_DeviceSize:$offset,
    HAL_DeviceSize:$length
  );
}

def HAL_BufferViewSliceOp : HAL_PureOp<"buffer_view.slice", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    SameVariadicOperandSize,
  ]> {
  let summary = [{buffer view slice operation}];
  let description = [{
    Returns a view into a buffer. The buffer is not copied and both the original
    and sliced references must be synchronized.
  }];

  let arguments = (ins
    RefPtrOf<HAL_Buffer>:$buffer,
    HAL_Shape:$shape,
    HAL_Dims:$indices,
    HAL_Dims:$lengths
  );
  let results = (outs
    RefPtrOf<HAL_Buffer>:$result
  );
}

//===----------------------------------------------------------------------===//
// iree::hal::CommandBuffer
//===----------------------------------------------------------------------===//

def HAL_CommandBufferCreateOp : HAL_Op<"command_buffer.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{command buffer allocation operation}];
  let description = [{
    Returns a command buffer from the device pool ready to begin recording.
  }];

  let arguments = (ins
    RefPtrOf<HAL_Device>:$device,
    HAL_CommandBufferModeBitfieldAttr:$modes,
    HAL_CommandCategoryBitfieldAttr:$command_categories
  );
  let results = (outs
    RefPtrOf<HAL_CommandBuffer>:$result
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value *device,
      IREE::HAL::CommandBufferModeBitfield modes,
      IREE::HAL::CommandCategoryBitfield commandCategories
    }]>,
  ];
}

def HAL_CommandBufferBeginOp : HAL_Op<"command_buffer.begin"> {
  let summary = [{command buffer recording begin operation}];
  let description = [{
    Resets and begins recording into the command buffer, clearing all previously
    recorded contents.
  }];

  let arguments = (ins
    RefPtrOf<HAL_CommandBuffer>:$command_buffer
  );
}

def HAL_CommandBufferEndOp : HAL_Op<"command_buffer.end"> {
  let summary = [{command buffer recording end operation}];
  let description = [{
    Ends recording into the command buffer.
  }];

  let arguments = (ins
    RefPtrOf<HAL_CommandBuffer>:$command_buffer
  );
}

def HAL_CommandBufferExecutionBarrierOp : HAL_Op<"command_buffer.execution_barrier", [
    AttrSizedOperandSegments,
  ]> {
  let summary = [{command buffer execution barrier recording operation}];
  let description = [{
    Defines a memory dependency between commands recorded before and after the
    barrier.
  }];

  let arguments = (ins
    RefPtrOf<HAL_CommandBuffer>:$command_buffer,
    HAL_ExecutionStageBitfieldAttr:$source_stage_mask,
    HAL_ExecutionStageBitfieldAttr:$target_stage_mask,
    Variadic<HAL_MemoryBarrier>:$memory_barriers,
    Variadic<HAL_BufferBarrier>:$buffer_barriers
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value *commandBuffer,
      IREE::HAL::ExecutionStageBitfield sourceStageMask,
      IREE::HAL::ExecutionStageBitfield targetStageMask,
      ArrayRef<Value *> memoryBarriers, ArrayRef<Value *> bufferBarriers
    }]>,
  ];
}

// TODO(benvanik): event ops.

def HAL_CommandBufferFillBufferOp : HAL_Op<"command_buffer.fill_buffer"> {
  let summary = [{command buffer buffer fill recording operation}];
  let description = [{
    Fills the target buffer with the given repeating value.
  }];

  let arguments = (ins
    RefPtrOf<HAL_CommandBuffer>:$command_buffer,
    RefPtrOf<HAL_Buffer>:$target_buffer,
    HAL_DeviceSize:$target_offset,
    HAL_DeviceSize:$length,
    I32:$pattern
  );
}

// TODO(benvanik): update buffer op.

def HAL_CommandBufferCopyBufferOp : HAL_Op<"command_buffer.copy_buffer"> {
  let summary = [{command buffer buffer copy recording operation}];
  let description = [{
    Copies a range of one buffer to another.
  }];

  let arguments = (ins
    RefPtrOf<HAL_CommandBuffer>:$command_buffer,
    RefPtrOf<HAL_Buffer>:$source_buffer,
    HAL_DeviceSize:$source_offset,
    RefPtrOf<HAL_Buffer>:$target_buffer,
    HAL_DeviceSize:$target_offset,
    HAL_DeviceSize:$length
  );
}

def HAL_CommandBufferBindDescriptorSetOp :
    HAL_Op<"command_buffer.bind_descriptor_set"> {
  let summary = [{command buffer descriptor set binding operation}];
  let description = [{
    Binds a descriptor set to the given set number. The provided descriptor set
    must not be modified once bound to a command buffer.
  }];

  let arguments = (ins
    RefPtrOf<HAL_CommandBuffer>:$command_buffer,
    RefPtrOf<HAL_Executable>:$executable,
    I32Attr:$set,
    RefPtrOf<HAL_DescriptorSet>:$descriptor_set,
    Variadic<HAL_DeviceSize>:$dynamic_offsets
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value *commandBuffer,
      Value *executable, uint32_t set, Value *descriptorSet,
      ArrayRef<Value *> dynamicOffsets = {}
    }]>,
  ];
}

def HAL_CommandBufferDispatchOp : HAL_Op<"command_buffer.dispatch"> {
  let summary = [{command buffer dispatch recording operation}];
  let description = [{
    Dispatches an execution request.
  }];

  let arguments = (ins
    RefPtrOf<HAL_CommandBuffer>:$command_buffer,
    RefPtrOf<HAL_Executable>:$executable,
    HAL_OrdinalAttr:$entry_point,
    HAL_Workgroups:$workgroups
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value *commandBuffer,
      Value *executable, IREE::HAL::ExecutableEntryPointOp entryPoint,
      Value *workgroups
    }]>,
  ];
}

def HAL_CommandBufferDispatchIndirectOp : HAL_Op<"command_buffer.dispatch.indirect"> {
  let summary = [{command buffer indirect dispatch recording operation}];
  let description = [{
    Dispatches an execution request with the dispatch parameters loaded from the
    given buffer.
  }];

  let arguments = (ins
    RefPtrOf<HAL_CommandBuffer>:$command_buffer,
    RefPtrOf<HAL_Executable>:$executable,
    HAL_OrdinalAttr:$entry_point,
    RefPtrOf<HAL_Buffer>:$workgroups_buffer,
    HAL_DeviceSize:$workgroups_offset
  );
}

//===----------------------------------------------------------------------===//
// iree::hal::DescriptorSet
//===----------------------------------------------------------------------===//

def HAL_DescriptorSetAllocateOp : HAL_PureOp<"descriptor_set.allocate", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{allocates a descriptor set from the device pool}];
  let description = [{
    Allocates a DescriptorSet from the device pool.
  }];

  let arguments = (ins
    RefPtrOf<HAL_Device>:$device,
    RefPtrOf<HAL_DescriptorSetLayout>:$set_layout
  );
  let results = (outs
    RefPtrOf<HAL_DescriptorSet>:$result
  );
}

def HAL_DescriptorSetMakeBindingOp :
  HAL_MakeTupleOp<"descriptor_set.make_binding", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{temporary buffer binding allocation operation}];
  let description = [{
    Allocates a temporary DescriptorSet binding struct that can be passed to the
    hal.descriptor_set.update operation.
  }];

  let arguments = (ins
    I32Attr:$binding,
    RefPtrOf<HAL_Buffer>:$buffer,
    HAL_DeviceSize:$offset,
    HAL_DeviceSize:$length,
    HAL_MemoryAccessBitfieldAttr:$access
  );
  let results = (outs
    HAL_DescriptorSetBinding:$result
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state,
      int32_t binding, Value *buffer, Value *offset, Value *length,
      IREE::HAL::MemoryAccessBitfield access
    }]>,
  ];
}

def HAL_DescriptorSetUpdateOp : HAL_Op<"descriptor_set.update"> {
  let summary = [{updates one or more bindings in a descriptor set}];
  let description = [{
    Updates bindings in a descriptor set. The descriptor set must not be
    recorded into any command buffer that has not yet completed; updating when
    in use will cause undefined behavior.
  }];

  let arguments = (ins
    RefPtrOf<HAL_Device>:$device,
    RefPtrOf<HAL_DescriptorSet>:$set,
    Variadic<HAL_DescriptorSetBinding>:$bindings
  );
}

//===----------------------------------------------------------------------===//
// iree::hal::Device
//===----------------------------------------------------------------------===//

def HAL_DeviceAllocatorOp : HAL_PureOp<"device.allocator", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{device allocator accessor operation}];
  let description = [{
    Returns the allocator that can be used to allocate buffers compatible with
    the device.
  }];

  let arguments = (ins
    RefPtrOf<HAL_Device>:$device
  );
  let results = (outs
    RefPtrOf<HAL_Allocator>:$result
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value *device
    }], [{
      state.addOperands({device});
      state.addTypes({RefPtrType::get(AllocatorType::get(builder->getContext()))});
    }]>,
  ];
}

// TODO(benvanik): additional factory functions and submission ops.

//===----------------------------------------------------------------------===//
// iree::hal::Executable
//===----------------------------------------------------------------------===//

// TODO(benvanik): executable runtime type.

def HAL_ExecutableOp : HAL_Op<"executable", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"IREE::HAL::ExecutableEndOp">,
    Symbol,
    SymbolTable,
  ]> {
  let summary = [{target-specific executable module}];
  let description = [{
    An executable module representing a target-specific compiled
    kernel/shader/etc.
  }];

  let arguments = (ins
    StrAttr:$sym_name
    // TODO(benvanik): entry point types for verification.
  );

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, StringRef name
    }]>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return body().front(); }
  }];

  let verifier = [{ return verifyExecutableOp(*this); }];
}

def HAL_ExecutableEndOp : HAL_Op<"executable_end", [
    HasParent<"IREE::HAL::ExecutableOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the executable op}];
  let parser = [{ return parseRegionEndOp(parser, &result); }];
  let printer = [{ return printRegionEndOp(p, *this); }];
}

def HAL_ExecutableEntryPointOp : HAL_Op<"executable.entry_point", [Symbol]> {
  let summary = [{compiled executable binary data}];
  let description = [{
    A compiled executable binary with an optional nested module containing the
    IR prior to serialization (for debugging).
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    HAL_OrdinalAttr:$ordinal,
    HAL_WorkgroupSizeAttr:$workgroup_size
  );
}

// TODO(benvanik): add HAL_ExecutableSourceOp for auto serialization.

def HAL_ExecutableBinaryOp : HAL_Op<"executable.binary", [
    IsolatedFromAbove,
    HasParent<"IREE::HAL::ExecutableOp">,
    SingleBlockImplicitTerminator<"IREE::HAL::ExecutableBinaryEndOp">,
  ]> {
  let summary = [{compiled executable binary data}];
  let description = [{
    A compiled executable binary with an optional nested module containing the
    IR prior to serialization (for debugging).
  }];

  let arguments = (ins
    HAL_ExecutableFormatAttr:$format,
    HAL_ExecutableDataAttr:$data
    // TODO(benvanik): add compatibility and versioning attributes.
  );

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, uint32_t format,
      std::vector<uint8_t> data
    }]>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return body().front(); }

    llvm::Optional<::mlir::ModuleOp> getInnerModule() {
      auto moduleOps = getBlock().getOps<::mlir::ModuleOp>();
      if (moduleOps.empty()) return llvm::None;
      return *moduleOps.begin();
    }
  }];

  let verifier = [{ return verifyExecutableBinaryOp(*this); }];
}

def HAL_ExecutableBinaryEndOp : HAL_Op<"executable.binary_end", [
    HasParent<"IREE::HAL::ExecutableBinaryOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the executable binary op}];
  let parser = [{ return parseRegionEndOp(parser, &result); }];
  let printer = [{ return printRegionEndOp(p, *this); }];
}

//===----------------------------------------------------------------------===//
// iree::hal::ExecutableCache
//===----------------------------------------------------------------------===//

// TODO(benvanik): executable cache.

//===----------------------------------------------------------------------===//
// iree::hal::Fence
//===----------------------------------------------------------------------===//

// TODO(benvanik): fence query.

//===----------------------------------------------------------------------===//
// iree::hal::RingBuffer
//===----------------------------------------------------------------------===//

// TODO(benvanik): ring buffer.

//===----------------------------------------------------------------------===//
// iree::hal::Semaphore
//===----------------------------------------------------------------------===//

// TODO(benvanik): semaphore query/set.

#endif  // IREE_DIALECT_HAL_OPS
