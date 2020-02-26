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
    HAL_Device:$result
  );

  let assemblyFormat = "attr-dict `:` type($result)";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state
    }], [{
      state.addTypes({DeviceType::get(builder->getContext())});
    }]>,
  ];
}

def HAL_ExCacheExecutableOp : HAL_PureOp<"ex.cache_executable", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let arguments = (ins
    HAL_Device:$device,
    FlatSymbolRefAttr:$executable
  );
  let results = (outs
    HAL_Executable:$result
  );

  let assemblyFormat = "$device `,` $executable attr-dict `:` type($result)";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state,
      Value device, StringRef executable
    }], [{
      state.addOperands({device});
      state.addAttribute("executable", builder->getSymbolRefAttr(executable));
      state.addTypes({ExecutableType::get(builder->getContext())});
    }]>,
  ];
}

// TODO(benvanik): remove and replace with descriptor sets.
def HAL_ExPushBindingOp : HAL_Op<"ex.push_binding"> {
  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    I32Attr:$ordinal,
    HAL_Buffer:$buffer,
    HAL_Shape:$shape,
    HAL_ElementTypeAttr:$element_type
  );

  let assemblyFormat = [{
    $command_buffer `,` $ordinal `,` $buffer `,` `shape` `=` `[` $shape `]` `,`
    `element_type` `=` $element_type attr-dict
  }];
}

// TODO(benvanik): replace with resource sets.
def HAL_ExDeferReleaseOp : HAL_Op<"ex.defer_release"> {
  let arguments = (ins
    HAL_ObjectType:$operand
  );

  let assemblyFormat = "$operand `:` type($operand) attr-dict";
}

def HAL_ExSubmitAndWaitOp : HAL_Op<"ex.submit_and_wait", [YieldPoint]> {
  let arguments = (ins
    HAL_Device:$device,
    HAL_CommandBuffer:$command_buffer
  );

  let assemblyFormat = "$device `,` $command_buffer attr-dict";
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

  let assemblyFormat = [{
    $source_scope `,` $target_scope attr-dict-with-keyword `:` type($result)
  }];

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
    HAL_Buffer:$buffer,
    HAL_DeviceSize:$offset,
    HAL_DeviceSize:$length
  );
  let results = (outs
    HAL_BufferBarrier:$result
  );

  let assemblyFormat = [{
    $source_scope `,` $target_scope `,` operands attr-dict-with-keyword `:`
    type($result)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state,
      IREE::HAL::AccessScopeBitfield sourceScope,
      IREE::HAL::AccessScopeBitfield targetScope,
      Value buffer, Value offset, Value length
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// Global variables
//===----------------------------------------------------------------------===//

def HAL_VariableOp : HAL_Op<"variable", [
    Symbol,
  ]> {
  let summary = [{stateful variable declaration}];
  let description = [{
    Declares a global variable that maintains its value across invocations.
    The value is tied to the execution context of the module and different
    contexts will have different variable storage.
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    TypeAttr:$type,
    UnitAttr:$is_mutable,
    // TODO(benvanik): verify matches $type.
    OptionalAttr<FlatSymbolRefAttr>:$initializer,
    // TODO(benvanik): verify matches $type.
    OptionalAttr<AnyAttr>:$initial_value
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result, StringRef name, bool isMutable,
      Type type,
      Optional<StringRef> initializer, Optional<Attribute> initialValue,
      ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      Builder *builder, OperationState &result, StringRef name, bool isMutable,
      mlir::FuncOp initializer, ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      Builder *builder, OperationState &result, StringRef name, bool isMutable,
      Type type, Attribute initialValue, ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      Builder *builder, OperationState &result, StringRef name, bool isMutable,
      Type type, ArrayRef<NamedAttribute> attrs = {}
    }]>,
  ];

  let verifier = [{ return verifyVariableOp(*this); }];

  let hasCanonicalizer = 1;
}

def HAL_VariableAddressOp : HAL_PureOp<"variable.address"> {
  let summary = [{returns an address reference to a variable}];
  let description = [{
    Returns the address of a variable as a typed reference. Can be used with the
    variable load and store indirect ops.
  }];

  let arguments = (ins
    HAL_VariableRefAttr:$variable
  );
  let results = (outs
    HAL_VariablePtr:$result
  );

  let assemblyFormat = "$variable attr-dict `:` type($result)";
}

def HAL_VariableLoadOp : HAL_Op<"variable.load"> {
  let summary = [{loads a value from a global variable}];
  let description = [{
    Returns a copy of the variable value.
  }];

  let arguments = (ins
    HAL_VariableRefAttr:$variable
  );
  let results = (outs
    HAL_VariableType:$result
  );

  let assemblyFormat = "$variable attr-dict `:` type($result)";

  let verifier = [{ return verifyVariableLoadOp(*this); }];

  let hasCanonicalizer = 1;
}

def FLOW_VariableLoadIndirectOp : HAL_Op<"variable.load.indirect"> {
  let summary = [{loads a value from a global variable}];
  let description = [{
    Returns a copy of the variable value.
  }];

  let arguments = (ins
    HAL_VariablePtr:$variable
  );
  let results = (outs
    HAL_VariableType:$result
  );

  let assemblyFormat = "$variable attr-dict `:` type($variable) `->` type($result)";

  let verifier = [{ return verifyVariableLoadIndirectOp(*this); }];

  let hasCanonicalizer = 1;
}

def HAL_VariableStoreOp : HAL_Op<"variable.store"> {
  let summary = [{stores a value into a global variable}];
  let description = [{
    Stores a copy of the value into a variable.
  }];

  let arguments = (ins
    HAL_VariableType:$value,
    HAL_VariableRefAttr:$variable
  );

  let assemblyFormat = "$value `,` $variable attr-dict `:` type($value)";

  let verifier = [{ return verifyVariableStoreOp(*this); }];

  let hasCanonicalizer = 1;
}

def HAL_VariableStoreIndirectOp : HAL_Op<"variable.store.indirect"> {
  let summary = [{stores a value into a global variable}];
  let description = [{
    Stores a copy of the value into a variable.
  }];

  let arguments = (ins
    HAL_VariableType:$value,
    HAL_VariablePtr:$variable
  );

  let assemblyFormat = "$value `,` $variable attr-dict `:` type($value) `->` type($variable)";

  let verifier = [{ return verifyVariableStoreIndirectOp(*this); }];

  let hasCanonicalizer = 1;
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
    This returns the same value as `hal.buffer_view.byte_length`.
  }];

  let arguments = (ins
    HAL_Allocator:$allocator,
    HAL_Shape:$shape,
    HAL_ElementTypeAttr:$element_type
  );
  let results = (outs
    HAL_DeviceSize:$result
  );

  let assemblyFormat = [{
    $allocator `,` `shape` `=` `[` $shape `]` `,` `element_type` `=`
    $element_type attr-dict
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value allocator,
      ValueRange shape, int32_t elementSize
    }]>,
  ];
}

def HAL_AllocatorComputeOffsetOp : HAL_PureOp<"allocator.compute_offset", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    SameVariadicOperandSize,
  ]> {
  let summary = [{buffer view indices to byte offset computation operation}];
  let description = [{
    Computes an element byte offset within a buffer produced by the allocator.
    This returns the same value as `hal.buffer_view.compute_offset`.
  }];

  let arguments = (ins
    HAL_Allocator:$allocator,
    HAL_Shape:$shape,
    HAL_ElementTypeAttr:$element_type,
    HAL_Dims:$indices
  );

  let results = (outs
    HAL_DeviceSize:$offset
  );

  let assemblyFormat = [{
    $allocator `,` `shape` `=` `[` $shape `]` `,` `element_type` `=`
    $element_type `,` `indices` `=` `[` $indices `]` attr-dict
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value allocator,
      ValueRange shape, int32_t elementType, ValueRange indices
    }]>,
  ];
}

def HAL_AllocatorComputeRangeOp : HAL_PureOp<"allocator.compute_range", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    SameVariadicOperandSize,
  ]> {
  let summary = [{buffer view byte range computation operation}];
  let description = [{
    Computes a byte range within a buffer for one or more elements.
    This returns the same value as `hal.buffer_view.compute_range`.
  }];

  let arguments = (ins
    HAL_Allocator:$allocator,
    HAL_Shape:$shape,
    HAL_ElementTypeAttr:$element_type,
    HAL_Dims:$indices,
    HAL_Dims:$lengths
  );
  let results = (outs
    // TODO(benvanik): return a strides tuple instead, or one per dim.
    HAL_DeviceSize:$offset,
    HAL_DeviceSize:$length
  );

  let assemblyFormat = [{
    $allocator `,` `shape` `=` `[` $shape `]` `,` `element_type` `=`
    $element_type `,` `indices` `=` `[` $indices `]` `,` `lengths` `=` `[`
    $lengths `]` attr-dict
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value allocator,
      ValueRange shape, int32_t elementType, ValueRange indices,
      ValueRange lengths
    }]>,
  ];
}

def HAL_AllocatorAllocateOp : HAL_Op<"allocator.allocate", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{empty buffer allocation operation}];
  let description = [{
    Allocates a buffer of the given size from the allocator.
    The size of the buffer returned may be larger than the requested size if the
    allocator has specific alignment requirements or minimum allocation sizes.
  }];

  let arguments = (ins
    HAL_Allocator:$allocator,
    HAL_MemoryTypeBitfieldAttr:$memory_types,
    HAL_BufferUsageBitfieldAttr:$buffer_usage,
    HAL_DeviceSize:$allocation_size
  );
  let results = (outs
    HAL_Buffer:$result
  );

  let assemblyFormat = [{
    $allocator `,` $memory_types `,` $buffer_usage `,` $allocation_size
    attr-dict-with-keyword `:` type($result)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value allocator,
      IREE::HAL::MemoryTypeBitfield memoryTypes,
      IREE::HAL::BufferUsageBitfield bufferUsage,
      Value allocationSize
    }]>,
  ];
}

def HAL_AllocatorAllocateConstOp : HAL_Op<"allocator.allocate.const", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{constant buffer allocation operation}];
  let description = [{
    Allocates a buffer from the allocator with the given constant contents.
    The buffer contents cannot change after the the point of allocation and in
    most cases should be cached so that the buffer is not reallocated
    repeatedly.
  }];

  let arguments = (ins
    HAL_Allocator:$allocator,
    HAL_MemoryTypeBitfieldAttr:$memory_types,
    HAL_BufferUsageBitfieldAttr:$buffer_usage,
    ElementsAttr:$value
  );
  let results = (outs
    HAL_Buffer:$result
  );

  let assemblyFormat = [{
    $allocator `,` $memory_types `,` $buffer_usage attr-dict-with-keyword `:`
    type($result) `=` $value
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value allocator,
      IREE::HAL::MemoryTypeBitfield memoryTypes,
      IREE::HAL::BufferUsageBitfield bufferUsage,
      ElementsAttr value
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// iree::hal::Buffer
//===----------------------------------------------------------------------===//

def HAL_BufferAllocatorOp : HAL_PureOp<"buffer.allocator", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{buffer allocator accessor operation}];
  let description = [{
    Returns the allocator this buffer was allocated from.
  }];

  let arguments = (ins
    HAL_Buffer:$buffer
  );
  let results = (outs
    HAL_Allocator:$result
  );

  let assemblyFormat = "$buffer `:` type($result) attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value buffer
    }]>,
  ];

  let hasCanonicalizer = 1;
}

// TODO(benvanik): clone buffer op.

def HAL_BufferSubspanOp : HAL_PureOp<"buffer.subspan", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{buffer subspan operation}];
  let description = [{
    Returns a reference to a subspan of the buffer.
  }];

  let arguments = (ins
    HAL_Buffer:$source_buffer,
    HAL_DeviceSize:$source_offset,
    HAL_DeviceSize:$length
  );
  let results = (outs
    HAL_Buffer:$result
  );

  let assemblyFormat = "operands attr-dict `:` type($result)";
}

def HAL_BufferFillOp : HAL_Op<"buffer.fill"> {
  let summary = [{buffer fill operation}];
  let description = [{
    Fills the target buffer with the given repeating value.
  }];

  let arguments = (ins
    HAL_Buffer:$target_buffer,
    HAL_DeviceSize:$target_offset,
    HAL_DeviceSize:$length,
    I32:$pattern
  );

  let assemblyFormat = "operands attr-dict";
}

def HAL_BufferReadDataOp : HAL_Op<"buffer.read_data"> {
  let summary = [{buffer-to-heap read operation}];
  let description = [{
    Reads a block of byte data from the resource at the given offset.
  }];

  let arguments = (ins
    HAL_Buffer:$source_buffer,
    HAL_DeviceSize:$source_offset,
    MutableByteBufferType:$target_buffer,
    HAL_DeviceSize:$target_offset,
    HAL_DeviceSize:$length
  );

  let assemblyFormat = "operands attr-dict `:` type($target_buffer)";
}

def HAL_BufferWriteDataOp : HAL_Op<"buffer.write_data"> {
  let summary = [{heap-to-buffer write operation}];
  let description = [{
    Writes a block of byte data into the resource at the given offset.
  }];

  let arguments = (ins
    HAL_HostBuffer:$source_buffer,
    HAL_DeviceSize:$source_offset,
    HAL_Buffer:$target_buffer,
    HAL_DeviceSize:$target_offset,
    HAL_DeviceSize:$length
  );

  let assemblyFormat = "operands attr-dict `:` type($source_buffer)";
}

def HAL_BufferCopyDataOp : HAL_Op<"buffer.copy_data"> {
  let summary = [{buffer-to-buffer copy operation}];
  let description = [{
    Copies data from the provided source_buffer into the buffer.
  }];

  let arguments = (ins
    HAL_Buffer:$source_buffer,
    HAL_DeviceSize:$source_offset,
    HAL_Buffer:$target_buffer,
    HAL_DeviceSize:$target_offset,
    HAL_DeviceSize:$length
  );

  let assemblyFormat = "operands attr-dict";
}

def HAL_BufferLoadOp : HAL_PureOp<"buffer.load"> {
  let summary = [{buffer element load operation}];
  let description = [{
    Loads a value from a buffer by mapping it.
  }];

  let arguments = (ins
    HAL_Buffer:$source_buffer,
    HAL_DeviceSize:$source_offset
  );
  let results = (outs
    AnyTypeOf<[HAL_PrimitiveType, AnyVector]>:$result
  );

  let assemblyFormat = [{
    $source_buffer `[` $source_offset `]` `:` type($result) attr-dict
  }];
}

def HAL_BufferStoreOp : HAL_Op<"buffer.store"> {
  let summary = [{buffer element store operation}];
  let description = [{
    Stores a value into a buffer by mapping it.
  }];

  let arguments = (ins
    AnyTypeOf<[HAL_PrimitiveType, AnyVector]>:$value,
    HAL_Buffer:$target_buffer,
    HAL_DeviceSize:$target_offset
  );

  let assemblyFormat = [{
    $value `,` $target_buffer `[` $target_offset `]` `:` type($value) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// iree::hal::BufferView
//===----------------------------------------------------------------------===//

def HAL_BufferViewConstOp : HAL_PureOp<"buffer_view.const", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{buffer view constant initializer}];
  let description = [{
    Pseudo-op for allocating a constant buffer view. Expands to a buffer
    allocation and a buffer view wrapper.
  }];

  let arguments = (ins
    HAL_Allocator:$allocator,
    HAL_MemoryTypeBitfieldAttr:$memory_types,
    HAL_BufferUsageBitfieldAttr:$buffer_usage,
    ElementsAttr:$value
  );
  let results = (outs
    HAL_BufferView:$result
  );

  let assemblyFormat = [{
    $allocator `,` $memory_types `,` $buffer_usage `:` type($result)
    attr-dict-with-keyword `=` $value
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value allocator,
      IREE::HAL::MemoryTypeBitfield memoryTypes,
      IREE::HAL::BufferUsageBitfield bufferUsage,
      ElementsAttr value
    }]>,
  ];

  let hasCanonicalizer = 1;
}

def HAL_BufferViewCreateOp : HAL_PureOp<"buffer_view.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{buffer view reference initializer}];
  let description = [{
    Creates a reference to a buffer with a particular shape and element type.
    The buffer is not copied and both the original and view references must be
    synchronized. This makes it easier to associate commonly-carried metadata
    along with the contents.
  }];

  let arguments = (ins
    HAL_Buffer:$buffer,
    HAL_Shape:$shape,
    HAL_ElementTypeAttr:$element_type
  );
  let results = (outs
    HAL_BufferView:$result
  );

  let assemblyFormat = [{
    $buffer `,` `shape` `=` `[` $shape `]` `,` `element_type` `=` $element_type
    `:` type($result) attr-dict
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value buffer,
      ValueRange shape, int32_t elementType
    }]>,
  ];
}

def HAL_BufferViewSubviewOp : HAL_PureOp<"buffer_view.subview", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    SameVariadicOperandSize,
  ]> {
  let summary = [{buffer view subview initializer}];
  let description = [{
    Returns a view into a another buffer view. The buffer is not copied and both
    the original and sliced references must be synchronized.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view,
    HAL_Dims:$indices,
    HAL_Dims:$lengths
  );
  let results = (outs
    HAL_BufferView:$result
  );

  let assemblyFormat = [{
    $buffer_view `,` `indices` `=` `[` $indices `]` `,` `lengths` `=` `[`
    $lengths `]` `:` type($result) attr-dict
  }];
}

def HAL_BufferViewBufferOp : HAL_PureOp<"buffer_view.buffer", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{buffer view buffer accessor}];
  let description = [{
    Returns the buffer backing this view's contents.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view
  );
  let results = (outs
    HAL_Buffer:$result
  );

  let assemblyFormat = "$buffer_view `:` type($result) attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value bufferView
    }]>,
  ];

  let hasCanonicalizer = 1;
}

def HAL_BufferViewByteLengthOp : HAL_PureOp<"buffer_view.byte_length", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{buffer view buffer byte length accessor}];
  let description = [{
    Returns the allocated size of a shaped buffer view in bytes.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view
  );
  let results = (outs
    HAL_DeviceSize:$result
  );

  let assemblyFormat = "$buffer_view `:` type($result) attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value bufferView
    }]>,
  ];
}

def HAL_BufferViewComputeOffsetOp : HAL_PureOp<"buffer_view.compute_offset", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{buffer view indices to byte offset computation operation}];
  let description = [{
    Computes an element byte offset within a buffer view.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view,
    HAL_Dims:$indices
  );

  let results = (outs
    HAL_DeviceSize:$offset
  );

  let assemblyFormat = [{
    $buffer_view `,` `indices` `=` `[` $indices `]` attr-dict
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value bufferView,
      ValueRange indices
    }]>,
  ];
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
    HAL_BufferView:$buffer_view,
    HAL_Dims:$indices,
    HAL_Dims:$lengths
  );
  let results = (outs
    // TODO(benvanik): return a strides tuple instead, or one per dim.
    HAL_DeviceSize:$offset,
    HAL_DeviceSize:$length
  );

  let assemblyFormat = [{
    $buffer_view `,` `indices` `=` `[` $indices `]` `,` `lengths` `=` `[`
    $lengths `]` attr-dict
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value bufferView,
      ValueRange indices, ValueRange lengths
    }]>,
  ];
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
    HAL_Device:$device,
    HAL_CommandBufferModeBitfieldAttr:$modes,
    HAL_CommandCategoryBitfieldAttr:$command_categories
  );
  let results = (outs
    HAL_CommandBuffer:$result
  );

  let assemblyFormat = [{
    $device `,` $modes `,` $command_categories attr-dict-with-keyword `:`
    type($result)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value device,
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
    HAL_CommandBuffer:$command_buffer
  );

  let assemblyFormat = "$command_buffer attr-dict";
}

def HAL_CommandBufferEndOp : HAL_Op<"command_buffer.end"> {
  let summary = [{command buffer recording end operation}];
  let description = [{
    Ends recording into the command buffer.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer
  );

  let assemblyFormat = "$command_buffer attr-dict";
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
    HAL_CommandBuffer:$command_buffer,
    HAL_ExecutionStageBitfieldAttr:$source_stage_mask,
    HAL_ExecutionStageBitfieldAttr:$target_stage_mask,
    Variadic<HAL_MemoryBarrier>:$memory_barriers,
    Variadic<HAL_BufferBarrier>:$buffer_barriers
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value commandBuffer,
      IREE::HAL::ExecutionStageBitfield sourceStageMask,
      IREE::HAL::ExecutionStageBitfield targetStageMask,
      ValueRange memoryBarriers, ValueRange bufferBarriers
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
    HAL_CommandBuffer:$command_buffer,
    HAL_Buffer:$target_buffer,
    HAL_DeviceSize:$target_offset,
    HAL_DeviceSize:$length,
    I32:$pattern
  );

  let assemblyFormat = "operands attr-dict";
}

// TODO(benvanik): update buffer op.

def HAL_CommandBufferCopyBufferOp : HAL_Op<"command_buffer.copy_buffer"> {
  let summary = [{command buffer buffer copy recording operation}];
  let description = [{
    Copies a range of one buffer to another.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    HAL_Buffer:$source_buffer,
    HAL_DeviceSize:$source_offset,
    HAL_Buffer:$target_buffer,
    HAL_DeviceSize:$target_offset,
    HAL_DeviceSize:$length
  );

  let assemblyFormat = "operands attr-dict";
}

def HAL_CommandBufferBindDescriptorSetOp :
    HAL_Op<"command_buffer.bind_descriptor_set"> {
  let summary = [{command buffer descriptor set binding operation}];
  let description = [{
    Binds a descriptor set to the given set number. The provided descriptor set
    must not be modified once bound to a command buffer.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    HAL_ExecutableLayout:$executable_layout,
    I32Attr:$set,
    HAL_DescriptorSet:$descriptor_set,
    Variadic<HAL_DeviceSize>:$dynamic_offsets
  );

  let assemblyFormat = [{
    $command_buffer `,` $executable_layout `,` `set` `=` $set `,`
    $descriptor_set (`,` `offsets` `=` `[` $dynamic_offsets^ `]`)?
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value commandBuffer,
      Value executableLayout, uint32_t set, Value descriptorSet,
      ValueRange dynamicOffsets = {}
    }]>,
  ];
}

def HAL_CommandBufferDispatchOp : HAL_Op<"command_buffer.dispatch"> {
  let summary = [{command buffer dispatch recording operation}];
  let description = [{
    Dispatches an execution request.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    HAL_Executable:$executable,
    HAL_OrdinalAttr:$entry_point,
    HAL_Dim:$workgroup_x,
    HAL_Dim:$workgroup_y,
    HAL_Dim:$workgroup_z
  );

  let assemblyFormat = [{
    $command_buffer `,` $executable `,` `entry_point` `=` $entry_point `,`
    `workgroup_xyz` `=` `[` $workgroup_x `,` $workgroup_y `,` $workgroup_z `]`
    attr-dict
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value commandBuffer,
      Value executable, IREE::HAL::ExecutableEntryPointOp entryPoint,
      Value workgroupX, Value workgroupY, Value workgroupZ
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
    HAL_CommandBuffer:$command_buffer,
    HAL_Executable:$executable,
    HAL_OrdinalAttr:$entry_point,
    HAL_Buffer:$workgroups_buffer,
    HAL_DeviceSize:$workgroups_offset
  );

  let assemblyFormat = [{
    $command_buffer `,` $executable `,` `entry_point` `=` $entry_point `,`
    `workgroups` `=` $workgroups_buffer `[` $workgroups_offset `]` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// iree::hal::DescriptorSet
//===----------------------------------------------------------------------===//

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
    HAL_Buffer:$buffer,
    HAL_DeviceSize:$offset,
    HAL_DeviceSize:$length
  );
  let results = (outs
    HAL_DescriptorSetBinding:$result
  );

  let assemblyFormat = [{
    `binding` `=` $binding `,` operands attr-dict `:` type($result)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state,
      int32_t binding, Value buffer, Value offset, Value length
    }]>,
  ];
}

def HAL_DescriptorSetCreateOp : HAL_PureOp<"descriptor_set.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{allocates a descriptor set from the device pool}];
  let description = [{
    Creates a DescriptorSet from the device pool.
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_DescriptorSetLayout:$set_layout,
    Variadic<HAL_DescriptorSetBinding>:$bindings
  );
  let results = (outs
    HAL_DescriptorSet:$result
  );

  let assemblyFormat = [{
    $device `,` $set_layout `,` `bindings` `=` `[` $bindings `]` attr-dict `:`
    type($result)
  }];
}

//===----------------------------------------------------------------------===//
// iree::hal::DescriptorSetLayout
//===----------------------------------------------------------------------===//

def HAL_DescriptorSetLayoutMakeBindingOp :
  HAL_MakeTupleOp<"descriptor_set_layout.make_binding", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{temporary layout binding allocation operation}];
  let description = [{
    Allocates a temporary DescriptorSetLayout binding struct that can be passed
    to the hal.descriptor_set_layout.create operation.
  }];

  let arguments = (ins
    I32Attr:$binding,
    HAL_DescriptorTypeAttr:$type,
    HAL_MemoryAccessBitfieldAttr:$access
  );
  let results = (outs
    HAL_DescriptorSetLayoutBinding:$result
  );

  let assemblyFormat = [{
    `binding` `=` $binding attr-dict-with-keyword `:` type($result)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state,
      int32_t binding, IREE::HAL::DescriptorType type,
      IREE::HAL::MemoryAccessBitfield access
    }]>,
  ];
}

def HAL_DescriptorSetLayoutCreateOp :
  HAL_PureOp<"descriptor_set_layout.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{creates a descriptor set layout}];
  let description = [{
    Creates a descriptor set layout that defines the bindings used within a set.
  }];

  let arguments = (ins
    HAL_Device:$device,
    Variadic<HAL_DescriptorSetLayoutBinding>:$bindings
  );
  let results = (outs
    HAL_DescriptorSetLayout:$result
  );

  let assemblyFormat = [{
    $device `,` `bindings` `=` `[` $bindings `]` attr-dict `:` type($result)
  }];
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
    HAL_Device:$device
  );
  let results = (outs
    HAL_Allocator:$result
  );

  let assemblyFormat = "$device attr-dict `:` type($result)";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, Value device
    }], [{
      state.addOperands({device});
      state.addTypes({AllocatorType::get(builder->getContext())});
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
  let assemblyFormat = "attr-dict";
}

def HAL_ExecutableEntryPointOp : HAL_Op<"executable.entry_point", [Symbol]> {
  let summary = [{executable entry point declaration}];
  let description = [{
    An entry point exported by the executable with statically-available
    information describing the IO interface it uses and other dispatch metadata.
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

    The `format` attribute specifies a four character code (FourCC) matching the
    executable format in `iree/hal/executable_format.h`.
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
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// iree::hal::ExecutableCache
//===----------------------------------------------------------------------===//

// TODO(benvanik): executable cache.

//===----------------------------------------------------------------------===//
// iree::hal::ExecutableLayout
//===----------------------------------------------------------------------===//

def HAL_ExecutableLayoutCreateOp :
  HAL_PureOp<"executable_layout.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{creates an executable layout}];
  let description = [{
    Creates an executable layout from the given descriptor sets and push
    constant required size. Executable layouts can be shared across any
    executable that uses the same layout and push constant information.
  }];

  let arguments = (ins
    HAL_Device:$device,
    Variadic<HAL_DescriptorSetLayout>:$set_layouts,
    OptionalAttr<I32Attr>:$push_constants
  );
  let results = (outs
    HAL_ExecutableLayout:$result
  );

  let assemblyFormat = [{
    $device `,` `set_layouts` `=` `[` $set_layouts `]`
    (`,` `push_constants` `=` $push_constants^)? attr-dict-with-keyword `:`
    type($result)
  }];
}


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
