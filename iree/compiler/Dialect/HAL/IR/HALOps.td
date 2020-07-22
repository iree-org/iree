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
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class HAL_PureOp<string mnemonic, list<OpTrait> traits = []> :
    HAL_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

class HAL_MakeTupleOp<string mnemonic, list<OpTrait> traits = []> :
    HAL_PureOp<mnemonic, traits>;

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
      OpBuilder &builder, OperationState &state
    }], [{
      state.addTypes({DeviceType::get(builder.getContext())});
    }]>,
  ];
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
      OpBuilder &builder, OperationState &state,
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
      OpBuilder &builder, OperationState &state,
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
      OpBuilder &builder, OperationState &result, StringRef name,
      bool isMutable, Type type,
      Optional<StringRef> initializer, Optional<Attribute> initialValue,
      ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef name,
      bool isMutable, mlir::FuncOp initializer,
      ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef name,
      bool isMutable, Type type, Attribute initialValue,
      ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef name,
      bool isMutable, Type type, ArrayRef<NamedAttribute> attrs = {}
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

def HAL_VariableLoadOp : HAL_Op<"variable.load", [MemoryEffects<[MemRead]>]> {
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
// Control flow
//===----------------------------------------------------------------------===//

def HAL_CheckSuccessOp : HAL_Op<"check_success"> {
  let summary = [{raises a global failure if a status is not 'ok'}];
  let description = [{
    When the status is not 'ok' this signals a runtime failure that causes the
    entire active invocation - and possibly *all* in-flight and pending
    invocations - to fail with the given status. The status will be propagated
    back via the available runtime error handling mechanisms such as semaphores
    or synchronous invocation results.

    As the IREE execution model is deeply pipelined it's possible that failures
    have a latency between when they are emitted and when the application can
    observe the failure. It's also possible that other work that is in-flight
    or pending when the failure occurs will complete.
  }];

  let arguments = (ins
    IREE_Status:$status,
    OptionalAttr<StrAttr>:$message
  );

  let assemblyFormat = [{
    $status (`,` $message^)? attr-dict
  }];

  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result,
      Value status, StringRef message = ""
    }], [{
      build(
        builder, result, status,
        message.empty() ? StringAttr{} : builder.getStringAttr(message));
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
      OpBuilder &builder, OperationState &state, Value allocator,
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
      OpBuilder &builder, OperationState &state, Value allocator,
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
      OpBuilder &builder, OperationState &state, Value allocator,
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
      OpBuilder &builder, OperationState &state, Value allocator,
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
      OpBuilder &builder, OperationState &state, Value allocator,
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
      OpBuilder &builder, OperationState &state, Value buffer
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
      OpBuilder &builder, OperationState &state, Value allocator,
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
      OpBuilder &builder, OperationState &state, Value buffer,
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
      OpBuilder &builder, OperationState &state, Value bufferView
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
      OpBuilder &builder, OperationState &state, Value bufferView
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
      OpBuilder &builder, OperationState &state, Value bufferView,
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
      OpBuilder &builder, OperationState &state, Value bufferView,
      ValueRange indices, ValueRange lengths
    }]>,
  ];
}

def HAL_BufferViewRankOp : HAL_PureOp<"buffer_view.rank"> {
  let summary = [{buffer view rank query}];
  let description = [{
    Returns the rank of the buffer view.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view
  );
  let results = (outs
    Index:$result
  );

  let assemblyFormat = [{$buffer_view attr-dict `:` type($result)}];
}

def HAL_BufferViewDimOp : HAL_PureOp<"buffer_view.dim"> {
  let summary = [{buffer view dimension value query}];
  let description = [{
    Returns the value of the given dimension.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view,
    I32Attr:$index
  );
  let results = (outs
    Index:$result
  );

  let assemblyFormat = [{$buffer_view `,` $index attr-dict `:` type($result)}];
}

def HAL_BufferViewDimsOp : HAL_PureOp<"buffer_view.dims"> {
  let summary = [{buffer view multi-value dimension query}];
  let description = [{
    Returns each dimension value of the buffer view.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view
  );
  let results = (outs
    Variadic<Index>:$result
  );

  let assemblyFormat = [{$buffer_view attr-dict `:` type($result)}];
}

def HAL_BufferViewTraceOp : HAL_Op<"buffer_view.trace", []> {
  let summary = [{trace value(s) operation}];
  let description = [{
    Trace point for dispatchable functions.
  }];

  let arguments = (ins Variadic<HAL_BufferView>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
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
      OpBuilder &builder, OperationState &state, Value device,
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
      OpBuilder &builder, OperationState &state, Value commandBuffer,
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

def HAL_CommandBufferPushConstantsOp :
    HAL_Op<"command_buffer.push_constants"> {
  let summary = [{command buffer push constants operation}];
  let description = [{
    Pushes an inline set of constants that can be accessed by subsequent
    dispatches using a compatible executable layout.

    Push constants are always 4-byte values and treated as opaque, meaning that
    they may be bit-casted floats, bit-packed booleans, etc.

    ```mlir
    hal.command_buffer.push_constants %cmd, %exe_layout,
                                      offset = 0,
                                      values = [%value0, %value1] : i32
    hal.command_buffer.push_constants %cmd, %exe_layout,
                                      offset = 2,
                                      values = [%value2, %value3] : i32
    ```
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    HAL_ExecutableLayout:$executable_layout,
    I32Attr:$offset,
    Variadic<I32>:$values
  );

  let assemblyFormat = [{
    $command_buffer `,` $executable_layout `,` `offset` `=` $offset `,`
    `values` `=` `[` $values `]` `:` `i32` attr-dict-with-keyword
  }];
}

def HAL_CommandBufferPushDescriptorSetOp :
    HAL_Op<"command_buffer.push_descriptor_set", [
      SameVariadicOperandSize,
    ]> {
  let summary = [{command buffer descriptor set push binding operation}];
  let description = [{
    Pushes an inline-defined descriptor set to the command buffer.

    ```mlir
    hal.command_buffer.push_descriptor_set %cmd, %executable_layout, set = 0, bindings = [
      0 = (%buffer_0, %buffer_offset_0, %buffer_length_0),
      1 = (%buffer_1, %buffer_offset_1, %buffer_length_1),
      2 = (%buffer_2, %buffer_offset_2, %buffer_length_2)
    ]
    ```
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    HAL_ExecutableLayout:$executable_layout,
    I32Attr:$set,
    I32ArrayAttr:$bindings,
    Variadic<HAL_Buffer>:$binding_buffers,
    Variadic<HAL_DeviceSize>:$binding_offsets,
    Variadic<HAL_DeviceSize>:$binding_lengths
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &state, Value commandBuffer,
      Value executableLayout, uint32_t set,
      ArrayRef<DescriptorSetBindingValue> bindings
    }]>,
  ];
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
      OpBuilder &builder, OperationState &state, Value commandBuffer,
      Value executableLayout, uint32_t set, Value descriptorSet,
      ValueRange dynamicOffsets = {}
    }]>,
  ];
}

def HAL_CommandBufferDispatchOp : HAL_Op<"command_buffer.dispatch"> {
  let summary = [{command buffer dispatch recording operation}];
  let description = [{
    Dispatches an execution request.

    ```mlir
    %x = constant 128 : index
    %y = constant 32 : index
    %z = constant 1 : index
    hal.command_buffer.dispatch %cmd, %executable,
                                entry_point = 0,
                                workgroup_xyz = [%x, %y, %z]
    ```
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
      OpBuilder &builder, OperationState &state, Value commandBuffer,
      Value executable, IREE::HAL::ExecutableEntryPointOp entryPoint,
      Value workgroupX, Value workgroupY, Value workgroupZ
    }]>,
    OpBuilder<[{
      Builder &builder, OperationState &state, Value commandBuffer,
      Value executable, unsigned entryPointOrdinal,
      Value workgroupX, Value workgroupY, Value workgroupZ
    }]>,
  ];
}

def HAL_CommandBufferDispatchIndirectOp : HAL_Op<"command_buffer.dispatch.indirect"> {
  let summary = [{command buffer indirect dispatch recording operation}];
  let description = [{
    Dispatches an execution request with the dispatch parameters loaded from the
    given buffer.

    ```mlir
    hal.command_buffer.dispatch.indirect %cmd, %executable,
                                         entry_point = 0,
                                         workgroups = %buffer[%offset]
    ```
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

def HAL_DescriptorSetCreateOp : HAL_PureOp<"descriptor_set.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    SameVariadicOperandSize,
  ]> {
  let summary = [{allocates a descriptor set from the device pool}];
  let description = [{
    Creates a DescriptorSet from the device pool.
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_DescriptorSetLayout:$set_layout,
    I32ArrayAttr:$bindings,
    Variadic<HAL_Buffer>:$binding_buffers,
    Variadic<HAL_DeviceSize>:$binding_offsets,
    Variadic<HAL_DeviceSize>:$binding_lengths
  );
  let results = (outs
    HAL_DescriptorSet:$result
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &state, Value device, Value setLayout,
      ArrayRef<DescriptorSetBindingValue> bindings
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// iree::hal::DescriptorSetLayout
//===----------------------------------------------------------------------===//

def HAL_DescriptorSetLayoutCreateOp :
  HAL_PureOp<"descriptor_set_layout.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{creates a descriptor set layout}];
  let description = [{
    Creates a descriptor set layout that defines the bindings used within a set.
    The same descriptor set layout may be shared with many different executable
    layouts and by doing so some runtime binding overhead when switching between
    executables that use the same set layouts can be reduced.

    ```mlir
    %layout = hal.descriptor_set_layout.create %device, "PushOnly", bindings = [
      #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">,
      #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Write">
    ] : !hal.descriptor_set_layout
    ```
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_DescriptorSetLayoutUsageTypeAttr:$usage_type,
    HAL_DescriptorSetLayoutBindingArrayAttr:$bindings
  );
  let results = (outs
    HAL_DescriptorSetLayout:$result
  );

  let assemblyFormat = [{
    $device `,` $usage_type `,` `bindings` `=` $bindings attr-dict `:` type($result)
  }];
}

def HAL_DescriptorSetLayoutLookupOp : HAL_PureOp<"descriptor_set_layout.lookup", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{descriptor set layout cache lookup pseudo-op}];
  let description = [{
    Used during conversion to provide a placeholder for a globally cached and
    possibly lazy-initialized descriptor set layout.

    ```mlir
    %layout = hal.descriptor_set_layout.lookup %device, "PushOnly", bindings = [
      #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">,
      #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Write">
    ] : !hal.descriptor_set_layout
    ```
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_DescriptorSetLayoutUsageTypeAttr:$usage_type,
    HAL_DescriptorSetLayoutBindingArrayAttr:$bindings
  );
  let results = (outs
    HAL_DescriptorSetLayout:$result
  );

  let assemblyFormat = [{
    $device `,` $usage_type `,` `bindings` `=` $bindings attr-dict `:` type($result)
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
      OpBuilder &builder, OperationState &state, Value device
    }], [{
      state.addOperands({device});
      state.addTypes({AllocatorType::get(builder.getContext())});
    }]>,
  ];
}

def HAL_DeviceSwitchOp : HAL_Op<"device.switch", [IsolatedFromAbove]> {
  let summary = [{runtime device switch pseudo op}];
  let description = [{
    Switches between multiple regions based on the runtime device type.
    The provided regions are matched against the runtime backend of the given
    device and executed only when the device matches the conditions.

    Conditions can match on wildcards and be folded to enable conditions that
    have similar bodies to be folded. The patterns themselves are only matched
    once at startup and then the results are cached; the runtime overhead is
    equivalent to a normal switch statement. In cases where the compiler can
    statically identify the device type entire cases can be folded away.

    Supported conditions:
    * `#hal.match...`: execute the region if the expression matches.

    Supported match expressions:
    * `#hal.match.always`: always matches; useful for defaults.
    * `#hal.match.any<[...]>`: matches if any of the nested expressions match.
    * `#hal.match.all<[...]>`: matches only if all of the nested expressions
      match.
    * `#hal.device.match.id<"pattern*-?-*">`: matches against the device
      identifier. The pattern is evaluated with standard file path wildcards
      (`*` for zero or more characters and `?` for one character).

    If more than one condition is satisfied the first listed will be chosen.
    More specific conditions should be earlier in the set. If no condition is
    matched but there are return values the switch will abort at runtime. It's
    strongly recommend that all switches that return values end with a trailing
    `#hal.match.always` condition to handle the fallthrough case.

    Upon creation each condition region will have an empty entry block with the
    specified operands available as arguments. Each region must be setup to
    return the same types.

    ```mlir
    %c0 = constant 0 : i32
    %c1 = constant 1 : i32
    %c2 = constant 2 : i32
    %device = ... : !hal.device
    %0 = hal.device.switch(%device : !hal.device) -> i32
      #hal.device.match.id<"vulkan-v1.?-*">(%c1a = %c1 : i32) {
        hal.return %c1a : i32
      },
      #hal.match.any<[#hal.device.match.id<"vmla">, #hal.device.match.id<"vulkan-*">]>(%c2a = %c2 : i32) {
        hal.return %c2a : i32
      },
      #hal.match.always(%c0a = %c0 : i32) {
        hal.return %c0a : i32
      }
    ```
  }];

  let arguments = (ins
    HAL_Device:$device,
    ArrayAttr:$conditions,
    Variadic<AnyType>:$args
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region VariadicRegion<AnyRegion>:$condition_regions);

  let extraClassDeclaration = [{
    /// Returns the index of the args() operand in the Operation operands list.
    unsigned mapArgOperandToOpOperand(unsigned i) { return i + 1; }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &state, TypeRange resultTypes,
      Value device, ArrayRef<Attribute> conditions,
      ArrayRef<ValueRange> conditionArgs,
      ArrayRef<NamedAttribute> attributes = {}
    }]>,
  ];

  let verifier = [{ return verifyDeviceSwitchOp(*this); }];
}

def HAL_ReturnOp : HAL_Op<"return", [Terminator]> {
  let summary = [{return from a hal.device.switch region}];
  let description = [{
    Returns the given values from the region and back to the host code.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result
    }], [{
      build(builder, result, llvm::None);
    }]>,
  ];
}

// TODO(benvanik): additional factory functions and submission ops.
// TODO(benvanik): %0 = hal.device.query %device, group, property : i32/etc

def HAL_DeviceMatchIDOp : HAL_PureOp<"device.match.id"> {
  let summary = [{returns true if the device ID matches the pattern}];
  let description = [{
    Pattern matches the device ID with the given wildcard pattern.
    This can be used to conditionally evaluate device-specific code when the
    device is not known at compile-time.

    ```mlir
    %is_match = hal.device.match.id %device, pattern = ["vulkan-*"] : (!hal.device) -> i1
    ```
  }];

  let arguments = (ins
    HAL_Device:$device,
    StrAttr:$pattern
  );
  let results = (outs
    I1:$result
  );

  let assemblyFormat = [{
    $device `,` `pattern` `=` `[` $pattern `]` attr-dict
    `:` `(` type($device) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// iree::hal::Executable
//===----------------------------------------------------------------------===//

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
      OpBuilder &builder, OperationState &state, StringRef name
    }]>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return body().front(); }
    IREE::HAL::InterfaceOp getInterfaceOp();
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
    FlatSymbolRefAttr:$interface,
    TypeAttr:$signature
  );
}

def HAL_ExecutableTargetOp : HAL_Op<"executable.target", [
    IsolatedFromAbove,
    HasParent<"IREE::HAL::ExecutableOp">,
    SingleBlockImplicitTerminator<"IREE::HAL::ExecutableTargetEndOp">,
  ]> {
  let summary = [{target executable IR}];
  let description = [{
    The target IR for the executable. This can be preserved for debugging but
    is usually removed during transformation.
  }];

  let arguments = (ins
    StrAttr:$target_backend
    // TODO(benvanik): add compatibility and versioning attributes.
  );

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &state, StringRef targetBackend
    }]>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return body().front(); }

    ::mlir::ModuleOp getInnerModule() {
      auto moduleOps = getBlock().getOps<::mlir::ModuleOp>();
      assert(!moduleOps.empty() && "source ops need inner modules");
      return *moduleOps.begin();
    }
  }];
}

def HAL_ExecutableTargetEndOp : HAL_Op<"executable.target_end", [
    HasParent<"IREE::HAL::ExecutableTargetOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the executable target op}];
  let assemblyFormat = "attr-dict";
}

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
      OpBuilder &builder, OperationState &state, uint32_t format,
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

def HAL_ExecutableLookupOp : HAL_PureOp<"executable.lookup", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{executable cache lookup pseudo-op}];
  let description = [{
    Used during conversion to provide a placeholder for a globally cached and
    possibly lazy-initialized executable.
  }];

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
      OpBuilder &builder, OperationState &state,
      Value device, StringRef executable
    }], [{
      state.addOperands({device});
      state.addAttribute("executable", builder.getSymbolRefAttr(executable));
      state.addTypes({ExecutableType::get(builder.getContext())});
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// iree::hal::Executable Interfaces
//===----------------------------------------------------------------------===//

def HAL_InterfaceOp : HAL_Op<"interface", [
    IsolatedFromAbove,
    // HasParent<"IREE::HAL::ExecutableOp">,
    SingleBlockImplicitTerminator<"IREE::HAL::InterfaceEndOp">,
    Symbol,
    SymbolTable,
  ]> {
  let summary = [{executable IO interface description}];
  let description = [{
    Defines a set of IO bindings and associated information required for the
    scheduler to correctly dispatch the executable. Multiple executable binaries
    can share the same interface to reduce the amount of scheduling code
    generated, or a single executable binary can expose entry points with
    multiple different interfaces to allow for runtime selection based on
    supported device capabilities.

    The design is modeled on the Vulkan binding model, which uses one or more
    descriptor sets containing one or more bindings as part of its API. By using
    the same representation here we can get toll-free lowering on the Vulkan and
    SPIR-V side, and for other backends that may have simpler models the cost is
    negligable (just an extra indirection during dispatch). For more information
    on the Vulkan binding model see the documentation (which contains GLSL,
    SPIR-V, and Vulkan API examples that directly map to our use here):
    https://vulkan.lunarg.com/doc/view/1.0.33.0/linux/vkspec.chunked/ch13s02.html

    For background as to how multiple descriptor sets are used by the scheduler
    see: https://developer.nvidia.com/vulkan-shader-resource-binding
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    OptionalAttr<I32Attr>:$push_constants
  );

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &state, StringRef name,
      IntegerAttr pushConstants = {}
    }]>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return body().front(); }

    // TODO(benvanik): replace with a nested typed attr that works.
    // Array of HAL_DescriptorSetLayoutBindingArrayAttr.
    ArrayAttr getExecutableSetLayoutsAttr();
  }];
}

def HAL_InterfaceEndOp : HAL_Op<"interface_end", [
    HasParent<"IREE::HAL::InterfaceOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the executable interface op}];
  let assemblyFormat = "attr-dict";
}

def HAL_InterfaceBindingOp : HAL_Op<"interface.binding", [
    HasParent<"IREE::HAL::InterfaceOp">,
    Symbol,
  ]> {
  let summary = [{executable IO binding description}];
  let description = [{
    Defines an IO binding used to pass buffers into the executable for loads and
    stores. These bindings will be mapped into descriptor sets by the scheduler
    based on device support and usage within the command buffer.

    Bindings are typeless and only indicate a resource that is made available to
    executables. The types that can be loaded-from and stored-into the binding
    buffers are provided to backends by way of the `hal.interface.load.*` and
    `hal.interface.store.*` ops.

    Bindings do not need to correlate 1:1 with entry point arguments/results and
    in many cases will not. For example, the same binding may be used with
    different offsets if the arguments were packed into a ringbuffer/arena by
    the scheduler.
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    I32Attr:$set,
    I32Attr:$binding,
    HAL_DescriptorTypeAttr:$type,
    HAL_MemoryAccessBitfieldAttr:$access
  );
}

def HAL_InterfaceLoadConstantOp : HAL_PureOp<"interface.load.constant"> {
  let summary = [{loads a constant value from the interface constant block}];
  let description = [{
    Loads a scalar constant value from an executable IO push constant block.
    The value will be loaded from the given constant offset and will be
    bitcast (possibly with truncation or zero-extension) to the result type.
  }];

  let arguments = (ins
    HAL_HostSizeAttr:$offset
  );
  let results = (outs
    HAL_PrimitiveType:$result
  );

  let assemblyFormat = [{
    `offset` `=` $offset attr-dict `:` type($result)
  }];
}

def HAL_InterfaceLoadTensorOp : HAL_PureOp<"interface.load.tensor"> {
  let summary = [{loads a tensor from an executable IO binding}];
  let description = [{
    Loads a tensor value from an executable IO binding. This is a pseudo op
    that can be used to tie SSA tensor values in the IR to the bindings that
    contain those tensors.

    Note that because there may not be a 1:1 mapping between original tensor
    arguments to the entry point function and the bindings in the interface the
    backend must use the offset provided on this op to properly compute the base
    address of the tensor data. The offset is in bytes relative to the base
    binding address, irrespective of the type of the tensor loaded by this
    operation.

    The offset provided, if non-zero, will have an alignment compatible with the
    tensor type represented. For example, a `tensor<16xf32>` will be aligned on
    at least a 4 byte boundary.
  }];

  let arguments = (ins
    SymbolRefAttr:$binding,
    HAL_DeviceSize:$offset
  );
  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    $binding `,` `offset` `=` $offset attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    /// Returns the hal.interface.binding op associated with this op.
    /// Returns null op if not found.
    IREE::HAL::InterfaceBindingOp queryBindingOp();
  }];
}

def HAL_InterfaceStoreTensorOp : HAL_Op<"interface.store.tensor"> {
  let summary = [{stores a tensor in an executable IO binding}];
  let description = [{
    Stores a tensor value into an executable IO binding. This is a pseudo op
    indicating that the value of the operand tensor should be stored into the
    specified binding.

    Note that because there may not be a 1:1 mapping between original tensor
    arguments to the entry point function and the bindings in the interface the
    backend must use the offset provided on this op to properly compute the base
    address of the tensor data. The offset is in bytes relative to the base
    binding address, irrespective of the type of the tensor loaded by this
    operation.

    The offset provided, if non-zero, will have an alignment compatible with the
    tensor type represented. For example, a `tensor<16xf32>` will be aligned on
    at least a 4 byte boundary.
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    SymbolRefAttr:$binding,
    HAL_DeviceSize:$offset
  );

  let assemblyFormat = [{
    $operand `,` $binding `,` `offset` `=` $offset attr-dict `:` type($operand)
  }];

  let extraClassDeclaration = [{
    /// Returns the hal.interface.binding op associated with this op.
    /// Returns null op if not found.
    IREE::HAL::InterfaceBindingOp queryBindingOp();
  }];
}

//===----------------------------------------------------------------------===//
// iree::hal::ExecutableCache
//===----------------------------------------------------------------------===//

def HAL_ExecutableCacheCreateOp : HAL_Op<"executable_cache.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{creates an executable cache}];
  let description = [{
    Caches may be shared across multiple devices from the same driver or
    specific to individual devices. Caches may persist prepared executables
    across process launches or re-prepare them each run. Callers should assume
    that the cache is a no-op and the returned hal.executables only live for as
    long as the cache does.

    The term 'cache' here is rather optimistic - it's perfectly acceptable for
    implementations to not cache at all and return new hal.executables for each
    preparation (even for the same executable). Callers should expect such
    behavior and try to retain the results of the preparation to reduce overhead
    from re-preparing executables.

    Currently caches are synchronous but the intent is to support asynchronous
    compilation of multiple executables with semaphores used to indicate when
    the executables are ready for use.

    ```mlir
    %cache = hal.executable_cache.create %device, identifier = "some_guid" : !hal.executable_cache
    ```
  }];

  let arguments = (ins
    HAL_Device:$device,
    StrAttr:$identifier
  );
  let results = (outs
    HAL_ExecutableCache:$result
  );

  let assemblyFormat = [{
    $device `,` `identifier` `=` $identifier
    attr-dict-with-keyword `:` type($result)
  }];
}

def HAL_ExecutableCacheSelectFormatOp :
  HAL_PureOp<"executable_cache.select_format", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{selects the preferred format from the given list}];
  let description = [{
    Returns the index of the preferred format of the cache from the given set or
    -1 if none can be used. Preparation may still fail if the particular version
    or features required by the executable are not supported.

    ```mlir
    // Returns indices 0 or 1 if either of the two inputs are usable or -1 otherwise.
    %index = hal.executable_cache.select_format %cache, available_formats = [1447906369 : i32, 123482395 : i32] : i32
    ```
  }];

  let arguments = (ins
    HAL_ExecutableCache:$executable_cache,
    HAL_ExecutableFormatArrayAttr:$available_formats
  );
  let results = (outs
    I32:$result
  );

  let assemblyFormat = [{
    $executable_cache `,` `available_formats` `=` $available_formats
    attr-dict-with-keyword `:` type($result)
  }];
}

def HAL_ExecutableCachePrepareOp : HAL_PureOp<"executable_cache.prepare", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{synchronously prepares an executable for use}];
  let description = [{
    The provided spec and data will be used to either lookup a previously
    prepared executable in the cache or prepare a new one.

    Depending on the driver preparation may take a non-trivial amount of time
    (such as when JITing/etc). As the cache is internally synchronized callers
    can issue preparation requests from multiple threads - even for the same
    executables - and calls will block until preparation completes.

    ```mlir
    %exe = hal.executable_cache.prepare %executable_cache,
                                        layout = %layout,
                                        caching_mode = "AllowPersistentCaching|AllowOptimization",
                                        @executable : !hal.executable
    ```
  }];

  let arguments = (ins
    HAL_ExecutableCache:$executable_cache,
    HAL_ExecutableLayout:$executable_layout,
    HAL_ExecutableCachingModeBitfieldAttr:$caching_mode,
    FlatSymbolRefAttr:$executable
  );
  let results = (outs
    HAL_Executable:$result
  );

  let assemblyFormat = [{
    $executable_cache `,` `layout` `=` $executable_layout `,`
    `caching_mode` `=` $caching_mode `,` $executable
    attr-dict-with-keyword `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// iree::hal::ExecutableLayout
//===----------------------------------------------------------------------===//

def HAL_ExecutableLayoutCreateOp : HAL_PureOp<"executable_layout.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{creates an executable layout}];
  let description = [{
    Creates an executable layout from the given descriptor sets and push
    constant required size. Executable layouts can be shared across any
    executable that uses the same layout and push constant information. Sharing
    the layout between executables will reduce runtime binding overhead and it
    is often worth the cost to allow a small number of unused bindings in one
    executable such that it can share layouts with others that will be scheduled
    adjacent to it.

    ```mlir
    %set0 = hal.descriptor_set_layout.create ...
    %set1 = hal.descriptor_set_layout.create ...
    %layout = hal.executable_layout.create %device,
                                           set_layouts = [%set0, %set1],
                                           push_constants = 3 : !hal.executable_layout
    ```
  }];

  let arguments = (ins
    HAL_Device:$device,
    Variadic<HAL_DescriptorSetLayout>:$set_layouts,
    I32Attr:$push_constants
  );
  let results = (outs
    HAL_ExecutableLayout:$result
  );

  let assemblyFormat = [{
    $device `,` `set_layouts` `=` `[` $set_layouts `]` `,`
    `push_constants` `=` $push_constants
    attr-dict-with-keyword `:` type($result)
  }];
}

def HAL_ExecutableLayoutLookupOp : HAL_PureOp<"executable_layout.lookup", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{executable layout cache lookup pseudo-op}];
  let description = [{
    Used during conversion to provide a placeholder for a globally cached and
    possibly lazy-initialized executable layout.

    ```mlir
    %layout = hal.executable_layout.lookup %device, set_layouts = [
      [
        #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">,
        #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Write">
      ]
    ] : !hal.executable_layout
    ```
  }];

  let arguments = (ins
    HAL_Device:$device,
    // TODO(benvanik): replace with a nested typed attr that works.
    // Array of HAL_DescriptorSetLayoutBindingArrayAttr.
    ArrayAttr:$set_layouts,
    OptionalAttr<I32Attr>:$push_constants
  );
  let results = (outs
    HAL_ExecutableLayout:$result
  );

  let assemblyFormat = [{
    $device `,` `set_layouts` `=` $set_layouts
    (`,` `push_constants` `=` $push_constants^)?
    attr-dict-with-keyword `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// iree::hal::RingBuffer
//===----------------------------------------------------------------------===//

// TODO(benvanik): ring buffer.

//===----------------------------------------------------------------------===//
// iree::hal::Semaphore
//===----------------------------------------------------------------------===//

def HAL_SemaphoreCreateOp : HAL_Op<"semaphore.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{semaphore allocation operation}];
  let description = [{
    Returns a semaphore from the device pool with the given initial value.
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_TimelineValue:$initial_value
  );
  let results = (outs
    HAL_Semaphore:$result
  );

  let assemblyFormat = [{
    $device `,` `initial_value` `=` $initial_value
    attr-dict-with-keyword `:` type($result)
  }];
}

def HAL_SemaphoreQueryOp : HAL_Op<"semaphore.query"> {
  let summary = [{semaphore payload value query}];
  let description = [{
    Queries the current payload and returns a tuple of `(status, value)`.
    As the payload is monotonically increasing it is guaranteed that
    the value is at least equal to the previous result of a
    `hal.semaphore.signal` call and coherent with any waits for a
    specified value via `hal.semaphore.await`.
  }];

  let arguments = (ins
    HAL_Semaphore:$semaphore
  );
  let results = (outs
    IREE_Status:$status,
    HAL_TimelineValue:$value
  );

  let assemblyFormat = [{
    $semaphore attr-dict-with-keyword `:` type($status) `,` type($value)
  }];
}

def HAL_SemaphoreSignalOp : HAL_Op<"semaphore.signal"> {
  let summary = [{semaphore payload value signal operation}];
  let description = [{
    Signals the semaphore to the given payload value.
    The call is ignored if the current payload value exceeds `new_value`.
  }];

  let arguments = (ins
    HAL_Semaphore:$semaphore,
    HAL_TimelineValue:$new_value
  );

  let assemblyFormat = [{
    $semaphore `,` `value` `=` $new_value attr-dict-with-keyword
  }];
}

def HAL_SemaphoreFailOp : HAL_Op<"semaphore.fail"> {
  let summary = [{semaphore asynchronous failure operation}];
  let description = [{
    Signals the semaphore with a failure. The `status` will be returned from
    `hal.semaphore.query` and `hal.semaphore.signal` for the lifetime
    of the semaphore.
  }];

  let arguments = (ins
    HAL_Semaphore:$semaphore,
    IREE_Status:$status
  );

  let assemblyFormat = [{
    $semaphore `,` `status` `=` $status attr-dict-with-keyword
  }];
}

// TODO(benvanik): rework this to make it a terminator with branch targets
// for ^success and ^fail(status).
def HAL_SemaphoreAwaitOp : HAL_Op<"semaphore.await", [YieldPoint]> {
  let summary = [{asynchronous semaphore wait operation}];
  let description = [{
    Yields the caller until the semaphore reaches or exceeds the specified
    payload `min_value`. Returns the `status` of the semaphore after the wait,
    with a non-zero value indicating failure.
  }];

  let arguments = (ins
    HAL_Semaphore:$semaphore,
    HAL_TimelineValue:$min_value
  );
  let results = (outs
    IREE_Status:$status
  );

  let assemblyFormat = [{
    $semaphore `,` `min_value` `=` $min_value attr-dict-with-keyword `:` type($status)
  }];
}

#endif  // IREE_DIALECT_HAL_OPS
