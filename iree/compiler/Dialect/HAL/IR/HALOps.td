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
include "mlir/Interfaces/ViewLikeInterface.td"

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
    OpBuilder<(ins),
    [{
      $_state.addTypes({DeviceType::get($_builder.getContext())});
    }]>,
  ];
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
    OpBuilder<(ins "IREE::HAL::AccessScopeBitfield":$sourceScope,
      "IREE::HAL::AccessScopeBitfield":$targetScope)>,
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
    OpBuilder<(ins "IREE::HAL::AccessScopeBitfield":$sourceScope,
      "IREE::HAL::AccessScopeBitfield":$targetScope, "Value":$buffer,
      "Value":$offset, "Value":$length)>,
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
    OpBuilder<(ins "StringRef":$name, "bool":$isMutable, "Type":$type,
      "Optional<StringRef>":$initializer, "Optional<Attribute>":$initialValue,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "StringRef":$name, "bool":$isMutable,
      "mlir::FuncOp":$initializer, CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "StringRef":$name, "bool":$isMutable, "Type":$type,
      "Attribute":$initialValue, CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "StringRef":$name, "bool":$isMutable, "Type":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
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

def HAL_VariableLoadOp : HAL_Op<"variable.load", [
    // HACK: works around the lack of symbol side effects in C++.
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  ]> {
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
    OpBuilder<(ins "Value":$status, CArg<"StringRef", [{""}]>:$message),
    [{
      build(
        $_builder, $_state, status,
        message.empty() ? StringAttr{} : $_builder.getStringAttr(message));
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
    HAL_ElementType:$element_type
  );
  let results = (outs
    HAL_DeviceSize:$result
  );

  let assemblyFormat = [{
    $allocator `,` `shape` `=` `[` $shape `]` `,` `element_type` `=`
    $element_type attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Value":$allocator, "ValueRange":$shape,
      "int32_t":$elementType)>,
    OpBuilder<(ins "Value":$allocator, "ValueRange":$shape,
      "Value":$elementType)>,
  ];

  let hasCanonicalizer = 1;
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
    HAL_ElementType:$element_type,
    HAL_Dims:$indices
  );

  let results = (outs
    HAL_DeviceSize:$offset
  );

  let assemblyFormat = [{
    $allocator `,` `shape` `=` `[` $shape `]` `,` `element_type` `=`
    $element_type `,` `indices` `=` `[` $indices `]` attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Value":$allocator, "ValueRange":$shape,
      "int32_t":$elementType, "ValueRange":$indices)>,
    OpBuilder<(ins "Value":$allocator, "ValueRange":$shape,
      "Value":$elementType, "ValueRange":$indices)>,
  ];

  let hasCanonicalizer = 1;
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
    HAL_ElementType:$element_type,
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

  let builders = [
    OpBuilder<(ins "Value":$allocator, "ValueRange":$shape,
      "int32_t":$elementType, "ValueRange":$indices, "ValueRange":$lengths)>,
    OpBuilder<(ins "Value":$allocator, "ValueRange":$shape,
      "Value":$elementType, "ValueRange":$indices, "ValueRange":$lengths)>,
  ];

  let hasCanonicalizer = 1;
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
    OpBuilder<(ins "Value":$allocator,
      "IREE::HAL::MemoryTypeBitfield":$memoryTypes,
      "IREE::HAL::BufferUsageBitfield":$bufferUsage, "Value":$allocationSize)>,
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
    OpBuilder<(ins "Value":$allocator,
      "IREE::HAL::MemoryTypeBitfield":$memoryTypes,
      "IREE::HAL::BufferUsageBitfield":$bufferUsage, "ElementsAttr":$value)>,
  ];

  let hasCanonicalizer = 1;
}

def HAL_AllocatorMapOp : HAL_Op<"allocator.map", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{allocator-supported host buffer wrapping operation}];
  let description = [{
    Wraps a !hal.buffer around host read-only memory backed by the given byte
    buffer. The returned buffer may be host-only and not directly usable on
    devices.
  }];

  let arguments = (ins
    HAL_Allocator:$allocator,
    HAL_MemoryTypeBitfieldAttr:$memory_types,
    HAL_BufferUsageBitfieldAttr:$buffer_usage,
    // TODO(benvanik): support other types (and mutable buffers).
    ByteBufferType:$source,
    HAL_DeviceSize:$offset,
    HAL_DeviceSize:$length
  );
  let results = (outs
    HAL_Buffer:$result
  );

  let assemblyFormat = [{
    $allocator `,` $memory_types `,` $buffer_usage `,`
    $source `[` $offset `,` $length `]` attr-dict-with-keyword
    `:` type($source) `->` type($result)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$allocator,
      "IREE::HAL::MemoryTypeBitfield":$memoryTypes,
      "IREE::HAL::BufferUsageBitfield":$bufferUsage, "Value":$source,
      "Value":$offset, "Value":$length)>,
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
    OpBuilder<(ins "Value":$buffer)>,
  ];

  let hasCanonicalizer = 1;
}

// TODO(benvanik): clone buffer op.

def HAL_BufferSubspanOp : HAL_PureOp<"buffer.subspan", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
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
    OpBuilder<(ins "Value":$allocator,
      "IREE::HAL::MemoryTypeBitfield":$memoryTypes,
      "IREE::HAL::BufferUsageBitfield":$bufferUsage, "ElementsAttr":$value)>,
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
    HAL_ElementType:$element_type,
    HAL_Shape:$shape
  );
  let results = (outs
    HAL_BufferView:$result
  );

  let assemblyFormat = [{
    $buffer `,` `element_type` `=` $element_type `,` `shape` `=` `[` $shape `]`
    `:` type($result) attr-dict
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$buffer, "int32_t":$elementType,
                  "ValueRange":$shape)>,
    OpBuilder<(ins "Value":$buffer, "Value":$elementType,
                  "ValueRange":$shape)>,
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

  let hasCanonicalizer = 1;
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
    OpBuilder<(ins "Value":$bufferView)>,
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
    OpBuilder<(ins "Value":$bufferView)>,
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
    OpBuilder<(ins "Value":$bufferView, "ValueRange":$indices)>,
  ];

  let hasCanonicalizer = 1;
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
    OpBuilder<(ins "Value":$bufferView, "ValueRange":$indices,
      "ValueRange":$lengths)>,
  ];

  let hasCanonicalizer = 1;
}

def HAL_BufferViewElementTypeOp : HAL_PureOp<"buffer_view.element_type"> {
  let summary = [{buffer view element type query}];
  let description = [{
    Returns the element type of the buffer view.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view
  );
  let results = (outs
    HAL_ElementType:$result
  );

  let assemblyFormat = [{$buffer_view attr-dict `:` type($result)}];
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

  let hasCanonicalizer = 1;
}

def HAL_BufferViewTraceOp : HAL_Op<"buffer_view.trace", []> {
  let summary = [{trace value(s) operation}];
  let description = [{
    Traces out to a runtime trace sink (console, log file, etc) the given buffer
    views and titles them with the given key. The key is informational only and
    useful for titling/marking specific sets of buffers for easier searching.
  }];

  let arguments = (ins
    StrAttr:$key,
    Variadic<HAL_BufferView>:$operands
  );

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
    OpBuilder<(ins "Value":$device,
      "IREE::HAL::CommandBufferModeBitfield":$modes,
      "IREE::HAL::CommandCategoryBitfield":$commandCategories)>,
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

def HAL_CommandBufferDeviceOp : HAL_PureOp<"command_buffer.device"> {
  let summary = [{command buffer device query operation}];
  let description = [{
    Used during conversion to access the device used to create a command buffer.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer
  );
  let results = (outs
    HAL_Device:$device
  );

  let assemblyFormat = "$command_buffer attr-dict `:` type($device)";

  let hasCanonicalizer = 1;
}

def HAL_CommandBufferExecutionBarrierOp : HAL_Op<"command_buffer.execution_barrier"> {
  let summary = [{command buffer execution barrier recording operation}];
  let description = [{
    Defines an execution dependency between all commands recorded before the
    barrier and all commands recorded after the barrier. Only the stages
    provided will be affected.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    HAL_ExecutionStageBitfieldAttr:$source_stage_mask,
    HAL_ExecutionStageBitfieldAttr:$target_stage_mask,
    HAL_ExecutionBarrierFlagBitfieldAttr:$flags
  );
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
    hal.command_buffer.push_descriptor_set %cmd, %executable_layout, set = %c0, bindings = [
      %c0 = (%buffer_0, %buffer_offset_0, %buffer_length_0),
      %c1 = (%buffer_1, %buffer_offset_1, %buffer_length_1),
      %c2 = (%buffer_2, %buffer_offset_2, %buffer_length_2)
    ]
    ```
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    HAL_ExecutableLayout:$executable_layout,
    Index:$set,
    Variadic<Index>:$binding_ordinals,
    Variadic<HAL_Buffer>:$binding_buffers,
    Variadic<HAL_DeviceSize>:$binding_offsets,
    Variadic<HAL_DeviceSize>:$binding_lengths
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$commandBuffer, "Value":$executableLayout,
      "int64_t":$set, "ArrayRef<DescriptorSetBindingValue>":$bindings)>,
    OpBuilder<(ins "Value":$commandBuffer, "Value":$executableLayout,
      "Value":$set, "ArrayRef<DescriptorSetBindingValue>":$bindings)>,
  ];

  let hasCanonicalizer = 1;
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
    Index:$set,
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
    OpBuilder<(ins "Value":$commandBuffer, "Value":$executableLayout,
      "int64_t":$set, "Value":$descriptorSet,
      CArg<"ValueRange", "{}">:$dynamicOffsets)>,
    OpBuilder<(ins "Value":$commandBuffer, "Value":$executableLayout,
      "Value":$set, "Value":$descriptorSet,
      CArg<"ValueRange", "{}">:$dynamicOffsets)>,
  ];
}

def HAL_CommandBufferDispatchSymbolOp : HAL_Op<"command_buffer.dispatch.symbol"> {
  let summary = [{command buffer dispatch recording operation, using symbolref}];
  let description = [{
    Dispatches an execution request, using a nested symbol reference to the entry point.

    ```mlir
    %x = constant 128 : index
    %y = constant 32 : index
    %z = constant 1 : index
    hal.command_buffer.dispatch.symbol %cmd, @executable::@target::@entry,
                                       workgroup_xyz = [%x, %y, %z]
    ```
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    SymbolRefAttr:$entry_point,
    HAL_Dim:$workgroup_x,
    HAL_Dim:$workgroup_y,
    HAL_Dim:$workgroup_z
  );

  let assemblyFormat = [{
    $command_buffer `,` $entry_point `,`
    `workgroup_xyz` `=` `[` $workgroup_x `,` $workgroup_y `,` $workgroup_z `]`
    attr-dict
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$commandBuffer,
      "IREE::HAL::ExecutableEntryPointOp":$entryPoint, "Value":$workgroupX,
      "Value":$workgroupY, "Value":$workgroupZ)>,
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
}

def HAL_CommandBufferDispatchIndirectSymbolOp : HAL_Op<"command_buffer.dispatch.indirect.symbol"> {
  let summary = [{command buffer indirect dispatch recording operation, using symbolref}];
  let description = [{
    Dispatches an execution request with the dispatch parameters loaded from the
    given buffer, using using a nested symbol reference to the entry point.

    ```mlir
    hal.command_buffer.dispatch.indirect.symbol %cmd, @executable::@target::@entry,
                                                workgroups = %buffer[%offset]
    ```
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    SymbolRefAttr:$entry_point,
    HAL_Buffer:$workgroups_buffer,
    HAL_DeviceSize:$workgroups_offset
  );

  let assemblyFormat = [{
    $command_buffer `,` $entry_point `,`
    `workgroups` `=` $workgroups_buffer `[` $workgroups_offset `]` attr-dict
  }];
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
// Constant pooling
//===----------------------------------------------------------------------===//

def HAL_ConstantPoolOp : HAL_Op<"constant_pool", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"IREE::HAL::ConstantPoolEndOp">,
    Symbol,
    SymbolTable,
  ]> {
  let summary = [{pool of constants with similar lifetimes}];
  let description = [{
    A pool of constants that share a similiar lifetime and that should be stored
    together both in the source files and at runtime. By logically grouping
    constants by their frequency and locality of access we can reduce the number
    of bindings required on hal.interface by sourcing constants from the same
    buffer. We can also optimize module loading by mapping or DMA transferring
    constant data (based on device).
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    HAL_BufferConstraintsAttr:$buffer_constraints
  );

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$name,
      "BufferConstraintsAttr":$bufferConstraints)>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return body().front(); }
  }];
}

def HAL_ConstantPoolEndOp : HAL_Op<"constant_pool_end", [
    HasParent<"IREE::HAL::ConstantPoolOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the constant pool op}];
  let assemblyFormat = "attr-dict-with-keyword";
}

def HAL_ConstantPoolValueOp : HAL_Op<"constant_pool.value", [
    Symbol,
    HasParent<"IREE::HAL::ConstantPoolOp">,
  ]> {
  let summary = [{constant value within a parent constant pool}];
  let description = [{
    Represents a constant value as part of a constant pool containing constants
    with a similar lifetime.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ElementsAttr:$value
  );

  let assemblyFormat = [{
    $sym_name attr-dict `=` $value
  }];
}

def HAL_ConstantPoolSpanOp : HAL_Op<"constant_pool.span", [
    Symbol,
    HasParent<"IREE::HAL::ConstantPoolOp">,
  ]> {
  let summary = [{constant span within a parent storage block}];
  let description = [{
    Represents a constant stored within a hal.constant_pool. Provides a
    symbol that can be used to reference the constant data as a stored range
    within the module file.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$tensor_type,
    SymbolRefAttr:$storage_buffer,
    HAL_ByteRangeAttr:$storage_range,
    OptionalAttr<SymbolRefAttr>:$runtime_buffer,
    OptionalAttr<HAL_ByteRangeAttr>:$runtime_range
  );

  let assemblyFormat = [{
    $sym_name `:` $tensor_type attr-dict
    `=` $storage_buffer `[` $storage_range `]`
    (`->` $runtime_buffer^ `[` $runtime_range `]`)?
  }];
}

def HAL_ConstantPoolSplatOp : HAL_Op<"constant_pool.splat", [
    Symbol,
    HasParent<"IREE::HAL::ConstantPoolOp">,
  ]> {
  let summary = [{constant splat within a parent storage block}];
  let description = [{
    Represents a splatted constant that has no representation in the storage
    but must be represented at runtime as splatted 4-byte value.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ElementsAttr:$value,
    OptionalAttr<SymbolRefAttr>:$runtime_buffer,
    OptionalAttr<HAL_ByteRangeAttr>:$runtime_range
  );

  let assemblyFormat = [{
    $sym_name attr-dict `=` $value
    (`->` $runtime_buffer^ `[` $runtime_range `]`)?
  }];
}

def HAL_ConstantPoolLoadOp : HAL_PureOp<"constant_pool.load", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{constant pool tensor load pseudo-op}];
  let description = [{
    Used during conversion to provide a placeholder for a globally cached and
    possibly lazy-initialized compile-time constants. Will be replaced with a
    direct variable access during transformation.
  }];

  let arguments = (ins
    SymbolRefAttr:$constant
  );
  let results = (outs
    TypeAlias<AnyRankedTensor>:$result
  );

  let assemblyFormat = "$constant attr-dict `:` type($result)";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$resultType, "SymbolRefAttr":$constant),
    [{
      $_state.addTypes({resultType});
      $_state.addAttribute("constant", constant);
    }]>,
  ];

  let hasCanonicalizer = 1;
}

def HAL_ConstantStorageOp : HAL_Op<"constant_storage", [
    Symbol,
  ]> {
  let summary = [{constant data storage block}];
  let description = [{
    Represents a packed constant storage buffer meeting the buffer constraints
    placed on the parent pool. Referenced by other constant pool ops.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ElementsAttr:$value
  );

  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword `=` $value
  }];
}

def HAL_ConstantStorageLookupOp :
    HAL_PureOp<"constant_storage.lookup", [
      DeclareOpInterfaceMethods<OpAsmOpInterface>,
    ]> {
  let summary = [{constant storage byte buffer accessor}];
  let description = [{
    Returns the read-only host byte buffer storing the constant data.
  }];

  let arguments = (ins
    SymbolRefAttr:$constant
  );
  let results = (outs
    ByteBufferType:$result
  );

  let assemblyFormat = [{
    $constant `:` type($result) attr-dict
  }];
}

def HAL_ConstantSubspanOp : HAL_PureOp<"constant.subspan", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{runtime constant buffer lookup pseudo-op}];
  let description = [{
    Used during conversion to resolve a runtime representation of a constant as
    a tensor backed by a buffer range.
  }];

  let arguments = (ins
    SymbolRefAttr:$runtime_buffer,
    HAL_ByteRangeAttr:$runtime_range
  );
  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    $runtime_buffer `[` $runtime_range `]` `:` type($result) attr-dict
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$resultType, "SymbolRefAttr":$runtimeBuffer,
      "ByteRangeAttr":$runtimeRange),
    [{
      $_state.addTypes({resultType});
      $_state.addAttribute("runtime_buffer", runtimeBuffer);
      $_state.addAttribute("runtime_range", runtimeRange);
    }]>,
  ];
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
    Variadic<Index>:$binding_ordinals,
    Variadic<HAL_Buffer>:$binding_buffers,
    Variadic<HAL_DeviceSize>:$binding_offsets,
    Variadic<HAL_DeviceSize>:$binding_lengths
  );
  let results = (outs
    HAL_DescriptorSet:$result
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$device, "Value":$setLayout,
      "ArrayRef<DescriptorSetBindingValue>":$bindings)>,
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
    OpBuilder<(ins "Value":$device),
    [{
      $_state.addOperands({device});
      $_state.addTypes({AllocatorType::get($_builder.getContext())});
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
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$device,
      "ArrayRef<Attribute>":$conditions,
      "ArrayRef<SmallVector<Value, 4>>":$conditionArgs,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
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
    OpBuilder<(ins),
    [{
      build($_builder, $_state, llvm::None);
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

def HAL_DeviceMatchMemoryModelOp : HAL_PureOp<"device.match.memory_model"> {
  let summary = [{returns true if the device memory model matches the value}];
  let description = [{
    Compares the device's memory model against the specified model.
    This can be used to conditionally evaluate device-specific code when the
    device is not known at compile-time.

    ```mlir
    %is_match = hal.device.match.memory_model %device, memory_model = "Unified" : (!hal.device) -> i1
    ```
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_MemoryModelAttr:$model
  );
  let results = (outs
    I1:$result
  );

  let assemblyFormat = [{
    $device `,` `model` `=` `[` $model `]` attr-dict
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
    OpBuilder<(ins "StringRef":$name)>,
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

def HAL_ExecutableEntryPointOp : HAL_Op<"executable.entry_point", [
    Symbol,
    HasParent<"IREE::HAL::ExecutableTargetOp">,
    IsolatedFromAbove
  ]> {
  let summary = [{executable entry point declaration}];
  let description = [{
    An entry point exported by the executable with statically-available
    information describing the IO interface it uses and other dispatch metadata.

    The optional `workgroup_count_region` region represents the
    computation that returns the number of workgroups to use. The
    arguments to the region represents the workload along x, y and
    z. It returns the number of workgroups along x, y, and z.

    TODO(ravishankarm): In reality there is no need to define what the
    arguments represent. They could be any values that are needed to
    compute the number of workgroups. Its unclear what these are in
    general and how to plumb them through. So for now just accepting
    the workload along x, y, and z.
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    HAL_OrdinalAttr:$ordinal,
    FlatSymbolRefAttr:$interface,
    TypeAttr:$signature,
    OptionalAttr<HAL_WorkgroupSizeAttr>:$workgroup_size
  );

  let regions = (region VariadicRegion<SizedRegion<1>>:$workgroup_count_region);

  let builders = [
    OpBuilder<
      (ins "::llvm::StringRef":$sym_name, "::llvm::APInt":$ordinal,
        "::llvm::StringRef":$interface,"::mlir::Type":$signature,
        "::mlir::ArrayAttr":$workgroup_size),
       [{build($_builder, $_state, sym_name, ordinal, interface, signature,
            workgroup_size, 0);}]>,
    OpBuilder<
      (ins "::mlir::StringAttr":$sym_name, "::mlir::IntegerAttr":$ordinal,
        "::mlir::FlatSymbolRefAttr":$interface, "::mlir::TypeAttr":$signature,
        "::mlir::ArrayAttr":$workgroup_size),
      [{build($_builder, $_state, sym_name, ordinal, interface, signature,
          workgroup_size, 0);}]>
  ];

  let verifier = [{ return verifyExecutableEntryPointOp(*this); }];

  let extraClassDeclaration = [{
    Region *getBody() {
      auto regions = workgroup_count_region();
      if (!regions.empty()) return &regions.front();
      return nullptr;
    }
    Block* getBlock() {
      if (Region *region = getBody()) return &region->front();
      return nullptr;
    }
  }];
}

def HAL_ExecutableTargetOp : HAL_Op<"executable.target", [
    IsolatedFromAbove,
    HasParent<"IREE::HAL::ExecutableOp">,
    SingleBlockImplicitTerminator<"IREE::HAL::ExecutableTargetEndOp">,
    Symbol,
    SymbolTable,
  ]> {
  let summary = [{target executable IR}];
  let description = [{
    The target IR for the executable. This can be preserved for debugging but
    is usually removed during transformation.
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    StrAttr:$target_backend_filter
    // TODO(benvanik): add compatibility and versioning attributes.
    // TODO(scotttodd): add linking / preserve_ordinals attribute(s) for targets
    //                  with special linking requirements
  );

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$name, "StringRef":$targetBackendFilter)>,
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
    Symbol,
  ]> {
  let summary = [{compiled executable binary data}];
  let description = [{
    A compiled executable binary with an optional nested module containing the
    IR prior to serialization (for debugging).

    The `format` attribute specifies a four character code (FourCC) matching the
    executable format in `iree/hal/executable_format.h`.
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    HAL_ExecutableFormatAttr:$format,
    HAL_ExecutableDataAttr:$data
    // TODO(benvanik): add compatibility and versioning attributes.
  );

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$name, "uint32_t":$format, "std::vector<uint8_t>":$data)>,
    OpBuilder<(ins "StringRef":$name, "uint32_t":$format, "DenseIntElementsAttr":$data)>,
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

def HAL_ExecutableCreateOp : HAL_PureOp<"executable.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{creates an executable}];
  let description = [{
    Creates a target-dependent executable cached on the provided device. Entry
    points contained within the executable can be dispatched using the resulting
    executable handle.

    Depending on the driver creation may take a non-trivial amount of time
    (such as when JITing/etc). As the cache is internally synchronized callers
    can issue preparation requests from multiple threads - even for the same
    executables - and calls will block until preparation completes.

    ```mlir
    %exe = hal.executable.create %device, @executable::@target,
                                 layouts = [%exe_layout_0] : !hal.executable
    ```
  }];

  let arguments = (ins
    HAL_Device:$device,
    SymbolRefAttr:$executable_target,
    Variadic<HAL_ExecutableLayout>:$layouts
  );
  let results = (outs
    HAL_Executable:$result
  );

  let assemblyFormat = [{
    $device `,` $executable_target `,`
    `layouts` `=` `[` $layouts `]`
    attr-dict-with-keyword `:` type($result)
  }];
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
    OpBuilder<(ins "Value":$device, "StringRef":$executable),
    [{
      $_state.addOperands({device});
      $_state.addAttribute("executable", $_builder.getSymbolRefAttr(executable));
      $_state.addTypes({ExecutableType::get($_builder.getContext())});
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
    OpBuilder<(ins "StringRef":$name,
      CArg<"IntegerAttr", "{}">:$pushConstants)>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return body().front(); }

    // TODO(benvanik): replace with a nested typed attr that works.
    // Array of HAL_DescriptorSetLayoutBindingArrayAttr.
    ArrayAttr getExecutableSetLayoutsAttr();

    // Returns true if the all bindings in the interface match exactly those
    // in |other| (including order).
    bool isEquivalentTo(IREE::HAL::InterfaceOp other);
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

def HAL_InterfaceWorkgroupIDOp : HAL_PureOp<"interface.workgroup.id", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{returns the index of the current workgroup in the grid}];
  let description = [{
    The global workgroup ID of the current tile in the range of
    `[0, hal.interface.workgroup.count)` along each XYZ dimension.

    Corresponds to the `WorkgroupId` SPIR-V built-in and the `blockIdx` CUDA
    built-in variable.

    ```mlir
    %x = hal.interface.workgroup.id[0] : index
    %y = hal.interface.workgroup.id[1] : index
    %z = hal.interface.workgroup.id[2] : index
    ```
  }];

  let arguments = (ins IndexAttr:$dimension);
  let results = (outs HAL_Dim:$result);

  let assemblyFormat = "`[` $dimension `]` attr-dict `:` type($result)";
}

def HAL_InterfaceWorkgroupCountOp : HAL_PureOp<"interface.workgroup.count", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{returns the total workgroup count of the grid}];
  let description = [{
    The total number of workgroups along each dimension in the dispatch grid.
    Matches what was passed to the `hal.command_buffer.dispatch` command (or
    what was indirectly specified).

    Corresponds to the `NumWorkgroups` SPIR-V built-in and the `gridDim` CUDA
    built-in variable.

    ```mlir
    %x = hal.interface.workgroup.count[0] : index
    %y = hal.interface.workgroup.count[1] : index
    %z = hal.interface.workgroup.count[2] : index
    ```
  }];

  let arguments = (ins IndexAttr:$dimension);
  let results = (outs HAL_Dim:$result);

  let assemblyFormat = "`[` $dimension `]` attr-dict `:` type($result)";
}

def HAL_InterfaceWorkgroupSizeOp : HAL_PureOp<"interface.workgroup.size", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = [{returns the size of each workgroup in invocations}];
  let description = [{
    The number of local invocations within the current workgroup along each
    dimension. Depending on backend this may map to the SIMT thread count or
    inner loop nest parameters.

    Corresponds to the `WorkgroupSize` SPIR-V built-in and the `blockDim` CUDA
    built-in variable.

    ```mlir
    %x = hal.interface.workgroup.size[0] : index
    %y = hal.interface.workgroup.size[1] : index
    %z = hal.interface.workgroup.size[2] : index
    ```
  }];

  let arguments = (ins IndexAttr:$dimension);
  let results = (outs HAL_Dim:$result);

  let assemblyFormat = "`[` $dimension `]` attr-dict `:` type($result)";
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

def HAL_InterfaceBindingSubspanOp : HAL_Op<"interface.binding.subspan", [
    MemoryEffects<[MemAlloc]>]> {
  let summary = [{returns an alias to a subspan of interface binding data}];
  let description = [{
    // TODO(benvanik): add description
  }];

  let arguments = (ins
    SymbolRefAttr:$binding,
    HAL_DeviceSize:$byte_offset,
    Optional<HAL_DeviceSize>:$byte_length
  );
  let results = (outs
    Res<AnyType, "", [MemAlloc]>:$result
  );

  let assemblyFormat = [{
    $binding `[` $byte_offset ( `,` $byte_length^ )? `]`
    attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    /// Returns the hal.interface.binding op associated with this op.
    /// Returns null op if not found.
    IREE::HAL::InterfaceBindingOp queryBindingOp();
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

def HAL_InterfaceLoadTensorTileOp : HAL_PureOp<
    "interface.load.tensor.tile",
    [AttrSizedOperandSegments, OffsetSizeAndStrideOpInterface]> {
  let summary = [{loads a tensor tile from an executable IO binding}];
  let description = [{
    Loads a tensor tile value from an executable IO binding at the given
    offset/size/stride. This is a pseudo op that can be used to tie SSA tensor
    values in the IR to the bindings that contain those tensors.

    Note that because there may not be a 1:1 mapping between original tensor
    arguments to the entry point function and the bindings in the interface the
    backend must use the offset provided on this op to properly compute the base
    address of the tensor data. The offset is in bytes relative to the base
    binding address, irrespective of the type of the tensor loaded by this
    operation.

    The base_offset provided, if non-zero, will have an alignment compatible
    with the tensor type represented. For example, a `tensor<16xf32>` will be
    aligned on at least a 4 byte boundary.

    The op follows the semantic of similar core MLIR strided subview / subtensor
    ops where offset-list, size-list and stride-list are SSA value or constants
    of type index.

    Example:
    ```
      `hal.interface.load.tensor.tile` $binding `,`
        `base_offset` `=` base_offset `,`
        `offsets` `=` `[` offset-list `]` `,`
        `sizes` `=` `[` size-list `]` `,`
        `strides` `=` `[` stride-list `]` attr-dict `:` type($result)
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$binding,
    HAL_DeviceSize:$base_offset,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    I64ArrayAttr:$static_offsets,
    I64ArrayAttr:$static_sizes,
    I64ArrayAttr:$static_strides
  );
  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    $binding `,` `base_offset` `=` $base_offset `,` `offsets` `=`
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    `,` `sizes` `=` custom<OperandsOrIntegersSizesList>($sizes, $static_sizes)
    `,` `strides` `=`
    custom<OperandsOrIntegersOffsetsOrStridesList>($strides, $static_strides)
    attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    /// Returns the hal.interface.binding op associated with this op.
    /// Returns null op if not found.
    IREE::HAL::InterfaceBindingOp queryBindingOp();

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned resultRank = getResult().getType().cast<ShapedType>().getRank();
      return {resultRank, resultRank, resultRank};
    }

    /// Special attribute names.
    static StringRef getBindingAttrName() {
      return "binding";
    }
    static ArrayRef<StringRef> getSpecialAttrNames() {
      static SmallVector<StringRef, 5> names{getBindingAttrName()};
      if (names.size() !=
          1 + OffsetSizeAndStrideOpInterface::getSpecialAttrNames().size()) {
        auto otherNames = OffsetSizeAndStrideOpInterface::getSpecialAttrNames();
        names.append(otherNames.begin(), otherNames.end());
      }
      return names;
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  }];
}

def HAL_InterfaceStoreTensorTileOp :  HAL_Op<
    "interface.store.tensor.tile",
    [AttrSizedOperandSegments, OffsetSizeAndStrideOpInterface]> {
  let summary = [{stores a tensor tile in an executable IO binding}];
  let description = [{
    Stores a tensor value into an executable IO binding. This is a pseudo op
    indicating that the value of the operand tensor should be stored into the
    specified binding at the given offset/size/stride.

    Note that because there may not be a 1:1 mapping between original tensor
    arguments to the entry point function and the bindings in the interface the
    backend must use the offset provided on this op to properly compute the base
    address of the tensor data. The offset is in bytes relative to the base
    binding address, irrespective of the type of the tensor loaded by this
    operation.

    The base_offset provided, if non-zero, will have an alignment compatible
    with the tensor type represented. For example, a `tensor<16xf32>` will be
    aligned on at least a 4 byte boundary.

    The op follows the semantic of similar core MLIR strided subview / subtensor
    ops where offset-list, size-list and stride-list are SSA value or constants
    of type index.

    Grammar:
    ```
      `hal.interface.store.tensor.tile` $operand `,` $binding `,`
        `base_offset` `=` base_offset `,`
        `offsets` `=` `[` offset-list `]` `,`
        `sizes` `=` `[` size-list `]` `,`
        `strides` `=` `[` stride-list `]` attr-dict `:` type($operand)
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    SymbolRefAttr:$binding,
    HAL_DeviceSize:$base_offset,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    I64ArrayAttr:$static_offsets,
    I64ArrayAttr:$static_sizes,
    I64ArrayAttr:$static_strides
  );

  let assemblyFormat = [{
    $operand `,` $binding `,` `base_offset` `=` $base_offset `,` `offsets` `=`
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    `,` `sizes` `=` custom<OperandsOrIntegersSizesList>($sizes, $static_sizes)
    `,` `strides` `=`
    custom<OperandsOrIntegersOffsetsOrStridesList>($strides, $static_strides)
    attr-dict `:` type($operand)
  }];

  let extraClassDeclaration = [{
    /// Returns the hal.interface.binding op associated with this op.
    /// Returns null op if not found.
    IREE::HAL::InterfaceBindingOp queryBindingOp();

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = operand().getType().cast<ShapedType>().getRank();
      return {rank, rank, rank};
    }

    /// Special attribute names.
    static StringRef getBindingAttrName() {
      return "binding";
    }
    static ArrayRef<StringRef> getSpecialAttrNames() {
      static SmallVector<StringRef, 5> names{getBindingAttrName()};
      if (names.size() !=
          1 + OffsetSizeAndStrideOpInterface::getSpecialAttrNames().size()) {
        auto otherNames = OffsetSizeAndStrideOpInterface::getSpecialAttrNames();
        llvm::append_range(names, otherNames);
      }
      return names;
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }
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
                                           push_constants = 3,
                                           set_layouts = [%set0, %set1] : !hal.executable_layout
    ```
  }];

  let arguments = (ins
    HAL_Device:$device,
    I32Attr:$push_constants,
    Variadic<HAL_DescriptorSetLayout>:$set_layouts
  );
  let results = (outs
    HAL_ExecutableLayout:$result
  );

  let assemblyFormat = [{
    $device `,`
    `push_constants` `=` $push_constants `,`
    `set_layouts` `=` `[` $set_layouts `]`
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
