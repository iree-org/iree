// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_HAL_OPS
#define IREE_DIALECT_HAL_OPS

include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "iree/compiler/Dialect/HAL/IR/HALInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilAttrs.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
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
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
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
// Pseudo ops for conversion support
//===----------------------------------------------------------------------===//

def HAL_TensorImportOp : HAL_PureOp<"tensor.import", [
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
  Util_ShapeAwareOp,
]> {
  let summary = [{imports a tensor from a HAL buffer view}];
  let description = [{
    Defines an import of an external HAL buffer view into a SSA-form tensor.
    An optional semaphore timepoint can be specified indicating when the
    buffer view is available for use. If no semaphore timepoint is provided it
    is assumed the buffer view is immediately available.

    The provided `target_encoding`, if different from the `target` type,
    indicates that the ABI-facing type may differ from the internal
    representation. The types must be bitcastable (same storage size) and
    dynamically shaped values must have the same number of dynamic dimensions.
    This allows for casting between rank-0 and rank-N types, different element
    types, etc.
  }];

  let arguments = (ins
    AnyTypeOf<[HAL_Buffer, HAL_BufferView]>:$source,
    TypeAttr:$target_encoding,
    HAL_ShapeDynamicDims:$target_dims
  );
  let results = (outs
    AnyTensor:$target
  );

  let assemblyFormat = [{
    $source `:` type($source)
    `->`
    custom<TypeAlias>($target_encoding, type($target)) (`{` $target_dims^ `}`)?
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins
      "Type":$resultType,
      "Value":$source
    )>,
  ];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return {}; }
    ValueRange getResultDynamicDims(unsigned idx) { return target_dims(); }
  }];

  let verifier = [{ return verifyTensorImportOp(*this); }];

  let hasFolder = 1;
}

def HAL_TensorExportOp : HAL_PureOp<"tensor.export", [
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
  Util_ShapeAwareOp,
]> {
  let summary = [{exports a tensor to a HAL buffer view}];
  let description = [{
    Defines an export of an SSA-form tensor to an external HAL buffer view.
    An optional semaphore timepoint can be specified indicating when the
    buffer view is available for use. If no semaphore timepoint is requested it
    is assumed execution blocks until the buffer view is available.

    The provided `source_encoding`, if different from the `source` type,
    indicates that the ABI-facing type may differ from the internal
    representation. The types must be bitcastable (same storage size) and
    dynamically shaped values must have the same number of dynamic dimensions.
    This allows for casting between rank-0 and rank-N types, different element
    types, etc.
  }];

  let arguments = (ins
    AnyTensor:$source,
    TypeAttr:$source_encoding,
    HAL_ShapeDynamicDims:$source_dims
  );
  let results = (outs
    AnyTypeOf<[HAL_Buffer, HAL_BufferView]>:$target
  );

  let assemblyFormat = [{
    $source `:`
    custom<TypeAlias>($source_encoding, type($source)) (`{` $source_dims^ `}`)?
    `->`
    type($target)
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins
      "Type":$resultType,
      "Value":$source
    )>,
  ];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return source_dims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return {}; }
  }];

  let verifier = [{ return verifyTensorExportOp(*this); }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// !hal.allocator / iree_hal_allocator_t
//===----------------------------------------------------------------------===//

def HAL_AllocatorAllocateOp : HAL_Op<"allocator.allocate", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<Util_SizeAwareOp>,
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
    HAL_DeviceSize:$result_size
  );
  let results = (outs
    HAL_Buffer:$result
  );

  // TODO(benvanik): change type/usage to ref params.
  let assemblyFormat = [{
    `<` $allocator `:` type($allocator) `>`
    `type` `(` $memory_types `)`
    `usage` `(` $buffer_usage `)`
    `:` custom<SizeAwareType>(type($result), $result_size)
    attr-dict-with-keyword
  }];
}

def HAL_AllocatorMapOp : HAL_Op<"allocator.map", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<Util_SizeAwareOp>,
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

  // TODO(benvanik): change type/usage to ref params.
  let assemblyFormat = [{
    `<` $allocator `:` type($allocator) `>`
    `source` `(` $source `:` type($source) `)` `` `[` $offset `,` $length `]`
    `type` `(` $memory_types `)`
    `usage` `(` $buffer_usage `)`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HAL_AllocatorTryMapOp : HAL_Op<"allocator.try_map", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<Util_SizeAwareOp>,
  ]> {
  let summary = [{allocator-supported host buffer wrapping operation}];
  let description = [{
    Tries wrapping a !hal.buffer around host read-only memory backed by the
    given byte buffer. The returned buffer may be host-only and not directly
    usable on devices. If the mapping cannot be completed (such as trying to
    map the host memory as device-local on devices with discrete memory) then
    did_map will indicate that the returned buffer is null.
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
    I1:$did_map,
    HAL_Buffer:$result
  );

  // TODO(benvanik): change type/usage to ref params.
  let assemblyFormat = [{
    `<` $allocator `:` type($allocator) `>`
    `source` `(` $source `:` type($source) `)` `` `[` $offset `,` $length `]`
    `type` `(` $memory_types `)`
    `usage` `(` $buffer_usage `)`
    `:` type($did_map) `,` type($result)
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// !hal.buffer / iree_hal_buffer_t
//===----------------------------------------------------------------------===//

def HAL_BufferAssertOp : HAL_Op<"buffer.assert", []> {
  let summary = [{buffer compatibility assertion}];
  let description = [{
    Asserts that the buffer is compatible with the given allocator and usage.
    Program execution will abort as if `std.assert` had been used.

    This only checks that the buffer can be used and not that it matches the
    given parameters exactly. Buffers may be from other allocators so long as
    the allocators are compatible (devices can address each other's memory),
    the type and usage contain all the requested bits (having more bits is ok),
    and the length is at least the requested minimum (as padding may be
    ignored).
  }];

  let arguments = (ins
    HAL_Buffer:$buffer,
    StrAttr:$message,
    HAL_Allocator:$allocator,
    HAL_DeviceSize:$minimum_length,
    HAL_MemoryTypeBitfieldAttr:$memory_types,
    HAL_BufferUsageBitfieldAttr:$buffer_usage
  );
  let results = (outs);

  let assemblyFormat = [{
    `<` $buffer `:` type($buffer) `>`
    `message` `(` $message `)`
    `allocator` `(` $allocator `:` type($allocator) `)`
    `minimum_length` `(` $minimum_length `)`
    `type` `(` $memory_types `)`
    `usage` `(` $buffer_usage `)`
    attr-dict-with-keyword
  }];

  // TODO(benvanik): fold away when we know some properties of the buffer
  // (such as when we create it ourselves earlier on) or we've already asserted.
}

def HAL_BufferAllocatorOp : HAL_PureOp<"buffer.allocator", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  ]> {
  let summary = [{buffer allocator accessor operation}];
  let description = [{
    Returns the allocator this buffer was allocated from.
  }];

  let arguments = (ins
    HAL_BufferType:$buffer
  );
  let results = (outs
    HAL_Allocator:$result
  );

  let assemblyFormat = [{
    `<` $buffer `:` type($buffer) `>`
    `:` type($result)
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
}

def HAL_BufferSubspanOp : HAL_PureOp<"buffer.subspan", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<Util_SizeAwareOp>,
  ]> {
  let summary = [{buffer subspan operation}];
  let description = [{
    Returns a reference to a subspan of the buffer.
  }];

  let arguments = (ins
    HAL_BufferType:$source_buffer,
    HAL_DeviceSize:$source_offset,
    HAL_DeviceSize:$length
  );
  let results = (outs
    HAL_BufferType:$result
  );

  let assemblyFormat = [{
    `<` $source_buffer `:` type($source_buffer) `>`
    `` `[` $source_offset `,` $length `]`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HAL_BufferLengthOp : HAL_PureOp<"buffer.length", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  ]> {
  let summary = [{buffer byte length accessor}];
  let description = [{
    Returns the allocated size of a buffer in bytes.
    May be less than the underlying buffer allocation if this is a subspan or
    view into another buffer.
  }];

  let arguments = (ins
    HAL_BufferType:$buffer
  );
  let results = (outs
    HAL_DeviceSize:$result
  );

  let assemblyFormat = [{
    `<` $buffer `:` type($buffer) `>`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HAL_BufferLoadOp : HAL_PureOp<"buffer.load"> {
  let summary = [{buffer element load operation}];
  let description = [{
    Loads a value from a buffer by mapping it.
  }];

  let arguments = (ins
    HAL_BufferType:$source_buffer,
    HAL_DeviceSize:$source_offset
  );
  let results = (outs
    AnyTypeOf<[HAL_PrimitiveType, AnyVector]>:$result
  );

  let assemblyFormat = [{
    `<` $source_buffer `:` type($source_buffer) `>`
    `` `[` $source_offset `]`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HAL_BufferStoreOp : HAL_Op<"buffer.store"> {
  let summary = [{buffer element store operation}];
  let description = [{
    Stores a value into a buffer by mapping it.
  }];

  let arguments = (ins
    AnyTypeOf<[HAL_PrimitiveType, AnyVector]>:$value,
    HAL_BufferType:$target_buffer,
    HAL_DeviceSize:$target_offset
  );

  let assemblyFormat = [{
    `<` $target_buffer `:` type($target_buffer) `>`
    `` `[` $target_offset `]`
    `value` `(` $value `:` type($value) `)`
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// !hal.buffer_view / iree_hal_buffer_view_t
//===----------------------------------------------------------------------===//

def HAL_BufferViewCreateOp : HAL_PureOp<"buffer_view.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  ]> {
  let summary = [{buffer view reference initializer}];
  let description = [{
    Creates a reference to a buffer with a particular shape and element type.
    The buffer is not copied and both the original and view references must be
    synchronized. This makes it easier to associate commonly-carried metadata
    along with the contents.
  }];

  let arguments = (ins
    HAL_BufferType:$buffer,
    HAL_ElementType:$element_type,
    HAL_EncodingType:$encoding_type,
    HAL_Shape:$shape
  );
  let results = (outs
    HAL_BufferView:$result
  );

  let assemblyFormat = [{
    `buffer` `(` $buffer `:` type($buffer) `)`
    `shape` `(` `[` $shape `]` `)`
    `type` `(` $element_type `)`
    `encoding` `(` $encoding_type `)`
    `:` type($result)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "Value":$buffer,
      "int32_t":$elementType,
      "int32_t":$encodingType,
      "ValueRange":$shape
    )>,
    OpBuilder<(ins
      "Value":$buffer,
      "Value":$elementType,
      "Value":$encodingType,
      "ValueRange":$shape
    )>,
  ];
}

def HAL_BufferViewAssertOp : HAL_Op<"buffer_view.assert", []> {
  let summary = [{buffer view contents assertion}];
  let description = [{
    Asserts that the buffer view contains a data compatible tensor with the
    given encoding. Program execution will abort as if `std.assert` had been
    used.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view,
    StrAttr:$message,
    HAL_ElementType:$element_type,
    HAL_EncodingType:$encoding_type,
    HAL_Shape:$shape
  );
  let results = (outs);

  let assemblyFormat = [{
    `<` $buffer_view `:` type($buffer_view) `>`
    `message` `(` $message `)`
    `shape` `(` `[` $shape `]` `)`
    `type` `(` $element_type `)`
    `encoding` `(` $encoding_type `)`
    attr-dict-with-keyword
  }];

  // TODO(benvanik): fold away when we know some properties of the buffer view
  // (such as when we create it ourselves earlier on) or we've already asserted.
}

def HAL_BufferViewBufferOp : HAL_PureOp<"buffer_view.buffer", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  ]> {
  let summary = [{buffer view buffer accessor}];
  let description = [{
    Returns the buffer backing this view's contents.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view
  );
  let results = (outs
    HAL_BufferType:$result
  );

  let assemblyFormat = [{
    `<` $buffer_view `:` type($buffer_view) `>`
    `:` type($result)
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
}

def HAL_BufferViewByteLengthOp : HAL_PureOp<"buffer_view.byte_length", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  ]> {
  let summary = [{buffer view buffer byte length accessor}];
  let description = [{
    Returns the allocated size of a shaped buffer view in bytes.
    This may exceed the logical size of the buffer view; for example sparse
    tensors will have an allocation size that may be for the maximum possible
    size of the tensor when fully dense.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view
  );
  let results = (outs
    HAL_DeviceSize:$result
  );

  let assemblyFormat = [{
    `<` $buffer_view `:` type($buffer_view) `>`
    `:` type($result)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$bufferView)>,
  ];
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

  let assemblyFormat = [{
    `<` $buffer_view `:` type($buffer_view) `>`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HAL_BufferViewEncodingTypeOp : HAL_PureOp<"buffer_view.encoding_type"> {
  let summary = [{buffer view encoding type query}];
  let description = [{
    Returns the encoding type of the buffer view.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view
  );
  let results = (outs
    HAL_EncodingType:$result
  );

  let assemblyFormat = [{
    `<` $buffer_view `:` type($buffer_view) `>`
    `:` type($result)
    attr-dict-with-keyword
  }];
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
    HAL_Dim:$result
  );

  let assemblyFormat = [{
    `<` $buffer_view `:` type($buffer_view) `>`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HAL_BufferViewDimOp : HAL_PureOp<"buffer_view.dim"> {
  let summary = [{buffer view dimension value query}];
  let description = [{
    Returns the value of the given dimension.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view,
    IndexAttr:$index
  );
  let results = (outs
    HAL_Dim:$result
  );

  let assemblyFormat = [{
    `<` $buffer_view `:` type($buffer_view) `>`
    `` `[` $index `]`
    `:` type($result)
    attr-dict-with-keyword
  }];
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

  let assemblyFormat = [{
    `<` $buffer_view `:` type($buffer_view) `>`
    `:` type($result)
    attr-dict-with-keyword
  }];

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

  let assemblyFormat = [{
    $operands `:` type($operands)
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// !hal.command_buffer / iree_hal_command_buffer_t
//===----------------------------------------------------------------------===//

def HAL_CommandBufferCreateOp : HAL_Op<"command_buffer.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
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
    `device` `(` $device `:` type($device) `)`
    `mode` `(` $modes `)`
    `categories` `(` $command_categories `)`
    `:` type($result)
    attr-dict-with-keyword
  }];
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

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    attr-dict-with-keyword
  }];
}

def HAL_CommandBufferEndOp : HAL_Op<"command_buffer.end"> {
  let summary = [{command buffer recording end operation}];
  let description = [{
    Ends recording into the command buffer.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer
  );

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    attr-dict-with-keyword
  }];
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

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    `:` type($device)
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
}

def HAL_CommandBufferBeginDebugGroupOp : HAL_Op<"command_buffer.begin_debug_group"> {
  let summary = [{pushes a command buffer debug group label}];
  let description = [{
    Pushes a new debug group with the given label.
    All commands between this and a mandatory matching call to
    `hal.command_buffer.end_debug_group` will be grouped together with the
    given label.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    StrAttr:$label
  );

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    `label` `(` $label `)`
    attr-dict-with-keyword
  }];
}

def HAL_CommandBufferEndDebugGroupOp : HAL_Op<"command_buffer.end_debug_group"> {
  let summary = [{pops a command buffer debug group label}];
  let description = [{
    Pops a debug group from the stack.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer
  );

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    attr-dict-with-keyword
  }];
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

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    `source` `(` $source_stage_mask `)`
    `target` `(` $target_stage_mask `)`
    `flags` `(` $flags `)`
    attr-dict-with-keyword
  }];
}

// TODO(benvanik): event ops.

def HAL_CommandBufferFillBufferOp : HAL_Op<"command_buffer.fill_buffer"> {
  let summary = [{command buffer buffer fill recording operation}];
  let description = [{
    Fills the target buffer with the given repeating value.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    HAL_BufferType:$target_buffer,
    HAL_DeviceSize:$target_offset,
    HAL_DeviceSize:$length,
    AnyTypeOf<[I1, I8, I16, I32, F16, F32]>:$pattern
  );

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    `target` `(` $target_buffer `:` type($target_buffer) `)`
    `` `[` $target_offset `,` $length `]`
    `pattern` `(` $pattern `:` type($pattern) `)`
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
}

// TODO(benvanik): update buffer op.

def HAL_CommandBufferCopyBufferOp : HAL_Op<"command_buffer.copy_buffer"> {
  let summary = [{command buffer buffer copy recording operation}];
  let description = [{
    Copies a range of one buffer to another.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    HAL_BufferType:$source_buffer,
    HAL_DeviceSize:$source_offset,
    HAL_BufferType:$target_buffer,
    HAL_DeviceSize:$target_offset,
    HAL_DeviceSize:$length
  );

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    `source` `(` $source_buffer `:` type($source_buffer) `)`
    `` `[` $source_offset `]`
    `target` `(` $target_buffer `:` type($target_buffer) `)`
    `` `[` $target_offset `]`
    `length` `(` $length `)`
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
}

def HAL_CommandBufferPushConstantsOp :
    HAL_Op<"command_buffer.push_constants"> {
  let summary = [{command buffer push constants operation}];
  let description = [{
    Pushes an inline set of constants that can be accessed by subsequent
    dispatches using a compatible executable layout.

    Push constants are always 4-byte values and treated as opaque, meaning that
    they may be bit-casted floats, bit-packed booleans, etc.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    HAL_ExecutableLayout:$executable_layout,
    IndexAttr:$offset,
    Variadic<I32>:$values
  );

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    `layout` `(` $executable_layout `:` type($executable_layout) `)`
    `offset` `(` $offset `)`
    `values` `(` `[` $values `]` `)`
    `:` type($values)
    attr-dict-with-keyword
  }];
}

def HAL_CommandBufferPushDescriptorSetOp :
    HAL_Op<"command_buffer.push_descriptor_set", [
      SameVariadicOperandSize,
    ]> {
  let summary = [{command buffer descriptor set push binding operation}];
  let description = [{
    Pushes an inline-defined descriptor set to the command buffer.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    HAL_ExecutableLayout:$executable_layout,
    Index:$set,
    Variadic<Index>:$binding_ordinals,
    Variadic<HAL_BufferType>:$binding_buffers,
    Variadic<HAL_DeviceSize>:$binding_offsets,
    Variadic<HAL_DeviceSize>:$binding_lengths
  );

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    `layout` `(` $executable_layout `:` type($executable_layout) `)`
    `` `[` $set `]`
    `bindings` `(` `[`
    custom<DescriptorSetBindings>($binding_ordinals,
                                  $binding_buffers,
                                  type($binding_buffers),
                                  $binding_offsets,
                                  $binding_lengths)
    `]` `)`
    attr-dict-with-keyword
  }];

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
    `<` $command_buffer `:` type($command_buffer) `>`
    `layout` `(` $executable_layout `:` type($executable_layout) `)`
    `` `[` $set `]`
    `set` `(` $descriptor_set `:` type($descriptor_set) `)`
    (`offsets` `(` `[` $dynamic_offsets^ `]` `)`)?
    attr-dict-with-keyword
  }];
}

def HAL_CommandBufferDispatchSymbolOp : HAL_Op<"command_buffer.dispatch.symbol"> {
  let summary = [{command buffer dispatch recording operation, using symbolref}];
  let description = [{
    Dispatches an execution request, using a nested symbol reference to the entry point.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    SymbolRefAttr:$entry_point,
    HAL_Dim:$workgroup_x,
    HAL_Dim:$workgroup_y,
    HAL_Dim:$workgroup_z
  );

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    `target` `(` $entry_point `)`
    `workgroups` `(` `[`
        $workgroup_x `,`
        $workgroup_y `,`
        $workgroup_z
    `]` `)`
    attr-dict-with-keyword
  }];
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
    `<` $command_buffer `:` type($command_buffer) `>`
    `target` `(` $executable `:` type($executable) `)`
    `` `[` $entry_point `]`
    `workgroups` `(` `[`
        $workgroup_x `,`
        $workgroup_y `,`
        $workgroup_z
    `]` `)`
    attr-dict-with-keyword
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
    HAL_BufferType:$workgroups_buffer,
    HAL_DeviceSize:$workgroups_offset
  );

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    `target` `(` $entry_point `)`
    `workgroups` `(` $workgroups_buffer `:` type($workgroups_buffer) `)`
    `` `[` $workgroups_offset `]`
    attr-dict-with-keyword
  }];
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
    HAL_BufferType:$workgroups_buffer,
    HAL_DeviceSize:$workgroups_offset
  );

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    `target` `(` $executable `:` type($executable) `)`
    `` `[` $entry_point `]`
    `workgroups` `(` $workgroups_buffer `:` type($workgroups_buffer) `)`
    `` `[` $workgroups_offset `]`
    attr-dict-with-keyword
  }];
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
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    Util_AnySerializableAttr:$value
  );

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name `=` $value attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins
      "StringRef":$sym_name,
      "::mlir::Attribute":$value
    ), [{
      build($_builder, $_state, nullptr, sym_name, value);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// !hal.descriptor_set / iree_hal_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

def HAL_DescriptorSetCreateOp : HAL_PureOp<"descriptor_set.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
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
    Variadic<HAL_BufferType>:$binding_buffers,
    Variadic<HAL_DeviceSize>:$binding_offsets,
    Variadic<HAL_DeviceSize>:$binding_lengths
  );
  let results = (outs
    HAL_DescriptorSet:$result
  );

  let assemblyFormat = [{
    `device` `(` $device `:` type($device) `)`
    `layout` `(` $set_layout `:` type($set_layout) `)`
    `bindings` `(` `[`
    custom<DescriptorSetBindings>($binding_ordinals,
                                  $binding_buffers,
                                  type($binding_buffers),
                                  $binding_offsets,
                                  $binding_lengths)
    `]` `)`
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$device,
      "Value":$setLayout,
      "ArrayRef<DescriptorSetBindingValue>":$bindings
    )>,
  ];
}

//===----------------------------------------------------------------------===//
// !hal.descriptor_set_layout / iree_hal_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

def HAL_DescriptorSetLayoutCreateOp :
  HAL_PureOp<"descriptor_set_layout.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  ]> {
  let summary = [{creates a descriptor set layout}];
  let description = [{
    Creates a descriptor set layout that defines the bindings used within a set.
    The same descriptor set layout may be shared with many different executable
    layouts and by doing so some runtime binding overhead when switching between
    executables that use the same set layouts can be reduced.
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
    `device` `(` $device `:` type($device) `)`
    `usage` `(` $usage_type `)`
    `bindings` `(` $bindings `)`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HAL_DescriptorSetLayoutLookupOp : HAL_PureOp<"descriptor_set_layout.lookup", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  ]> {
  let summary = [{descriptor set layout cache lookup pseudo-op}];
  let description = [{
    Used during conversion to provide a placeholder for a globally cached and
    possibly lazy-initialized descriptor set layout.
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
    `device` `(` $device `:` type($device) `)`
    `usage` `(` $usage_type `)`
    `bindings` `(` $bindings `)`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// !hal.device / iree_hal_device_t
//===----------------------------------------------------------------------===//

def HAL_DeviceAllocatorOp : HAL_PureOp<"device.allocator", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
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

  let assemblyFormat = [{
    `<` $device `:` type($device) `>` `:` type($result) attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$device),
    [{
      $_state.addOperands({device});
      $_state.addTypes({AllocatorType::get($_builder.getContext())});
    }]>,
  ];
}

def HAL_DeviceSwitchOp : HAL_Op<"device.switch", [
    NoRegionArguments,
    RecursiveSideEffects,
  ]> {
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
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %device = ... : !hal.device
    %0 = hal.device.switch<%device : !hal.device> -> i32
      #hal.device.match.id<"vulkan-v1.?-*"> {
        hal.return %c1 : i32
      },
      #hal.match.any<[#hal.device.match.id<"vmvx">, #hal.device.match.id<"vulkan-*">]> {
        hal.return %c2 : i32
      },
      #hal.match.always {
        hal.return %c0 : i32
      }
    ```
  }];

  let arguments = (ins
    HAL_Device:$device,
    ArrayAttr:$conditions
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region VariadicRegion<AnyRegion>:$condition_regions);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "TypeRange":$resultTypes,
      "Value":$device,
      "ArrayRef<Attribute>":$conditions,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes
    )>,
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

  let assemblyFormat = [{
    ($operands^ `:` type($operands))? attr-dict
  }];

  let builders = [
    OpBuilder<(ins),
    [{
      build($_builder, $_state, llvm::None);
    }]>,
  ];
}

def HAL_DeviceQueryOp :
    HAL_PureOp<"device.query", [HAL_DeviceQuery]> {
  let summary = [{returns a runtime configuration parameter from the device}];
  let description = [{
    Queries a device configuration parameter with the given key.
    Returns a status indicating whether the pair was recognized/available and if
    it was the value converted to the specified type. Queries must return the
    same value for the lifetime of the module though may vary from run to run.

    This is roughly equivalent to the `sysconf` linux syscall
    (https://man7.org/linux/man-pages/man3/sysconf.3.html) in that the exact
    set of keys available and their interpretation is target-dependent. If there
    is a HAL match attribute (`#hal.device.match.*`) or op
    (`hal.device.match.*`) prefer to use that in order to get compile-time
    propagation when the target is specified and elide the runtime query and
    get compile-time verification when a runtime query is required.

    Users of the op must check the `ok` result before using the value as what
    set of keys is available may change over time. If in doubt: don't use this.
    Each key used adds additional versioning and testing complexity as runtime
    code path changes will explode combinatorially and should be treated with as
    much care as a binary file format change. Keys should be prefixed with `ex.`
    when experimental indicating that they are not expected to be present
    forever; all non-experimental keys should be vetted.
  }];

  let arguments = (ins
    HAL_Device:$device,
    StrAttr:$category,
    StrAttr:$key,
    OptionalAttr<AnyAttr>:$default_value
  );
  let results = (outs
    I1:$ok,
    AnyType:$value
  );

  let assemblyFormat = [{
    `<` $device `:` type($device) `>`
    `key` `(` $category `:` `` `:` $key `)`
    `:` type($ok) `,` type($value)
    (`=` $default_value^)?
    attr-dict-with-keyword
  }];

  let verifier = [{ return verifyDeviceQueryOp(*this); }];
}

//===----------------------------------------------------------------------===//
// !hal.executable / iree_hal_executable_t
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
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name
    // TODO(benvanik): entry point types for verification.
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name
    attr-dict-with-keyword
    ``
    regions
  }];

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
    HasParent<"IREE::HAL::ExecutableVariantOp">,
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
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    HAL_OrdinalAttr:$ordinal,
    FlatSymbolRefAttr:$interface,
    OptionalAttr<HAL_WorkgroupSizeAttr>:$workgroup_size,
    OptionalAttr<IndexAttr>:$workgroup_local_memory
  );

  let regions = (region VariadicRegion<SizedRegion<1>>:$workgroup_count_region);

  let builders = [
    OpBuilder<(ins
      "::llvm::StringRef":$sym_name,
      "::llvm::APInt":$ordinal,
      "::llvm::StringRef":$interface,
      "::mlir::ArrayAttr":$workgroup_size,
      "::mlir::IntegerAttr":$workgroup_local_memory
    ), [{
      build($_builder, $_state, nullptr, sym_name, ordinal, interface,
            workgroup_size, workgroup_local_memory, 0);
    }]>,
    OpBuilder<(ins
      "::mlir::StringAttr":$sym_name,
      "::mlir::IntegerAttr":$ordinal,
      "::mlir::FlatSymbolRefAttr":$interface,
      "::mlir::ArrayAttr":$workgroup_size,
      "::mlir::IntegerAttr":$workgroup_local_memory
    ), [{
      build($_builder, $_state, nullptr, sym_name, ordinal, interface,
            workgroup_size, workgroup_local_memory, 0);
    }]>,
    OpBuilder<(ins
      "::mlir::StringAttr":$sym_name,
      "::mlir::IntegerAttr":$ordinal,
      "::mlir::FlatSymbolRefAttr":$interface,
      "::mlir::ArrayAttr":$workgroup_size,
      "::mlir::IntegerAttr":$workgroup_local_memory,
      "int":$workgroup_count_regionCount
    ), [{
      build($_builder, $_state, nullptr, sym_name, ordinal, interface,
            workgroup_size, workgroup_local_memory, workgroup_count_regionCount);
    }]>
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

def HAL_ExecutableVariantOp : HAL_Op<"executable.variant", [
    IsolatedFromAbove,
    HasParent<"IREE::HAL::ExecutableOp">,
    SingleBlockImplicitTerminator<"IREE::HAL::ExecutableVariantEndOp">,
    Symbol,
    SymbolTable,
  ]> {
  let summary = [{target-specific variant of an executable op}];
  let description = [{
    The target IR for the executable. This can be preserved for debugging but
    is usually removed during transformation.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    HAL_ExecutableTargetAttr:$target
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name
    `,` `target` `=` $target
    attr-dict-with-keyword
    ``
    regions
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$name, "IREE::HAL::ExecutableTargetAttr":$target)>,
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

def HAL_ExecutableVariantEndOp : HAL_Op<"executable.variant_end", [
    HasParent<"IREE::HAL::ExecutableVariantOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the executable variant op}];
  let assemblyFormat = "attr-dict";
}

def HAL_ExecutableBinaryOp : HAL_Op<"executable.binary", [
    HasParent<"IREE::HAL::ExecutableOp">,
    Symbol,
  ]> {
  let summary = [{compiled executable binary data}];
  let description = [{
    A compiled executable binary with an optional nested module containing the
    IR prior to serialization (for debugging).
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    StrAttr:$format,
    HAL_ExecutableDataAttr:$data,
    OptionalAttr<StrAttr>:$mime_type
    // TODO(benvanik): add compatibility and versioning attributes.
  );

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "StringRef":$name,
      "StringRef":$format,
      "std::vector<uint8_t>":$data
    )>,
    OpBuilder<(ins
      "StringRef":$name,
      "StringAttr":$format,
      "DenseIntElementsAttr":$data
    )>,
  ];
}

def HAL_ExecutableCreateOp : HAL_PureOp<"executable.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
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
    `device` `(` $device `:` type($device) `)`
    `target` `(` $executable_target `)`
    `layouts` `(` `[` $layouts `]` `)`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HAL_ExecutableLookupOp : HAL_PureOp<"executable.lookup", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
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

  let assemblyFormat = [{
    `device` `(` $device `:` type($device) `)`
    `executable` `(` $executable `)`
    `:` type($result)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$device, "StringRef":$executable),
    [{
      $_state.addOperands({device});
      $_state.addAttribute("executable", mlir::SymbolRefAttr::get($_builder.getContext(), executable));
      $_state.addTypes({ExecutableType::get($_builder.getContext())});
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// hal.interface
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
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    OptionalAttr<IndexAttr>:$push_constants
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name
    attr-dict-with-keyword
    ``
    regions
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "StringRef":$name,
      CArg<"IntegerAttr", "{}">:$pushConstants
    )>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return body().front(); }

    // TODO(benvanik): replace with a nested typed attr that works.
    // Array of HAL_DescriptorSetLayoutBindingArrayAttr.
    ArrayAttr getExecutableSetLayoutsAttr();

    // Returns true if the all bindings in the interface match exactly those
    // in |other| (including order).
    bool isEquivalentTo(IREE::HAL::InterfaceOp other);

    // Returns a hash for the interface, considering the push constant and
    // all bindings.
    llvm::hash_code getInterfaceHash();
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
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    IndexAttr:$set,
    IndexAttr:$binding,
    HAL_DescriptorTypeAttr:$type
  );

  // TODO(scotttodd): enable assemblyFormat after figuring out how to keep
  //   quotes around "type". The custom printer uses stringify*
  // let assemblyFormat = [{
  //   custom<SymbolVisibility>($sym_visibility)
  //   $sym_name
  //   `,` `set` `` `=` `` $set
  //   `,` `binding` `` `=` `` $binding
  //   `,` `type` `` `=` `` $type
  //   attr-dict-with-keyword
  // }];

  let builders = [
    OpBuilder<(ins
      "StringRef":$sym_name,
      "::llvm::APInt":$set,
      "::llvm::APInt":$binding,
      "IREE::HAL::DescriptorType":$type
    ), [{
      build($_builder, $_state, nullptr, sym_name, set, binding, type);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns a hash for the descriptor, considering the set, binding,
    /// and type.
    llvm::hash_code getDescriptorHash();
  }];
}

def HAL_InterfaceWorkgroupIDOp : HAL_PureOp<"interface.workgroup.id", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
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

  let assemblyFormat = [{
    `[` $dimension `]` attr-dict `:` type($result)
  }];
}

def HAL_InterfaceWorkgroupCountOp : HAL_PureOp<"interface.workgroup.count", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
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

  let assemblyFormat = [{
    `[` $dimension `]` attr-dict `:` type($result)
  }];
}

def HAL_InterfaceWorkgroupSizeOp : HAL_PureOp<"interface.workgroup.size", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
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

  let assemblyFormat = [{
    `[` $dimension `]` attr-dict `:` type($result)
  }];
}

def HAL_InterfaceLoadConstantOp : HAL_PureOp<"interface.load.constant"> {
  let summary = [{loads a constant value from the interface constant block}];
  let description = [{
    Loads a scalar constant value from an executable IO push constant block.
    The value will be loaded from the given constant offset and will be
    bitcast (possibly with truncation or zero-extension) to the result type.

    An optional alignment indicates the byte alignment of potential values for
    the constant when it could be determined from analysis. If omitted the value
    may be anything and its interpretation is up to the usage. This is intended
    to provide pointer alignment-like semantics to constants that are used to
    index into binding resources.

    An optional set of values indicates all possible values that can be passed
    to the constant from all dispatch sites in the program. If omitted the value
    may be from an unanalyzable source (outside of the program, indirect, etc)
    and must be assumed to have any value.
  }];

  let arguments = (ins
    HAL_HostSizeAttr:$offset,
    OptionalAttr<IndexAttr>:$alignment,
    OptionalAttr<ArrayAttr>:$values
  );
  let results = (outs
    HAL_PrimitiveType:$result
  );

  let assemblyFormat = [{
    `offset` `=` $offset attr-dict `:` type($result)
  }];
}

def HAL_InterfaceBindingSubspanOp : HAL_Op<"interface.binding.subspan", [
    AttrSizedOperandSegments, MemoryEffects<[MemAlloc]>,
    Util_ShapeAwareOp,
  ]> {
  let summary = [{returns an alias to a subspan of interface binding data}];
  let description = [{
    Returns a subspan of an interface binding storage buffer in a generic type.
    The exact shape, type, and alignment of the returned type are defined by
    the result type (tensor, memref, etc).

    An optional alignment indicates the byte alignment of the base binding
    resource. Note that the byte offset is added to the base and the alignment
    will be the minimum of the two.
  }];

  let arguments = (ins
    SymbolRefAttr:$binding,
    HAL_DeviceSize:$byte_offset,
    Optional<HAL_DeviceSize>:$byte_length,
    HAL_ShapeDynamicDims:$dynamic_dims,
    OptionalAttr<IndexAttr>:$alignment
  );
  let results = (outs
    Res<AnyType, "", [MemAlloc]>:$result
  );

  let assemblyFormat = [{
    $binding `[` $byte_offset ( `,` $byte_length^ )? `]`
    attr-dict `:` type($result) (`{` $dynamic_dims^ `}`)?
  }];

  let verifier = [{ return verifyInterfaceBindingSubspanOp(*this); }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return dynamic_dims(); }

    /// Returns the hal.interface.binding op associated with this op.
    /// Returns null op if not found.
    IREE::HAL::InterfaceBindingOp queryBindingOp();

    // Attempts to calculate an alignment of the final subspan offset in the
    // parent storage buffer. This is a combination of both the binding
    // alignment and the byte offset and may be as small as the natural
    // alignment of the element type being accessed.
    llvm::Align calculateAlignment();
  }];
}

//===----------------------------------------------------------------------===//
// !hal.executable_layout / iree_hal_executable_layout_t
//===----------------------------------------------------------------------===//

def HAL_ExecutableLayoutCreateOp : HAL_PureOp<"executable_layout.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
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
  }];

  let arguments = (ins
    HAL_Device:$device,
    IndexAttr:$push_constants,
    Variadic<HAL_DescriptorSetLayout>:$set_layouts
  );
  let results = (outs
    HAL_ExecutableLayout:$result
  );

  // TODO(benvanik): include descriptor set layout types.
  let assemblyFormat = [{
    `device` `(` $device `:` type($device) `)`
    `push_constants` `(` $push_constants `)`
    `layouts` `(` `[` $set_layouts `]` `)`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HAL_ExecutableLayoutLookupOp : HAL_PureOp<"executable_layout.lookup", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  ]> {
  let summary = [{executable layout cache lookup pseudo-op}];
  let description = [{
    Used during conversion to provide a placeholder for a globally cached and
    possibly lazy-initialized executable layout.
  }];

  let arguments = (ins
    HAL_Device:$device,
    OptionalAttr<IndexAttr>:$push_constants,
    // TODO(benvanik): replace with a nested typed attr that works.
    // Array of HAL_DescriptorSetLayoutBindingArrayAttr.
    ArrayAttr:$set_layouts
  );
  let results = (outs
    HAL_ExecutableLayout:$result
  );

  let assemblyFormat = [{
    `device` `(` $device `:` type($device) `)`
    (`push_constants` `(` $push_constants^ `)`)?
    `layouts` `(` $set_layouts `)`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// !hal.semaphore / iree_hal_semaphore_t
//===----------------------------------------------------------------------===//

def HAL_SemaphoreCreateOp : HAL_Op<"semaphore.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
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
    `device` `(` $device `:` type($device) `)`
    `initial` `(` $initial_value `)`
    `:` type($result)
    attr-dict-with-keyword
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
    Util_Status:$status,
    HAL_TimelineValue:$value
  );

  let assemblyFormat = [{
    `<` $semaphore `:` type($semaphore) `>`
    `:` type($status) `,` type($value)
    attr-dict-with-keyword
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
    `<` $semaphore `:` type($semaphore) `>`
    `value` `(` $new_value `)`
    attr-dict-with-keyword
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
    Util_Status:$status
  );

  let assemblyFormat = [{
    `<` $semaphore `:` type($semaphore) `>`
    `status` `(` $status `)`
    attr-dict-with-keyword
  }];
}

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
    Util_Status:$status
  );

  let assemblyFormat = [{
    `<` $semaphore `:` type($semaphore) `>`
    `until` `(` $min_value `)`
    `:` type($status)
    attr-dict-with-keyword
  }];
}

#endif  // IREE_DIALECT_HAL_OPS
