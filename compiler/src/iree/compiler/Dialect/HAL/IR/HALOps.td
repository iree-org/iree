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
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class HAL_PureOp<string mnemonic, list<Trait> traits = []> :
    HAL_Op<mnemonic, !listconcat(traits, [Pure])>;

class HAL_MakeTupleOp<string mnemonic, list<Trait> traits = []> :
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

//===----------------------------------------------------------------------===//
// Pseudo ops for conversion support
//===----------------------------------------------------------------------===//

def HAL_TensorImportOp : HAL_PureOp<"tensor.import", [
  AttrSizedOperandSegments,
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
    HAL_ShapeDynamicDims:$target_dims,
    Optional<HAL_Fence>:$wait_fence,
    OptionalAttr<StrAttr>:$name
  );
  let results = (outs
    AnyTensor:$target
  );

  let assemblyFormat = [{
    (`wait` `(` $wait_fence^ `)` `=` `` `>`)?
    $source
    ($name^)?
    `:` type($source) `->`
    custom<TypeAlias>($target_encoding, type($target)) (`{` $target_dims^ `}`)?
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins
      "Type":$resultType,
      "Value":$source,
      "StringAttr":$name
    )>,
    OpBuilder<(ins
      "Type":$resultType,
      "Value":$source,
      "Value":$waitFence,
      "StringAttr":$name
    )>,
  ];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return {}; }
    ValueRange getResultDynamicDims(unsigned idx) { return getTargetDims(); }
  }];

  let hasVerifier = 1;

  let hasFolder = 1;
}

def HAL_TensorExportOp : HAL_PureOp<"tensor.export", [
  AttrSizedOperandSegments,
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

    The provided `source_encoding`, if different from the `source` type,
    indicates that the ABI-facing type may differ from the internal
    representation. The types must be bitcastable (same storage size) and
    dynamically shaped values must have the same number of dynamic dimensions.
    This allows for casting between rank-0 and rank-N types, different element
    types, etc.

    An optional `target_storage` buffer can be provided to hold the exported
    result. The export will fail at runtime if the storage is null or if it has
    insufficient capacity to store the output. The storage must be
    device-visible and defined for transfer-target and dispatch usage.
  }];

  let arguments = (ins
    AnyTensor:$source,
    TypeAttr:$source_encoding,
    HAL_ShapeDynamicDims:$source_dims,
    Optional<HAL_Buffer>:$target_storage,
    OptionalAttr<StrAttr>:$name
  );
  let results = (outs
    AnyTypeOf<[HAL_Buffer, HAL_BufferView]>:$target
  );

  let assemblyFormat = [{
    $source
    ($name^)?
    (`into` $target_storage^)?
    `:`
    custom<TypeAlias>($source_encoding, type($source)) (`{` $source_dims^ `}`)?
    `->`
    type($target)
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins
      "Type":$resultType,
      "Value":$source,
      "StringAttr":$name
    )>,
  ];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getSourceDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasFolder = 1;
}

def HAL_TensorBarrierOp : HAL_Op<"tensor.barrier", [
  AllTypesMatch<["sources", "results"]>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{signals a fence when all tensors are available}];
  let description = [{
    Defines a barrier that is used to indicate availability of an entire set of
    tensors by signaling a fence. The source tensors are returned for chaining.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$sources,
    HAL_Fence:$signal_fence
  );
  let results = (outs
    Variadic<AnyTensor>:$results
  );

  let assemblyFormat = [{
    `join` `` `(` $sources `:` type($sources) `)`
    `=` `` `>`
    $signal_fence `:` type($signal_fence)
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins
      "ValueRange":$sources,
      "Value":$signalFence
    )>,
  ];

  let hasCanonicalizer = 1;
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

def HAL_AllocatorAllocateInitializedOp : HAL_Op<"allocator.allocate.initialized", [
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
    Util_BufferType:$source,
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
    Util_BufferType:$source,
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
    HAL_BufferType:$source_buffer,
    HAL_DeviceSize:$source_offset,
    HAL_DeviceSize:$source_length,
    HAL_ElementType:$element_type,
    HAL_EncodingType:$encoding_type,
    HAL_Shape:$shape
  );
  let results = (outs
    HAL_BufferView:$result
  );

  let assemblyFormat = [{
    `buffer` `(` $source_buffer `:` type($source_buffer) `)`
    `` `[` $source_offset `,` $source_length `]`
    `shape` `(` `[` $shape `]` `)`
    `type` `(` $element_type `)`
    `encoding` `(` $encoding_type `)`
    `:` type($result)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "Value":$sourceBuffer,
      "Value":$sourceOffset,
      "Value":$sourceLength,
      "int32_t":$elementType,
      "int32_t":$encodingType,
      "ValueRange":$shape
    )>,
    OpBuilder<(ins
      "Value":$sourceBuffer,
      "Value":$sourceOffset,
      "Value":$sourceLength,
      "Value":$elementType,
      "Value":$encodingType,
      "ValueRange":$shape
    )>,
  ];

  let hasCanonicalizer = 1;
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
// !hal.channel / iree_hal_channel_t
//===----------------------------------------------------------------------===//

def HAL_ChannelCreateOp : HAL_Op<"channel.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  ]> {
  let summary = [{creates a new channel for collective communication}];
  let description = [{
    Returns a new channel with the given rank associated with the given device
    queue. Collective operations using this channel must only be submitted on
    compatible queues.

    A rank or count of -1 can be used to indicate a default inherited from the
    environment or device configuration.
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_DeviceQueueAffinity:$queue_affinity,
    // TODO(benvanik): optional discriminator for groupings.
    // TODO(benvanik): channel ID as a `Util_BufferType:$id`.
    // TODO(benvanik): make rank/count optional (but then would need a custom vm
    // import pattern).
    I32:$rank,
    I32:$count
  );
  let results = (outs
    HAL_Channel:$result
  );

  let assemblyFormat = [{
    `device` `(` $device `:` type($device) `)`
    `affinity` `(` $queue_affinity `)`
    `rank` `(` $rank `)`
    `count` `(` $count `)`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HAL_ChannelRankAndCountOp : HAL_PureOp<"channel.rank_and_count", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  ]> {
  let summary = [{returns the rank of the local participant in the group}];
  let description = [{
    Returns the rank the channel represents as a participant in a collective
    group in `[0, count)` and the total participant count.
  }];

  let arguments = (ins
    HAL_Channel:$channel
  );
  let results = (outs
    I32:$rank,
    I32:$count
  );

  let assemblyFormat = [{
    `<` $channel `:` type($channel) `>`
    `:` type($rank) `,` type($count)
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
    HAL_CommandCategoryBitfieldAttr:$command_categories,
    Optional<Index>:$binding_capacity
  );
  let results = (outs
    HAL_CommandBuffer:$result
  );

  let assemblyFormat = [{
    `device` `(` $device `:` type($device) `)`
    `mode` `(` $modes `)`
    `categories` `(` $command_categories `)`
    (`bindings` `(` $binding_capacity^ `)`)?
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HAL_CommandBufferFinalizeOp : HAL_Op<"command_buffer.finalize"> {
  let summary = [{finalizes command buffer recording}];
  let description = [{
    Ends recording into the command buffer and prepares it for submission.
    No more commands may be recorded into the command buffer.
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
    HAL_FillPatternType:$pattern
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

def HAL_CommandBufferCollectiveOp :
  HAL_Op<"command_buffer.collective", [
    AttrSizedOperandSegments,
  ]> {
  let summary = [{command buffer collective dispatch recording operation}];
  let description = [{
    Dispatches a collective operation defined by op using the given buffers.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    HAL_Channel:$channel,
    HAL_CollectiveAttr:$op,
    HAL_DeviceSize:$element_count,
    Optional<I32>:$param,
    // TODO(benvanik): change this to take descriptor set + binding instead.
    // This would let us use indirect bindings.
    Optional<HAL_BufferType>:$send_buffer,
    Optional<HAL_DeviceSize>:$send_offset,
    Optional<HAL_DeviceSize>:$send_length,
    Optional<HAL_BufferType>:$recv_buffer,
    Optional<HAL_DeviceSize>:$recv_offset,
    Optional<HAL_DeviceSize>:$recv_length
  );

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    `channel` `(` $channel `:` type($channel) `)`
    `op` `(` $op `)`
    (`param` `(` $param^ `:` type($param) `)`)?
    (`send` `(` $send_buffer^ `:` type($send_buffer) `)`
     `` `[` $send_offset `,` $send_length `]`)?
    (`recv` `(` $recv_buffer^ `:` type($recv_buffer) `)`
     `` `[` $recv_offset `,` $recv_length `]`)?
    `count` `(` $element_count `)`
    attr-dict-with-keyword
  }];
}

def HAL_CommandBufferPushConstantsOp :
    HAL_Op<"command_buffer.push_constants"> {
  let summary = [{command buffer push constants operation}];
  let description = [{
    Pushes an inline set of constants that can be accessed by subsequent
    dispatches using a compatible pipeline layout.

    Push constants are always 4-byte values and treated as opaque, meaning that
    they may be bit-casted floats, bit-packed booleans, etc.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    HAL_PipelineLayout:$pipeline_layout,
    IndexAttr:$offset,
    Variadic<I32>:$values
  );

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    `layout` `(` $pipeline_layout `:` type($pipeline_layout) `)`
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
    The provided buffers may either be HAL buffers or indirect references into
    the command buffer binding table.
  }];

  let arguments = (ins
    HAL_CommandBuffer:$command_buffer,
    HAL_PipelineLayout:$pipeline_layout,
    Index:$set,
    Variadic<Index>:$binding_ordinals,
    Variadic<AnyTypeOf<[Index, HAL_BufferType]>>:$binding_buffers,
    Variadic<HAL_DeviceSize>:$binding_offsets,
    Variadic<HAL_DeviceSize>:$binding_lengths
  );

  let assemblyFormat = [{
    `<` $command_buffer `:` type($command_buffer) `>`
    `layout` `(` $pipeline_layout `:` type($pipeline_layout) `)`
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
    OpBuilder<(ins "Value":$commandBuffer, "Value":$pipelineLayout,
      "int64_t":$set, "ArrayRef<DescriptorSetBindingValue>":$bindings)>,
    OpBuilder<(ins "Value":$commandBuffer, "Value":$pipelineLayout,
      "Value":$set, "ArrayRef<DescriptorSetBindingValue>":$bindings)>,
  ];

  let hasCanonicalizer = 1;
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
    HAL_DescriptorSetLayoutFlagsAttr:$flags,
    HAL_DescriptorSetLayoutBindingArrayAttr:$bindings
  );
  let results = (outs
    HAL_DescriptorSetLayout:$result
  );

  let assemblyFormat = [{
    `device` `(` $device `:` type($device) `)`
    `flags` `(` $flags `)`
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
    HAL_DescriptorSetLayoutFlagsAttr:$flags,
    HAL_DescriptorSetLayoutBindingArrayAttr:$bindings
  );
  let results = (outs
    HAL_DescriptorSetLayout:$result
  );

  let assemblyFormat = [{
    `device` `(` $device `:` type($device) `)`
    `flags` `(` $flags `)`
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
    RecursiveMemoryEffects,
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


  let hasVerifier = 1;
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
      build($_builder, $_state, std::nullopt);
    }]>,
  ];

  let hasVerifier = 1;
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

    Well-known keys:

    * hal.executable.format :: {some format}
      Returns 1 if the given format is supported by the device loader.

    * hal.device :: concurrency
      The maximum concurrently executable submissions, mapping roughly to the
      queue count. The actual concurrency available may be less than this based
      on dynamic runtime parameters such as power/thermal modes, quota limits,
      or user choice.

    * hal.dispatch :: concurrency
      The maximum concurrently executable workgroups for a particular dispatch.
      The actual concurrency available may be less depending on device state.
  }];

  let arguments = (ins
    HAL_Device:$device,
    StrAttr:$category,
    StrAttr:$key,
    OptionalAttr<TypedAttrInterface>:$default_value
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

  let hasVerifier = 1;
}

def HAL_DeviceQueueAllocaOp : HAL_Op<"device.queue.alloca", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<Util_SizeAwareOp>,
  ]> {
  let summary = [{allocates a queue-ordered transient buffer}];
  let description = [{
    Returns a queue-ordered transient buffer that will be available for use when
    the signal fence is reached. The allocation will not be made until the
    wait fence has been reached.

    The size of the buffer returned may be larger than the requested size if the
    allocator has specific alignment requirements or minimum allocation sizes.

    The buffer handle will remain live so long as there are retainers but the
    contents are undefined before the allocation signal fence has been signaled
    and after the deallocation wait fence has been reached.
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_DeviceQueueAffinity:$queue_affinity,
    HAL_Fence:$wait_fence,
    HAL_Fence:$signal_fence,
    HAL_DeviceQueuePool:$pool,
    HAL_MemoryTypeBitfieldAttr:$memory_types,
    HAL_BufferUsageBitfieldAttr:$buffer_usage,
    HAL_DeviceSize:$result_size
  );
  let results = (outs
    HAL_Buffer:$result
  );

  // TODO(benvanik): change type/usage to ref params.
  let assemblyFormat = [{
    `<` $device `:` type($device) `>`
    `affinity` `(` $queue_affinity `)`
    `wait` `(` $wait_fence `)`
    `signal` `(` $signal_fence `)`
    `pool` `(` $pool `)`
    `type` `(` $memory_types `)`
    `usage` `(` $buffer_usage `)`
    `:` custom<SizeAwareType>(type($result), $result_size)
    attr-dict-with-keyword
  }];
}

def HAL_DeviceQueueDeallocaOp : HAL_Op<"device.queue.dealloca"> {
  let summary = [{deallocates a queue-ordered transient buffer}];
  let description = [{
    Deallocates a queue-ordered transient buffer.
    The deallocation will not be made until the wait fence has been reached and
    once the storage is available for reuse the signal fence will be signaled.

    After deallocation the contents of the buffer may still be accessible but
    will have undefined contents as other operations reuse the memory.
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_DeviceQueueAffinity:$queue_affinity,
    HAL_Fence:$wait_fence,
    HAL_Fence:$signal_fence,
    HAL_Buffer:$buffer
  );
  let results = (outs);

  let assemblyFormat = [{
    `<` $device `:` type($device) `>`
    `affinity` `(` $queue_affinity `)`
    `wait` `(` $wait_fence `)`
    `signal` `(` $signal_fence `)`
    `buffer` `(` $buffer `:` type($buffer) `)`
    attr-dict-with-keyword
  }];
}

def HAL_DeviceQueueExecuteOp : HAL_Op<"device.queue.execute"> {
  let summary = [{enqueues command buffer execution}];
  let description = [{
    Executes one or more command buffers on a device queue.
    The command buffers are executed in order as if they were recorded as one.
    No commands will execute until the wait fence has been reached and the
    signal fence will be signaled when all commands have completed.
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_DeviceQueueAffinity:$queue_affinity,
    HAL_Fence:$wait_fence,
    HAL_Fence:$signal_fence,
    Variadic<HAL_CommandBuffer>:$command_buffers
  );
  let results = (outs);

  let assemblyFormat = [{
    `<` $device `:` type($device) `>`
    `affinity` `(` $queue_affinity `)`
    `wait` `(` $wait_fence `)`
    `signal` `(` $signal_fence `)`
    (`commands` `(` `[` $command_buffers^ `]` `)`)?
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
}

def HAL_DeviceQueueFlushOp : HAL_Op<"device.queue.flush"> {
  let summary = [{flushes locally-pending submissions to the queue}];
  let description = [{
    Flushes any locally-pending submissions in the queue.
    When submitting many queue operations this can be used to eagerly flush
    earlier submissions while later ones are still being constructed.
    This may be a no-op.
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_DeviceQueueAffinity:$queue_affinity
  );
  let results = (outs);

  let assemblyFormat = [{
    `<` $device `:` type($device) `>`
    `affinity` `(` $queue_affinity `)`
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// !hal.executable / iree_hal_executable_t
//===----------------------------------------------------------------------===//

def HAL_ExecutableSourceOp : HAL_Op<"executable.source", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"IREE::HAL::ExecutableSourceEndOp">,
    Symbol,
    SymbolTable,
  ]> {
  let summary = [{generic source contents of an executable op}];
  let description = [{
    This is an unspecialized source representation of an executable
    module without an assigned target. This is useful for hand-authoring
    executables prior to device specification.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    OptionalAttr<HAL_ExecutableObjectsAttr>:$objects
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name
    attr-dict-with-keyword
    ``
    regions
  }];

  let extraClassDeclaration = [{
    Block& getBlock() { return getBody().front(); }

    bool isExternal() {
      return getBlock().getOps<::mlir::ModuleOp>().empty();
    }

    ::mlir::ModuleOp getInnerModule() {
      auto moduleOps = getBlock().getOps<::mlir::ModuleOp>();
      assert(!moduleOps.empty() && "source ops need inner modules");
      return *moduleOps.begin();
    }
  }];
}

def HAL_ExecutableSourceEndOp : HAL_Op<"executable.source_end", [
    HasParent<"IREE::HAL::ExecutableSourceOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the executable source op}];
  let assemblyFormat = "attr-dict";
}

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
    regions
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$name)>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return getBody().front(); }
  }];

  let hasVerifier = 1;
}

def HAL_ExecutableEndOp : HAL_Op<"executable_end", [
    HasParent<"IREE::HAL::ExecutableOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the executable op}];
  let assemblyFormat = "attr-dict";
}

def HAL_ExecutableExportOp : HAL_Op<"executable.export", [
    Symbol,
    ParentOneOf<[
      "IREE::HAL::ExecutableSourceOp",
      "IREE::HAL::ExecutableVariantOp",
    ]>,
    IsolatedFromAbove,
  ]> {
  let summary = [{executable entry point declaration}];
  let description = [{
    An entry point exported by the executable with statically-available
    information describing the IO interface it uses and other dispatch metadata.

    The `calculate_workgroup_count` region represents the computation that
    returns the number of workgroups to use in the 3D grid dispatch.
    The arguments to the region represents the workload as captured by each
    dispatch.
    It returns the number of workgroups along x, y, and z.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    OptionalAttr<HAL_OrdinalAttr>:$ordinal,
    HAL_PipelineLayoutAttr:$layout,
    OptionalAttr<HAL_WorkgroupSizeAttr>:$workgroup_size,
    OptionalAttr<HAL_SubgroupSizeAttr>:$subgroup_size,
    OptionalAttr<IndexAttr>:$workgroup_local_memory
  );

  let regions = (region AnyRegion:$workgroup_count);

  let builders = [
    OpBuilder<(ins
      "::mlir::StringAttr":$sym_name,
      "::mlir::IntegerAttr":$ordinal,
      "IREE::HAL::PipelineLayoutAttr":$layout,
      "::mlir::ArrayAttr":$workgroup_size,
      "::mlir::IntegerAttr":$subgroup_size,
      "::mlir::IntegerAttr":$workgroup_local_memory
    ), [{
      build($_builder, $_state, nullptr, sym_name, ordinal, layout,
            workgroup_size, subgroup_size, workgroup_local_memory);
    }]>,
  ];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// For now assume that the workload is at max 3D.
    /// Arguments to the region are workload along x, y and z.
    // TODO: Propogate this and avoid magic constants.
    static int64_t getNumWorkgroupDims() { return 3; }

    Block* getWorkgroupCountBody() {
      if (getWorkgroupCount().empty()) return nullptr;
      return &getWorkgroupCount().front();
    }

    // Calculates an XYZ workgroup count based on the given |workload|.
    std::array<Value, 3> calculateWorkgroupCount(
        Location loc, Value device, ValueRange workload, OpBuilder &builder);

    // Calculates an XYZ workgroup size based on the given |workload|.
    std::array<Value, 3> calculateWorkgroupSize(
        Location loc, Value device, ValueRange workload, OpBuilder &builder);
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
    HAL_ExecutableTargetAttr:$target,
    OptionalAttr<HAL_ExecutableObjectArrayAttr>:$objects
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name
    `,` `target` `=` $target
    (`,` `objects` `=` $objects^ )?
    attr-dict-with-keyword
    regions
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$name, "IREE::HAL::ExecutableTargetAttr":$target)>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return getBody().front(); }

    bool isExternal() {
      return getBlock().getOps<::mlir::ModuleOp>().empty();
    }

    ::mlir::ModuleOp getInnerModule() {
      auto moduleOps = getBlock().getOps<::mlir::ModuleOp>();
      assert(!moduleOps.empty() && "source ops need inner modules");
      return *moduleOps.begin();
    }

    // Returns a map of constant key attributes to ordinals across all constant
    // blocks inside the variant.
    DenseMap<Attribute, int> gatherConstantOrdinals();
  }];

  let hasCanonicalizer = 1;
}

def HAL_ExecutableVariantEndOp : HAL_Op<"executable.variant_end", [
    HasParent<"IREE::HAL::ExecutableVariantOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the executable variant op}];
  let assemblyFormat = "attr-dict";
}

def HAL_ExecutableConstantBlockOp :
  HAL_Op<"executable.constant.block", [
    ParentOneOf<[
      "IREE::HAL::ExecutableSourceOp",
      "IREE::HAL::ExecutableVariantOp",
    ]>,
    IsolatedFromAbove,
    CallableOpInterface,
    FunctionOpInterface,
  ]> {
  let summary = [{executable constant block initializer}];
  let description = [{
    Initializes one or more constants in the executable constant block by
    returning one value per identified constant. Each constant block is
    evaluated on the host prior to instantiating the executable for a given
    device and allows for the executable to be specialized based on device
    capabilities and limits.

    The keys specified are unique per variant and will be deduplicated across
    multiple constant blocks when present. They are only used during lowering
    and will not survive to runtime so they need only have descriptive enough
    names to avoid collisions and represent the semantics of the value.

    Constant values can be loaded in the device code with the
    `hal.executable.constant.load` op:

    ```mlir
    hal.executable.variant public @target {
      hal.executable.constant.block(%device: !hal.device) -> (i32, i32) as ("foo", "bar") {
        %0 = hal.device.query<%device> key("some.device.prop")...
        %1 = hal.device.query<%device> key("another.device.prop")...
        hal.return %0, %1 : i32, i32
      }
      builtin.module {
        func @dispatch0() {
          %0 = hal.executable.constant.load "foo" : i32
          %1 = hal.executable.constant.load "bar" : i32
          return
        }
      }
    }
    ```

    Each target backend will implement the constant initialization and access in
    a way compatible with its execution model. Examples:
    - CPU: read-only buffer initialized on load and passed to each dispatch
    - CUDA: read-only buffer initialized on load and passed to each dispatch
    - SPIR-V: specialization constants
    - Metal: function constants
    - WebGPU: pipeline-overridable constants
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    ArrayAttr:$keys,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins
      "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>,
  ];

  let extraClassDeclaration = [{
    // CallableOpInterface:
    ::mlir::Region *getCallableRegion() { return &getBody(); }
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }
    // FunctionOpInterface:
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    // Placeholder attribute added to ops that look up constants by key.
    // Ordinal assignment passes use this to identify and update usage.
    static StringRef getKeyAttrName() { return "hal.executable.constant.key"; }

    /// Make symbol optional as this op has no symbol.
    bool isOptionalSymbol() { return true; }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def HAL_ExecutableConstantLoadOp : HAL_PureOp<"executable.constant.load"> {
  let summary = [{loads a constant value from the executable constant block}];
  let description = [{
    Loads a scalar constant value from the static executable constant block.
    The value provided by a constant block with the given key will be loaded and
    bitcast (possibly with truncation or zero-extension) to the result type.

    Note that backends are allowed to implement their own mechanisms for
    referencing constant block values and this is provided only as a default for
    those not needing special behavior.
  }];

  let arguments = (ins
    StrAttr:$key
  );
  let results = (outs
    HAL_PrimitiveType:$result
  );

  let assemblyFormat = [{
    $key attr-dict `:` type($result)
  }];
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
    AttrSizedOperandSegments,
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

    Optional constants provide for specialization of the executable based on
    runtime-derived parameters.
  }];

  let arguments = (ins
    HAL_Device:$device,
    SymbolRefAttr:$executable_target,
    Variadic<HAL_PipelineLayout>:$layouts,
    Variadic<I32>:$constants
  );
  let results = (outs
    HAL_Executable:$result
  );

  let assemblyFormat = [{
    `device` `(` $device `:` type($device) `)`
    `target` `(` $executable_target `)`
    `layouts` `(` `[` $layouts `]` `)`
    (`constants` `(` `[` $constants^ `]` `)`)?
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

def HAL_ExecutableCalculateWorkgroupsOp : HAL_PureOp<"executable.calculate_workgroups"> {
  let summary = [{calculates workgroup count from workload for an exported function}];
  let description = [{
    Calculates the workgroup count (grid XYZ) based on the given workload using
    the workgroup count calculation region of the target
    `hal.executable.export` op.
  }];

  let arguments = (ins
    HAL_Device:$device,
    SymbolRefAttr:$entry_point,
    Variadic<Index>:$workload
  );
  let results = (outs
    HAL_Dim:$workgroup_x,
    HAL_Dim:$workgroup_y,
    HAL_Dim:$workgroup_z
  );

  let assemblyFormat = [{
    `device` `(` $device `:` type($device) `)`
    `target` `(` $entry_point `)`
    (`workload` `(` `[` $workload^ `]` `)`)?
    `:` type($workgroup_x) `,` type($workgroup_y) `,` type($workgroup_z)
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// hal.instrument.*
//===----------------------------------------------------------------------===//

def HAL_InstrumentWorkgroupOp : HAL_Op<"instrument.workgroup", []> {
  let summary = [{emits a dispatch workgroup instrumentation event}];
  let description = [{
    Emits an `iree_instrument_dispatch_workgroup_t` event into the
    instrumentation stream. The workgroup event identifies the unique dispatch,
    its workgroup count, and the ID of the emitting workgroup within the
    dispatch. Optionally targets that support querying the processor ID
    executing the workgroup can attach that information for tracking purposes.

    On targets such as CPUs where entire workgroups execute as atomic units
    only one workgroup event should be emitted. On targets such as GPUs where
    there may be multiple invocations executing as part of a single workgroup
    only the first invocation within the workgroup should emit the workgroup
    event (by checking if the LocalInvocationIndex or threadIdx == 0, etc).

    The resulting workgroup key is used by subsequent workgroup-specific
    instrumentation events.
  }];

  let arguments = (ins
    AnyMemRef:$buffer,
    I32:$dispatchId
  );
  let results = (outs
    Index:$workgroupKey
  );

  let assemblyFormat = [{
    `` `[` $buffer `:` type($buffer) `]`
    `dispatch` `(` $dispatchId `)`
    attr-dict `:` type($workgroupKey)
  }];
}

def HAL_InstrumentPrintOp : HAL_Op<"instrument.print", []> {
  let summary = [{emits a human-readable printf-style string event}];
  let description = [{
    Formats a string using a limited subset of printf format specifiers and the
    provided values and then emits an `iree_instrument_dispatch_print_t` event. Final
    formatted string lengths may be limited to as much as 1024 characters and
    should be kept as small as possible to avoid easily exceeding the
    instrumentation storage buffers with redundant strings.
  }];

  let arguments = (ins
    AnyMemRef:$buffer,
    Index:$workgroupKey,
    StrAttr:$format,
    Variadic<AnyType>:$values
  );

  let assemblyFormat = [{
    `` `[` $buffer `:` type($buffer) `for` $workgroupKey `]`
    $format (`*` `(` $values^ `:` type($values) `)`)?
    attr-dict
  }];
}

def HAL_InstrumentValueOp : HAL_Op<"instrument.value", [
    AllTypesMatch<["operand", "result"]>,
  ]> {
  let summary = [{emits a scalar value instrumentation event}];
  let description = [{
    Emits a workgroup-specific typed value with the given workgroup-relative
    ordinal.

    This op will be preserved even if the output is not used as it is only for
    debugging purposes.
  }];

  let arguments = (ins
    AnyMemRef:$buffer,
    Index:$workgroupKey,
    AnyI8Attr:$ordinal,
    AnyType:$operand
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    `` `[` $buffer `:` type($buffer) `for` $workgroupKey `]`
    $ordinal `=` $operand attr-dict `:` type($operand)
  }];
}

def HAL_InstrumentMemoryLoadOp : HAL_PureOp<"instrument.memory.load", [
    AllTypesMatch<["loadValue", "result"]>,
  ]> {
  let summary = [{emits a memory load instrumentation event}];
  let description = [{
    Emits a workgroup-specific memory load event indicating that a number of
    bytes from the given resolved pointer have been loaded by the workgroup.
  }];

  let arguments = (ins
    AnyMemRef:$buffer,
    Index:$workgroupKey,
    AnyType:$loadValue,
    AnyMemRef:$base,
    Variadic<Index>:$indices
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    `` `[` $buffer `:` type($buffer) `for` $workgroupKey `]`
    $base `[` $indices `]` `,` $loadValue
    attr-dict `:` type($base) `,` type($result)
  }];
}

def HAL_InstrumentMemoryStoreOp : HAL_PureOp<"instrument.memory.store", [
    AllTypesMatch<["storeValue", "result"]>,
  ]> {
  let summary = [{emits a memory store instrumentation event}];
  let description = [{
    Emits a workgroup-specific memory store event indicating that a number of
    bytes have been stored to the given resolved pointer by the workgroup.
  }];

  let arguments = (ins
    AnyMemRef:$buffer,
    Index:$workgroupKey,
    AnyType:$storeValue,
    AnyMemRef:$base,
    Variadic<Index>:$indices
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    `` `[` $buffer `:` type($buffer) `for` $workgroupKey `]`
    $base `[` $indices `]` `,` $storeValue
    attr-dict `:` type($base) `,` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// hal.interface
//===----------------------------------------------------------------------===//

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

  let builders = [
    OpBuilder<(ins "unsigned":$dim),
    [{
      build($_builder, $_state, $_builder.getIndexType(), $_builder.getIndexAttr(dim));
    }]>,
  ];

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

  let builders = [
    OpBuilder<(ins "unsigned":$dim),
    [{
      build($_builder, $_state, $_builder.getIndexType(), $_builder.getIndexAttr(dim));
    }]>,
  ];

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

  let builders = [
    OpBuilder<(ins "unsigned":$dim),
    [{
      build($_builder, $_state, $_builder.getIndexType(), $_builder.getIndexAttr(dim));
    }]>,
  ];

  let assemblyFormat = [{
    `[` $dimension `]` attr-dict `:` type($result)
  }];
}

def HAL_InterfaceConstantLoadOp : HAL_PureOp<"interface.constant.load"> {
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
    HAL_HostSizeAttr:$index,
    OptionalAttr<IndexAttr>:$alignment,
    OptionalAttr<ArrayAttr>:$values
  );
  let results = (outs
    HAL_PrimitiveType:$result
  );

  let assemblyFormat = [{
    `` `[` $index `]`
    (`alignment` `(` $alignment^ `)`)?
    (`values` `(` $values^ `)`)?
    attr-dict `:` type($result)
  }];
}

def HAL_InterfaceBindingSubspanOp : HAL_PureOp<"interface.binding.subspan", [
    AttrSizedOperandSegments,
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
    IndexAttr:$set,
    IndexAttr:$binding,
    HAL_DescriptorTypeAttr:$descriptor_type,
    Optional<HAL_DeviceSize>:$byte_offset,
    HAL_ShapeDynamicDims:$dynamic_dims,
    OptionalAttr<IndexAttr>:$alignment,
    OptionalAttr<HAL_DescriptorFlagsAttr>:$descriptor_flags
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    `set` `(` $set `)`
    `binding` `(` $binding `)`
    `type` `(` custom<DescriptorType>($descriptor_type) `)`
    (`alignment` `(` $alignment^ `)`)?
    (`offset` `(` $byte_offset^ `)`)?
    (`flags` `(` $descriptor_flags^ `)`)?
    attr-dict `:` type($result) (`{` $dynamic_dims^ `}`)?
  }];

  let builders = [
    OpBuilder<(ins
      "Type":$resultType,
      "APInt":$set,
      "APInt":$binding,
      "IREE::HAL::DescriptorType":$descriptor_type,
      "Value":$byte_offset,
      "ValueRange":$dynamic_dims,
      "IntegerAttr":$alignment,
      CArg<"mlir::Optional<DescriptorFlags>", "llvm::None">:$flags
    )>,
  ];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return getDynamicDims(); }

    // Returns the alignment of the base buffer pointer (before offset).
    llvm::MaybeAlign getBaseAlignment();

    // Attempts to calculate an alignment of the final subspan offset in the
    // parent storage buffer. This is a combination of both the binding
    // alignment and the byte offset and may be as small as the natural
    // alignment of the element type being accessed.
    llvm::Align calculateAlignment();
  }];
}

//===----------------------------------------------------------------------===//
// !hal.pipeline_layout / iree_hal_pipeline_layout_t
//===----------------------------------------------------------------------===//

def HAL_PipelineLayoutCreateOp : HAL_PureOp<"pipeline_layout.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  ]> {
  let summary = [{creates an pipeline layout}];
  let description = [{
    Creates an pipeline layout from the given descriptor sets and push
    constant required size. Pipeline layouts can be shared across any
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
    HAL_PipelineLayout:$result
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

def HAL_PipelineLayoutLookupOp : HAL_PureOp<"pipeline_layout.lookup", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  ]> {
  let summary = [{pipeline layout cache lookup pseudo-op}];
  let description = [{
    Used during conversion to provide a placeholder for a globally cached and
    possibly lazy-initialized pipeline layout.
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_PipelineLayoutAttr:$layout
  );
  let results = (outs
    HAL_PipelineLayout:$result
  );

  let assemblyFormat = [{
    `device` `(` $device `:` type($device) `)`
    `layout` `(` $layout `)`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// !hal.fence / iree_hal_fence_t
//===----------------------------------------------------------------------===//

def HAL_FenceCreateOp : HAL_Op<"fence.create", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    MemoryEffects<[MemAlloc]>,
  ]> {
  let summary = [{creates an unsignaled fence}];
  let description = [{
    Returns a fence that defines a point in time. By default fences will remain
    unsignaled unless they are explicitly signaled with `hal.fence.signal` or
    asynchronously signaled by the device by passing them as an operand to
    queue submission ops.
  }];

  let arguments = (ins
    HAL_Device:$device,
    HAL_FenceFlagBitfieldAttr:$flags
  );
  let results = (outs
    HAL_Fence:$result
  );

  let assemblyFormat = [{
    `device` `(` $device `:` type($device) `)`
    `flags` `(` $flags `)`
    `:` type($result)
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
}

def HAL_FenceJoinOp : HAL_Op<"fence.join", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  ]> {
  let summary = [{creates a fence from the given timepoints}];
  let description = [{
    Returns a fence that joins the input fences as a wait-all operation.
  }];

  let arguments = (ins
    Variadic<HAL_Fence>:$fences
  );
  let results = (outs
    HAL_Fence:$result
  );

  let assemblyFormat = [{
    `at` `(` `[` $fences `]` `)`
    `->` type($result)
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
}

def HAL_FenceQueryOp : HAL_Op<"fence.query"> {
  let summary = [{fence query operation}];
  let description = [{
    Queries whether the fence has been reached and its status.
    Returns OK if the fence has been signaled successfully, DEFERRED if it is
    unsignaled, and otherwise an error indicating the failure.
  }];

  let arguments = (ins
    HAL_Fence:$fence
  );
  let results = (outs
    Util_Status:$status
  );

  let assemblyFormat = [{
    `<` $fence `:` type($fence) `>`
    `:` type($status)
    attr-dict-with-keyword
  }];
}

def HAL_FenceSignalOp : HAL_Op<"fence.signal"> {
  let summary = [{fence signal operation}];
  let description = [{
    Signals the fence to indicate that the timepoints contained have been
    reached. Waiting work may begin immediately.
  }];

  let arguments = (ins
    HAL_Fence:$fence
  );

  let assemblyFormat = [{
    `<` $fence `:` type($fence) `>`
    attr-dict-with-keyword
  }];
}

def HAL_FenceFailOp : HAL_Op<"fence.fail"> {
  let summary = [{fence failure operation}];
  let description = [{
    Signals the fence with a failure. The `status` will be returned from
    each timepoint semaphores `hal.semaphore.query` and `hal.semaphore.signal`
    for the lifetime of each semaphore.
  }];

  let arguments = (ins
    HAL_Fence:$fence,
    Util_Status:$status
  );

  let assemblyFormat = [{
    `<` $fence `:` type($fence) `>`
    `status` `(` $status `)`
    attr-dict-with-keyword
  }];
}

def HAL_FenceAwaitOp : HAL_Op<"fence.await", [
    Util_YieldPoint,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  ]> {
  let summary = [{asynchronous fence wait operation}];
  let description = [{
    Yields the caller until all fences is reached. Returns the `status` of the
    fence after the wait, with a non-zero value indicating failure.
  }];

  let arguments = (ins
    I32:$timeout_millis,
    Variadic<HAL_Fence>:$fences
  );
  let results = (outs
    Util_Status:$status
  );

  let assemblyFormat = [{
    `until` `(` `[` $fences `]` `)`
    `timeout_millis` `(` $timeout_millis `)`
    `:` type($status)
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
}

#endif  // IREE_DIALECT_HAL_OPS
