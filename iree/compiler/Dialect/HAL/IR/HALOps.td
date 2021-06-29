// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_HAL_OPS
#define IREE_DIALECT_HAL_OPS

include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "iree/compiler/Dialect/HAL/IR/HALInterfaces.td"
include "iree/compiler/Dialect/IREE/IR/IREEInterfaces.td"
include "iree/compiler/Dialect/Shape/IR/ShapeInterfaces.td"
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
// Pseudo ops for conversion support
//===----------------------------------------------------------------------===//

def HAL_TensorCastOp : HAL_PureOp<"tensor.cast", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<IREE_TiedOpInterface, [
        "getTiedResult",
        "getTiedResultOperandIndex",
        "getTiedResultOperandIndices",
    ]>,
    DeclareOpInterfaceMethods<Shape_ShapeCarryingOpInterface>,
  ]> {
  let summary = [{conversion placeholder for HAL<->tensor type conversion}];
  let description = [{
    Defines a conversion from a higher-level dialect type such as `tensor` that
    is resolved during lowering into the HAL. This can be used to interoperate
    between levels of the stack that require specifying HAL types and those that
    prior to lowering do not handle them.
  }];

  let arguments = (ins
    AnyType:$source,
    HAL_ShapeDynamicDims:$source_dims,
    HAL_ShapeDynamicDims:$target_dims
  );
  let results = (outs
    AnyType:$target
  );

  let assemblyFormat = [{
    $source `:`
    type($source) (`{` $source_dims^ `}`)? `->`
    type($target) (`{` $target_dims^ `}`)?
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "Type":$resultType,
      "Value":$source,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins
      "Type":$resultType,
      "Value":$source,
      "ValueRange":$dynamicDims,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let hasFolder = 1;
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
    OpBuilder<(ins
      "StringRef":$name,
      "bool":$isMutable,
      "Type":$type,
      "Optional<StringRef>":$initializer,
      "Optional<Attribute>":$initialValue,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
    OpBuilder<(ins
      "StringRef":$name,
      "bool":$isMutable,
      "mlir::FuncOp":$initializer,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
    OpBuilder<(ins
      "StringRef":$name,
      "bool":$isMutable,
      "Type":$type,
      "Attribute":$initialValue,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
    OpBuilder<(ins
      "StringRef":$name,
      "bool":$isMutable,
      "Type":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
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

  let assemblyFormat = [{
    $variable attr-dict `:` type($result)
  }];
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

  let assemblyFormat = [{
    $variable attr-dict `:` type($result)
  }];

  let verifier = [{ return verifyVariableLoadOp(*this); }];
}

def HAL_VariableLoadIndirectOp : HAL_Op<"variable.load.indirect"> {
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

  let assemblyFormat = [{
    $variable attr-dict `:` type($variable) `->` type($result)
  }];

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

  let assemblyFormat = [{
    $value `,` $variable attr-dict `:` type($value)
  }];

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

  let assemblyFormat = [{
    $value `,` $variable attr-dict `:` type($value) `->` type($variable)
  }];

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
// !hal.allocator / iree_hal_allocator_t
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
    `<` $allocator `:` type($allocator) `>`
    `shape` `(` `[` $shape `]` `)`
    `type` `(` $element_type `)`
    `:` type($result)
    attr-dict-with-keyword
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
    `<` $allocator `:` type($allocator) `>`
    `indices` `(` `[` $indices `]` `)`
    `shape` `(` `[` $shape `]` `)`
    `type` `(` $element_type `)`
    `:` type($offset)
    attr-dict-with-keyword
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
    `<` $allocator `:` type($allocator) `>`
    `indices` `(` `[` $indices `]` `)`
    `lengths` `(` `[` $lengths `]` `)`
    `shape` `(` `[` $shape `]` `)`
    `type` `(` $element_type `)`
    `:` type($offset) `,` type($length)
    attr-dict-with-keyword
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
    DeclareOpInterfaceMethods<HAL_SizeAwareOp>,
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

def HAL_AllocatorConstantOp : HAL_Op<"allocator.constant", [
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
    AnyTypeOf<[HAL_Buffer, HAL_BufferView]>:$result
  );

  // TODO(benvanik): change type/usage to ref params.
  let assemblyFormat = [{
    `<` $allocator `:` type($allocator) `>`
    `type` `(` $memory_types `)`
    `usage` `(` $buffer_usage `)`
    `:` type($result) `=` $value
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
}

def HAL_AllocatorMapOp : HAL_Op<"allocator.map", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    DeclareOpInterfaceMethods<HAL_SizeAwareOp>,
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

def HAL_AllocatorPackOp : HAL_PureOp<"allocator.pack", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    AttrSizedOperandSegments,
  ]> {
  let summary = [{packs variable-sized slices into a single slab}];
  let description = [{
    Performs a greedy packing of one or more sized slices with specified
    lifetimes and returns their relative offsets in an aliased linear space.

    Slices are `[start, end] = %slice_byte_size`, where the start and end values
    define an inclusive lifetime range and the size is the total number of bytes
    required to be live for that range.

    ```mlir
    // Computes thte total length required for the packed values and the offsets
    // of the 3 slices requested relative to the base of the packed memory:
    %total_length, %offset_0, %offset_1, %offset_2 =
        hal.allocator.pack<%allocator : !hal.allocator>
            // Each slice gets one result offset:
            slices([
              // 3 slices where A and B overlap and will get unique offsets
              // while B and C do not overlap and are allowed to alias.
              [0, 10] = %size_0,  // A => %offset_0
              [3,  8] = %size_1,  // B => %offset_1
              [9, 10] = %size_2,  // C => %offset_2
              ...
            ]) : index
    ```

    The lifetime start and end points (inclusive) are only used for relative
    comparisons and may originate with any meaning (op order in block, epoch,
    phase of the moon, etc). The packing algorithm uses the intervals to
    determine slice liveness and when aliasing is safe.

    The size of each slice may either be a constant or runtime-computed dynamic
    value. Constant slices can achieve more dense packing than the dynamic
    values and CSE/canonicalization should be applied to ensure that as many of
    the dynamic values are equivalent if possible.

    The total length required to pack all slices is returned and can be used to
    acquire storage. The individual slice offsets are 0-based and as such if are
    directly used as buffer offsets may need additional offseting. This can
    either be applied via the optional `offset` operand or slicing of the
    underlying allocation buffer.
  }];

  let arguments = (ins
    HAL_Allocator:$allocator,
    Optional<HAL_DeviceSize>:$offset,
    HAL_IndexArrayAttr:$lifetime_intervals,
    Variadic<HAL_DeviceSize>:$dynamic_slice_sizes
  );
  let results = (outs
    HAL_DeviceSize:$total_length,
    Variadic<HAL_DeviceSize>:$packed_offsets
  );

  let assemblyFormat = [{
    `<` $allocator `:` type($allocator) `>`
    (`offset` `(` $offset^ `)`)?
    `slices` `(` `{`
    custom<PackSliceRanges>($lifetime_intervals,
                            $dynamic_slice_sizes,
                            type($packed_offsets))
    `}` `)`
    `:` type($total_length)
    attr-dict-with-keyword
  }];

  let verifier = [{ return verifyAllocatorPackOp(*this); }];

  let extraClassDeclaration = [{
    struct Slice {
      int64_t lifetimeStart;
      int64_t lifetimeEnd;
      Value dynamicSize;
      Value packedOffset;

      bool operator==(const Slice &rhs) const {
        return lifetimeStart == rhs.lifetimeStart &&
               lifetimeEnd == rhs.lifetimeEnd;
      }
      bool operator!=(const Slice &rhs) const {
        return !(*this == rhs);
      }
      bool operator<(const Slice &rhs) const {
        return std::make_pair(lifetimeStart, lifetimeEnd) <
               std::make_pair(rhs.lifetimeStart, rhs.lifetimeEnd);
      }
      bool intersects(const Slice &rhs) const {
        return lifetimeEnd >= rhs.lifetimeStart &&
               rhs.lifetimeEnd >= lifetimeStart;
      }
    };

    /// Returns all of the slices to be packed.
    /// Order is ascending by lifetime interval (post-canonicalization).
    SmallVector<Slice> getSlices();
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// !hal.buffer / iree_hal_buffer_t
//===----------------------------------------------------------------------===//

def HAL_BufferAllocatorOp : HAL_PureOp<"buffer.allocator", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
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
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    DeclareOpInterfaceMethods<HAL_SizeAwareOp>,
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
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
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
    HAL_BufferType:$buffer,
    HAL_ElementType:$element_type,
    HAL_Shape:$shape
  );
  let results = (outs
    HAL_BufferView:$result
  );

  let assemblyFormat = [{
    $buffer `,`
    `element_type` `=` $element_type `,`
    `shape` `=` `[` $shape `]`
    attr-dict `:` type($buffer) `->` type($result)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "Value":$buffer,
      "int32_t":$elementType,
      "ValueRange":$shape
    )>,
    OpBuilder<(ins
      "Value":$buffer,
      "Value":$elementType,
      "ValueRange":$shape
    )>,
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
    $buffer_view `,`
    `indices` `=` `[` $indices `]` `,`
    `lengths` `=` `[` $lengths `]`
    attr-dict `:` type($result)
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
    HAL_BufferType:$result
  );

  let assemblyFormat = [{
    $buffer_view attr-dict `:` type($result)
  }];

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

  let assemblyFormat = [{
    $buffer_view attr-dict `:` type($result)
  }];

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
    $buffer_view `,`
    `indices` `=` `[` $indices `]`
    attr-dict `:` type($offset)
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
    $buffer_view `,`
    `indices` `=` `[` $indices `]` `,`
    `lengths` `=` `[` $lengths `]`
    attr-dict `:` type($offset) `,` type($length)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "Value":$bufferView,
      "ValueRange":$indices,
      "ValueRange":$lengths
    )>,
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

  let assemblyFormat = [{
    $buffer_view attr-dict `:` type($result)
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
    $buffer_view attr-dict `:` type($result)
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
    $buffer_view `,` $index attr-dict `:` type($result)
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
    $buffer_view attr-dict `:` type($result)
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
    attr-dict ($operands^ `:` type($operands))?
  }];
}

//===----------------------------------------------------------------------===//
// !hal.command_buffer / iree_hal_command_buffer_t
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
    I32:$pattern
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
    OpBuilder<(ins
      "StringRef":$name,
      "BufferConstraintsAttr":$bufferConstraints
    )>,
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
    $sym_name `=` $value attr-dict-with-keyword
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
    $sym_name `:` $tensor_type
    `=` $storage_buffer `[` $storage_range `]`
    (`->` $runtime_buffer^ `[` $runtime_range `]`)?
    attr-dict-with-keyword
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
    $sym_name `=` $value
    (`->` $runtime_buffer^ `[` $runtime_range `]`)?
    attr-dict-with-keyword
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

  let assemblyFormat = [{
    $constant `:` type($result) attr-dict-with-keyword
  }];

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
    $sym_name `=` $value attr-dict-with-keyword
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
    $constant `:` type($result) attr-dict-with-keyword
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
    $runtime_buffer `[` $runtime_range `]` `:` type($result)
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// !hal.descriptor_set / iree_hal_descriptor_set_layout_t
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
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
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
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
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
    %0 = hal.device.switch<%device : !hal.device> -> i32
      #hal.device.match.id<"vulkan-v1.?-*">(%c1a = %c1 : i32) {
        hal.return %c1a : i32
      },
      #hal.match.any<[#hal.device.match.id<"vmvx">, #hal.device.match.id<"vulkan-*">]>(%c2a = %c2 : i32) {
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

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "TypeRange":$resultTypes,
      "Value":$device,
      "ArrayRef<Attribute>":$conditions,
      "ArrayRef<SmallVector<Value, 4>>":$conditionArgs,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes
    )>,
  ];

  let extraClassDeclaration = [{
    /// Returns the index of the args() operand in the Operation operands list.
    unsigned mapArgOperandToOpOperand(unsigned i) { return i + 1; }
  }];


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

// TODO(benvanik): additional factory functions and submission ops.
// TODO(benvanik): %0 = hal.device.query %device, group, property : i32/etc

def HAL_DeviceMatchIDOp : HAL_PureOp<"device.match.id"> {
  let summary = [{returns true if the device ID matches the pattern}];
  let description = [{
    Pattern matches the device ID with the given wildcard pattern.
    This can be used to conditionally evaluate device-specific code when the
    device is not known at compile-time.

    ```mlir
    %is_match = hal.device.match.id<%device : !hal.device>
                            pattern("vulkan-*") : i1
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
    `<` $device `:` type($device) `>`
    `pattern` `(` $pattern `)`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HAL_DeviceMatchMemoryModelOp : HAL_PureOp<"device.match.memory_model"> {
  let summary = [{returns true if the device memory model matches the value}];
  let description = [{
    Compares the device's memory model against the specified model.
    This can be used to conditionally evaluate device-specific code when the
    device is not known at compile-time.

    ```mlir
    %is_match = hal.device.match.memory_model<%device : !hal.device>
                                        value("Unified") : i1
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
    `<` $device `:` type($device) `>`
    `value` `(` $model `)`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HAL_DeviceQueryOp : HAL_PureOp<"device.query"> {
  let summary = [{returns a runtime configuration parameter from the device}];
  let description = [{
    Queries a device configuration parameter with the given key.
    Returns a status indicating whether the key was recognized/available and if
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
    StrAttr:$key
  );
  let results = (outs
    I1:$ok,
    AnyType:$value
  );

  let assemblyFormat = [{
    `<` $device `:` type($device) `>`
    `key` `(` $key `)`
    `:` type($ok) `,` type($value)
    attr-dict-with-keyword
  }];
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
    StrAttr:$sym_name,
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
      build($_builder, $_state, sym_name, ordinal, interface,
            workgroup_size, workgroup_local_memory, 0);
    }]>,
    OpBuilder<(ins
      "::mlir::StringAttr":$sym_name,
      "::mlir::IntegerAttr":$ordinal,
      "::mlir::FlatSymbolRefAttr":$interface,
      "::mlir::ArrayAttr":$workgroup_size,
      "::mlir::IntegerAttr":$workgroup_local_memory
    ), [{
      build($_builder, $_state, sym_name, ordinal, interface,
            workgroup_size, workgroup_local_memory, 0);
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

def HAL_ExecutableVariantEndOp : HAL_Op<"executable.variant_end", [
    HasParent<"IREE::HAL::ExecutableVariantOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the executable variant op}];
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
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    StrAttr:$format,
    HAL_ExecutableDataAttr:$data,
    OptionalAttr<StrAttr>:$mime_type
    // TODO(benvanik): add compatibility and versioning attributes.
  );

  let regions = (region SizedRegion<1>:$body);

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
      $_state.addAttribute("executable", $_builder.getSymbolRefAttr(executable));
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
    StrAttr:$sym_name,
    OptionalAttr<IndexAttr>:$push_constants
  );

  let regions = (region SizedRegion<1>:$body);

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
    IndexAttr:$set,
    IndexAttr:$binding,
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

  let assemblyFormat = [{
    `[` $dimension `]` attr-dict `:` type($result)
  }];
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

  let assemblyFormat = [{
    `[` $dimension `]` attr-dict `:` type($result)
  }];
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

//===----------------------------------------------------------------------===//
// !hal.executable_layout / iree_hal_executable_layout_t
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
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
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
    IREE_Status:$status,
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
    IREE_Status:$status
  );

  let assemblyFormat = [{
    `<` $semaphore `:` type($semaphore) `>`
    `status` `(` $status `)`
    attr-dict-with-keyword
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
    `<` $semaphore `:` type($semaphore) `>`
    `until` `(` $min_value `)`
    `:` type($status)
    attr-dict-with-keyword
  }];
}

#endif  // IREE_DIALECT_HAL_OPS
