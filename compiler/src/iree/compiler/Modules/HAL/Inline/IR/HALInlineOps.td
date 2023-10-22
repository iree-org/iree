// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_HAL_INLINE_OPS
#define IREE_DIALECT_MODULES_HAL_INLINE_OPS

include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "iree/compiler/Modules/HAL/Inline/IR/HALInlineBase.td"
include "iree/compiler/Dialect/Util/IR/UtilAttrs.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class HALInline_PureOp<string mnemonic, list<Trait> traits = []> :
    HALInline_Op<mnemonic, !listconcat(traits, [Pure])>;

//===----------------------------------------------------------------------===//
// !hal.buffer / iree_hal_buffer_t
//===----------------------------------------------------------------------===//

def OpGroupBufferOps : OpDocGroup {
  let summary = "Buffer ops";
  let description = "Ops for `!hal.buffer` / `iree_hal_buffer_t`.";
}

let opDocGroup = OpGroupBufferOps in {

def HALInline_BufferAllocateOp : HALInline_Op<"buffer.allocate", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  DeclareOpInterfaceMethods<Util_SizeAwareOp>,
]> {
  let summary = [{empty buffer allocation operation}];
  let description = [{
    Allocates a buffer of the given size.
    The size of the buffer returned may be larger than the requested size if the
    allocator has specific alignment requirements or minimum allocation sizes.
  }];

  let arguments = (ins
    HAL_DeviceSize:$minimum_alignment,
    HAL_DeviceSize:$allocation_size
  );
  let results = (outs
    HAL_Buffer:$result,
    Util_BufferType:$storage
  );

  let assemblyFormat = [{
    `alignment` `(` $minimum_alignment `)`
    `:` custom<SizeAwareType>(type($result), $allocation_size) `in` type($storage)
    attr-dict-with-keyword
  }];
}

def HALInline_BufferAllocateInitializedOp : HALInline_Op<"buffer.allocate.initialized", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  DeclareOpInterfaceMethods<Util_SizeAwareOp>,
]> {
  let summary = [{buffer allocation with cloning}];
  let description = [{
    Allocates a buffer with a copy of the provided contents.
  }];

  let arguments = (ins
    HAL_DeviceSize:$minimum_alignment,
    Util_BufferType:$source,
    HAL_DeviceSize:$offset,
    HAL_DeviceSize:$length
  );
  let results = (outs
    HAL_Buffer:$result,
    Util_BufferType:$storage
  );

  let assemblyFormat = [{
    `source` `(` $source `:` type($source) `)` `` `[` $offset `,` $length `]`
    `alignment` `(` $minimum_alignment `)`
    `:` custom<SizeAwareType>(type($result), ref($length)) `in` type($storage)
    attr-dict-with-keyword
  }];
}

def HALInline_BufferWrapOp : HALInline_Op<"buffer.wrap", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  DeclareOpInterfaceMethods<Util_SizeAwareOp>,
]> {
  let summary = [{host buffer wrapping operation}];
  let description = [{
    Tries wrapping a !hal.buffer around host memory backed by the given byte
    buffer.
  }];

  let arguments = (ins
    Util_BufferType:$source,
    HAL_DeviceSize:$offset,
    HAL_DeviceSize:$length
  );
  let results = (outs
    HAL_Buffer:$result
  );

  // TODO(benvanik): change type/usage to ref params.
  let assemblyFormat = [{
    `source` `(` $source `:` type($source) `)` `` `[` $offset `,` $length `]`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def HALInline_BufferSubspanOp : HALInline_PureOp<"buffer.subspan", [
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

  // TODO(benvanik): folder to elide when offset is 0 and length is all.
}

def HALInline_BufferLengthOp : HALInline_PureOp<"buffer.length", [
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

  let builders = [
    OpBuilder<(ins "Value":$buffer),
    [{
      build($_builder, $_state, $_builder.getIndexType(), buffer);
    }]>,
  ];

  let hasFolder = 1;
}

def HALInline_BufferStorageOp : HALInline_PureOp<"buffer.storage", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{buffer backing storage accessor}];
  let description = [{
    Returns the host backing storage of the HAL buffer as a subspan limited to
    to the buffer's logical range (meaning that byte 0 of the returned buffer is
    byte 0 of the HAL buffer).
  }];

  let arguments = (ins
    HAL_BufferType:$buffer
  );
  let results = (outs
    Util_BufferType:$storage
  );

  let assemblyFormat = [{
    `<` $buffer `:` type($buffer) `>`
    `:` type($storage)
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins "Value":$buffer),
    [{
      build($_builder, $_state, $_builder.getType<IREE::Util::BufferType>(), buffer);
    }]>,
  ];

  let hasFolder = 1;
}

} // OpGroupBufferOps

//===----------------------------------------------------------------------===//
// !hal.buffer_view / iree_hal_buffer_view_t
//===----------------------------------------------------------------------===//

def OpGroupBufferViewOps : OpDocGroup {
  let summary = "Buffer view ops";
  let description = "Ops for `!hal.buffer_view` / `iree_hal_buffer_view_t`.";
}

let opDocGroup = OpGroupBufferViewOps in {

def HALInline_BufferViewCreateOp : HALInline_PureOp<"buffer_view.create", [
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

def HALInline_BufferViewAssertOp : HALInline_Op<"buffer_view.assert"> {
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

def HALInline_BufferViewBufferOp : HALInline_PureOp<"buffer_view.buffer", [
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

def HALInline_BufferViewElementTypeOp : HALInline_PureOp<"buffer_view.element_type"> {
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

def HALInline_BufferViewEncodingTypeOp : HALInline_PureOp<"buffer_view.encoding_type"> {
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

def HALInline_BufferViewRankOp : HALInline_PureOp<"buffer_view.rank"> {
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

def HALInline_BufferViewDimOp : HALInline_PureOp<"buffer_view.dim"> {
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

def HALInline_BufferViewTraceOp : HALInline_Op<"buffer_view.trace", []> {
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
    $key `=`
    $operands `:` type($operands)
    attr-dict-with-keyword
  }];
}

} // OpGroupBufferViewOps

//===----------------------------------------------------------------------===//
// !hal.device / iree_hal_device_t
//===----------------------------------------------------------------------===//

def OpGroupDeviceOps : OpDocGroup {
  let summary = "Device ops";
  let description = "Ops for `!hal.device` / `iree_hal_device_t`.";
}

let opDocGroup = OpGroupDeviceOps in {

def HALInline_DeviceQueryOp :
    HALInline_PureOp<"device.query"> {
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

    Well-known keys: (none yet)
  }];

  let arguments = (ins
    StrAttr:$category,
    StrAttr:$key,
    OptionalAttr<AnyAttr>:$default_value
  );
  let results = (outs
    I1:$ok,
    AnyType:$value
  );

  let assemblyFormat = [{
    `key` `(` $category `:` `` `:` $key `)`
    `:` type($ok) `,` type($value)
    (`=` $default_value^)?
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;
}

} // OpGroupDeviceOps

#endif  // IREE_DIALECT_MODULES_HAL_INLINE_OPS
