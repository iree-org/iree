// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_IO_STREAM_OPS
#define IREE_DIALECT_MODULES_IO_STREAM_OPS

include "iree/compiler/Dialect/Util/IR/UtilAttrs.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilTypes.td"
include "iree/compiler/Modules/IO/Stream/IR/IOStreamBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class IOStream_PureOp<string mnemonic, list<Trait> traits = []> :
    IOStream_Op<mnemonic, !listconcat(traits, [Pure])>;

//===----------------------------------------------------------------------===//
// Console I/O handles
//===----------------------------------------------------------------------===//

def OpGroupConsoleOps : OpDocGroup {
  let summary = "Console I/O handle ops";
  let description = "Console standard input/output/error handle access.";
}

let opDocGroup = OpGroupConsoleOps in {

def IOStream_ConsoleStdinOp : IOStream_PureOp<"console.stdin", [
  OpAsmOpInterface,
]> {
  let summary = [{returns the console stdin handle (if available)}];

  let arguments = (ins);
  let results = (outs
    IOStream_HandleType:$result
  );

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
      setNameFn(getResult(), "stdin");
    }
  }];
}

def IOStream_ConsoleStdoutOp : IOStream_PureOp<"console.stdout", [
  OpAsmOpInterface,
]> {
  let summary = [{returns the console stdout handle (if available)}];

  let arguments = (ins);
  let results = (outs
    IOStream_HandleType:$result
  );

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
      setNameFn(getResult(), "stdout");
    }
  }];
}

def IOStream_ConsoleStderrOp : IOStream_PureOp<"console.stderr", [
  OpAsmOpInterface,
]> {
  let summary = [{returns the console stderr handle (if available)}];

  let arguments = (ins);
  let results = (outs
    IOStream_HandleType:$result
  );

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
      setNameFn(getResult(), "stderr");
    }
  }];
}

} // OpGroupConsoleOps

//===----------------------------------------------------------------------===//
// Console I/O handles
//===----------------------------------------------------------------------===//

def OpGroupStreamOps : OpDocGroup {
  let summary = "Stream I/O ops";
  let description = "Stream I/O operations.";
}

let opDocGroup = OpGroupStreamOps in {

def IOStream_OffsetOp : IOStream_Op<"offset", [
  OpAsmOpInterface,
]> {
  let summary = [{returns the current byte offset of the stream}];
  let description = [{
    The offset in bytes of the stream read/write head in the underlying
    resource. Depending on the stream type the offset may be physical or virtual
    or may even always remain as zero.
  }];

  let arguments = (ins
    IOStream_HandleType:$handle
  );
  let results = (outs
    I64:$result
  );

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
    `(` type(operands) `)` `->` type($result)
  }];

  let extraClassDeclaration = [{
    void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
      setNameFn(getResult(), "offset");
    }
  }];
}

def IOStream_LengthOp : IOStream_Op<"length", [
  OpAsmOpInterface,
]> {
  let summary = [{returns the current length of the stream or -1}];
  let description = [{
    The length in bytes of the underlying resource defining the valid range
    of bytes. Returns -1 if the underlying resource has no length concept (an
    infinite streaming pipe, etc).
  }];

  let arguments = (ins
    IOStream_HandleType:$handle
  );
  let results = (outs
    I64:$result
  );

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
    `(` type(operands) `)` `->` type($result)
  }];

  let extraClassDeclaration = [{
    void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
      setNameFn(getResult(), "length");
    }
  }];
}

def IOStream_ReadByteOp : IOStream_Op<"read.byte", [
  OpAsmOpInterface,
]> {
  let summary = [{reads a single byte from the stream}];
  let description = [{
    Reads and returns a single byte from the stream and a booleaning indicating
    whether the end-of-stream was reached. If the end-of-stream was reached the
    byte returned will have undefined contents.
  }];

  let arguments = (ins
    IOStream_HandleType:$handle
  );
  let results = (outs
    I8:$result,
    I1:$eos
  );

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
    `(` type(operands) `)` `->` `(` type(results) `)`
  }];

  let extraClassDeclaration = [{
    void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
      setNameFn(getResult(), "byte");
      setNameFn(getEos(), "is_eos");
    }
  }];
}

def IOStream_ReadBytesOp : IOStream_Op<"read.bytes", [
  OpAsmOpInterface,
  SameVariadicOperandSize,
]> {
  let summary = [{reads a span of bytes into the target buffer}];
  let description = [{
    Reads up to the specified length into the buffer at the given offset.
    Returns the total number of bytes read which may be less than the requested
    length if the stream reaches its end prior to completing the read. When an
    end-of-stream is hit all bytes until then are populated and the remaining
    range between the read bytes and requested length will be unmodified.
  }];

  let arguments = (ins
    IOStream_HandleType:$handle,
    Util_BufferType:$buffer,
    Optional<Util_Offset>:$offset,
    Optional<Util_Offset>:$length
  );
  let results = (outs
    Util_Offset:$result
  );

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
    `(` type(operands) `)` `->` type($result)
  }];

  let extraClassDeclaration = [{
    void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
      setNameFn(getResult(), "read_length");
    }
  }];
}

def IOStream_ReadDelimiterOp : IOStream_Op<"read.delimiter", [
  OpAsmOpInterface,
]> {
  let summary = [{reads until the given delimiter byte is reached}];
  let description = [{
    Returns a buffer containing all bytes between the current stream offset and
    the first occurrence of the given delimiter byte (exclusive) or the
    end-of-stream. Returns an empty buffer if the next byte in the stream is
    the delimiter. Returns null if the stream is already at end-of-stream and
    no more bytes are available. The result does not contain the delimiter.
  }];

  let arguments = (ins
    IOStream_HandleType:$handle,
    I8:$delimiter
  );
  let results = (outs
    Util_BufferType:$result
  );

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
    `(` type(operands) `)` `->` type($result)
  }];

  let extraClassDeclaration = [{
    void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
      setNameFn(getResult(), "read_bytes");
    }
  }];
}

def IOStream_ReadLineOp : IOStream_Op<"read.line", [
  OpAsmOpInterface,
]> {
  let summary = [{reads until the next newline character is reached}];
  let description = [{
    Returns a buffer containing all bytes between the current stream offset and
    the first occurrence of a newline character (`0x0A`/`10`/`\n`) or the
    end-of-stream. Returns an empty buffer if the next byte in the stream is
    the newline character. Returns null if the stream is already at
    end-of-stream and no more bytes are available. The result does not contain
    the newline character.
  }];

  let arguments = (ins
    IOStream_HandleType:$handle
  );
  let results = (outs
    Util_BufferType:$result
  );

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
    `(` type(operands) `)` `->` type($result)
  }];

  let extraClassDeclaration = [{
    void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
      setNameFn(getResult(), "read_line");
    }
  }];
}

def IOStream_WriteByteOp : IOStream_Op<"write.byte"> {
  let summary = [{writes a single byte to the stream}];
  let description = [{
    Writes the given byte to the stream and advances the offset.
  }];

  let arguments = (ins
    IOStream_HandleType:$handle,
    I8:$byte
  );

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
    `(` type(operands) `)` `->` `(` `` `)`
  }];
}

def IOStream_WriteNewlineOp : IOStream_Op<"write.newline"> {
  let summary = [{writes a newline character to the stream}];
  let description = [{
    Writes a newline character (`0x0A`/`10`/`\n`) to the stream.
  }];

  let arguments = (ins
    IOStream_HandleType:$handle
  );

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
    `(` type(operands) `)` `->` `(` `` `)`
  }];
}

def IOStream_WriteBytesOp : IOStream_Op<"write.bytes", [
  SameVariadicOperandSize,
]> {
  let summary = [{writes a span of bytes to the stream}];
  let description = [{
    Writes the range of bytes in the source buffer starting at offset for length
    to the stream.
  }];

  let arguments = (ins
    IOStream_HandleType:$handle,
    Util_BufferType:$buffer,
    Optional<Util_Offset>:$offset,
    Optional<Util_Offset>:$length
  );

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
    `(` type(operands) `)` `->` `(` `` `)`
  }];
}

def IOStream_WriteLineOp : IOStream_Op<"write.line"> {
  let summary = [{writes the given string buffer and a newline to the stream}];
  let description = [{
    Writes the entire contents of the provided buffer to the stream and adds a
    single newline character to the end.
  }];

  let arguments = (ins
    IOStream_HandleType:$handle,
    Util_BufferType:$buffer
  );

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
    `(` type(operands) `)` `->` `(` `` `)`
  }];
}

} // OpGroupStreamOps

#endif  // IREE_DIALECT_MODULES_IO_STREAM_OPS
