// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_IO_STREAM_BASE
#define IREE_DIALECT_MODULES_IO_STREAM_BASE

include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"

//===----------------------------------------------------------------------===//
// I/O stream operations
//===----------------------------------------------------------------------===//

def IOStream_Dialect : Dialect {
  let name = "io_stream";
  let cppNamespace = "::mlir::iree_compiler::IREE::IO::Stream";

  let summary = [{
    I/O stream operations.
  }];
  let description = [{
    Provides read/write operations on file-like stream resources, roughly
    modeling fread/fwrite in C. Each stream handle statefully models a position
    within an underlying file-like resource with byte access.

    Some basic string-like operations are supported but most access is assumed
    to be binary and additional string handling of unicode/etc needs to be
    handled by users prior to/after performing I/O on the byte stream.

    For debugging and testing purposes the special stream handles of `stdin`,
    `stdout`, and `stderr` can be used for interactive consoles or pipe-based
    multi-processing. These may not always be compiled into a runtime as the
    concept of a console or pipes does not exist on bare-metal systems, usually.
    There are also security and privacy implications of allowing console I/O
    that discourages their use outside of test environments. The behavior of
    multiple IREE contexts interacting with the same console streams is
    undefined and only one context should ever claim exclusive control over
    them.

    See `io_stream.imports.mlir` for the full list of exported functions.
  }];
}

//===----------------------------------------------------------------------===//
// IOStream enums
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IOStream types
//===----------------------------------------------------------------------===//

def IOStream_HandleType : TypeDef<IOStream_Dialect, "Handle", [
  Util_GlobalType,
]> {
  let mnemonic = "handle";

  let summary = [{a handle to a stream object}];
  let description = [{
    A stateful stream handle keeping track of the current position within an
    underlying file-like resource. Streams are not thread-safe but are assumed
    to be detached from any statefulness of underlying resources and multiple
    streams can be opened to the same resource with independent positions (even
    if behavior is undefined).
  }];

  let parameters = (ins);
}

//===----------------------------------------------------------------------===//
// IOStream op traits
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Base IOStream op classes
//===----------------------------------------------------------------------===//

class IOStream_Op<string mnemonic, list<Trait> traits = []> :
    Op<IOStream_Dialect, mnemonic, !listconcat(traits, [])> {}

#endif  // IREE_DIALECT_MODULES_IO_STREAM_BASE
