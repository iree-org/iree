// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_ENCODING_BASE
#define IREE_DIALECT_ENCODING_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def IREEEncoding_Dialect : Dialect {
  let name = "iree_encoding";
  let cppNamespace = "::mlir::iree_compiler::IREE::Encoding";
  let summary = [{
    A dialect designed for tensor encoding attributes and ops.
  }];
  let description = [{
    The encoding dialect defines IREE tensor encoding attributes and related
    ops. These encoding hints enable optimization of compute-heavy operations
    like matmul. E.g., IREE implements data-tiling using the encoding dialect,
    by introducing encodings on matmul ops and fuse encoding ops with producers.

    There are two core operations in this dialect, and they are very simple:
    - SetEncodingOp: Adds an encoding attribute to a tensor value.
    - UnsetEncodingOp: Removes the encoding attribute from a tensor value.

    Encoding attributes fall into two categories: encoding type attribute and
    encoding resolver.

    An encoding type attribute is expected to be attached on tensor types. They
    typically go through three states:

    +------------------+    +---------------------+    +--------------------+
    | Verbose Encoding | -> | Serialized Encoding | -> | Physical Ops/Types |
    +------------------+    +---------------------+    +--------------------+

    Encoding resolvers are responsible for transforming the encodings between
    states. Each backend can define its own resolver to interpret encodings
    based on target-specific details.

    To support encoding transformations better, IREE defines three encoding
    interfaces:
    - LayoutResolverAttr: Converts verbose encodings into serialized ones, based
      on target-specific information.
    - SerializableAttr: Represents serialized encoding formats with enough
      detail for host-side compilation.
    - LayoutMaterializerAttr: Lowers encodings into physical operations and
      types.

    An encoding resolver must implement `LayoutResolverAttr`, as it needs to
    convert verbose encodings into serialized encodings with target specifics.

    Sometimes, an encoding resolver also implements `SerializableAttr` because
    the layout can be customized format, e.g., packing, etc. Only the resolver
    itself knows how to interpret the layout.

    An encoding resolver must implement `LayoutMaterializerAttr`, because only
    the resolver itself knows what the final operations and types are.

    Encoding resolvers are conventionally named with a `_resolver` suffix.

    The transition from verbose to serialized encoding happens during the
    SpecializeEncoding pass within the stream dialect. At stream level, the pass
    has access to all necessary context (e.g., affinity, etc.) to retrieve
    encoding resolvers. It bridges the gap between host and device compilation,
    as it allows IREE to specialize storage size based on the serialized
    encoding result.
  }];
  let extraClassDeclaration = [{
    void registerAttributes();
  }];
  let useDefaultAttributePrinterParser = 1;
}

#endif // IREE_DIALECT_ENCODING_BASE
