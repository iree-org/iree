// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_STREAM_OPS
#define IREE_DIALECT_STREAM_OPS

include "iree/compiler/Dialect/Stream/IR/StreamBase.td"
include "iree/compiler/Dialect/Stream/IR/StreamInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class Stream_PureOp<string mnemonic, list<Trait> traits = []> :
    Stream_Op<mnemonic, !listconcat(traits, [Pure])>;

//===----------------------------------------------------------------------===//
// Generic resource ops
//===----------------------------------------------------------------------===//

def Stream_ResourceAllocOp : Stream_PureOp<"resource.alloc", [
  DeclareOpInterfaceMethods<Stream_AffinityOp, [
    "getAffinity",
    "setAffinity",
  ]>,
  Util_SizeAwareOp,
  MemoryEffects<[MemAlloc]>,
]> {
  let summary = [{allocates a persistent value with undefined contents}];
  let description = [{
    Allocates a persistent value (one that is long-lived and possibly external
    to the program) with undefined contents. Consumers of the allocated
    result must assume nothing of the contents and use `discard` access.

    Uninitialized allocations will have undefined contents and must only be used
    when all bytes are discarded prior to any reads. Runtimes decide what
    "undefined contents" means and here it only indicates that execution will be
    correct even if the memory starts with non-zero values.

    If multiple values are allocated from the same operation it implies that
    they have matching lifetimes. When lowering to execution environments the
    separate allocations may be fused into one or more slab allocations in order
    to reduce overheads. How many allocations can be fused is based on the size
    of the individual resources and the target constraints (how large any single
    buffer may be, etc). At the stream dialect level treat a multi-result alloc
    as a way to indicate similar lifetimes.
  }];

  let arguments = (ins
    Variadic<Stream_Size>:$storage_sizes,
    UnitAttr:$uninitialized,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<Stream_AnyResource>:$results
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    (`uninitialized` $uninitialized^)?
    attr-dict `:` custom<SizeAwareTypeList>(type($results), $storage_sizes)
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getStorageSizes()[idx]; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_ResourceAllocaOp : Stream_PureOp<"resource.alloca", [
  DeclareOpInterfaceMethods<Stream_AffinityOp, [
    "getAffinity",
    "setAffinity",
  ]>,
  Stream_TimelineOp,
  Util_SizeAwareOp,
  MemoryEffects<[MemAlloc]>,
]> {
  let summary = [{allocates a transient value with undefined contents}];
  let description = [{
    Allocates a transient value (one that is short-lived and local to the
    current computation) with undefined contents. Consumers of the allocated
    result must assume nothing of the contents and use `discard` access.

    The resource returned is not valid for use until the timepoint is reached;
    execution using this resource must await on the timepoint.
  }];

  let arguments = (ins
    Stream_Size:$storage_size,
    Optional<Stream_Timepoint>:$await_timepoint,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    AnyTypeOf<[
      Stream_StagingResource,
      Stream_TransientResource,
    ]>:$result,
    Stream_Timepoint:$result_timepoint
  );

  let assemblyFormat = [{
    `uninitialized`
    (`on` `(` $affinity^ `)`)?
    (`await` `(` $await_timepoint^ `)` `=` `` `>`):(`:`)?
    attr-dict
    type($result) `{` $storage_size `}`
    `=` `` `>`
    type($result_timepoint)
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getStorageSize(); }
    SmallVector<Value> getAwaitTimepoints() {
      if (getAwaitTimepoint()) return {getAwaitTimepoint()}; else return {};
    }
  }];

  let hasCanonicalizer = 1;
}

def Stream_ResourceDeallocaOp : Stream_Op<"resource.dealloca", [
  DeclareOpInterfaceMethods<Stream_AffinityOp, [
    "getAffinity",
    "setAffinity",
  ]>,
  Stream_TimelineOp,
  Util_SizeAwareOp,
  MemoryEffects<[MemFree]>,
]> {
  let summary = [{frees a transient value when available}];
  let description = [{
    Deallocates a transient value (one that is short-lived and local to the
    current computation) previously allocated using `stream.resource.alloca`.

    The resource is considered live and valid until the provided timepoint is
    reached and the memory is only made available for future requests after
    the result timepoint is reached.
  }];

  let arguments = (ins
    AnyTypeOf<[
      Stream_StagingResource,
      Stream_TransientResource,
    ]>:$operand,
    Stream_Size:$operand_size,
    Optional<Stream_Timepoint>:$await_timepoint,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_Timepoint:$result_timepoint
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    (`await` `(` $await_timepoint^ `)` `=` `` `>`)?
    $operand `:` type($operand) `{` $operand_size `}`
    `=` `` `>` type($result_timepoint)
    attr-dict
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getOperandSize(); }
    Value getResultSize(unsigned idx) { return {}; }
    SmallVector<Value> getAwaitTimepoints() {
      if (getAwaitTimepoint()) return {getAwaitTimepoint()}; else return {};
    }
  }];

  let hasCanonicalizer = 1;
}

def Stream_ResourceSizeOp : Stream_PureOp<"resource.size", [
  Stream_AffinityOp,
  Util_SizeAwareOp,
]> {
  let summary = [{returns the size of the resource storage in bytes}];
  let description = [{
    Returns a possibly runtime-dynamic byte size of the resource backing
    storage. This may differ from the logical storage size of a value based on
    the alignment requirements of the target as well as encoding of higher level
    values such as sparse tensor formats.
  }];

  let arguments = (ins
    Stream_AnyResource:$operand,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_Size:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $operand
    attr-dict `:` type($operand)
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getResult(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Stream_ResourceMapOp : Stream_Op<"resource.map", [
  Stream_AffinityOp,
  Util_SizeAwareOp,
  MemoryEffects<[MemAlloc]>,
]> {
  let summary = [{maps read-only memory into a staging resource}];
  let description = [{
    Synchronously maps a host heap buffer into a stream-accessible staging
    resource. Will never fail but may induce a copy if required and as such the
    mapped resource is not coherent with the original source buffer: changing
    the source buffer after mapping has undefined behavior.
  }];

  let arguments = (ins
    Util_BufferType:$source,
    Stream_Offset:$source_offset,
    Stream_Size:$result_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_StagingResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `[` $source_offset `]` `:`
    type($source)
    `->`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_ResourceTryMapOp : Stream_PureOp<"resource.try_map", [
  Stream_AffinityOp,
  Util_SizeAwareOp,
  MemoryEffects<[MemAlloc]>,
]> {
  let summary = [{maps read-only memory into a resource}];
  let description = [{
    Synchronously maps a host heap buffer into a stream-accessible resource
    with constant lifetime. If the given source cannot be mapped into a constant
    a failure is returned and the resulting resource value is null. As with
    `stream.resource.map` the resulting resource is not coherent with the source
    and changes will not be reflected.
  }];

  let arguments = (ins
    Util_BufferType:$source,
    Stream_Offset:$source_offset,
    Stream_Size:$result_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    I1:$did_map,
    Stream_ConstantResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `[` $source_offset `]` `:`
    type($source)
    `->`
    type($did_map) `,` type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_ResourceLoadOp : Stream_Op<"resource.load", [
  Util_SizeAwareOp,
]> {
  let summary = [{loads a value from a staging resource}];
  let description = [{
    Returns the element(s) at the given offset in the staging resource.
    The operation will complete synchronously against the resource though it may
    introduce a yield point if the staging resource needs to be transferred.
  }];

  let arguments = (ins
    Stream_StagingResource:$source,
    Stream_Size:$source_size,
    Stream_Offset:$source_offset
  );
  let results = (outs
    AnyTypeOf<[Stream_PrimitiveType, AnyVector]>:$result
  );

  let assemblyFormat = [{
    $source `[` $source_offset `]` `:`
    type($source) `` `{` $source_size `}`
    `->`
    type($result)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_ResourceStoreOp : Stream_Op<"resource.store", [
  Util_SizeAwareOp,
  MemoryEffects<[MemWrite]>,
]> {
  let summary = [{stores a value into a staging resource}];
  let description = [{
    The operation will complete synchronously against the resource though it may
    introduce a yield point if the staging resource needs to be acquired.
  }];

  let arguments = (ins
    Stream_StagingResource:$target,
    Stream_Size:$target_size,
    Stream_Offset:$target_offset,
    AnyTypeOf<[Stream_PrimitiveType, AnyVector]>:$value
  );

  let assemblyFormat = [{
    $value `,`
    $target `[` $target_offset `]` `:`
    type($value)
    `->`
    type($target) `{` $target_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_ResourcePackOp : Stream_PureOp<"resource.pack", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  Stream_AffinityOp,
]> {
  let summary = [{packs variable-sized slices into a single slab}];
  let description = [{
    Performs a greedy packing of one or more sized slices with specified
    lifetimes and returns their relative offsets in an aliased linear space.

    Slices are `[start, end] = %slice_byte_size`, where the start and end values
    define an inclusive lifetime range and the size is the total number of bytes
    required to be live for that range.

    ```mlir
    // Computes the total length required for the packed values and the offsets
    // of the 3 slices requested relative to the base of the packed memory:
    %total_length, %offset_0, %offset_1, %offset_2 =
        stream.resource.pack
            // Each slice gets one result offset:
            slices({
              // 3 slices where A and B overlap and will get unique offsets
              // while B and C do not overlap and are allowed to alias.
              [0, 10] = %size_0,  // A => %offset_0
              [3,  8] = %size_1,  // B => %offset_1
              [9, 10] = %size_2,  // C => %offset_2
              ...
            }) : index
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
    directly used as buffer offsets may need additional offsetting. This can
    either be applied via the optional `offset` operand or slicing of the
    underlying allocation buffer.
  }];

  let arguments = (ins
    Optional<Stream_Offset>:$offset,
    Stream_IndexArrayAttr:$lifetime_intervals,
    Variadic<Stream_Size>:$dynamic_slice_sizes,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_Size:$total_length,
    Variadic<Stream_Offset>:$packed_offsets
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    (`offset` `(` $offset^ `)`)?
    `slices` `(` `{`
    custom<PackSliceRanges>($lifetime_intervals,
                            $dynamic_slice_sizes,
                            type($packed_offsets))
    `}` `)`
    `:` type($total_length)
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;

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

def Stream_ResourceConstantsOp : Stream_PureOp<"resource.constants", [
  SameVariadicResultSize,
  Stream_AffinityOp,
  Stream_TimelineOp,
  Util_SizeAwareOp,
]> {
  let summary = [{asynchronously uploads or maps constant values}];
  let description = [{
    Represents an upload of constant resources that may be packed, suballocated,
    and mapped depending on the final lowering target.

    In runtime environments where memory is shared between host and device this
    turns into a mapping operation that avoids additional memory allocation and
    copies. When memory cannot be shared an asynchronous stream will be created
    to allocate and copy all of the constant values.

    Though this op returns a unique resource for each constant value it's
    expected that almost all end up aliasing into the same storage. The exact
    packing and number of storage resources that are needed are not known until
    lowering to a particular backend, though, so they are separate here for
    proper usage tracking.

    Both constant and variable resources can be produced; a constant is
    immutable while a variable will be treated as a constant-value initializer
    for a mutable resource. By modeling these together it's not required that
    variable initializers first be allocated, copied to the target, and then
    copied into the variable storage if the target is capable of doing a direct
    upload or mapping.
  }];

  let arguments = (ins
    TypedArrayAttrBase<ElementsAttr, "constant value array attribute">:$values,
    Variadic<Stream_Size>:$result_sizes,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<AnyTypeOf<[
      Stream_ConstantResource,
      Stream_VariableResource,
    ]>>:$results,
    Stream_Timepoint:$result_timepoint
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    attr-dict `:`
    custom<ConstantValueList>(type($results),
                              $result_sizes,
                              $values)
    `\n` ` ` ` ` `=` `` `>` type($result_timepoint)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSizes()[idx]; }
    SmallVector<Value> getAwaitTimepoints() { return {}; }
  }];
}

def Stream_ResourceSubviewOp : Stream_PureOp<"resource.subview", [
  AllTypesMatch<["source", "result"]>,
  DeclareOpInterfaceMethods<ViewLikeOpInterface>,
  DeclareOpInterfaceMethods<Stream_StreamableOp, [
    "isMetadata",
  ]>,
  Util_SizeAwareOp,
  Util_SubrangeOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{slices out a cloned subview of a value}];
  let description = [{
    Aliases a byte subrange of a resource.
  }];

  let arguments = (ins
    Stream_AnyResource:$source,
    Stream_Size:$source_size,
    Stream_Offset:$source_offset,
    Stream_Size:$result_size
  );
  let results = (outs
    Stream_AnyResource:$result
  );

  let assemblyFormat = [{
    $source `[` $source_offset `]` `:`
    type($source) `` `{` $source_size `}` `->`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return getResultSize(); }

    Value getSubrangeResource() { return getSource(); }
    Value getSubrangeResourceSize() { return getSourceSize(); }
    Value getSubrangeOffset() { return getSourceOffset(); }
    Value getSubrangeLength() { return getResultSize(); }
    Value getSubrangeResult() { return getResult(); }

    // Walks up the use-def chain to find a subview op that feeds into |value|.
    // Stops at any timeline op that may indicate a change in resource contents.
    static IREE::Stream::ResourceSubviewOp findSubviewOp(Value value);
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Pseudo ops for conversion support
//===----------------------------------------------------------------------===//

def Stream_TensorImportOp : Stream_PureOp<"tensor.import", [
  Stream_AffinityOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{conversion placeholder for other->stream type conversion}];
  let description = [{
    Defines a conversion from a higher-level dialect type such as `tensor` that
    is resolved during lowering into the stream dialect. This can be used to
    interoperate between levels of the stack that require specifying stream
    types and those that prior to lowering do not handle them.
  }];

  let arguments = (ins
    AnyType:$source,
    TypeAttr:$result_encoding,
    Stream_ShapeDynamicDims:$result_encoding_dims,
    Stream_Size:$result_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_ExternalResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `:`
    type($source)
    `->`
    $result_encoding (`` `{` $result_encoding_dims^ `}`)?
    `in`
    type($result) `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultEncodingDims(); }
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Stream_TensorExportOp : Stream_PureOp<"tensor.export", [
  Stream_AffinityOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{conversion placeholder for stream->other type conversion}];
  let description = [{
    Defines a conversion to a higher-level dialect type such as `tensor` that
    is resolved during lowering into the stream dialect. This can be used to
    interoperate between levels of the stack that require specifying stream
    types and those that prior to lowering do not handle them.
  }];

  let arguments = (ins
    Stream_ExternalResource:$source,
    TypeAttr:$source_encoding,
    Stream_ShapeDynamicDims:$source_encoding_dims,
    Stream_Size:$source_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `:`
    $source_encoding (`` `{` $source_encoding_dims^ `}`)?
    `in`
    type($source) `` `{` $source_size `}`
    `->`
    type($result)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getSourceEncodingDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return ValueRange{}; }
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// High-level tensor ops
//===----------------------------------------------------------------------===//

def Stream_TensorSizeOfOp : Stream_PureOp<"tensor.sizeof", [
  Stream_AffinityOp,
  Stream_TensorPhaseOp,
]> {
  let summary = [{calculates the storage size of a given high-level type}];
  let description = [{
    Target-dependent storage size calculation using a high-level annotated type.
    While within the stream dialect the storage size of a value is left as a
    placeholder using this op. The requisite target-specific parameters for
    expanding the size calculation are only available after affinities have been
    assigned.
  }];

  let arguments = (ins
    TypeAttr:$encoding,
    Stream_ShapeDynamicDims:$encoding_dims,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_Size:$storage_size
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $encoding (`{` $encoding_dims^ `}`)?
    attr-dict `:` type($storage_size)
  }];

  let hasVerifier = 1;
}

def Stream_TensorEmptyOp : Stream_PureOp<"tensor.empty", [
  Stream_AffinityOp,
  Stream_StreamableOp,
  Stream_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{defines an empty tensor value}];
  let description = [{
    Returns a typed resource initialized with no contents. This still carries
    shape metadata and may encode to a non-empty resource such as in cases
    where the empty representation still has data (e.g. sparse tensors).
    Subsequent writes must populate any ranges of the tensor that are later
    read.
  }];

  let arguments = (ins
    TypeAttr:$result_encoding,
    Stream_ShapeDynamicDims:$result_encoding_dims,
    Stream_Size:$result_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    `:`
    $result_encoding (`` `{` $result_encoding_dims^ `}`)?
    `in`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultEncodingDims(); }
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;
}

def Stream_TensorConstantOp : Stream_PureOp<"tensor.constant", [
  Stream_AffinityOp,
  Stream_StreamableOp,
  Stream_TensorPhaseOp,
  Util_ShapeAwareOp,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{defines a constant tensor value}];
  let description = [{
    Returns a typed resource initialized to the given constant value.
  }];

  let arguments = (ins
    ElementsAttr:$value,
    TypeAttr:$result_encoding,
    Stream_ShapeDynamicDims:$result_encoding_dims,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    `:`
    $result_encoding (`` `{` $result_encoding_dims^ `}`)?
    `in`
    type($result)
    `=`
    $value
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultEncodingDims(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_TensorSplatOp : Stream_PureOp<"tensor.splat", [
  Stream_AffinityOp,
  Stream_StreamableOp,
  Stream_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
]> {
  let summary = [{splats a value into a shaped tensor}];
  let description = [{
    Returns a typed resource initialized to the given primitive value.
  }];

  let arguments = (ins
    Stream_PrimitiveType:$value,
    TypeAttr:$result_encoding,
    Stream_ShapeDynamicDims:$result_encoding_dims,
    Stream_Size:$result_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $value
    `:` type($value)
    `->`
    $result_encoding (`` `{` $result_encoding_dims^ `}`)?
    `in`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultEncodingDims(); }
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_TensorCloneOp : Stream_PureOp<"tensor.clone", [
  AttrSizedOperandSegments,
  Stream_AffinityOp,
  Stream_StreamableOp,
  Stream_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
]> {
  let summary = [{clones the contents of a value}];
  let description = [{
    Clones the contents of a value at a snapshot in time. Future changes to the
    cloned value will not effect the result. Acts as a copy-on-write operation.
  }];

  let arguments = (ins
    Stream_AnyStreamResource:$source,
    TypeAttr:$source_encoding,
    Stream_ShapeDynamicDims:$source_encoding_dims,
    Stream_Size:$source_size,
    TypeAttr:$result_encoding,
    Stream_ShapeDynamicDims:$result_encoding_dims,
    Stream_Size:$result_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `:`
    $source_encoding (`` `{` $source_encoding_dims^ `}`)?
    `in`
    type($source) `` `{` $source_size `}`
    `->`
    $result_encoding (`` `{` $result_encoding_dims^ `}`)?
    `in`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getSourceEncodingDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultEncodingDims(); }
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Stream_TensorSliceOp : Stream_PureOp<"tensor.slice", [
  AttrSizedOperandSegments,
  Stream_AffinityOp,
  Stream_StreamableOp,
  Stream_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
]> {
  let summary = [{slices out a cloned subview of a value}];
  let description = [{
    Slices a subrange of a stream resource based on a tensor encoding. Acts as a
    copy-on-write operation.
  }];

  let arguments = (ins
    Stream_AnyStreamResource:$source,
    TypeAttr:$source_encoding,
    Stream_ShapeDynamicDims:$source_encoding_dims,
    Stream_Size:$source_size,
    Variadic<Stream_Dim>:$start_indices,
    Variadic<Stream_Dim>:$lengths,
    TypeAttr:$result_encoding,
    Stream_ShapeDynamicDims:$result_encoding_dims,
    Stream_Size:$result_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `[` $start_indices `for` $lengths `]` `:`
    $source_encoding (`` `{` $source_encoding_dims^ `}`)?
    `in`
    type($source) `` `{` $source_size `}`
    `->`
    $result_encoding (`` `{` $result_encoding_dims^ `}`)?
    `in`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getSourceEncodingDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultEncodingDims(); }
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Stream_TensorFillOp : Stream_Op<"tensor.fill", [
  AttrSizedOperandSegments,
  AllTypesMatch<["target", "result"]>,
  Stream_AffinityOp,
  Stream_StreamableOp,
  Stream_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{fills a subview of a stream resource with a value}];
  let description = [{
    Splats a value into a subview of the given stream resource and returns the
    resource with the update applied.

    Equivalent to a stream.tensor.splat + stream.tensor.update.
  }];

  let arguments = (ins
    Stream_AnyStreamResource:$target,
    TypeAttr:$target_encoding,
    Stream_ShapeDynamicDims:$target_encoding_dims,
    Stream_Size:$target_size,
    Variadic<Stream_Dim>:$start_indices,
    Variadic<Stream_Dim>:$lengths,
    Stream_PrimitiveType:$value,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $value `,` $target `[` $start_indices `for` $lengths `]` `:`
    type($value)
    `->`
    $target_encoding (`` `{` $target_encoding_dims^ `}`)?
    `in`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getTargetEncodingDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return getTargetEncodingDims(); }
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_TensorUpdateOp : Stream_Op<"tensor.update", [
  AttrSizedOperandSegments,
  AllTypesMatch<["target", "result"]>,
  Stream_AffinityOp,
  Stream_StreamableOp,
  Stream_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{updates a slice of a subview of a resource in-place}];
  let description = [{
    Copies a value into a resource based on tensor encodings. The returned value
    is the entire updated target value.
  }];

  let arguments = (ins
    Stream_AnyStreamResource:$target,
    TypeAttr:$target_encoding,
    Stream_ShapeDynamicDims:$target_encoding_dims,
    Stream_Size:$target_size,
    Variadic<Stream_Dim>:$start_indices,
    Stream_AnyStreamResource:$update,
    TypeAttr:$update_encoding,
    Stream_ShapeDynamicDims:$update_encoding_dims,
    Stream_Size:$update_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $update `,` $target `[` $start_indices `]` `:`
    $update_encoding (`` `{` $update_encoding_dims^ `}`)?
    `in`
    type($update) `` `{` $update_size `}`
    `->`
    $target_encoding (`` `{` $target_encoding_dims^ `}`)?
    `in`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) {
      return idx == 0 ? getTargetEncodingDims() : getUpdateEncodingDims();
    }
    ValueRange getResultDynamicDims(unsigned idx) { return getTargetEncodingDims(); }
    Value getOperandSize(unsigned idx) {
      return idx == 0 ? getTargetSize() : getUpdateSize();
    }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Stream_TensorLoadOp : Stream_PureOp<"tensor.load", [
  AttrSizedOperandSegments,
  Stream_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
]> {
  let summary = [{loads a value from a tensor element}];
  let description = [{
    Returns the element at the given location from within the tensor.
  }];

  let arguments = (ins
    Stream_StagingResource:$source,
    TypeAttr:$source_encoding,
    Stream_ShapeDynamicDims:$source_encoding_dims,
    Stream_Size:$source_size,
    Variadic<Stream_Dim>:$indices
  );
  let results = (outs
    AnyTypeOf<[Stream_PrimitiveType, AnyVector]>:$result
  );

  let assemblyFormat = [{
    $source (`[` $indices^ `]`)? `:`
    $source_encoding (`` `{` $source_encoding_dims^ `}`)?
    `in`
    type($source) `` `{` $source_size `}`
    `->`
    type($result)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getSourceEncodingDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return ValueRange{}; }
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_TensorStoreOp : Stream_PureOp<"tensor.store", [
  AttrSizedOperandSegments,
  AllTypesMatch<["target", "result"]>,
  Stream_TensorPhaseOp,
  Util_ShapeAwareOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{stores a value into a tensor element}];
  let description = [{
    Returns a tensor with the element at the given index set to the given value.
  }];

  let arguments = (ins
    Stream_StagingResource:$target,
    TypeAttr:$target_encoding,
    Stream_ShapeDynamicDims:$target_encoding_dims,
    Stream_Size:$target_size,
    Variadic<Stream_Dim>:$indices,
    AnyTypeOf<[Stream_PrimitiveType, AnyVector]>:$value
  );
  let results = (outs
    Stream_StagingResource:$result
  );

  let assemblyFormat = [{
    $value `,`
    $target (`[` $indices^ `]`)? `:`
    type($value)
    `->`
    $target_encoding (`` `{` $target_encoding_dims^ `}`)?
    `in`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getTargetEncodingDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return getTargetEncodingDims(); }
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_TensorTraceOp : Stream_Op<"tensor.trace", []> {
  let summary = [{trace value(s) operation}];
  let description = [{
    Traces out to a runtime trace sink (console, log file, etc) the given
    tensors and titles them with the given key. The key is informational only
    and useful for titling/marking specific sets of tensors for easier
    searching.
  }];

  let arguments = (ins
    StrAttr:$key,
    Variadic<AnyTensor>:$operands
  );

  // TODO(benvanik): stream.tensor.trace assembly format (custom).
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// Builtins
//===----------------------------------------------------------------------===//
// Builtins are added by frontends or during tensor encoding to handle ops or
// data types not supported by the lower-level stream ops (stream.async.* and
// stream.cmd.*).

def Stream_BuiltinSplatI64Op : Stream_Op<"builtin.splat.i64", [
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Stream_BuiltinOp, [
    "mergeBuiltinModule",
    "convertBuiltinOp",
  ]>,
  Util_SizeAwareOp,
]> {
  let summary = [{splats a value into a resource}];
  let description = [{
    Generated builtin for emulating 64-bit splats.
    As with stream.async.splat it returns a new resource with the given
    value splatted out to fill the entire contents.
  }];

  let arguments = (ins
    I64:$value,
    Stream_Size:$result_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $value `:` type($value) `->` type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;
}

def Stream_BuiltinFillI64Op : Stream_Op<"builtin.fill.i64", [
  AllTypesMatch<["target", "result"]>,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Stream_BuiltinOp, [
    "mergeBuiltinModule",
    "convertBuiltinOp",
  ]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{fills a subview of a stream resource with a value}];
  let description = [{
    Generated builtin for emulating 64-bit fills.
    As with stream.async.fill it splats a value into a subview of the given
    stream resource and returns the resource with the update applied.
  }];

  let arguments = (ins
    Stream_AnyStreamResource:$target,
    Stream_Size:$target_size,
    Stream_Offset:$target_offset,
    Stream_Offset:$target_end,
    Stream_Size:$target_length,
    I64:$value,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $value `,`
    $target `[` $target_offset `to` $target_end `for` $target_length `]` `:`
    type($value) `->`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Resource transfer ops
//===----------------------------------------------------------------------===//

def Stream_AsyncAllocaOp : Stream_PureOp<"async.alloca", [
  DeclareOpInterfaceMethods<Stream_AffinityOp, [
    "getAffinity",
    "setAffinity",
  ]>,
  Stream_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Stream_StreamableOp, [
    "isMetadata",
    "preferCloneToConsumers",
  ]>,
  Util_SizeAwareOp,
  MemoryEffects<[MemAlloc]>,
]> {
  let summary = [{allocates a transient value with undefined contents}];
  let description = [{
    Allocates a transient value (one that is short-lived and local to the
    current computation) with undefined contents. Consumers of the allocated
    result must assume nothing of the contents and use `discard` access.
  }];

  let arguments = (ins
    Stream_Size:$storage_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    attr-dict `:` type($result) `{` $storage_size `}`
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getStorageSize(); }
  }];

  let hasCanonicalizer = 1;
}

def Stream_AsyncConstantOp : Stream_PureOp<"async.constant", [
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Stream_StreamableOp, [
    "isMetadata",
  ]>,
  DeclareOpInterfaceMethods<Stream_AsyncAccessOp, [
    "getAsyncAccessRanges",
  ]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{defines a constant resource}];
  let description = [{
    Returns a new resource with the given constant value.
  }];

  let arguments = (ins
    ElementsAttr:$value,
    Stream_Size:$result_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    `:`
    type($result) `` `{` $result_size `}`
    `=`
    $value
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_AsyncSplatOp : Stream_Op<"async.splat", [
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Stream_StreamableOp, [
    "preferCloneToConsumers",
  ]>,
  DeclareOpInterfaceMethods<Stream_AsyncAccessOp, [
    "getAsyncAccessRanges",
  ]>,
  Util_SizeAwareOp,
]> {
  let summary = [{splats a value into a resource}];
  let description = [{
    Returns a new resource with the given primitive value splatted out to fill
    the entire contents.
  }];

  let arguments = (ins
    Stream_FillPatternType:$value,
    Stream_Size:$result_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $value `:` type($value) `->` type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_AsyncCloneOp : Stream_Op<"async.clone", [
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Stream_StreamableOp, [
    "preferCloneToConsumers",
  ]>,
  DeclareOpInterfaceMethods<Stream_AsyncAccessOp, [
    "getAsyncAccessRanges",
  ]>,
  Util_SizeAwareOp,
]> {
  let summary = [{clones the contents of a value}];
  let description = [{
    Clones the contents of a value at a snapshot in time. Future changes to the
    cloned value will not effect the result. Acts as a copy-on-write operation.
  }];

  let arguments = (ins
    Stream_AnyStreamResource:$source,
    Stream_Size:$source_size,
    Stream_Size:$result_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `:`
    type($source) `` `{` $source_size `}` `->`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Stream_AsyncSliceOp : Stream_PureOp<"async.slice", [
  AllTypesMatch<["source", "result"]>,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  Stream_StreamableOp,
  DeclareOpInterfaceMethods<Stream_AsyncAccessOp, [
    "getAsyncAccessRanges",
  ]>,
  Util_SizeAwareOp,
]> {
  let summary = [{slices out a cloned subview of a value}];
  let description = [{
    Slices a subrange of a stream resource based on a byte range. Acts as a
    copy-on-write operation.
  }];

  let arguments = (ins
    Stream_AnyStreamResource:$source,
    Stream_Size:$source_size,
    Stream_Offset:$source_offset,
    Stream_Offset:$source_end,
    Stream_Size:$result_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `[` $source_offset `to` $source_end `]` `:`
    type($source) `` `{` $source_size `}` `->`
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Stream_AsyncFillOp : Stream_Op<"async.fill", [
  AllTypesMatch<["target", "result"]>,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  Stream_StreamableOp,
  DeclareOpInterfaceMethods<Stream_AsyncAccessOp, [
    "getAsyncAccessRanges",
  ]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{fills a subview of a stream resource with a value}];
  let description = [{
    Splats a value into a subview of the given stream resource and returns the
    resource with the update applied.

    Equivalent to a stream.async.splat + stream.async.update.
  }];

  let arguments = (ins
    Stream_AnyStreamResource:$target,
    Stream_Size:$target_size,
    Stream_Offset:$target_offset,
    Stream_Offset:$target_end,
    Stream_Size:$target_length,
    Stream_FillPatternType:$value,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $value `,`
    $target `[` $target_offset `to` $target_end `for` $target_length `]` `:`
    type($value) `->`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_AsyncUpdateOp : Stream_Op<"async.update", [
  AllTypesMatch<["target", "result"]>,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  Stream_StreamableOp,
  DeclareOpInterfaceMethods<Stream_AsyncAccessOp, [
    "getAsyncAccessRanges",
  ]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{updates a slice of a subview of a resource in-place}];
  let description = [{
    Copies a value into a resource based on a byte range. The returned value
    is the entire updated target value. Updates can be turned into placement
    allocations and avoid copies.
  }];

  let arguments = (ins
    Stream_AnyStreamResource:$target,
    Stream_Size:$target_size,
    Stream_Offset:$target_offset,
    Stream_Offset:$target_end,
    Stream_AnyStreamResource:$update,
    Stream_Size:$update_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $update `,`
    $target `[` $target_offset `to` $target_end `]` `:`
    type($update) `` `{` $update_size `}` `->`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return idx == 0 ? getTargetSize() : getUpdateSize();
    }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Stream_AsyncCopyOp : Stream_Op<"async.copy", [
  AllTypesMatch<["target", "result"]>,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  Stream_StreamableOp,
  DeclareOpInterfaceMethods<Stream_AsyncAccessOp, [
    "getAsyncAccessRanges",
  ]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{copies a subview of a stream resource to another}];
  let description = [{
    Copies a subview of a resource into a subview of another.
    As with memcpy this does not support overlapping updates into the same
    resource. Unlike `stream.async.update` copy sources cannot be allocated
    in-place.

    Equivalent to a stream.async.slice + stream.async.update.
  }];

  let arguments = (ins
    Stream_AnyStreamResource:$target,
    Stream_Size:$target_size,
    Stream_Offset:$target_offset,
    Stream_Offset:$target_end,
    Stream_AnyStreamResource:$source,
    Stream_Size:$source_size,
    Stream_Offset:$source_offset,
    Stream_Offset:$source_end,
    Stream_Size:$length,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $source `[` $source_offset `to` $source_end `]` `,`
    $target `[` $target_offset `to` $target_end `]` `,`
    $length `:`
    type($source) `` `{` $source_size `}` `->`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return idx == 0 ? getTargetSize() : getSourceSize();
    }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_AsyncCollectiveOp : Stream_Op<"async.collective", [
  AllTypesMatch<["target", "result"]>,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  Stream_StreamableOp,
  DeclareOpInterfaceMethods<Stream_AsyncAccessOp, [
    "getAsyncAccessRanges",
  ]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{performs a collective operation}];
  let description = [{
    TODO: document different usage. For now this should be considered a
    prototype and that modeling of collective operations may change in the
    future to better ensure in-place operations (where send/recv is a subset of
    recv/send). We may have dedicated operations for the send and recv verbs as
    they have sequencing implications - or we could add optional sequencing to
    this base op.
  }];

  let arguments = (ins
    Stream_CollectiveAttr:$op,
    Stream_AnyStreamResource:$target,
    Stream_Size:$target_size,
    Stream_Offset:$target_offset,
    Stream_Offset:$target_end,
    Stream_Size:$target_length,
    Stream_AnyStreamResource:$source,
    Stream_Size:$source_size,
    Stream_Offset:$source_offset,
    Stream_Offset:$source_end,
    Stream_Size:$source_length,
    Stream_Size:$element_count,
    Stream_Channel:$channel,
    Optional<I32>:$param,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_AnyStreamResource:$result
  );

  let assemblyFormat = [{
    `` $op `` `[` $element_count `]`
    (`on` `(` $affinity^ `)`)?
    `channel` `(` $channel `)`
    custom<CollectiveParam>(ref($op), $param)
    $source `[` $source_offset `to` $source_end `for` $source_length `]` `,`
    $target `[` $target_offset `to` $target_end `for` $target_length `]` `:`
    type($source) `` `{` $source_size `}` `->`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return idx == 0 ? getTargetSize() : getSourceSize();
    }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_AsyncTransferOp : Stream_Op<"async.transfer", [
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  Stream_StreamableOp,
  DeclareOpInterfaceMethods<Stream_AsyncAccessOp, [
    "getAsyncAccessRanges",
  ]>,
  Util_SizeAwareOp,
]> {
  let summary = [{transfers a resource from one location/state to another}];
  let description = [{
    Transfers a resource between different states (such as a `staging` lifetime
    to a `local` lifetime) or different affinities. This is roughly equivalent
    to a cast but may have special semantics when later lowered to one or more
    devices with discrete memory spaces or pools.
  }];

  let arguments = (ins
    AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_StagingResource,
    ]>:$source,
    Stream_Size:$source_size,
    Stream_Size:$result_size,
    OptionalAttr<Stream_AffinityAttr>:$source_affinity,
    OptionalAttr<Stream_AffinityAttr>:$result_affinity
  );
  let results = (outs
    AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_StagingResource,
    ]>:$result
  );

  let assemblyFormat = [{
    (`from` `(` $source_affinity^ `)`)?
    $source `:`
    type($source) `` `{` $source_size `}` `->`
    (`to` `(` $result_affinity^ `)`)?
    type($result) `` `{` $result_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return getResultSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Stream_AsyncLoadOp : Stream_PureOp<"async.load", [
  Stream_AsyncPhaseOp,
  Util_SizeAwareOp,
]> {
  let summary = [{loads a value from a resource}];
  let description = [{
    Returns the element at the given location from within the resource.
  }];

  let arguments = (ins
    Stream_StagingResource:$source,
    Stream_Size:$source_size,
    Stream_Offset:$source_offset
  );
  let results = (outs
    AnyTypeOf<[Stream_PrimitiveType, AnyVector]>:$result
  );

  let assemblyFormat = [{
    $source `[` $source_offset `]` `:`
    type($source) `` `{` $source_size `}`
    `->`
    type($result)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getSourceSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_AsyncStoreOp : Stream_PureOp<"async.store", [
  AllTypesMatch<["target", "result"]>,
  Stream_AsyncPhaseOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{stores a value into a resource}];
  let description = [{
    Returns a resource with the element at the given offset set to the given
    value.
  }];

  let arguments = (ins
    Stream_StagingResource:$target,
    Stream_Size:$target_size,
    Stream_Offset:$target_offset,
    AnyTypeOf<[Stream_PrimitiveType, AnyVector]>:$value
  );
  let results = (outs
    Stream_StagingResource:$result
  );

  let assemblyFormat = [{
    $value `,`
    $target `[` $target_offset `]` `:`
    type($value)
    `->`
    custom<ShapedTiedResult>(type($target), $target_size)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return getTargetSize(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_AsyncDispatchOp : Stream_Op<"async.dispatch", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  Stream_StreamableOp,
  DeclareOpInterfaceMethods<Stream_AsyncAccessOp, [
    "getAsyncAccessRanges",
  ]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedOperandsIndexAndLength",
  ]>,
]> {
  let summary = [{dispatches a parallelized grid of work}];
  let description = [{
    Calls the specified entry point function once for each element in the
    specified workgroup count. Each workgroup has access to the same operands
    and results and is able to load/store at will.
  }];

  let arguments = (ins
    Variadic<Index>:$workload,
    SymbolRefAttr:$entry_point,
    Variadic<AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_PrimitiveType,
    ]>>:$resource_operands,
    Variadic<Stream_Size>:$resource_operand_sizes,
    Variadic<Stream_Offset>:$resource_operand_offsets,
    Variadic<Stream_Offset>:$resource_operand_ends,
    Variadic<Stream_Size>:$resource_operand_lengths,
    Variadic<Stream_Size>:$result_sizes,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<Stream_AnyStreamResource>:$results
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $entry_point
    (`[` $workload^ `]`)? ``
    custom<DispatchOperands>($resource_operands,
                             $resource_operand_offsets,
                             $resource_operand_ends,
                             $resource_operand_lengths) attr-dict `:`
    custom<ShapedFunctionType>(ref($resource_operands),
                               type($resource_operands), $resource_operand_sizes,
                               type($results), $result_sizes,
                               $tied_operands)
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), getResourceOperandSizes());
    }
    Value getResultSize(unsigned idx) {
      return findValueSizeInList(idx, getResults(), getResultSizes());
    }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Async control flow ops
//===----------------------------------------------------------------------===//

// TODO(benvanik): stream.async.if
// TODO(benvanik): stream.async.select
// TODO(benvanik): stream.async.for

def Stream_AsyncFuncOp : Stream_Op<"async.func", [
  CallableOpInterface,
  FunctionOpInterface,
  IsolatedFromAbove,
  Stream_AsyncPhaseOp,
]> {
  let summary = [{streamable function declaration}];
  let description = [{
    Declares a function that can be called as an asynchronous streaming
    operation via `stream.async.call`. Today only external functions are
    allowed.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name
    ``
    custom<ShapedFunctionSignature>($function_type,
                                    $tied_operands,
                                    $arg_attrs,
                                    $res_attrs)
    attr-dict-with-keyword
    ($body^)?
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "StringRef":$name,
      "FunctionType":$type,
      CArg<"ArrayAttr", "{}">:$tied_operands,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$resAttrs
    )>,
  ];

  let extraClassDeclaration = [{
    static AsyncFuncOp create(Location location, StringRef name,
                              FunctionType type,
                              ArrayRef<int64_t> tiedOperands = {},
                              ArrayRef<DictionaryAttr> argAttrs = {},
                              ArrayRef<DictionaryAttr> resAttrs = {});

    bool isDeclaration() { return true; }

    ::mlir::Region *getCallableRegion() { return nullptr; }
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }
    ::mlir::ArrayAttr getCallableArgAttrs() { return getArgAttrs().value_or(nullptr); }
    ::mlir::ArrayAttr getCallableResAttrs() { return getResAttrs().value_or(nullptr); }

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    // Returns true if the given result is tied to an argument.
    bool isResultTied(int resultIndex);
  }];
}

def Stream_AsyncCallOp : Stream_Op<"async.call", [
  AttrSizedOperandSegments,
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  Stream_StreamableOp,
  DeclareOpInterfaceMethods<Stream_AsyncAccessOp, [
    "getAsyncAccessRanges",
  ]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedOperandsIndexAndLength",
  ]>,
]> {
  let summary = [{calls a streamable external host function}];
  let description = [{
    Calls a function taking/returning resource values with stream semantics.
    Asynchronous calls must have no side-effects.

    Note that returned resources must have their sizes declared prior to the
    call as this is what allows the call to be made on the stream. If external
    host logic is required to compute the size (avoid at all costs!) a separate
    func.call can be used outside of the stream to do so. If sizes are
    unknownable until the operation is performed it should be made as a normal
    asynchronous host call with 'coarse-fences' instead.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_PrimitiveType,
      AnyType,
    ]>>:$resource_operands,
    Variadic<Stream_Size>:$resource_operand_sizes,
    Variadic<Stream_Offset>:$resource_operand_offsets,
    Variadic<Stream_Offset>:$resource_operand_ends,
    Variadic<Stream_Size>:$resource_operand_lengths,
    Variadic<Stream_Size>:$result_sizes,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_PrimitiveType,
    ]>>:$results
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $callee ``
    custom<DispatchOperands>($resource_operands,
                             $resource_operand_offsets,
                             $resource_operand_ends,
                             $resource_operand_lengths) attr-dict `:`
    custom<ShapedFunctionType>(ref($resource_operands),
                               type($resource_operands), $resource_operand_sizes,
                               type($results), $result_sizes,
                               $tied_operands)
  }];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return getResourceOperands().begin(); }
    operand_iterator arg_operand_end() { return getResourceOperands().end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }

    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), getResourceOperandSizes());
    }
    Value getResultSize(unsigned idx) {
      return findValueSizeInList(idx, getResults(), getResultSizes());
    }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_AsyncExecuteOp : Stream_Op<"async.execute", [
  AttrSizedOperandSegments,
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getSuccessorEntryOperands",
  ]>,
  SingleBlockImplicitTerminator<"IREE::Stream::YieldOp">,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Stream_AsyncAccessOp, [
    "getAsyncAccessRanges",
  ]>,
  Stream_TimelineOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_ClosureOpInterface>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResultsIndexAndLength",
  ]>,
]> {
  let summary = [{executes a dependency-aware sequence of streamable ops}];
  let description = [{
    Evaluates the operations within the region by dependency order while obeying
    ties when present. Nested ops execute serially in block order and nested
    `stream.async.concurrent` ops can be used to run multiple ops concurrently
    within the stream. All resource inputs must be captured explicitly. All
    results are only ready once all nested ops complete execution and the
    returned timepoint is reached. Zero or more timepoints may be provided to
    block execution until they are all reached; zero timepoints indicates that
    execution may begin immediately.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_StagingResource,
    ]>>:$resource_operands,
    Variadic<Stream_Size>:$resource_operand_sizes,
    Variadic<Stream_Size>:$result_sizes,
    Optional<Stream_Timepoint>:$await_timepoint,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_StagingResource,
    ]>>:$results,
    Stream_Timepoint:$result_timepoint
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    (`await` `(` $await_timepoint^ `)` `=` `` `>`)?
    `with` ``
    custom<ResourceRegion>($resource_operands,
                           type($resource_operands), $resource_operand_sizes,
                           type($results), $result_sizes,
                           $tied_operands, $body)
    `=` `` `>` type($result_timepoint)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "TypeRange":$resultTypes, "ValueRange":$resultSizes,
      "Value":$awaitTimepoint,
      "ValueRange":$resourceOperands, "ValueRange":$resourceOperandSizes,
      "ArrayRef<int64_t>":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), getResourceOperandSizes());
    }
    Value getResultSize(unsigned idx) {
      return findValueSizeInList(idx, getResults(), getResultSizes());
    }
    SmallVector<Value> getAwaitTimepoints() {
      if (getAwaitTimepoint()) return {getAwaitTimepoint()}; else return {};
    }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_AsyncConcurrentOp : Stream_Op<"async.concurrent", [
  ParentOneOf<[
    "IREE::Stream::AsyncExecuteOp",
    "IREE::Stream::AsyncConcurrentOp",
  ]>,
  AttrSizedOperandSegments,
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getSuccessorEntryOperands",
  ]>,
  SingleBlockImplicitTerminator<"IREE::Stream::YieldOp">,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  Stream_StreamableOp,
  DeclareOpInterfaceMethods<Stream_AsyncAccessOp, [
    "getAsyncAccessRanges",
  ]>,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_ClosureOpInterface>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface>,
]> {
  let summary = [{executes all ops concurrently}];
  let description = [{
    Represents a wave of work scheduled concurrently (each op executing at the
    same time). All resource inputs must be captured explicitly. All results are
    only ready once all nested ops complete execution.

    Waves can be nested to create a DAG. For example, take the following graph:
    ```
                      |
            v---------+---------v
    +-------|-------+   +-------|-------+
    |    v--+--v    |   |    v--+--v    |
    | +----+ +----+ |   | +----+ +----+ |
    | | %a | | %b | |   | | %c | | %d | |
    | +----+ +----+ |   | +----+ +----+ |
    |    +--v--+    |   |    +--v--+    |
    +-------|-------+   +-------|-------+
            +---------v---------+
                      |
    ```

    Represented with nested waves:
    ```mlir
      %0 = stream.async.concurrent with(%arg) -> ... {
        %1 = stream.async.concurrent with(%arg as %arg0) -> ... {
          %a = ...
          %b = ...
          stream.yield %a, %b
        }
        %2 = stream.async.concurrent with(%arg as %arg1) -> ... {
          %c = ...
          %d = ...
          stream.yield %c, %d
        }
        stream.yield %1, %2
      }
    ```
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_StagingResource,
    ]>>:$resource_operands,
    Variadic<Stream_Size>:$resource_operand_sizes,
    Variadic<Stream_Size>:$result_sizes,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_StagingResource,
    ]>>:$results
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    `with` ``
    custom<ResourceRegion>($resource_operands,
                           type($resource_operands), $resource_operand_sizes,
                           type($results), $result_sizes,
                           $tied_operands, $body)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "TypeRange":$resultTypes, "ValueRange":$resultSizes,
      "ValueRange":$resourceOperands, "ValueRange":$resourceOperandSizes,
      "ArrayRef<int64_t>":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), getResourceOperandSizes());
    }
    Value getResultSize(unsigned idx) {
      return findValueSizeInList(idx, getResults(), getResultSizes());
    }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Explicit command ops
//===----------------------------------------------------------------------===//

def Stream_CmdFlushOp : Stream_Op<"cmd.flush", [
  Stream_CmdPhaseOp,
  Stream_StreamableOp,
  Stream_SubviewEffectOp,
  Util_SizeAwareOp,
]> {
  let summary = [{flushes a subview of a resource}];
  let description = [{
    Transfers a resource to an external target. The resource memory is made
    available to the target and can be made visible there using
    `stream.cmd.invalidate`.
  }];

  let arguments = (ins
    Stream_AnyResource:$target,
    Stream_Size:$target_size,
    Stream_Offset:$target_offset,
    Stream_Size:$target_length,
    OptionalAttr<Stream_AffinityAttr>:$source_affinity
  );
  let results = (outs);

  let assemblyFormat = [{
    (`to` `(` $source_affinity^ `)`)?
    $target `[` $target_offset `for` $target_length `]` `:`
    type($target) `` `{` $target_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_CmdInvalidateOp : Stream_Op<"cmd.invalidate", [
  Stream_CmdPhaseOp,
  Stream_StreamableOp,
  Stream_SubviewEffectOp,
  Util_SizeAwareOp,
]> {
  let summary = [{invalidates a subview of a resource}];
  let description = [{
    Transfers a resource from an external source into the current target. The
    resource memory is assumed to have been made available at the source via
    `stream.cmd.flush`.
  }];

  let arguments = (ins
    Stream_AnyResource:$target,
    Stream_Size:$target_size,
    Stream_Offset:$target_offset,
    Stream_Size:$target_length,
    OptionalAttr<Stream_AffinityAttr>:$source_affinity
  );
  let results = (outs);

  let assemblyFormat = [{
    (`from` `(` $source_affinity^ `)`)?
    $target `[` $target_offset `for` $target_length `]` `:`
    type($target) `` `{` $target_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_CmdDiscardOp : Stream_Op<"cmd.discard", [
  Stream_CmdPhaseOp,
  Stream_StreamableOp,
  Stream_SubviewEffectOp,
  Util_SizeAwareOp,
]> {
  let summary = [{discards a subview of a resource}];
  let description = [{
    Discards a subview of a resource, indicating that after this command the
    specified contents are no longer needed. This can be used to trim memory
    or invalidate caches.
  }];

  let arguments = (ins
    Stream_AnyResource:$target,
    Stream_Size:$target_size,
    Stream_Offset:$target_offset,
    Stream_Size:$target_length
  );
  let results = (outs);

  let assemblyFormat = [{
    $target `[` $target_offset `for` $target_length `]` `:`
    type($target) `` `{` $target_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_CmdFillOp : Stream_Op<"cmd.fill", [
  Stream_CmdPhaseOp,
  Stream_StreamableOp,
  Stream_SubviewEffectOp,
  Util_SizeAwareOp,
]> {
  let summary = [{fills a subview of a stream resource with a value}];
  let description = [{
    Splats a value into a subview of the given stream resource and returns the
    resource with the update applied.
  }];

  let arguments = (ins
    Stream_AnyStreamResource:$target,
    Stream_Size:$target_size,
    Stream_Offset:$target_offset,
    Stream_Size:$target_length,
    Stream_FillPatternType:$value
  );
  let results = (outs);

  let assemblyFormat = [{
    $value `,`
    $target `[` $target_offset `for` $target_length `]` `:`
    type($value) `->`
    type($target) `` `{` $target_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getTargetSize(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_CmdCopyOp : Stream_Op<"cmd.copy", [
  Stream_CmdPhaseOp,
  Stream_StreamableOp,
  Stream_SubviewEffectOp,
  Util_SizeAwareOp,
]> {
  let summary = [{copies a subview of a stream resource to another}];
  let description = [{
    Copies a subview of a resource into a subview of another.
    As with memcpy this does not support overlapping updates into the same
    resource.
  }];

  let arguments = (ins
    Stream_AnyResource:$source,
    Stream_Size:$source_size,
    Stream_Offset:$source_offset,
    Stream_AnyResource:$target,
    Stream_Size:$target_size,
    Stream_Offset:$target_offset,
    Stream_Size:$length
  );
  let results = (outs);

  let assemblyFormat = [{
    $source `[` $source_offset `]` `,`
    $target `[` $target_offset `]` `,`
    $length `:`
    type($source) `` `{` $source_size `}` `->`
    type($target) `` `{` $target_size `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return idx == 0 ? getSourceSize() : getTargetSize();
    }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_CmdCollectiveOp : Stream_Op<"cmd.collective", [
  AttrSizedOperandSegments,
  Stream_CmdPhaseOp,
  Stream_StreamableOp,
  Stream_SubviewEffectOp,
  Util_SizeAwareOp,
]> {
  let summary = [{dispatches a collective operation}];
  let description = [{
    Dispatches a collective operation specified against the device. If grouped
    with other collectives in a `stream.cmd.concurrent` region the collective
    operations may fuse and execute more efficiently.
  }];

  let arguments = (ins
    Stream_CollectiveAttr:$op,
    Stream_Channel:$channel,
    Stream_Size:$element_count,
    Optional<I32>:$param,
    Variadic<Stream_AnyStreamResource>:$resources,
    Variadic<Stream_Size>:$resource_sizes,
    Variadic<Stream_Offset>:$resource_offsets,
    Variadic<Stream_Size>:$resource_lengths,
    Stream_ResourceAccessArrayAttr:$resource_accesses
  );
  let results = (outs);

  let assemblyFormat = [{
    `` $op `` `[` $element_count `]`
    `channel` `(` $channel `)`
    (`param` `(` $param^ `:` type($param) `)`)? `{`
    custom<DispatchResources>($resources, type($resources), $resource_sizes,
                              $resource_offsets, $resource_lengths,
                              $resource_accesses)
    `\n` `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(
          idx - getODSOperandIndexAndLength(4).first,
          getResources(), getResourceSizes());
    }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_CmdDispatchOp : Stream_Op<"cmd.dispatch", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Stream_CmdPhaseOp,
  Stream_StreamableOp,
  Stream_SubviewEffectOp,
  Util_SizeAwareOp,
]> {
  let summary = [{dispatches a parallelized grid of work}];
  let description = [{
    Calls the specified entry point function once for each element in the
    specified workgroup count. Each workgroup has access to the same operands
    and results and is able to load/store at will.
  }];

  let arguments = (ins
    Variadic<Index>:$workload,
    SymbolRefArrayAttr:$entry_points,
    Variadic<Stream_PrimitiveType>:$uniform_operands,
    Variadic<Stream_AnyStreamResource>:$resources,
    Variadic<Stream_Size>:$resource_sizes,
    Variadic<Stream_Offset>:$resource_offsets,
    Variadic<Stream_Size>:$resource_lengths,
    Stream_ResourceAccessArrayAttr:$resource_accesses
  );
  let results = (outs);

  let assemblyFormat = [{
    custom<DispatchEntryPoints>($entry_points)
    (`[` $workload^ `]`)? ``
    (`(` $uniform_operands^ `:` type($uniform_operands) `)`)? `{`
    custom<DispatchResources>($resources, type($resources), $resource_sizes,
                              $resource_offsets, $resource_lengths,
                              $resource_accesses)
    `\n` `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    auto getEntryPointRefs() {
      return getEntryPoints().getAsRange<SymbolRefAttr>();
    }
    void forEachEntryPointAttr(std::function<void(SymbolRefAttr)> fn) {
      for (auto entryPointAttr : getEntryPointRefs()) fn(entryPointAttr);
    }

    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(
          idx - getODSOperandIndexAndLength(2).first,
          getResources(), getResourceSizes());
    }
    Value getResultSize(unsigned idx) { return {}; }

    // Builds a map of operand index to argument index.
    static SmallVector<unsigned> makeOperandToArgMap(mlir::func::FuncOp funcOp);
    // Builds a map of resource to argument index of the corresponding binding.
    static SmallVector<unsigned> makeResourceToArgMap(mlir::func::FuncOp funcOp);
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_CmdFuncOp : Stream_Op<"cmd.func", [
  CallableOpInterface,
  FunctionOpInterface,
  IsolatedFromAbove,
  Stream_CmdPhaseOp,
]> {
  let summary = [{streamable function declaration}];
  let description = [{
    Declares a function that can be called as an asynchronous streaming
    operation via `stream.cmd.call`. Today only external functions are
    allowed.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name ``
    custom<DispatchFunctionSignature>($function_type,
                                      $arg_attrs,
                                      $res_attrs)
    attr-dict-with-keyword
    ($body^)?
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "StringRef":$name,
      "FunctionType":$type,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$resAttrs
    )>,
  ];

  let extraClassDeclaration = [{
    static CmdFuncOp create(Location location, StringRef name,
                            FunctionType type,
                            ArrayRef<DictionaryAttr> argAttrs = {},
                            ArrayRef<DictionaryAttr> resAttrs = {});

    bool isDeclaration() { return true; }

    ::mlir::Region *getCallableRegion() { return nullptr; }
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }
    ::mlir::ArrayAttr getCallableArgAttrs() { return getArgAttrs().value_or(nullptr); }
    ::mlir::ArrayAttr getCallableResAttrs() { return getResAttrs().value_or(nullptr); }

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
}

def Stream_CmdCallOp : Stream_Op<"cmd.call", [
  AttrSizedOperandSegments,
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Stream_CmdPhaseOp,
  Stream_StreamableOp,
  Stream_SubviewEffectOp,
  Util_SizeAwareOp,
]> {
  let summary = [{calls a streamable external host function}];
  let description = [{
    Calls a function operating on resource values with stream semantics.
    Asynchronous calls must have no side-effects.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    // TODO(benvanik): rework this to make more sense - it's mostly a copy of
    // stream.cmd.dispatch mashed together with stream.async.call and it's kind
    // of confusing.
    Variadic<AnyTypeOf<[
      Stream_PrimitiveType,
      Stream_AnyStreamResource,
      AnyType,  // for custom types
    ]>>:$resource_operands,
    Variadic<Stream_Size>:$resource_operand_sizes,
    Variadic<Stream_Offset>:$resource_operand_offsets,
    Variadic<Stream_Size>:$resource_operand_lengths,
    Variadic<Stream_Size>:$result_sizes,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands,
    Stream_ResourceAccessArrayAttr:$resource_operand_accesses
  );
  let results = (outs
    Variadic<Stream_PrimitiveType>:$results
  );

  let assemblyFormat = [{
    $callee ``
    custom<CmdCallOperands>($resource_operands,
                            $resource_operand_offsets,
                            $resource_operand_lengths,
                            $resource_operand_accesses) attr-dict `:`
    custom<ShapedFunctionType>(ref($resource_operands),
                               type($resource_operands),
                               $resource_operand_sizes,
                               type($results),
                               $result_sizes,
                               $tied_operands)
  }];

  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return getOperands().begin(); }
    operand_iterator arg_operand_end() { return getOperands().end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }

    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), getResourceOperandSizes());
    }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_CmdExecuteOp : Stream_Op<"cmd.execute", [
  AttrSizedOperandSegments,
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getSuccessorEntryOperands",
  ]>,
  SingleBlockImplicitTerminator<"IREE::Stream::YieldOp">,
  Stream_AffinityOp,
  Stream_CmdPhaseOp,
  Stream_TimelineOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_ClosureOpInterface>,
]> {
  let summary = [{executes a dependency-aware sequence of streamable ops}];
  let description = [{
    Evaluates the operations within the region by dependency order while obeying
    ties when present. Nested ops execute serially in block order and nested
    `stream.cmd.concurrent` ops can be used to run multiple ops concurrently
    within the stream. All resource inputs must be captured explicitly. All
    results are only ready once all nested ops complete execution and the
    returned timepoint is reached. Zero or more timepoints may be provided to
    block execution until they are all reached; zero timepoints indicates that
    execution may begin immediately.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_StagingResource,
    ]>>:$resource_operands,
    Variadic<Stream_Size>:$resource_operand_sizes,
    Optional<Stream_Timepoint>:$await_timepoint,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_Timepoint:$result_timepoint
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    (`await` `(` $await_timepoint^ `)` `=` `` `>`)?
    `with` ``
    custom<ExplicitResourceRegion>($resource_operands,
                                   type($resource_operands), $resource_operand_sizes,
                                   $body)
    `=` `` `>` type($result_timepoint)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "Value":$awaitTimepoint,
      "ValueRange":$resourceOperands, "ValueRange":$resourceOperandSizes,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), getResourceOperandSizes());
    }
    Value getResultSize(unsigned idx) {
      return {};
    }
    SmallVector<Value> getAwaitTimepoints() {
      if (getAwaitTimepoint()) return {getAwaitTimepoint()}; else return {};
    }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_CmdSerialOp : Stream_Op<"cmd.serial", [
  ParentOneOf<[
    "IREE::Stream::CmdExecuteOp",
    "IREE::Stream::CmdSerialOp",
    "IREE::Stream::CmdConcurrentOp",
  ]>,
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface>,
  SingleBlockImplicitTerminator<"IREE::Stream::YieldOp">,
  Stream_CmdPhaseOp,
  Stream_StreamableOp,
]> {
  let summary = [{executes all ops serially (in-order)}];
  let description = [{
    Represents a sequence of work scheduled serially (each op executing one
    after the other).

    Regions can be nested to create a DAG. For example, take the following graph:
    ```
                      |
            v---------+-----v
    +-------|-------+   +---|----+
    |    v--+--v    |   |   v    |
    | +----+ +----+ |   | +----+ |
    | | @a | | @b | |   | | @c | |
    | +----+ +----+ |   | +----+ |
    |    |     |    |   |   |    |
    |    |     |    |   | +-v--+ |
    |    |     |    |   | | @d | |
    |    |     |    |   | +----+ |
    |    +--v--+    |   |   |    |
    +-------|-------+   +---|----+
            +---------v-----+
                      |
    ```

    Represented with nested regions:
    ```mlir
      stream.cmd.concurrent {
        stream.cmd.concurrent {
          stream.cmd.dispatch @a
          stream.cmd.dispatch @b
        }
        stream.cmd.serial {
          stream.cmd.dispatch @c
          stream.cmd.dispatch @d
        }
      }
    ```
  }];

  let arguments = (ins);
  let results = (outs);

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    $body
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_CmdConcurrentOp : Stream_Op<"cmd.concurrent", [
  ParentOneOf<[
    "IREE::Stream::CmdExecuteOp",
    "IREE::Stream::CmdSerialOp",
    "IREE::Stream::CmdConcurrentOp",
  ]>,
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface>,
  SingleBlockImplicitTerminator<"IREE::Stream::YieldOp">,
  Stream_CmdPhaseOp,
  Stream_StreamableOp,
]> {
  let summary = [{executes all ops concurrently}];
  let description = [{
    Represents a wave of work scheduled concurrently (each op executing at the
    same time).

    Waves can be nested to create a DAG. For example, take the following graph:
    ```
                      |
            v---------+---------v
    +-------|-------+   +-------|-------+
    |    v--+--v    |   |    v--+--v    |
    | +----+ +----+ |   | +----+ +----+ |
    | | @a | | @b | |   | | @c | | @d | |
    | +----+ +----+ |   | +----+ +----+ |
    |    +--v--+    |   |    +--v--+    |
    +-------|-------+   +-------|-------+
            +---------v---------+
                      |
    ```

    Represented with nested waves:
    ```mlir
      stream.cmd.concurrent {
        stream.cmd.concurrent {
          stream.cmd.dispatch @a
          stream.cmd.dispatch @b
        }
        stream.cmd.concurrent {
          stream.cmd.dispatch @c
          stream.cmd.dispatch @d
        }
      }
    ```
  }];

  let arguments = (ins);
  let results = (outs);

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    $body
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Synchronization ops
//===----------------------------------------------------------------------===//

def Stream_TimepointImmediateOp : Stream_PureOp<"timepoint.immediate", [
  ConstantLike,
  Stream_TimelineOp,
]> {
  let summary = [{results an immediately-available timepoint}];
  let description = [{
    Timepoints indicate a point in the execution timeline and this op can be
    used to get a placeholder representing the start of the timeline. Any waits
    on the returned timepoint will resolve immediately. This generally folds
    away but can be useful if needing to initialize globals or branch args.
  }];

  let arguments = (ins);
  let results = (outs
    Stream_Timepoint:$result_timepoint
  );

  let assemblyFormat = [{
    attr-dict
    `=` `` `>` type($result_timepoint)
  }];

  let extraClassDeclaration = [{
    SmallVector<Value> getAwaitTimepoints() { return {}; }
  }];

  let hasFolder = 1;
}

def Stream_TimepointImportOp : Stream_PureOp<"timepoint.import", [
  Stream_AffinityOp,
]> {
  let summary = [{imports a timepoint from an external dialect type}];
  let description = [{
    Defines a conversion from an external dialect type such as `hal.semaphore`
    that is resolved during lowering into the stream dialect. This can be used
    to interoperate between levels of the stack that require specifying stream
    types and those that prior to lowering do not handle them.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_Timepoint:$result_timepoint
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $operands `:` `(` type($operands) `)`
    `=` `` `>`
    type($result_timepoint)
    attr-dict-with-keyword
  }];
}

def Stream_TimepointExportOp : Stream_PureOp<"timepoint.export", [
  Stream_AffinityOp,
]> {
  let summary = [{exports a timepoint to an external dialect type}];
  let description = [{
    Defines a conversion to an external dialect type such as `hal.fence`
    that is resolved during lowering into the stream dialect. This can be used
    to interoperate between levels of the stack that require specifying stream
    types and those that prior to lowering do not handle them.
  }];

  let arguments = (ins
    Stream_Timepoint:$await_timepoint,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $await_timepoint
    `=` `` `>`
    `(` type($results) `)`
    attr-dict-with-keyword
  }];

  let hasFolder = 1;
}

def Stream_TimepointChainExternalOp :
    Stream_Op<"timepoint.chain_external", [
      Stream_AffinityOp,
    ]> {
  let summary = [{exports a timepoint to an external dialect type}];
  let description = [{
    Defines a conversion to an external dialect type such as `hal.fence`
    that is resolved during lowering into the stream dialect. This can be used
    to interoperate between levels of the stack that require specifying stream
    types and those that prior to lowering do not handle them.
  }];

  let arguments = (ins
    Stream_Timepoint:$await_timepoint,
    Variadic<AnyType>:$external_values,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $await_timepoint
    `=` `` `>`
    `(` $external_values `:` type($external_values) `)`
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
}

def Stream_TimepointJoinOp : Stream_PureOp<"timepoint.join", [
  Stream_TimelineOp,
]> {
  let summary = [{joins one or more timepoints into the max of all of them}];
  let description = [{
    Returns a timepoint that indicates that all of the input timepoints have
    been reached.
  }];

  let arguments = (ins
    Variadic<Stream_Timepoint>:$await_timepoints
  );
  let results = (outs
    Stream_Timepoint:$result_timepoint
  );

  let assemblyFormat = [{
    `max` `(` $await_timepoints `)` `=` `` `>` type($result_timepoint)
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Stream_TimepointBarrierOp : Stream_PureOp<"timepoint.barrier", [
  AllTypesMatch<["resource", "result"]>,
  Stream_AffinityOp,
  Stream_TimelineOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
    "getTiedResultsIndexAndLength",
  ]>,
]> {
  let summary = [{returns a timepoint indicating when a resource is available}];
  let description = [{
    After asynchronous execution scheduling resources may exist in different
    states at different points in the execution timeline. This op enables
    identifying when the version of a resource after a particular point in the
    timeline is available. As timepoints transitively chain the timepoint must
    only cover the resource availability but not be limited to its original
    production timepoint.
  }];

  let arguments = (ins
    AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_StagingResource,
    ]>:$resource,
    Stream_Size:$resource_size,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_StagingResource,
    ]>:$result,
    Stream_Timepoint:$result_timepoint
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $resource `:` type($resource) `` `{` $resource_size `}`
    `=` `` `>`
    type($result_timepoint)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return getResourceSize(); }
    Value getResultSize(unsigned idx) { return getResourceSize(); }
    SmallVector<Value> getAwaitTimepoints() { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def Stream_TimepointAwaitOp : Stream_PureOp<"timepoint.await", [
  AttrSizedOperandSegments,
  Stream_AffinityOp,
  Stream_TimelineOp,
  Util_SizeAwareOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{awaits a timepoint before returning a set of resources}];
  let description = [{
    After asynchronous execution scheduling resources may exist in different
    states at different points in the execution timeline. This op enables
    resolving the version of a resource after a particular point in the
    timeline. As timepoints transitively chain the timepoint must only cover the
    resource availability but not be limited to its original production
    timepoint.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_StagingResource,
    ]>>:$resource_operands,
    Variadic<Stream_Size>:$resource_operand_sizes,
    Stream_Timepoint:$await_timepoint,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_StagingResource,
    ]>>:$results
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $await_timepoint `=` `` `>`
    $resource_operands `:`
    custom<SizeAwareTypeList>(type($resource_operands),
                              type($results), $resource_operand_sizes)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "ValueRange":$resourceOperands, "ValueRange":$resourceOperandSizes,
      "Value":$await_timepoint,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return getResourceOperandSizes()[idx];
    }
    Value getResultSize(unsigned idx) {
      return getResourceOperandSizes()[idx];
    }
    SmallVector<Value> getAwaitTimepoints() {
      if (getAwaitTimepoint()) return {getAwaitTimepoint()}; else return {};
    }
    Value getResultTimepoint() { return {}; }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Channels
//===----------------------------------------------------------------------===//

def Stream_ChannelDefaultOp : Stream_PureOp<"channel.default", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  Stream_AffinityOp,
]> {
  let summary = [{returns a default collective communication channel}];
  let description = [{
    Returns a channel initialized using the runtime environment for the
    particular affinity.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$group,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_Channel:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    (`group` `(` $group^ `)`)?
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def Stream_ChannelCreateOp : Stream_Op<"channel.create", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  Stream_AffinityOp,
]> {
  let summary = [{creates a new channel for collective communication}];
  let description = [{
    Returns a new channel with the given rank associated with the specified
    affinity. Collective operations using this channel must only be submitted on
    compatible affinities.

    The group and ID are optional and may be null. The rank and count can be
    omitted to indicate a default inherited from the environment or device
    configuration at runtime.
  }];

  let arguments = (ins
    Optional<Util_BufferType>:$id,
    OptionalAttr<StrAttr>:$group,
    Optional<Index>:$rank,
    Optional<Index>:$count,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Stream_Channel:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    (`id` `(` $id^ `)`)?
    (`group` `(` $group^ `)`)?
    (`rank` `(` $rank^ `)`)?
    (`count` `(` $count^ `)`)?
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def Stream_ChannelRankOp : Stream_PureOp<"channel.rank", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{returns the rank of the local participant in the group}];
  let description = [{
    Returns the rank the channel represents as a participant in a collective
    group in `[0, count)`.
  }];

  let arguments = (ins
    Stream_Channel:$channel
  );
  let results = (outs
    Index:$result
  );

  let assemblyFormat = [{
     $channel `:` type($result)
    attr-dict-with-keyword
  }];
}

def Stream_ChannelCountOp : Stream_PureOp<"channel.count", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{returns the total number of participants in the group}];
  let description = [{
    Returns the total participant count in the collective communicator group.
  }];

  let arguments = (ins
    Stream_Channel:$channel
  );
  let results = (outs
    Index:$result
  );

  let assemblyFormat = [{
    $channel `:` type($result)
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// Executables
//===----------------------------------------------------------------------===//

def Stream_ExecutableOp : Stream_Op<"executable", [
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"IREE::Stream::ExecutableEndOp">,
  NativeOpTrait<"SymbolTable">,
  Symbol,
]> {
  let summary = [{generic executable module}];
  let description = [{
    An executable module containing one or more public functions. The contents
    of the functions are safe to dispatch and can be lowered further to
    target-specific backend IR representations.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name
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
    OpBuilder<(ins "StringRef":$sym_name)>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return getBody().front(); }
    ::mlir::ModuleOp getInnerModule() {
      return *getBlock().getOps<::mlir::ModuleOp>().begin();
    }
  }];

  let hasVerifier = 1;
}

def Stream_ExecutableEndOp : Stream_Op<"executable.end", [
  HasParent<"IREE::Stream::ExecutableOp">,
  Terminator,
]> {
  let summary = [{terminator pseudo-op for the executable op}];
  let assemblyFormat = "attr-dict";
}

def Stream_ExecutableExportOp : Stream_Op<"executable.export", [
  HasParent<"IREE::Stream::ExecutableOp">,
  Symbol,
  IsolatedFromAbove,
]> {
  let summary = [{defines an executable entry point for dispatch operations}];
  let description = [{
    Specifies an exported function with an externally-visible alias. Multiple
    exports can reference the same internal function.

    Each entry point can have a unique workgroup count calculation region.
    This region takes the workload parameters passed to each flow.dispatch and
    produces an XYZ workgroup count for the 3D grid dispatch.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$function_ref
  );

  let regions = (region AnyRegion:$workgroup_count);

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    custom<SymbolAlias>($sym_name, $function_ref)
    custom<WorkgroupCountRegion>($workgroup_count)
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins
      "StringRef":$sym_name,
      "FlatSymbolRefAttr":$function_ref
    )>,
  ];

  let extraClassDeclaration = [{
    ::mlir::func::FuncOp lookupFunctionRef();
  }];

  let hasVerifier = 1;
}

def Stream_BindingSubspanOp : Stream_PureOp<"binding.subspan", [
    Util_ShapeAwareOp,
  ]> {
  let summary = [{returns an alias to a subspan of interface binding data}];
  let description = [{
    Returns a subview to a tensor or memref-like type from a binding. The same
    binding may have multiple subviews at different byte offsets.
  }];

  let arguments = (ins
    Stream_AnyBinding:$binding,
    Stream_Offset:$byte_offset,
    Stream_ShapeDynamicDims:$dynamic_dims
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $binding `` `[` $byte_offset `]`
    attr-dict `:` type($binding) `->` type($result) (`{` $dynamic_dims^ `}`)?
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return getDynamicDims(); }
  }];
}

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

def Stream_ReturnOp : Stream_Op<"return", [
  ParentOneOf<[
    "IREE::Stream::ExecutableExportOp",
  ]>,
  Pure,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface, [
    "getMutableSuccessorOperands",
  ]>,
  ReturnLike,
  Terminator,
]> {
  let summary = [{returns results from a region}];
  let description = [{
    The values returned are copied by-value.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let assemblyFormat = [{
    attr-dict
    $operands `:` type($operands)
  }];
}

def Stream_YieldOp : Stream_Op<"yield", [
  ParentOneOf<[
    "IREE::Stream::AsyncExecuteOp",
    "IREE::Stream::AsyncConcurrentOp",
    "IREE::Stream::CmdExecuteOp",
    "IREE::Stream::CmdSerialOp",
    "IREE::Stream::CmdConcurrentOp",
  ]>,
  Pure,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface, [
    "getMutableSuccessorOperands",
  ]>,
  ReturnLike,
  Terminator,
  SameVariadicOperandSize,
  Util_SizeAwareOp,
]> {
  let summary = [{yields stream values from an execution region}];
  let description = [{
    The values returned represent the asynchronous value at the point in time
    the SSA value is defined (or tied).
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_StagingResource,
    ]>>:$resource_operands,
    Variadic<Index>:$resource_operand_sizes
  );

  let assemblyFormat = [{
    attr-dict
    ($resource_operands^ `:`
        custom<SizeAwareTypeList>(type($resource_operands),
                                  $resource_operand_sizes))?
  }];

  let builders = [
    OpBuilder<(ins),
    [{
      build($_builder, $_state, ValueRange{}, ValueRange{});
    }]>,
  ];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), getResourceOperandSizes());
    }
    Value getResultSize(unsigned idx) { return {}; }
  }];
}

#endif  // IREE_DIALECT_STREAM_OPS
