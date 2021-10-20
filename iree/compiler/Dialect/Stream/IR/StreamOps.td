// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_STREAM_OPS
#define IREE_DIALECT_STREAM_OPS

include "iree/compiler/Dialect/Shape/IR/ShapeInterfaces.td"
include "iree/compiler/Dialect/Stream/IR/StreamBase.td"
include "iree/compiler/Dialect/Stream/IR/StreamInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class Stream_PureOp<string mnemonic, list<OpTrait> traits = []> :
    Stream_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

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
    Value getResultSize(unsigned idx) { return storage_sizes()[idx]; }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
}

def Stream_ResourceAllocaOp : Stream_PureOp<"resource.alloca", [
  DeclareOpInterfaceMethods<Stream_AffinityOp, [
    "getAffinity",
    "setAffinity",
  ]>,
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
    (`on` `(` $affinity^ `)`)?
    `uninitialized`
    attr-dict `:`
    type($result) `{` $storage_size `}`
    `=` `` `>`
    type($result_timepoint)
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return storage_size(); }
  }];

  let hasCanonicalizer = 1;
}

def Stream_ResourceDeallocaOp : Stream_Op<"resource.dealloca", [
  DeclareOpInterfaceMethods<Stream_AffinityOp, [
    "getAffinity",
    "setAffinity",
  ]>,
  Util_SizeAwareOp,
  MemoryEffects<[MemFree]>,
]> {
  let summary = [{frees a transient value when available}];
  let description = [{
    Deallocates a transient value (one that is short-lived and local to the
    current computation) previously allocated using `stream.resource.alloca`.

    The resource is considered live and valid until the provided timepoint is
    reached and the memory is only made available for future requests afterward.
  }];

  let arguments = (ins
    AnyTypeOf<[
      Stream_StagingResource,
      Stream_TransientResource,
    ]>:$operand,
    Stream_Size:$operand_size,
    Stream_Timepoint:$timepoint,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs);

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    `await` `(` $timepoint `)`
    `=` `` `>`
    $operand `:` type($operand) `{` $operand_size `}`
    attr-dict
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return operand_size(); }
    Value getResultSize(unsigned idx) { return {}; }
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
    Value getOperandSize(unsigned idx) { return result(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

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
    ByteBufferType:$source,
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
    Value getResultSize(unsigned idx) { return result_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

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
    ByteBufferType:$source,
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
    Value getResultSize(unsigned idx) { return result_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

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
    Value getOperandSize(unsigned idx) { return source_size(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let verifier = [{ return verifyOp(*this); }];

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
    Value getOperandSize(unsigned idx) { return target_size(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let verifier = [{ return verifyOp(*this); }];

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

  let verifier = [{ return verifyOp(*this); }];

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

  let verifier = [{ return verifyOp(*this); }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return result_sizes()[idx]; }
  }];
}

def Stream_ResourceSubviewOp : Stream_PureOp<"resource.subview", [
  AllTypesMatch<["source", "result"]>,
  DeclareOpInterfaceMethods<ViewLikeOpInterface>,
  DeclareOpInterfaceMethods<Stream_StreamableOp, [
    "isMetadata",
  ]>,
  Util_SizeAwareOp,
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
    Value getOperandSize(unsigned idx) { return source_size(); }
    Value getResultSize(unsigned idx) { return result_size(); }

    // Walks up the use-def chain to find a subview op that feeds into |value|.
    static IREE::Stream::ResourceSubviewOp findSubviewOp(Value value);
  }];

  let verifier = [{ return verifyOp(*this); }];

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
    ValueRange getResultDynamicDims(unsigned idx) { return result_encoding_dims(); }
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return result_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

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
    ValueRange getOperandDynamicDims(unsigned idx) { return source_encoding_dims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return ValueRange{}; }
    Value getOperandSize(unsigned idx) { return source_size(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Builtin tensor ops
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

  let verifier = [{ return verifyOp(*this); }];
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
    ValueRange getResultDynamicDims(unsigned idx) { return result_encoding_dims(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

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
    ValueRange getResultDynamicDims(unsigned idx) { return result_encoding_dims(); }
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return result_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

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
    ValueRange getOperandDynamicDims(unsigned idx) { return source_encoding_dims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return result_encoding_dims(); }
    Value getOperandSize(unsigned idx) { return source_size(); }
    Value getResultSize(unsigned idx) { return result_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

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
    ValueRange getOperandDynamicDims(unsigned idx) { return source_encoding_dims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return result_encoding_dims(); }
    Value getOperandSize(unsigned idx) { return source_size(); }
    Value getResultSize(unsigned idx) { return result_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

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
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands,
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
    custom<ShapedTiedResult>(type($target), $target_size, $tied_operands)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return target_encoding_dims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return target_encoding_dims(); }
    Value getOperandSize(unsigned idx) { return target_size(); }
    Value getResultSize(unsigned idx) { return target_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

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
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands,
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
    custom<ShapedTiedResult>(type($target), $target_size, $tied_operands)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) {
      return idx == 0 ? target_encoding_dims() : update_encoding_dims();
    }
    ValueRange getResultDynamicDims(unsigned idx) { return target_encoding_dims(); }
    Value getOperandSize(unsigned idx) {
      return idx == 0 ? target_size() : update_size();
    }
    Value getResultSize(unsigned idx) { return target_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

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
    ValueRange getOperandDynamicDims(unsigned idx) { return source_encoding_dims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return ValueRange{}; }
    Value getOperandSize(unsigned idx) { return source_size(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
}

def Stream_TensorStoreOp : Stream_Op<"tensor.store", [
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
    AnyTypeOf<[Stream_PrimitiveType, AnyVector]>:$value,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
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
    custom<ShapedTiedResult>(type($target), $target_size, $tied_operands)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return target_encoding_dims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return target_encoding_dims(); }
    Value getOperandSize(unsigned idx) { return target_size(); }
    Value getResultSize(unsigned idx) { return target_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
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
    Stream_TransientResource:$result
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    attr-dict `:` type($result) `{` $storage_size `}`
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return {}; }
    Value getResultSize(unsigned idx) { return storage_size(); }
  }];

  let hasCanonicalizer = 1;
}

def Stream_AsyncConstantOp : Stream_PureOp<"async.constant", [
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Stream_StreamableOp, [
    "isMetadata",
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
    Value getResultSize(unsigned idx) { return result_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
}

def Stream_AsyncSplatOp : Stream_Op<"async.splat", [
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Stream_StreamableOp, [
    "preferCloneToConsumers",
  ]>,
  Util_SizeAwareOp,
]> {
  let summary = [{splats a value into a resource}];
  let description = [{
    Returns a new resource with the given primitive value splatted out to fill
    the entire contents.
  }];

  let arguments = (ins
    Stream_PrimitiveType:$value,
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
    Value getResultSize(unsigned idx) { return result_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
}

def Stream_AsyncCloneOp : Stream_Op<"async.clone", [
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Stream_StreamableOp, [
    "preferCloneToConsumers",
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
    Value getOperandSize(unsigned idx) { return source_size(); }
    Value getResultSize(unsigned idx) { return result_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Stream_AsyncSliceOp : Stream_PureOp<"async.slice", [
  AllTypesMatch<["source", "result"]>,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Stream_StreamableOp, [
    "isMetadata",
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
    Value getOperandSize(unsigned idx) { return source_size(); }
    Value getResultSize(unsigned idx) { return result_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Stream_AsyncFillOp : Stream_Op<"async.fill", [
  AllTypesMatch<["target", "result"]>,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  Stream_StreamableOp,
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
    Stream_PrimitiveType:$value,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands,
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
    custom<ShapedTiedResult>(type($target), $target_size, $tied_operands)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) { return target_size(); }
    Value getResultSize(unsigned idx) { return target_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
}

def Stream_AsyncUpdateOp : Stream_Op<"async.update", [
  AllTypesMatch<["target", "result"]>,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  DeclareOpInterfaceMethods<Stream_StreamableOp, [
    "isMetadata",
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
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands,
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
    custom<ShapedTiedResult>(type($target), $target_size, $tied_operands)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return idx == 0 ? target_size() : update_size();
    }
    Value getResultSize(unsigned idx) { return target_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Stream_AsyncCopyOp : Stream_Op<"async.copy", [
  AllTypesMatch<["target", "result"]>,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  Stream_StreamableOp,
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
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands,
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
    custom<ShapedTiedResult>(type($target), $target_size, $tied_operands)
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return idx == 0 ? target_size() : source_size();
    }
    Value getResultSize(unsigned idx) { return target_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
}

def Stream_AsyncTransferOp : Stream_Op<"async.transfer", [
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  Stream_StreamableOp,
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
    Value getOperandSize(unsigned idx) { return source_size(); }
    Value getResultSize(unsigned idx) { return result_size(); }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Stream_AsyncDispatchOp : Stream_Op<"async.dispatch", [
  AttrSizedOperandSegments,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  Stream_StreamableOp,
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
    Variadic<Index>:$workgroup_count,
    SymbolRefAttr:$entry_point,
    Variadic<AnyTypeOf<[
      Stream_AnyStreamResource,
      Stream_PrimitiveType,
    ]>>:$operands,
    Variadic<Stream_Size>:$operand_sizes,
    Variadic<Stream_Size>:$result_sizes,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands,
    OptionalAttr<Stream_AffinityAttr>:$affinity
  );
  let results = (outs
    Variadic<Stream_AnyStreamResource>:$results
  );

  let assemblyFormat = [{
    (`on` `(` $affinity^ `)`)?
    $entry_point `[` $workgroup_count `]` ``
    `(` $operands `)` attr-dict `:`
    custom<ShapedFunctionType>(ref($operands),
                               type($operands), $operand_sizes,
                               type($results), $result_sizes,
                               $tied_operands)
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), operand_sizes());
    }
    Value getResultSize(unsigned idx) {
      return findValueSizeInList(idx, getResults(), result_sizes());
    }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Async control flow ops
//===----------------------------------------------------------------------===//

// TODO(benvanik): stream.async.if
// TODO(benvanik): stream.async.select
// TODO(benvanik): stream.async.for

def Stream_AsyncExecuteOp : Stream_Op<"async.execute", [
  AttrSizedOperandSegments,
  RecursiveSideEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getSuccessorEntryOperands",
  ]>,
  SingleBlockImplicitTerminator<"IREE::Stream::YieldOp">,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
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
    ]>>:$operands,
    Variadic<Stream_Size>:$operand_sizes,
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
    custom<ResourceRegion>($operands, type($operands), $operand_sizes,
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
      "ValueRange":$operands, "ValueRange":$operandSizes,
      "ArrayRef<int64_t>":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), operand_sizes());
    }
    Value getResultSize(unsigned idx) {
      return findValueSizeInList(idx, getResults(), result_sizes());
    }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
}

def Stream_AsyncConcurrentOp : Stream_Op<"async.concurrent", [
  ParentOneOf<[
    "IREE::Stream::AsyncExecuteOp",
    "IREE::Stream::AsyncConcurrentOp",
  ]>,
  AttrSizedOperandSegments,
  RecursiveSideEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getSuccessorEntryOperands",
  ]>,
  SingleBlockImplicitTerminator<"IREE::Stream::YieldOp">,
  Stream_AffinityOp,
  Stream_AsyncPhaseOp,
  Stream_StreamableOp,
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
    ]>>:$operands,
    Variadic<Stream_Size>:$operand_sizes,
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
    custom<ResourceRegion>($operands, type($operands), $operand_sizes,
                           type($results), $result_sizes,
                           $tied_operands, $body)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "TypeRange":$resultTypes, "ValueRange":$resultSizes,
      "ValueRange":$operands, "ValueRange":$operandSizes,
      "ArrayRef<int64_t>":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), operand_sizes());
    }
    Value getResultSize(unsigned idx) {
      return findValueSizeInList(idx, getResults(), result_sizes());
    }
  }];

  let verifier = [{ return verifyOp(*this); }];

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
    Value getOperandSize(unsigned idx) { return target_size(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let verifier = [{ return verifyOp(*this); }];

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
    Value getOperandSize(unsigned idx) { return target_size(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let verifier = [{ return verifyOp(*this); }];

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
    Value getOperandSize(unsigned idx) { return target_size(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let verifier = [{ return verifyOp(*this); }];

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
    Stream_PrimitiveType:$value
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
    Value getOperandSize(unsigned idx) { return target_size(); }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let verifier = [{ return verifyOp(*this); }];

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
      return idx == 0 ? source_size() : target_size();
    }
    Value getResultSize(unsigned idx) { return {}; }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
}

def Stream_CmdDispatchOp : Stream_Op<"cmd.dispatch", [
  AttrSizedOperandSegments,
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
    Variadic<Index>:$workgroup_count,
    SymbolRefAttr:$entry_point,
    Variadic<Stream_PrimitiveType>:$operands,
    Variadic<Stream_AnyStreamResource>:$resources,
    Variadic<Stream_Size>:$resource_sizes,
    Variadic<Stream_Offset>:$resource_offsets,
    Variadic<Stream_Size>:$resource_lengths,
    Stream_ResourceAccessArrayAttr:$resource_accesses
  );
  let results = (outs);

  let assemblyFormat = [{
    $entry_point `[` $workgroup_count `]` ``
    (`(` $operands^ `:` type($operands) `)`)? `{`
    custom<DispatchResources>($resources, type($resources), $resource_sizes,
                              $resource_offsets, $resource_lengths,
                              $resource_accesses)
    `\n` `}`
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(
          idx - getODSOperandIndexAndLength(2).first,
          resources(), resource_sizes());
    }
    Value getResultSize(unsigned idx) { return {}; }

    // Builds a map of operand index to argument index.
    static SmallVector<unsigned> makeOperandToArgMap(mlir::FuncOp funcOp);
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
}

def Stream_CmdExecuteOp : Stream_Op<"cmd.execute", [
  AttrSizedOperandSegments,
  RecursiveSideEffects,
  DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getSuccessorEntryOperands",
  ]>,
  SingleBlockImplicitTerminator<"IREE::Stream::YieldOp">,
  Stream_AffinityOp,
  Stream_CmdPhaseOp,
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
    ]>>:$operands,
    Variadic<Stream_Size>:$operand_sizes,
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
    custom<ExplicitResourceRegion>($operands, type($operands), $operand_sizes,
                                   $body)
    `=` `` `>` type($result_timepoint)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "Value":$awaitTimepoint,
      "ValueRange":$operands, "ValueRange":$operandSizes,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), operand_sizes());
    }
    Value getResultSize(unsigned idx) {
      return {};
    }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
}

// TODO(benvanik): stream.cmd.serial for serialized execution.
def Stream_CmdSerialOp : Stream_Op<"cmd.serial", [
  ParentOneOf<[
    "IREE::Stream::CmdExecuteOp",
    "IREE::Stream::CmdSerialOp",
    "IREE::Stream::CmdConcurrentOp",
  ]>,
  RecursiveSideEffects,
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
    `` $body
    attr-dict-with-keyword
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
}

def Stream_CmdConcurrentOp : Stream_Op<"cmd.concurrent", [
  ParentOneOf<[
    "IREE::Stream::CmdExecuteOp",
    "IREE::Stream::CmdSerialOp",
    "IREE::Stream::CmdConcurrentOp",
  ]>,
  RecursiveSideEffects,
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
    `` $body
    attr-dict-with-keyword
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Synchronization ops
//===----------------------------------------------------------------------===//

def Stream_TimepointImmediateOp : Stream_PureOp<"timepoint.immediate", [
  ConstantLike,
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
    Stream_Timepoint:$timepoint
  );

  let assemblyFormat = [{
    attr-dict
    `=` `` `>` type($timepoint)
  }];

  let hasFolder = 1;
}

def Stream_TimepointJoinOp : Stream_PureOp<"timepoint.join", []> {
  let summary = [{joins one or more timepoints into the max of all of them}];
  let description = [{
    Returns a timepoint that indicates that all of the input timepoints have
    been reached.
  }];

  let arguments = (ins
    Variadic<Stream_Timepoint>:$timepoints
  );
  let results = (outs
    Stream_Timepoint:$result
  );

  let assemblyFormat = [{
    `max` `(` $timepoints `)` `=` `` `>` type($result)
    attr-dict-with-keyword
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Stream_TimepointAwaitOp : Stream_PureOp<"timepoint.await", [
  AttrSizedOperandSegments,
  Stream_AffinityOp,
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
    ]>>:$operands,
    Variadic<Stream_Size>:$operand_sizes,
    Stream_Timepoint:$timepoint,
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
    $timepoint `=` `` `>`
    $operands `:`
    custom<SizeAwareTypeList>(type($operands), type($results), $operand_sizes)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "ValueRange":$operands, "ValueRange":$operandSizes,
      "Value":$timepoint,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return operand_sizes()[idx];
    }
    Value getResultSize(unsigned idx) {
      return operand_sizes()[idx];
    }
  }];

  let verifier = [{ return verifyOp(*this); }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
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
    ``
    regions
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$sym_name)>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return body().front(); }
    ::mlir::ModuleOp getInnerModule() {
      return *getBlock().getOps<::mlir::ModuleOp>().begin();
    }
  }];

  let verifier = [{ return verifyOp(*this); }];
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
]> {
  let summary = [{defines an executable entry point for dispatch operations}];
  let description = [{
    Specifies an exported function with an externally-visible alias. Multiple
    exports can reference the same internal function.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$function_ref
  );

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    custom<SymbolAlias>($sym_name, $function_ref)
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins
      "StringRef":$sym_name,
      "FlatSymbolRefAttr":$function_ref
    )>,
  ];
}

def Stream_BindingSubspanOp : Stream_PureOp<"binding.subspan", [
    DeclareOpInterfaceMethods<Shape_ShapeCarryingOpInterface>,
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

  let verifier = [{ return verifyOp(*this); }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return dynamic_dims(); }
  }];
}

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

def Stream_YieldOp : Stream_Op<"yield", [
  ParentOneOf<[
    "IREE::Stream::AsyncExecuteOp",
    "IREE::Stream::AsyncConcurrentOp",
    "IREE::Stream::CmdExecuteOp",
    "IREE::Stream::CmdSerialOp",
    "IREE::Stream::CmdConcurrentOp",
  ]>,
  NoSideEffect,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface, [
    "getMutableSuccessorOperands",
  ]>,
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
    ]>>:$operands,
    Variadic<Index>:$operand_sizes
  );

  let assemblyFormat = [{
    attr-dict
    ($operands^ `:` custom<SizeAwareTypeList>(type($operands), $operand_sizes))?
  }];

  let builders = [
    OpBuilder<(ins),
    [{
      build($_builder, $_state, ValueRange{}, ValueRange{});
    }]>,
  ];

  let extraClassDeclaration = [{
    Value getOperandSize(unsigned idx) {
      return findValueSizeInList(idx, getOperands(), operand_sizes());
    }
    Value getResultSize(unsigned idx) { return {}; }
  }];
}

#endif  // IREE_DIALECT_STREAM_OPS
