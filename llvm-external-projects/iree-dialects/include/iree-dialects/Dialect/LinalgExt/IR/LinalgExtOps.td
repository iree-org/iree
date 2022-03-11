// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_OPS
#define IREE_DIALECT_LINALGEXT_OPS

include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtBase.td"
include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtInterfaces.td"
include "iree-dialects/Dialect/LinalgExt/IR/TiledOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"


//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class IREELinalgExt_PureOp<string mnemonic, list<Trait> traits = []> :
    Op<IREELinalgExt_Dialect, mnemonic, traits> {
}

class IREELinalgExt_Op<string mnemonic, list<Trait> traits = []> :
    IREELinalgExt_PureOp<mnemonic, !listconcat(traits,
        [AttrSizedOperandSegments,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
         LinalgExtInterface,
         SingleBlockImplicitTerminator<"::mlir::iree_compiler::IREE::LinalgExt::YieldOp">
  ])> {
  let verifier = [{ return verify$cppClass(*this); }];
  let printer = [{ return print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
  code extraLinalgExtOpClassDeclaration = [{
    SmallVector<Value> getDestinationOperands(OpBuilder &b) {
      SmallVector<Value> dest(outputs().begin(), outputs().end());
      return dest;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Non-structured ops
//===----------------------------------------------------------------------===//

def IREELinalgExt_ScatterOp : IREELinalgExt_Op<"scatter",
    [DeclareOpInterfaceMethods<TiledOpInterface,
        ["getTiledImplementation", "generateScalarImplementation"]>]> {
  let summary = "Scatter operator";
  let description = [{
    Based on XLA operation semantics, takes two `inputs` (`update` and
    `indices`) and `outputs` value (`original`). The operation updates
    the value at the slices specified by `indices` by combining the
    current value with the value in `updates` using the computation
    specified in `region`. The `region` specifies a binary operation
    of signature (T, T) -> T, where `T` is the element-type of
    `updates` (and `original`). The first argument correspond the
    value to be updated (i.e. from `updates`), and the second the
    current value (i.e. value from `original`).

    The `indices` is a 2D tensor/memref type. The first dim is the number of
    updates, and the second dim is index depth. The index depth should always be
    static.

    The first dim of `updates` and `indices` is identical, since they represent
    the number of updates.

    The rank of the `original`/`result` is at least
    `index_depth + rank(%updates) - 1`. The first `index_depth` indices are
    derived from `indices` and the shape of update value has the last
    rank(%original) - index_depth values match %(originals) last dimensions,
    with the previous dims extending from the index offsets.

    The unique_indices attribute carries the information whether all the indices
    are unique. If there are repeated indices, the first iteration loop will be
    marked as reduction.

    The shapes definition follows tensorflow operations execept that it force
    batch dims to be 1D. See more information in
      https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update
  }];
  let arguments = (ins
      Variadic<AnyRankedTensorOrMemRefType>:$inputs,
      Variadic<AnyRankedTensorOrMemRefType>:$outputs,
      DefaultValuedAttr<BoolAttr, "true">:$unique_indices
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    attr-dict `unique_indices` `(` $unique_indices `)`
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];
  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{

    int64_t getIndexDepth() {
      return getInputOperand(1)
          ->get()
          .getType()
          .cast<ShapedType>()
          .getShape()
          .back();
    }

    Value updates() {
      return getInputOperand(0)->get();
    }

    ShapedType getUpdateType() {
      return updates().getType().cast<ShapedType>();
    }

    Value indices() {
      return getInputOperand(1)->get();
    }

    ShapedType getIndicesType() {
      return indices().getType().cast<ShapedType>();
    }

    Value original() {
      return getOutputOperand(0)->get();
    }

    ShapedType getOriginalType() {
      return original().getType().cast<ShapedType>();
    }

    int64_t getUpdateSliceRank() {
      return updates().getType().cast<ShapedType>().getRank() - 1;
    }

    bool isScalarUpdate() {
      return getUpdateSliceRank() == 0;
    }
  }];
}

def IREELinalgExt_SortOp : IREELinalgExt_Op<"sort",
    [DeclareOpInterfaceMethods<TiledOpInterface,
        ["getPartitionableLoops", "generateScalarImplementation",
         "getTiledImplementation"]>]> {
  let summary = "Sort operator";
  let description = [{
    Based on XLA operation semantics, sorts the given `operands` at the given
    `dimension` with the given `comparator`.

    See https://www.tensorflow.org/xla/operation_semantics#sort.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       I64Attr:$dimension
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    attr-dict
    `dimension` `(` $dimension `)`
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];
  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value operand(int index) {
      return outputs()[index];
    }
    ShapedType getOperandType(int index) {
      return operand(index).getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType(0).getRank();
    }
    ArrayRef<int64_t> getOperandShape() {
      return getOperandType(0).getShape();
    }
  }];
}

def IREELinalgExt_FftOp : IREELinalgExt_Op<"fft", [
  DeclareOpInterfaceMethods<TiledOpInterface,
                            [
                              "getPartitionableLoops", "getTiledImplementation",
                              "generateScalarImplementation"
                            ]>,
  DeclareOpInterfaceMethods<LinalgExtInterface>
]> {
  let summary = "Fft operator";
  let description = [{
    Apply 1D FFT to innermost dim. This is an iterative FFT, not recurrsive.
    Thus, the bit reversal is assumed applied on the input. The op carries an
    input -- stage, which indicates the level of reduction loop in the
    algorithm. It represents the computation body. For more details, see
    "Data reordering, bit reversal, and in-place algorithms" section in
    https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm

    The size of innermost dim is expected to be a power of 2.

    It is optional to carry coefficient tensors/buffers as inputs. In this
    context, they will be the second and third inputs.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs,
                       Variadic<AnyShaped>:$outputs
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    attr-dict (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    `outs` `(` $outputs `:` type($outputs) `)`
    (`:` type($results)^)?
  }];
  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value getStage() { return inputs()[0]; }
    Value getReal() { return outputs()[0]; }
    Value getImag() { return outputs()[1]; }
    bool hasCoeff() { return getNumInputs() > 1; }
    void generateScalarImplWithoutCoeffBuf(
        OpBuilder & b, Location loc, ArrayRef<Value> operands, Value wholeSize);
    void generateScalarImplWithCoeffBuf(OpBuilder & b, Location loc,
                                        ArrayRef<Value> operands);
    Value getRealCoeff() {
      if (!hasCoeff()) return Value();
      return inputs()[1];
    }
    Value getImagCoeff() {
      if (!hasCoeff()) return Value();
      return inputs()[2];
    }
    ShapedType getOperandType() {
      return getReal().getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }
    ArrayRef<int64_t> getOperandShape() {
      return getOperandType().getShape();
    }
    int64_t getFftLength() {
      return getOperandShape().back();
    }
  }];
}

def IREELinalgExt_ScanOp : IREELinalgExt_Op<"scan",
    [DeclareOpInterfaceMethods<TiledOpInterface,
      ["getPartitionableLoops", "generateScalarImplementation",
       "getTiledImplementation"]>]> {
  let summary = "Scan operator";
  let description = [{
    Computes the inclusive/exclusive scan along a given dimension.
  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       I64Attr:$dimension,
                       BoolAttr:$inclusive
  );

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs,
      CArg<"int64_t", "0">:$dimension, CArg<"bool", "true">:$inclusive)>
  ];

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `dimension` `(` $dimension `)`
    `inclusive` `(` $inclusive `)`
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value input() {
      return getInputOperand(0)->get();
    }
    Value accumulator() {
      return getOutputOperand(1)->get();
    }
    Value output() {
      return getOutputOperand(0)->get();
    }
    ShapedType getOperandType() {
      return input().getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }
  }];
}

def IREELinalgExt_ReverseOp : IREELinalgExt_Op<"reverse", [
  DeclareOpInterfaceMethods<
      TiledOpInterface,
      ["generateScalarImplementation", "getTiledImplementation"]>,
  DeclareOpInterfaceMethods<LinalgExtInterface>]> {
  let summary = "Reverse operator";
  let description = [{
    A temporary solution for lowering reverse ops into IREE, allowing IREE to
    tile and distribute them.
    }
  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       I64ElementsAttr:$dimensions
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    attr-dict `dimensions` `(` $dimensions `)`
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^ `:` type($outputs) `)`)?
    (`:` type($results)^)?
  }];
  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value input() {
      return getInputOperand(0)->get();
    }
    Value output() {
      return getOutputOperand(0)->get();
    }
    ShapedType getOperandType() {
      return input().getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }
    ArrayRef<int64_t> getOprerandShape() {
      return getOperandType().getShape();
    }
    SmallVector<int64_t> dims() {
      SmallVector<int64_t> ret;
      for (const APInt& elem : dimensions()) {
        ret.push_back(elem.getLimitedValue());
      }
      return ret;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Pure ops
//===----------------------------------------------------------------------===//

def IREELinalgExt_YieldOp : IREELinalgExt_PureOp<"yield", [NoSideEffect, ReturnLike, Terminator]> {
  let summary = "LinalgExt yield op";
  let description = [{
    `iree_linalg_ext.yield` is a special terminator operation for blocks inside
    regions in `iree_linalg_ext` ops.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// Ops supporting concurrency with tensors.
//===----------------------------------------------------------------------===//

def IREELinalgExt_TileOp : IREELinalgExt_PureOp<"tile",
      [
       // TODO: enable to allow hoisting, LICM and isDefinedOutside
       // DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       SingleBlockImplicitTerminator<"::mlir::iree_compiler::IREE::LinalgExt::TileYieldOp">,
       RecursiveSideEffects
      ]> {
  let summary = "tile operation";
  let description = [{
    `iree_linalg_ext.tile` is a 1-D loop construct that operates on tensors and
    evaluates its body once for each tile. The number and size of tiles is
    specified by the `tile_size` operand.

    The `tile` op takes a list of destination-passing style tensors and returns
    a matching list of tensors of the same size.

    Every instance of the body is expected to return a tile with leading
    dimension matching the corresponding tile size.

    The default terminator behavior is such that tiles yielded by individual
    iterations are concatenated along the `tiled_dim` dimension.
    This is the canonical way to perform "subset insertions".
    Note, if `tiled_dim` has the value `0`, it may be elided from pretty
    pinting and parsing.

    All return tiles are concatenated into forming the matching full result
    tensor according to the terminator.

    When the `tile_size` operand is a `tensor<..index>`, the `tile` op
    evaluates its body `dim(tile_size, 0)` times. Each iteration `i` produces a
    tile of leading size `tile_size[i]`.

    The induced `offset` block argument captures the running sum of `tile_size`
    for all the previous iterations.

    When the `tile_size` operand is a single index, it is interpreted as a
    sequence of tile sizes given by the following formula:
    ```
      N = tensor.dim(...)
      S = sizes
      T, R = divmod(N, S)
      [T] * S + ([R] if R != 0 else [])
    ```

    All tiles except the last are of the same size.
  }];
  let arguments = (ins AnyTypeOf<[// TODO: allow TensorOf<[Index]>,
                                  Index]>:$tile_size,
                       Variadic<AnyRankedTensor>:$outs,
                       I64Attr:$tiled_dim);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);
  let skipDefaultBuilders = 1;
  let builders = [
    // Builder that builds a tile on the implicit first dimension (i.e. `0`).
    OpBuilder<(ins "Value":$tileSizes, "ValueRange":$outs,
      CArg<"function_ref<void(OpBuilder &, Location, Value, Value, ValueRange)>",
           "nullptr">)>,
    // Builder that builds a tile with a specified integral dimension.
    OpBuilder<(ins "Value":$tileSizes, "ValueRange":$outs, "int64_t":$tiledDims,
      CArg<"function_ref<void(OpBuilder &, Location, Value, Value, ValueRange)>",
           "nullptr">)>,
  ];

  let extraClassDeclaration = [{
    static StringRef getTiledDimAttrName() { return "tiled_dim";}
    using TileOpBodyBuilderFn =
      function_ref<void(OpBuilder &, Location, Value /*offset*/, Value /*size*/,
                        ValueRange /*outs*/)>;
    // TODO: helper for getting named region args without magic constants etc.
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def IREELinalgExt_TileYieldOp : IREELinalgExt_PureOp<"tile_yield", [
    NoSideEffect, ReturnLike, Terminator]> {
  let summary = "LinalgExt tile_yield op";
  let description = [{
    `iree_linalg_ext.tile_yield` is a special terminator operation for blocks inside
    regions in `iree_linalg_ext.tile`.
    The tiles yielded by individual iterations are concatenated along the first
    dimension. This is the canonical way to perform "subset insertions"
    (TODO: allow dim permutations).
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def IREELinalgExt_InParallelOp : IREELinalgExt_PureOp<"in_parallel", [
       SingleBlockImplicitTerminator<"::mlir::iree_compiler::IREE::LinalgExt::PerformConcurrentlyOp">,
       RecursiveSideEffects,
       AutomaticAllocationScope,
      ]> {
  let summary = "evaluate a block multiple times in parallel";
  let description = [{
    `iree_linalg_ext.in_parallel` is a target-independent parallel function application
    operation. It has exactly one block that represents the parallel function body
    and it takes a single index operand that indicates how many parallel instances
    of that function should get instantiated.

    When the parallel function body is pure (i.e. has no side effects) then the only
    allowed terminator is `iree_linalg_ext.perform_concurrently`, which dictates
    how the results of all parallel invocations should be reconciled into a full
    value that will be returned from `in_parallel`. Multi-value returns are encoded
    by including multiple operations inside the `perform_concurrently` block.

    When the parallel function body has side effects, the order of reads and writes
    to memory is unspecified across iterations.

    This op resembles `scf.for` to a large degree, but crucially differs in that it
    (1) doesn't have `iter_args` and (2) has a special terminator, both of which
    enable reasoning about its parallel semantics. Another difference is that
    `in_parallel` always iterates over a range between 0 and an upper bound, but
    that's insignificant.
  }];
  let arguments = (ins Index:$num_threads);

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  // The default builder does not add the proper body BBargs, roll our own.
  let skipDefaultBuilders = 1;
  let builders = [
    // Bodyless builder, result types must be specified.
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$num_threads)>,
    // Builder that takes a bodyBuilder lambda, result types are inferred from
    // the terminator.
    OpBuilder<(ins "Value":$num_threads,
              "function_ref<void(OpBuilder &, Location, Value)>":$bodyBuilder)>
  ];
  let extraClassDeclaration = [{
    Value getThreadIndex() { return getBody()->getArgument(0); }
    static void ensureTerminator(Region &region, Builder &builder, Location loc);
    PerformConcurrentlyOp getTerminator();
  }];
}

def IREELinalgExt_PerformConcurrentlyOp : IREELinalgExt_PureOp<"perform_concurrently", [
       NoSideEffect,
       Terminator,
       SingleBlockImplicitTerminator<"::mlir::iree_compiler::IREE::LinalgExt::EndPerformConcurrentlyOp">,
      ]> {
  let summary = "terminates a `in_parallel` block";
  let description = [{
    `iree_linalg_ext.perform_concurrently` is a designated terminator for the blocks
    of `iree_linalg_ext.in_parallel` operations. The terminator contains a single block
    itself, which describes how the results of each parallel invocation are to be
    reconciled into a single value to be returned from the parallel invocation.
    One operation in this terminator's block corresponds to a single return of
    `in_parallel`.
  }];

  let regions = (region SizedRegion<1>:$region);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  // TODO(apaszke, ntv): Add an interface for ops that can appear inside
  // perform_concurrently.
  let extraClassDeclaration = [{
    SmallVector<Type> yieldedTypes();
    SmallVector<ParallelInsertSliceOp> yieldingOps();
  }];
}

def IREELinalgExt_EndPerformConcurrentlyOp : IREELinalgExt_PureOp<"end_perform_concurrently", [
       NoSideEffect, Terminator]> {
  let summary = "terminates a `perform_concurrently` block";
  let description = [{
    A designated terminator for `perform_concurrently`. It's not expected to appear
    in the textual form of the IR.
  }];
}

def IREELinalgExt_ParallelInsertSliceOp : IREELinalgExt_PureOp<"parallel_insert_slice", [
       AttrSizedOperandSegments, OffsetSizeAndStrideOpInterface]> {
  let summary = "updates slices of a tensor concurrently";
  let description = [{
    Updates slices of a full tensor with multiple sub-slices concurrently.

    Conflicting writes result in undefined semantics, in that the indices written
    to by multiple parallel updates might contain data from any of the updates, or
    even a malformed bit pattern (in reality the semantics might end up depending
    on the memory model of the parallel hardware that `in_parallel` will be lowered to).

    If an index is updated by exactly one updates, the value contained at that index
    in the resulting tensor will be equal to the value at a corresponding index of a
    slice that was used for the updated. If an index is not updated at all, its value
    will be equal to the one in the original tensor.

    Note that we cannot mark this operation as pure (NoSideEffects), even
    though it has no side effects, because it will get DCEd during
    canonicalization. Ideally we would use attributes instead of those funny
    terminating ops, but attributes cannot refer to SSA values at the moment, so
    it's the best we can do for now.
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    AnyRankedTensor:$dest,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    I64ArrayAttr:$static_offsets,
    I64ArrayAttr:$static_sizes,
    I64ArrayAttr:$static_strides
  );
  let assemblyFormat = [{
    $source `into` $dest ``
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    custom<OperandsOrIntegersSizesList>($sizes, $static_sizes)
    custom<OperandsOrIntegersOffsetsOrStridesList>($strides, $static_strides)
    attr-dict `:` type($source) `into` type($dest)
  }];

  let extraClassDeclaration = [{
    Type yieldedType() { return dest().getType(); }

    RankedTensorType getSourceType() {
      return source().getType().cast<RankedTensorType>();
    }

    /// Return the expected rank of each of the `static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before `offsets`, `sizes` and
    /// `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  }];

  let builders = [
    // Build a ParallelInsertSliceOp with mixed static and dynamic entries.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a ParallelInsertSliceOp with dynamic entries.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let hasCanonicalizer = 1;
}

#endif  // IREE_DIALECT_LINALGEXT_OPS
