// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_PCF_OPS
#define IREE_CODEGEN_DIALECT_PCF_OPS

include "iree/compiler/Codegen/Dialect/PCF/IR/PCFBase.td"
include "iree/compiler/Utils/CommonTypeConstraints.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/Properties.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Alloc
//===----------------------------------------------------------------------===//

def OpGroupAllocOps : OpDocGroup {
  let summary = "Alloc ops";
  let description = "";
}

let opDocGroup = OpGroupAllocOps in {

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

def AllocOp : PCF_Op<"alloc", []> {
  let summary = [{
    Shaped ref allocation operation
  }];
  let description = [{
    Allocates a `pcf.sref` with the given element type and shape. Dynamic
    dimensions in the result type must have corresponding dynamic size
    operands. The allocation scope is determined by the scope attribute of
    the result type.

    Example:
    ```mlir
      %sref = pcf.alloc() : !pcf.sref<4x8xf32, #foo.scope>
      %sref_dyn = pcf.alloc(%d0, %d1) : !pcf.sref<?x?xf32, #foo.scope>
    ```
  }];

  let arguments = (ins Variadic<Index>:$dynamicSizes);
  let results = (outs PCF_AnyShapedRef:$result);

  let builders = [
    OpBuilder<(ins "ShapedRefType":$srefType), [{
      return build($_builder, $_state, srefType, {});
    }]>
  ];

  let assemblyFormat = [{
    `(`$dynamicSizes`)` attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    ShapedRefType getResultType() {
      return cast<ShapedRefType>(getResult().getType());
    }
    SmallVector<OpFoldResult> getMixedSizes();
  }];

  let hasVerifier = 1;
}

} // OpGroupAllocOps

//===----------------------------------------------------------------------===//
// ParallelOps
//===----------------------------------------------------------------------===//

def OpGroupParallelOps : OpDocGroup {
  let summary = "Parallel execution ops";
  let description = "";
}

let opDocGroup = OpGroupParallelOps in {

def GenericOp : PCF_Op<"generic", [
       AttrSizedOperandSegments,
       AutomaticAllocationScope,
       RecursiveMemoryEffects,
       DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
     ]> {
  let summary = [{
    Execute a set of workers in parallel on a region.
  }];
  let description = [{
    Executes a region across a set of workers at a specified scope. When
    control flow reaches this op, `nproc` workers of the specified scope are
    spawned and begin executing the region. The scope is given by an attribute
    implementing the `ScopeAttrInterface` interface and is responsible for the semantics
    of all pcf primitives at the same scope. Further details about scopes are
    included in the docs for the interface.

    The optional `initialize` region is executed once when control flow first
    reaches the op. Values yielded from the initializer become block arguments
    available to the execute region. This is useful for setting up per-op
    state that persists across all worker invocations.

    Results are produced by snapshotting the value of each result's tied sref
    once all workers have returned. Results can either be:
    1. Tied to initial values (tensor or memref) - the init value provides the
       initial contents and the result captures the final state.
    2. Allocated by the op itself - dynamic sizes must be provided for
       untied results with dynamic dimensions.

    Basic example with tied results:
    ```mlir
      %0 = ... : tensor<4x8xf32>
      %1 = pcf.generic scope(#foo.scope)
        execute(%ref = %0)[%id: index, %num_workers: index]
             : (!pcf.sref<4x8xf32, #foo.scope>) -> (tensor<4x8xf32>) {
        // Each worker can read/write %ref.
        pcf.return
      }
    ```

    Example with initializer:
    ```mlir
      %result = pcf.generic scope(#foo.scope)
        initialize {
          %scratch = pcf.alloc() : !pcf.sref<16xf32, #foo.scope>
          pcf.yield %scratch : !pcf.sref<16xf32, #foo.scope>
        } -> (%scratch_arg: !pcf.sref<16xf32, #foo.scope>)
        execute(%ref = %init)[%id: index, %num_workers: index]
             : (!pcf.sref<4x8xf32, #foo.scope>) -> (tensor<4x8xf32>) {
        // %scratch_arg is available here, initialized once.
        pcf.return
      }
    ```

    Example with untied (allocated) results:
    ```mlir
      %d0, %d1 = ... : index
      %result = pcf.generic scope(#foo.scope)
        execute[%id: index, %num_workers: index]
             : () -> (tensor<?x?xf32>{%d0, %d1}) {
        // Result sref is allocated by the op, not tied to any init.
        pcf.return
      }
    ```
  }];
  let arguments = (ins PCF_ScopeAttrInterface:$scope,
      Variadic<AnyRankedTensorOrMemRef>:$inits, Variadic<Index>:$dynamic_sizes,
      ArrayProp<BoolProp>:$is_tied,
      DefaultValuedProp<BoolProp, "false">:$sync_on_return,
      IntProp<"int64_t">:$num_index_args, IntProp<"int64_t">:$num_leading_args);

  let results = (outs Variadic<AnyRankedTensorOrMemRef>:$results);
  let regions = (region
    MaxSizedRegion<1>:$initializer,
    MinSizedRegion<1>:$region
  );

  let assemblyFormat = [{
    (`sync` $sync_on_return^)?
    `scope` `(` $scope `)`
    (`initialize` $initializer^)?
    custom<ParallelExecutionBody>($inits,
                                  type($inits),
                                  $dynamic_sizes,
                                  type($results),
                                  $is_tied,
                                  $region,
                                  $num_leading_args,
                                  "true")
    custom<InferNumIndexArgs>(ref($region), ref($num_leading_args), $num_index_args)
    prop-dict attr-dict
  }];
  let hasVerifier = 1;

  // The default builder does not add the proper body BBargs, roll our own.
  let skipDefaultBuilders = 1;
  let builders = [
      // Builder with no return values.
      OpBuilder<(ins "ScopeAttrInterface":$scope, "int64_t":$num_iterators,
          CArg<"bool", "false">:$sync_on_return)>,
      // Builder with return types inferred from inits.
      OpBuilder<(ins "ScopeAttrInterface":$scope, "ValueRange":$inits,
          "int64_t":$num_iterators, CArg<"bool", "false">:$sync_on_return)>,
      // Builder with no inits.
      OpBuilder<(ins "TypeRange":$result_types, "ScopeAttrInterface":$scope,
          "ValueRange":$dynamic_sizes, "int64_t":$num_iterators,
          CArg<"bool", "false">:$sync_on_return)>,
      // Builder with everything.
      OpBuilder<(ins "TypeRange":$result_types, "ScopeAttrInterface":$scope,
          "ValueRange":$inits, "ValueRange":$dynamic_sizes,
          "ArrayRef<bool>":$is_tied, "int64_t":$num_iterators,
          CArg<"bool", "false">:$sync_on_return)>,
  ];

  let extraClassDeclaration = [{
    int64_t getNumIterators() {
      return getNumIndexArgs() / 2;
    }
    MutableArrayRef<BlockArgument> getLeadingArgsMutable() {
      return getRegion().getArguments().take_front(getNumLeadingArgs());
    }
    ArrayRef<BlockArgument> getLeadingArgs() {
      return getLeadingArgsMutable();
    }
    MutableArrayRef<BlockArgument> getIdArgsMutable() {
      return getRegion().getArguments().take_back(getNumIndexArgs()).take_front(getNumIterators());
    }
    ArrayRef<BlockArgument> getIdArgs() {
      return getIdArgsMutable();
    }
    MutableArrayRef<BlockArgument> getCountArgsMutable() {
      return getRegion().getArguments().take_back(getNumIterators());
    }
    ArrayRef<BlockArgument> getCountArgs() {
      return getCountArgsMutable();
    }
    MutableArrayRef<BlockArgument> getIdAndCountArgsMutable() {
      return getRegion().getArguments().take_back(getNumIndexArgs());
    }
    ArrayRef<BlockArgument> getIdAndCountArgs() {
      return getIdAndCountArgsMutable();
    }
    MutableArrayRef<BlockArgument> getRegionRefArgsMutable() {
      return getRegion().getArguments().drop_front(getNumLeadingArgs()).take_front(getNumResults());
    }
    ArrayRef<BlockArgument> getRegionRefArgs() {
      return getRegionRefArgsMutable();
    }

    bool isRegionRefArg(BlockArgument b);
    bool isInitializedArg(BlockArgument b) {
      assert(b.getOwner() == &getRegion().front() && "unexpected non-entry block arg");
      return b.getArgNumber() < getNumLeadingArgs();
    }
    SmallVector<int64_t> getInitTiedResultIndices();
    OpResult getTiedResult(OpOperand &operand);
    OpResult getTiedResult(BlockArgument b) {
      assert(isRegionRefArg(b) && "unexpected non region ref arg");
      return (*this)->getOpResult(b.getArgNumber() - getNumLeadingArgs());
    }
    OpOperand *getTiedInit(int64_t i);
    ShapedType getResultType(int64_t i) {
      return cast<ShapedType>(getResults()[i].getType());
    }
    ValueRange getResultDims(int64_t i);
  }];
}

def LoopOp : PCF_Op<"loop", [
       AttrSizedOperandSegments,
       AutomaticAllocationScope,
       RecursiveMemoryEffects,
       DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
       DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
       SingleBlockImplicitTerminator<"mlir::iree_compiler::IREE::PCF::ReturnOp">
     ]> {
  let summary = [{
    Execute a set of workers in parallel on a region.
  }];
  let description = [{
    Executes a region for each point in the iteration space defined by the
    `count` operands. Unlike `pcf.generic` which spawns workers equal to the
    native parallelism of the scope, `pcf.loop` explicitly specifies the
    iteration count and maps iterations to workers according to the scope's
    scheduling policy.

    When control flow reaches this op, the scope determines how to distribute
    the iterations across available workers. The scope is given by an attribute
    implementing the `ScopeAttrInterface` interface. Further details about scopes are
    included in the docs for the interface.

    The execute region receives one index block argument per count operand,
    representing the current iteration's coordinates in the iteration space.

    Results are produced by snapshotting the value of each result's tied sref
    once all iterations have completed. Results can either be:
    1. Tied to initial values (tensor or memref) - the init value provides the
       initial contents and the result captures the final state.
    2. Allocated by the op itself - dynamic sizes must be provided for
       untied results with dynamic dimensions.

    Basic example with 1D iteration:
    ```mlir
      %n = ... : index
      %0 = ... : tensor<4x8xf32>
      %1 = pcf.loop scope(#foo.scope) count(%n)
        execute(%ref = %0)[%id: index]
             : (!pcf.sref<4x8xf32, #foo.scope>) -> (tensor<4x8xf32>) {
        // %id ranges from 0 to %n-1.
        pcf.return
      }
    ```

    Example with multi-dimensional iteration:
    ```mlir
      %m, %n = ... : index
      %result = pcf.loop scope(#foo.scope) count(%m, %n)
        execute(%ref = %init)[%i: index, %j: index]
             : (!pcf.sref<?x?xf32, #foo.scope>) -> (tensor<?x?xf32>) {
        // %i ranges from 0 to %m-1, %j ranges from 0 to %n-1.
        pcf.return
      }
    ```
  }];
  let arguments = (ins PCF_ScopeAttrInterface:$scope, Variadic<Index>:$count,
      Variadic<AnyRankedTensorOrMemRef>:$inits, Variadic<Index>:$dynamic_sizes,
      ArrayProp<BoolProp>:$is_tied,
      DefaultValuedProp<BoolProp, "false">:$sync_on_return);

  let results = (outs Variadic<AnyRankedTensorOrMemRef>:$results);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    (`sync` $sync_on_return^)?
    `scope` `(` $scope `)`
    `count` `(` $count `)`
    custom<ParallelExecutionBody>($inits,
                                  type($inits),
                                  $dynamic_sizes,
                                  type($results),
                                  $is_tied,
                                  $region)
    prop-dict attr-dict
  }];
  let hasVerifier = 1;

  // The default builder does not add the proper body BBargs, roll our own.
  let skipDefaultBuilders = 1;
  let builders = [
      // Builder with no return values.
      OpBuilder<(ins "ScopeAttrInterface":$scope, "ValueRange":$count,
          CArg<"bool", "false">:$sync_on_return)>,
      // Builder with return types inferred from inits.
      OpBuilder<(ins "ScopeAttrInterface":$scope, "ValueRange":$count,
          "ValueRange":$inits, CArg<"bool", "false">:$sync_on_return)>,
      // Builder with no inits.
      OpBuilder<(ins "TypeRange":$result_types, "ScopeAttrInterface":$scope,
          "ValueRange":$count, "ValueRange":$dynamic_sizes,
          CArg<"bool", "false">:$sync_on_return)>,
      // Builder with everything.
      OpBuilder<(ins "TypeRange":$result_types, "ScopeAttrInterface":$scope,
          "ValueRange":$count, "ValueRange":$inits, "ValueRange":$dynamic_sizes,
          "ArrayRef<bool>":$is_tied, CArg<"bool", "false">:$sync_on_return)>,
  ];

  let extraClassDeclaration = [{
    int64_t getNumIdArgs() {
      return getCount().size();
    }
    MutableArrayRef<BlockArgument> getIdArgsMutable() {
      return getRegion().getArguments().take_back(getNumIdArgs());
    }
    ArrayRef<BlockArgument> getIdArgs() {
      return getIdArgsMutable();
    }
    MutableArrayRef<BlockArgument> getRegionRefArgsMutable() {
      return getRegion().getArguments().take_front(getNumResults());
    }
    ArrayRef<BlockArgument> getRegionRefArgs() {
      return getRegionRefArgsMutable();
    }

    SmallVector<int64_t> getInitTiedResultIndices();
    OpResult getTiedResult(OpOperand &operand);
    OpResult getTiedResult(BlockArgument b) {
      return (*this)->getOpResult(b.getArgNumber());
    }
    OpOperand *getTiedInit(int64_t i);
    ShapedType getResultType(int64_t i) {
      return cast<ShapedType>(getResults()[i].getType());
    }
    ValueRange getResultDims(int64_t i);
  }];
}

} // OpGroupParallelOps


//===----------------------------------------------------------------------===//
// Control Flow Ops
//===----------------------------------------------------------------------===//

def OpGroupControlFlowOps : OpDocGroup {
  let summary = "Parallel execution ops";
  let description = "";
}

let opDocGroup = OpGroupControlFlowOps in {

//===----------------------------------------------------------------------===//
// BranchCondReturnOp
//===----------------------------------------------------------------------===//

def BranchCondReturnOp : PCF_Op<"br.cond_return", [
    ParentOneOf<[
      "IREE::PCF::GenericOp",
    ]>,
    DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
    Pure, Terminator
  ]> {
  let summary = "Branch operation with conditional return";
  let description = [{
    The `pcf.br.cond_return` operation represents a conditional branch operation
    to a given block, or return from the parent.

    Example:

    ```mlir
    pcf.<scoped op> #foo.scope {
      ^bb0(%0: !foo.type)
        %1 = ... %0 : !foo.type
        pcf.br.cond_return %cond ^bb0(%0: !foo.type)
    }
    ```
  }];

  let arguments = (ins
    I1:$condition,
    Variadic<AnyType>:$dest_operands
  );
  let successors = (successor AnySuccessor:$dest);

  let extraClassDeclaration = [{
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];

  let assemblyFormat = [{
    $condition $dest (`(` $dest_operands^ `:` type($dest_operands) `)`)? attr-dict
  }];
}

} // OpGroupControlFlowOps

//===----------------------------------------------------------------------===//
// WriteOps
//===----------------------------------------------------------------------===//

def OpGroupWriteOps : OpDocGroup {
  let summary = "Write ops";
  let description = "";
}

let opDocGroup = OpGroupWriteOps in {

  //===----------------------------------------------------------------------===//
  // WriteSliceOp
  //===----------------------------------------------------------------------===//

  def PCF_WriteSliceOp
      : PCF_Op<"write_slice", [AttrSizedOperandSegments,
                               OffsetSizeAndStrideOpInterface,
                               AllRanksMatch<["source", "dest"]>,
                               AllElementTypesMatch<["source", "dest"]>,
  ]> {
    let summary = [{
    Submit a write of a tensor, vector, or memref to a slice of a `pcf.sref`.
  }];
    let description = [{
    The `pcf.write_slice` operation supports the following arguments:

    * source: the shaped value that is written.
    * dest: the sref into which the source is written.
    * offsets: shaped-rank number of offsets into the `dest` into which
               the slice is inserted.
    * sizes: shaped-rank number of sizes which specify the sizes of the source
             tensor type.
    * strides: shaped-rank number of strides that specify subsampling in each
               dimension.
  }];

    let arguments = (ins AnyRankedTensorVectorOrMemRef:$source,
        PCF_AnyShapedRef:$dest, Variadic<Index>:$offsets,
        Variadic<Index>:$sizes, Variadic<Index>:$strides,
        DenseI64ArrayAttr:$static_offsets, DenseI64ArrayAttr:$static_sizes,
        DenseI64ArrayAttr:$static_strides);
    let assemblyFormat = [{
    $source `into` $dest ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `into` type($dest)
  }];

    let extraClassDeclaration = [{
    ShapedType getSourceType() {
      return ::llvm::cast<ShapedType>(getSource().getType());
    }

    PCF::ShapedRefType getDestType() {
      return ::llvm::cast<PCF::ShapedRefType>(getDest().getType());
    }

    /// Return the expected rank of each of the `static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getDestType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before `offsets`, `sizes` and
    /// `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }
  }];

    let builders = [
        // Build a WriteSliceOp with mixed static and dynamic entries.
        OpBuilder<(ins "Value":$source, "Value":$dest,
            "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
            "ArrayRef<OpFoldResult>":$strides,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
        // Build a WriteSliceOp with mixed static and dynamic entries
        // packed into a Range vector.
        OpBuilder<(ins "Value":$source, "Value":$dest,
            "ArrayRef<Range>":$ranges,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
        // Build a WriteSliceOp with dynamic entries.
        OpBuilder<(ins "Value":$source, "Value":$dest, "ValueRange":$offsets,
            "ValueRange":$sizes, "ValueRange":$strides,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];

    let hasFolder = 1;
  }

} // OpGroupWriteOps

//===----------------------------------------------------------------------===//
// ReadOps
//===----------------------------------------------------------------------===//

def OpGroupReadOps : OpDocGroup {
  let summary = "Read ops";
  let description = "";
}

let opDocGroup = OpGroupReadOps in {

//===----------------------------------------------------------------------===//
// ReadSliceOp
//===----------------------------------------------------------------------===//

def PCF_ReadSliceOp : PCF_Op<"read_slice", [
       AttrSizedOperandSegments,
       OffsetSizeAndStrideOpInterface,
       AllRanksMatch<["source", "result"]>,
       AllElementTypesMatch<["source", "result"]>,
  ]> {
  let summary = [{
    Read a tensor or vector from a `pcf.sref` based on the provided slice
    parameters.
  }];
  let description = [{
    Read a slice from a `pcf.sref`. If this is reading a vector, the sizes
    may be smaller than the return vector type. In this case out of bounds
    elements have undefined value.

    The `pcf.read_slice` operation supports the following arguments:
    * source: the shaped value that is written.
    * dest: the sref into which the source is written.
    * offsets: shaped-rank number of offsets into the `dest` into which
               the slice is inserted.
    * sizes: shaped-rank number of sizes which specify the sizes of the source
             tensor type.
    * strides: shaped-rank number of strides that specify subsampling in each
               dimension.
  }];

  let arguments = (ins
    PCF_AnyShapedRef:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );

  let results = (outs AnyTypeOf<[AnyRankedTensor, AnyVectorOfAnyRank]>:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
  }];

  let extraClassDeclaration = [{
    PCF::ShapedRefType getSourceType() {
      return ::llvm::cast<PCF::ShapedRefType>(getSource().getType());
    }

    ShapedType getResultType() {
      return ::llvm::cast<ShapedType>(getResult().getType());
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
    // Build a ReadSliceOp with mixed static and dynamic entries.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a ReadSliceOp with mixed static and dynamic entries
    // packed into a Range vector.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ArrayRef<Range>":$ranges,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a ReadSliceOp with dynamic entries.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// GetMemrefOp
//===----------------------------------------------------------------------===//

def PCF_GetMemrefOp : PCF_Op<"get_memref", [
       AttrSizedOperandSegments,
       OffsetSizeAndStrideOpInterface,
       AllRanksMatch<["source", "result"]>,
       AllElementTypesMatch<["source", "result"]>,
  ]> {
  let summary = [{
    Extract a memref view from a slice of a `pcf.sref`.
  }];
  let description = [{
    The `pcf.get_memref` operation extracts a memref view from a slice of a
    sref, breaking the synchronization guarantees of the source.

    The returned memref must have a maximally dynamic layout (all strides and
    offset dynamic) and no memory space. Layout and memory space information
    is determined by the ConvertSRefToMemRef analysis pass.

    The operation supports the following arguments:
    * source: the sref from which to extract a view.
    * offsets: shaped-rank number of offsets into the `source` from which
               the slice begins.
    * sizes: shaped-rank number of sizes which specify the sizes of the result
             memref type.
    * strides: shaped-rank number of strides that specify subsampling in each
               dimension.
  }];

  let arguments = (ins
    PCF_AnyShapedRef:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );

  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
  }];

  let extraClassDeclaration = [{
    PCF::ShapedRefType getSourceType() {
      return ::llvm::cast<PCF::ShapedRefType>(getSource().getType());
    }

    MemRefType getResultType() {
      return ::llvm::cast<MemRefType>(getResult().getType());
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
    // Build a GetMemrefOp with mixed static and dynamic entries.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a GetMemrefOp with mixed static and dynamic entries
    // packed into a Range vector.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ArrayRef<Range>":$ranges,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a GetMemrefOp with dynamic entries.
    OpBuilder<(ins "Type":$resultType, "Value":$source,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let hasFolder = 1;
  let hasVerifier = 1;
}

} // OpGroupReadOps

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

def PCF_YieldOp : PCF_Op<"yield", [ParentOneOf<["IREE::PCF::GenericOp"]>, Pure,
                                   ReturnLike, Terminator,
]> {
  let summary = [{Yields results from a region.}];
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

def PCF_ReturnOp : PCF_Op<"return", [
  ParentOneOf<[
    "IREE::PCF::GenericOp",
    "IREE::PCF::LoopOp",
  ]>,
  Pure,
  Terminator,
  // Note: This op is not return-like because it does not produce the results
  // of the region. The parent is responsible for that.
]> {
  let summary = [{Returns from a thread.}];
  let description = [{
    Returns control flow to the parent without fencing memory. If the parent
    carries an implicit fence one may still occur after the parent has finished.
  }];

  let arguments = (ins);

  let assemblyFormat = [{attr-dict}];
}

#endif  // IREE_CODEGEN_DIALECT_PCF_OPS
