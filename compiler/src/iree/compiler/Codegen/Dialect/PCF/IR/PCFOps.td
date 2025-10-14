// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_PCF_OPS
#define IREE_CODEGEN_DIALECT_PCF_OPS

include "iree/compiler/Codegen/Dialect/PCF/IR/PCFBase.td"
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
    // DO NOT SUBMIT
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
    SmallVector<OpFoldResult> getMixedSizes() {
      SmallVector<OpFoldResult> result;
      unsigned ctr = 0;
      OpBuilder b(getContext());
      for (int64_t i = 0, e = getResultType().getRank(); i < e; ++i) {
        if (getResultType().isDynamicDim(i)) {
          result.push_back(getDynamicSizes()[ctr++]);
        } else {
          result.push_back(b.getIndexAttr(getResultType().getShape()[i]));
        }
      }
      return result;
    }
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
       RecursiveMemoryEffects,
       DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
     ]> {
  let summary = [{
    Execute a set of workers in parallel on a region.
  }];
  let description = [{
    Executes a region across a set of workers at a specified scope. The scope
    is given by an attribute implementing the `ScopeAttr` interface and is
    responsible for the semantics of all pcf primitives at the same scope.
    Further details about scopes are included in the docs for the interface.

// DO NOT SUBMIT

    Example:
    ```mlir
      %0 = ... : tensor<?>
      %1 = ... : tensor<?x?>
      %d0, %d1 = ... : index
      %n = ... : index
      %2:4 = pcf.generic scope(#foo.scope) count(%n)
        initialize {
          %foo = ... !foo.type
          pcf.yield %foo
        } -> (%foo_arg: !foo.type)
        execute(%ref = %0, %ref_1, %ref_2, %ref_3 = %1)[%id: index, %num_threads: index]
             : (!pcf.sref<?, #foo.scope>, !pcf.sref<?, #foo.scope>, !pcf.sref<?x?, #foo.scope>, !pcf.sref<?x?, #foo.scope>)
            -> (tensor<?>, tensor<?>{%d0}, tensor<?x?>{%d0, %d1}, tensor<?x?>) {
        ...
      } {attr-dict-with-keyword}
    ```
  }];
  let arguments = (ins
    PCF_ScopeAttr:$scope,
    Variadic<PCF_AnyTensorOrMemRef>:$inits,
    Variadic<Index>:$dynamic_sizes,
    ArrayProp<BoolProp>:$is_tied,
    DefaultValuedProp<BoolProp, "false">:$sync_on_return,
    IntProp<"int64_t">:$num_index_args,
    IntProp<"int64_t">:$num_leading_args
  );

  let results = (outs Variadic<PCF_AnyTensorOrMemRef>:$results);
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
    OpBuilder<(ins "ScopeAttr":$scope, "int64_t":$num_iterators,
                   CArg<"bool", "false">:$sync_on_return)>,
    // Builder with return types inferred from inits.
    OpBuilder<(ins "ScopeAttr":$scope, "ValueRange":$inits,
                   "int64_t":$num_iterators,
                   CArg<"bool", "false">:$sync_on_return)>,
    // Builder with no inits.
    OpBuilder<(ins "TypeRange":$result_types, "ScopeAttr":$scope,
                   "ValueRange":$dynamic_sizes, "int64_t":$num_iterators,
                   CArg<"bool", "false">:$sync_on_return)>,
    // Builder with everything.
    OpBuilder<(ins "TypeRange":$result_types, "ScopeAttr":$scope,
                   "ValueRange":$inits, "ValueRange":$dynamic_sizes,
                   "ArrayRef<bool>":$is_tied, "int64_t":$num_iterators,
                   CArg<"bool", "false">:$sync_on_return)>,
  ];

  let extraClassDeclaration = [{
    int64_t getNumIterators() {
      return getNumIndexArgs() / 2;
    }
    MutableArrayRef<BlockArgument> getLeadingArgs() {
      return getRegion().getArguments().take_front(getNumLeadingArgs());
    }
    MutableArrayRef<BlockArgument> getIdArgs() {
      return getRegion().getArguments().take_back(getNumIndexArgs()).take_front(getNumIterators());
    }
    MutableArrayRef<BlockArgument> getCountArgs() {
      return getRegion().getArguments().take_back(getNumIterators());
    }
    MutableArrayRef<BlockArgument> getIdAndCountArgs() {
      return getRegion().getArguments().take_back(getNumIndexArgs());
    }
    MutableArrayRef<BlockArgument> getRegionRefArgs() {
      return getRegion().getArguments().drop_front(getNumLeadingArgs()).take_front(getNumResults());
    }

    bool isRegionRefArg(BlockArgument b) {
      assert(b.getOwner() == &getRegion().front() && "unexpected non-entry block arg");
      int64_t rangeBegin = getNumLeadingArgs();
      int64_t rangeEnd = getNumLeadingArgs() + getNumResults();
      return b.getArgNumber() >= rangeBegin && b.getArgNumber() < rangeEnd;
    }

    bool isInitializedArg(BlockArgument b) {
      assert(b.getOwner() == &getRegion().front() && "unexpected non-entry block arg");
      return b.getArgNumber() < getNumLeadingArgs();
    }

    SmallVector<int64_t> getInitTiedResultIndices() {
      SmallVector<int64_t> tiedResults;
      for (auto [i, isTied] : llvm::enumerate(getIsTied())) {
        if (isTied) {
          tiedResults.push_back(i);
        }
      }
      return tiedResults;
    }

    OpResult getTiedResult(OpOperand &operand) {
      int64_t beginIndex = getInits().getBeginOperandIndex();
      int64_t operandIndex = operand.getOperandNumber();
      if (operandIndex < beginIndex || operandIndex >= getInits().size() + beginIndex) {
        return OpResult();
      }

      int64_t initIndex = operandIndex - beginIndex;
      for (auto [i, isTied] : llvm::enumerate(getIsTied())) {
        if (isTied) {
          if (initIndex == 0) {
            return (*this)->getOpResult(i);
          }
          --initIndex;
        }
      }

      return OpResult();
    }

    OpResult getTiedResult(BlockArgument b) {
      assert(isRegionRefArg(b) && "unexpected non region ref arg");
      return (*this)->getOpResult(b.getArgNumber() - getNumLeadingArgs());
    }

    OpOperand *getTiedInit(int64_t i) {
      if (i < 0 || i >= getNumResults() || !getIsTied()[i]) {
        return nullptr;
      }

      int64_t initIndex = llvm::count(getIsTied().take_front(i), true);
      return &getInitsMutable()[initIndex];
    }

    ShapedType getResultType(int64_t i) {
      return cast<ShapedType>(getResults()[i].getType());
    }

    ValueRange getResultDims(int64_t i) {
      if (getIsTied()[i]) {
        return {};
      }

      int64_t startIndex = 0;
      for (auto [curr, isTied] : llvm::enumerate(getIsTied())) {
        if (curr == i) {
          break;
        }
        if (!isTied) {
          startIndex += getResultType(curr).getNumDynamicDims();
        }
      }

      return ValueRange(getDynamicSizes().slice(startIndex, startIndex + getResultType(i).getNumDynamicDims()));
    }
  }];
}

def LoopOp : PCF_Op<"loop", [
       AttrSizedOperandSegments,
       RecursiveMemoryEffects,
       DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       SingleBlockImplicitTerminator<"mlir::iree_compiler::IREE::PCF::ReturnOp">
     ]> {
  let summary = [{
    Execute a set of workers in parallel on a region.
  }];
  let description = [{
    Executes a region across a set of workers at a specified scope. The scope
    is given by an attribute implementing the `ScopeAttr` interface and is
    responsible for the semantics of all pcf primitives at the same scope.
    Further details about scopes are included in the docs for the interface.

// DO NOT SUBMIT

    Example:
    ```mlir
      %0 = ... : tensor<?>
      %1 = ... : tensor<?x?>
      %d0, %d1 = ... : index
      %n = ... : index
      %2:4 = pcf.loop scope(#foo.scope) count(%n)
        execute(%ref = %0, %ref_1, %ref_2, %ref_3 = %1)[%id: index]
             : (!pcf.sref<?, #foo.scope>, !pcf.sref<?, #foo.scope>, !pcf.sref<?x?, #foo.scope>, !pcf.sref<?x?, #foo.scope>)
            -> (tensor<?>, tensor<?>{%d0}, tensor<?x?>{%d0, %d1}, tensor<?x?>) {
        ...
      } {attr-dict-with-keyword}
    ```
  }];
  let arguments = (ins
    PCF_ScopeAttr:$scope,
    Variadic<Index>:$count,
    Variadic<PCF_AnyTensorOrMemRef>:$inits,
    Variadic<Index>:$dynamic_sizes,
    ArrayProp<BoolProp>:$is_tied,
    DefaultValuedProp<BoolProp, "false">:$sync_on_return
  );

  let results = (outs Variadic<PCF_AnyTensorOrMemRef>:$results);
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
    OpBuilder<(ins "ScopeAttr":$scope, "ValueRange":$count,
                   CArg<"bool", "false">:$sync_on_return)>,
    // Builder with return types inferred from inits.
    OpBuilder<(ins "ScopeAttr":$scope, "ValueRange":$count,
                   "ValueRange":$inits,
                   CArg<"bool", "false">:$sync_on_return)>,
    // Builder with no inits.
    OpBuilder<(ins "TypeRange":$result_types, "ScopeAttr":$scope,
                   "ValueRange":$count,
                   "ValueRange":$dynamic_sizes,
                   CArg<"bool", "false">:$sync_on_return)>,
    // Builder with everything.
    OpBuilder<(ins "TypeRange":$result_types, "ScopeAttr":$scope,
                   "ValueRange":$count, "ValueRange":$inits,
                   "ValueRange":$dynamic_sizes, "ArrayRef<bool>":$is_tied,
                   CArg<"bool", "false">:$sync_on_return)>,
  ];

  let extraClassDeclaration = [{
    int64_t getNumIdArgs() {
      return getCount().size();
    }
    MutableArrayRef<BlockArgument> getIdArgs() {
      return getRegion().getArguments().take_back(getNumIdArgs());
    }
    MutableArrayRef<BlockArgument> getRegionRefArgs() {
      return getRegion().getArguments().take_front(getNumResults());
    }

    SmallVector<int64_t> getInitTiedResultIndices() {
      SmallVector<int64_t> tiedResults;
      for (auto [i, isTied] : llvm::enumerate(getIsTied())) {
        if (isTied) {
          tiedResults.push_back(i);
        }
      }
      return tiedResults;
    }

    OpResult getTiedResult(OpOperand &operand) {
      int64_t beginIndex = getInits().getBeginOperandIndex();
      int64_t operandIndex = operand.getOperandNumber();
      if (operandIndex < beginIndex || operandIndex >= getInits().size() + beginIndex) {
        return OpResult();
      }

      int64_t initIndex = operandIndex - beginIndex;
      for (auto [i, isTied] : llvm::enumerate(getIsTied())) {
        if (isTied) {
          if (initIndex == 0) {
            return (*this)->getOpResult(i);
          }
          --initIndex;
        }
      }

      return OpResult();
    }

    OpResult getTiedResult(BlockArgument b) {
      return (*this)->getOpResult(b.getArgNumber());
    }

    OpOperand *getTiedInit(int64_t i) {
      if (i < 0 || i >= getNumResults() || !getIsTied()[i]) {
        return nullptr;
      }

      int64_t initIndex = llvm::count(getIsTied().take_front(i), true);
      return &getInitsMutable()[initIndex];
    }

    ShapedType getResultType(int64_t i) {
      return cast<ShapedType>(getResults()[i].getType());
    }

    ValueRange getResultDims(int64_t i) {
      if (getIsTied()[i]) {
        return {};
      }

      int64_t startIndex = 0;
      for (auto [curr, isTied] : llvm::enumerate(getIsTied())) {
        if (curr == i) {
          break;
        }
        if (!isTied) {
          startIndex += getResultType(curr).getNumDynamicDims();
        }
      }

      return ValueRange(getDynamicSizes().slice(startIndex, startIndex + getResultType(i).getNumDynamicDims()));
    }
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
// ParallelInsertSliceOp
//===----------------------------------------------------------------------===//

def PCF_WriteSliceOp : PCF_Op<"write_slice", [
       AttrSizedOperandSegments,
       OffsetSizeAndStrideOpInterface,
       AllRanksMatch<["source", "dest"]>,
       AllElementTypesMatch<["source", "dest"]>,
  ]> {
  let summary = [{
    Submit a write of a tensor, vector, or memref to a slice of a `pcf.shaped_ref`.
  }];
  let description = [{
    The `pcf.write_slice` operation supports the following arguments:

    * source: the shaped value that is written.
    * dest: the shaped_ref into which the source is written.
    * offsets: shaped-rank number of offsets into the `dest` into which
               the slice is inserted.
    * sizes: shaped-rank number of sizes which specify the sizes of the source
             tensor type.
    * strides: shaped-rank number of strides that specify subsampling in each
               dimension.
  }];

  let arguments = (ins
    PCF_AnyTensorVectorOrMemRef:$source,
    PCF_AnyShapedRef:$dest,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
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
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];
}

} // OpGroupWriteOps

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//

def PCF_YieldOp : PCF_Op<"yield", [
// DO NOT SUBMIT
  // ParentOneOf<[]>,
  Pure,
  ReturnLike,
  Terminator,
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
