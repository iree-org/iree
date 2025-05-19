// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_TENSOR_EXT_OPS
#define IREE_DIALECT_TENSOR_EXT_OPS

include "iree/compiler/Dialect/TensorExt/IR/TensorExtBase.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class IREETensorExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<IREETensorExt_Dialect, mnemonic, traits> {
  // let hasCustomAssemblyFormat = 1;
}

class IREETensorExt_PureOp<string mnemonic, list<Trait> traits = []> :
    IREETensorExt_Op<mnemonic, !listconcat(traits, [Pure])>;

def IREETensorExt_DispatchTensorLoadOp : IREETensorExt_PureOp<"dispatch.tensor.load", [
  AttrSizedOperandSegments,
  OffsetSizeAndStrideOpInterface,
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
      "getTiedResult",
      "getTiedResultOperandIndex",
      "getTiedResultOperandIndices",
  ]>,
  Util_ShapeAwareOp,
]> {
  let summary = [{Loads a tensor from a dispatch input placeholder.}];
  let description = [{
    Loads an input tensor or subtensor from an input placeholder. As each
    workgroup executes concurrently all workgroups will receive identical loaded
    results of regions that may overlap.
  }];

  let arguments = (ins
    IREETensorExt_DispatchTensor:$source,
    IREETensorExt_ShapeDynamicDims:$source_dims,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    $source
    `,` `offsets` `=` custom<DynamicIndexList>(
      $offsets, $static_offsets)
    `,` `sizes` `=` custom<DynamicIndexList>(
      $sizes, $static_sizes)
    `,` `strides` `=` custom<DynamicIndexList>(
      $strides, $static_strides)
    attr-dict `:` type($source) (`{` $source_dims^ `}`)?  `->` type($result)
  }];

  let builders = [
    // Builder for tensor.load with empty offset, sizes and strides operands.
    // This is used to load an entire tensor.
    OpBuilder<(ins
      "RankedTensorType":$resultType,
      "Value":$source,
      "ValueRange":$sourceDynamicDims,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
    OpBuilder<(ins
      "RankedTensorType":$resultType,
      "Value":$source,
      "ValueRange":$sourceDynamicDims,
      "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
    // Builder for tensor.load with mixed static/dynamic opperands.
    OpBuilder<(ins
      "Value":$source,
      "ValueRange":$sourceDynamicDims,
      "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>
  ];

  let extraClassDeclaration = [{
    /// Return the expected rank of each of the `static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned sourceRank = cast<IREE::TensorExt::DispatchTensorType>(
        getSource().getType()).asRankedTensorType().getRank();
      return {sourceRank, sourceRank, sourceRank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }

    // Workaround for OffsetSizeAndStrideOpInterface being incompatible with
    // prefixed accessors.
    OperandRange offsets() { return getOffsets(); }
    OperandRange sizes() { return getSizes(); }
    OperandRange strides() { return getStrides(); }

    /// Returns the type of the result based on the sizes.
    static RankedTensorType inferResultType
        (IREE::TensorExt::DispatchTensorType sourceType,
         ArrayRef<OpFoldResult> mixedSizes);

    /// Returns the list of dimensions that are dropped if the
    /// !iree_tensor_ext.dispatch.tensor.load is rank-reducing.
    llvm::SmallBitVector getDroppedDims();

    /// Return the result type as a `RankedTensorType`.
    RankedTensorType getType() {
      return cast<RankedTensorType>(getResult().getType());
    }

    /// Return the type of the source as `IREE::TensorExt::DispatchTensorType`.
    IREE::TensorExt::DispatchTensorType getSourceType() {
      return cast<IREE::TensorExt::DispatchTensorType>(getSource().getType());
    }

    ValueRange getOperandDynamicDims(unsigned idx) { return getSourceDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return getSizes(); }

    /// Returns true if the load is loading the whole source. This is
    /// best effort since this can miss some dynamic cases.
    bool isLoadOfWholeSource();
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def IREETensorExt_DispatchTensorStoreOp : IREETensorExt_Op<"dispatch.tensor.store", [
  AttrSizedOperandSegments,
  OffsetSizeAndStrideOpInterface,
  Util_ShapeAwareOp,
]> {
  let summary = [{Stores a tensor into a dispatch output placeholder.}];
  let description = [{
    Stores a tensor or subtensor into an output tensor placeholder. As each
    workgroup executes concurrently behavior is undefined if more than one
    workgroup stores into overlapping regions of the full output tensor.
  }];

  let arguments = (ins
    AnyRankedTensor:$value,
    IREETensorExt_WritableDispatchTensor:$target,
    IREETensorExt_ShapeDynamicDims:$target_dims,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
  let results = (outs);

  let assemblyFormat = [{
    $value `,` $target
    `,` `offsets` `=` custom<DynamicIndexList>(
      $offsets, $static_offsets)
    `,` `sizes` `=` custom<DynamicIndexList>(
      $sizes, $static_sizes)
    `,` `strides` `=` custom<DynamicIndexList>(
      $strides, $static_strides)
    attr-dict `:` type($value) `->` type($target) (`{` $target_dims^ `}`)?
  }];

  let builders = [
    // Builder for tensor.store with empty offset, sizes and strides operands.
    // This is used to store an entire tensor.
    OpBuilder<(ins
      "Value":$value,
      "Value":$target,
      "ValueRange":$targetDynamicDims,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
    // Builder for tensor.store with mixed static and dynamic offset, sizes and strides.
    OpBuilder<(ins
      "Value":$value,
      "Value":$target,
      "ValueRange":$targetDynamicDims,
      "ArrayRef<OpFoldResult>":$mixedOffsets,
      "ArrayRef<OpFoldResult>":$mixedSizes,
      "ArrayRef<OpFoldResult>":$mixedStrides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>
  ];

  let extraClassDeclaration = [{
    /// Return the expected rank of each of the `static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned resultRank =
           cast<IREE::TensorExt::DispatchTensorType>(getTarget().getType()).asRankedTensorType().getRank();
      return {resultRank, resultRank, resultRank};
    }

    /// Return the type of the value type as a RankedTensorType.
    RankedTensorType getValueType() { return cast<RankedTensorType>(getValue().getType()); }

    /// Return the type of the target.
    IREE::TensorExt::DispatchTensorType getTargetType() {
      return cast<IREE::TensorExt::DispatchTensorType>(getTarget().getType());
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 3; }

    // Workaround for OffsetSizeAndStrideOpInterface being incompatible with
    // prefixed accessors.
    OperandRange offsets() { return getOffsets(); }
    OperandRange sizes() { return getSizes(); }
    OperandRange strides() { return getStrides(); }

    ValueRange getOperandDynamicDims(unsigned idx) {
      return idx == 0 ? getSizes() : getTargetDims();
    }
    ValueRange getResultDynamicDims(unsigned idx) { return {}; }

    /// Returns the list of dimensions that are dropped if the
    /// !iree_tensor_ext.dispatch.tensor.load is rank-reducing.
    llvm::SmallBitVector getDroppedDims();

    /// Checks if the store covers the entire target. Note that
    /// this is best effort, especially in cases where the the shapes are dynamic.
    bool isStoreToWholeTarget();
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

//===---------------------------------------------------------------------===//
// Parameterization Ops
//===---------------------------------------------------------------------===//

class IREETensorExt_DispatchWorkgroupCountOp<string mnemonic, list<Trait> traits = []> :
    IREETensorExt_PureOp<mnemonic, traits> {
  let arguments = (ins Variadic<Index>:$operands);
  let results = (outs Index:$x, Index:$y, Index:$z);

  let assemblyFormat = [{
    attr-dict ($operands^)?
  }];
}

def IREETensorExt_DispatchWorkgroupCountFromDagRootOp :
    IREETensorExt_DispatchWorkgroupCountOp<"dispatch.workgroup_count_from_dag_root"> {
  let summary = [{
    Workgroup count computed based on iteration range of the root of the DAG
    for ops within the dispatch.
  }];
  let description = [{
    When using tile + distribution of the root of the DAG (Directed Acyclic
    Graph) of ops within the dispatch to split the work amongst workgroups. The
    workload captured is the size of the iteration space of the root of the DAG.
    This op represents the computation that given the workload returns the
    number of workgroups to use. The backends are responsible for lowering this
    op into actual computation (typically based on the tile sizes used to tile
    and distribute the root of the DAG).
  }];
}

def IREETensorExt_DispatchWorkgroupCountFromSliceOp :
    IREETensorExt_DispatchWorkgroupCountOp<"dispatch.workgroup_count_from_slice"> {
  let summary = [{
    Place holder to signify default workgroup count calculation.
  }];
  let description = [{
    The default computation of the number of workgroups (or workgroup count)
    assumes that the dispatch + captured values is enough to compute the
    workgroup count. It does so by using a program slice of the values
    within the dispatch that represent the number of workgroups when available
    within the dispatch.
    Currently the arguments of index types captured by the
    `flow.dispatch.workgroups` is treated as the workload for the operation.
    It is a requirement that the slice of the program that computes the
    number of workgroups will need to have its leaves be these captured values.

    TODO: This could be generalized in future to allow the slices to encompass
    arbitrary computation. The computation of the workgroup count can then be
    done on the device itself, if this is data dependent. In such cases the
    workload could be more than just values of index types.
  }];
}

def IREETensorExt_DispatchWorkloadOrdinalOp :
    IREETensorExt_PureOp<"dispatch.workload.ordinal", [
      DeclareOpInterfaceMethods<InferIntDivisibilityOpInterface>,
      DeclareOpInterfaceMethods<InferIntRangeInterface,
        ["inferResultRanges"]>
    ]> {
  let arguments = (ins
    Index:$operand,
    IndexAttr:$ordinal
  );
  let results = (outs Index:$result);

  let summary = [{
    Annotates the values captured as workload within the body of
    `flow.dispatch.workgroups` op.
  }];
  let description = [{
    The arguments that represent the captured/returned values of the
    `flow.dispatch.workgroups`, i.e. the signature of the body of the op is not
    preserved during IREEs compilation. Since the workloads are derived from
    the operands captured by the operation, this op denotes the values captured
    as workloads. This can be used in the backends to map back to the workload
    values while materializing the workgroup count computation.

    TODO: Find a better way to represent this information, either by somehow
    propagating the signature of the created dispatch workgroup op through
    the compilation stack until the codegen backends, or as a separate
    list/attribute that can be plumbed through without using explicit ops.
  }];

  let assemblyFormat = [{
    attr-dict $operand `,` $ordinal `:` type($operand)
  }];

  let hasCanonicalizer = 1;

  let hasFolder = 1;
}

#endif  // IREE_DIALECT_TENSOR_EXT_OPS
