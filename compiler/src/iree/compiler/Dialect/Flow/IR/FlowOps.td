// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_FLOW_OPS
#define IREE_DIALECT_FLOW_OPS

include "iree/compiler/Dialect/Flow/IR/FlowBase.td"
include "iree/compiler/Dialect/Flow/IR/FlowInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class FLOW_PureOp<string mnemonic, list<Trait> traits = []> :
    FLOW_Op<mnemonic, !listconcat(traits, [Pure])>;

//===----------------------------------------------------------------------===//
// Partitioned regions
//===----------------------------------------------------------------------===//

def OpGroupPartitionedRegionOps : OpDocGroup {
  let summary = "Partitioned region ops";
  let description = "";
}

let opDocGroup = OpGroupPartitionedRegionOps in {

def FLOW_DispatchRegionOp : FLOW_PureOp<"dispatch.region", [
    Util_ShapeAwareOp,
    AttrSizedOperandSegments]> {
  let summary = [{a group of ops}];
  let description = [{
    This op is a container/grouping of ops. It represents a fusion group before
    being lowered to a dispatch region. Ops are collected inside of the region
    body of the op. Values from parent regions can be captured. Results are
    yielded with a `return` terminator and returned from this op.

    `dispatch.region` ops are lowered to `dispatch.workgroups` ops. Workgroups
    isolated from above. `dispatch.region` ops are a more lightweight
    abstraction for implementing fusion heuristics, i.e., the process of
    deciding which ops should form a dispatch region.

    This op also has a second region: `workload_count`. The arguments to the
    region represent the workload for the dispatch, and returns the number of
    workgroups for the dispatch. The region is lowered directly to
    `workload_count` region of `dispatch.workgroups`.
  }];

  let arguments = (ins FLOW_ShapeDynamicDims:$result_dims,
                       Variadic<FLOW_Dim>:$workload);

  let results = (outs Variadic<AnyType>:$result);

  let regions = (region AnyRegion:$body,
                        AnyRegion:$workgroup_count);

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx);
  }];
}

def FLOW_DispatchWorkgroupsOp : FLOW_PureOp<"dispatch.workgroups", [
  IsolatedFromAbove,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<Util_ClosureOpInterface>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedOperandsIndexAndLength",
  ]>,
  Util_ShapeAwareOp,
]> {
  let summary = [{a dispatch of workgroups across a 3-dimensional grid}];
  let description = [{
    Dispatches some number of workgroups across a 3-dimensional grid. The
    body region will be invoked for each workgroup with a unique
    `flow.dispatch.workgroup.id` in the range of
    `[0, flow.dispatch.workgroup.count)` (along each dimension XYZ).

    From the outside the dispatch operation has value semantics: some tensors
    (and optionally other primitive types) are consumed and one or more new
    result tensors are produced. Inside each workgroup, however, the input and
    output tensors are available for arbitrary loads and stores. In many cases
    each workgroup will load some particular tile(s) from the input tensors and
    store some particular tile(s) to the output tensors unique to that
    workgroup. Though it's possible for multiple workgroups to load the same
    regions of the input tensors behavior is undefined if multiple workgroups
    store to the same regions of the output tensors.

    Though the representation is similar to the GPU-style grid dispatch model
    here we still have not yet allocated buffers, determined the target device
    for execution, or even completed fully resolving shapes/types/etc. Because
    of this it's important that the workgroup body use the
    `flow.dispatch.workgroup.*` ops to query the workgroup ID/count/size instead
    of hardcoding them to a particular set of values. Assume that any workgroup
    dispatch may end up being specialized for several different target devices
    and even several different variants for a particular target device
    (differing workgroup sizes, etc).

    Because at this point in the layering devices have not yet been selected the
    workgroup count cannot be fully evaluated. Instead workload parameters are
    captured that are then passed to a function that when later evaluated
    computes the actual workgroup count based on target information. The
    workload is not limited to the 3D XYZ grid dispatch of the workgroup count
    and can contain any number of parameters used to compute it.

    ```mlir
    %r = flow.dispatch.workgroups[%c5, %c5](%0, %1)
        : (tensor<5x5xf32>, tensor<5xf32>) -> tensor<5x5xf32> =
              (%arg0: !flow.dispatch.tensor<readonly:tensor<5x5xf32>>,
               %arg1: !flow.dispatch.tensor<readonly:tensor<5xf32>>,
               %arg2: !flow.dispatch.tensor<writeonly:tensor<5x5xf32>>) {
      ...
    }
    ```

    The number of results of the operation is equal to the number of results
    in the type signature (`(tensor<5x5xf32>, tensor<5xf32>) -> tensor<5x5xf32>`).
    Each tensor argument and result in the type signature has a corresponding
    block argument of type `!flow.dispatch.tensor`. Furthermore, each argument
    has a corresponding `arguments` operand.

    There are no `arguments` operands for results, but a result can be tied an
    argument by writing the argument operand's SSA value instead of its type:
    E.g., in the above example, `-> %0` would tie the first argument to the
    result. In that case, there would be no separate block argument for the
    result.
  }];

  let arguments = (ins
    Variadic<FLOW_Dim>:$workload,
    Variadic<AnyType>:$arguments,
    FLOW_ShapeDynamicDims:$argument_dims,
    FLOW_ShapeDynamicDims:$result_dims,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region
    AnyRegion:$workgroup_body,
    AnyRegion:$workgroup_count
  );

  let assemblyFormat = [{
    (`[` $workload^ `]`)? ``
    `(` $arguments `)` `:`
    custom<ShapedFunctionType>(ref($arguments),
                               type($arguments), $argument_dims,
                               type($results), $result_dims,
                               $tied_operands)
    attr-dict-with-keyword
    `=` `\n` ` ` ` ` ` `
    custom<DispatchWorkgroupBody>(ref(type($arguments)),
                                  ref(type($results)),
                                  $workgroup_body)
    `` custom<DispatchWorkgroupsCountRegion>($workgroup_count)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "ValueRange":$workload,
      "TypeRange":$resultTypes, "ValueRange":$resultDims,
      "ValueRange":$arguments, "ValueRange":$argumentDims,
      "ArrayRef<int64_t>":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    FunctionType getDispatchType() {
      return FunctionType::get(
          getContext(), llvm::map_to_vector(getArguments(), [](Value value) {
            return value.getType();
          }),
          getResultTypes());
    }

    /// Returns the index of the args() operand in the Operation operands list.
    unsigned mapArgOperandToOpOperand(unsigned i) { return i + getWorkload().size(); };

    ValueRange getOperandDynamicDims(unsigned idx) {
      return IREE::Util::findVariadicDynamicDims(idx - getWorkload().size(), getArguments(), getArgumentDims());
    }

    ValueRange getResultDynamicDims(unsigned idx) {
      return IREE::Util::findVariadicDynamicDims(idx, getResults(), getResultDims());
    }

    /// Returns the BlockArguments corresponding to the inputs in the type
    /// signature.
    Block::BlockArgListType getInputBlockArguments() {
      return getWorkgroupBody().getArguments().take_front(getArguments().size());
    }

    /// Returns the BlockArgument corresponding to the idx-th input in the type
    /// signature.
    BlockArgument getInputBlockArgument(unsigned idx) {
      return getWorkgroupBody().getArguments()[idx];
    }

    /// Returns the BlockArgument corresponding to the idx-th output in the
    /// type signature. If the output is tied to an input, the returned value
    /// is also an input BlockArgument.
    BlockArgument getOutputBlockArgument(unsigned idx);

    /// Returns the BlockArguments corresponding to the outputs in the type
    /// signature. In case an output is tied to an input, the return list
    /// overlaps with `getInputBlockArguments`.
    SmallVector<BlockArgument> getOutputBlockArguments();

    /// Returns the tiedOperands attribute as a list of integers.
    SmallVector<int64_t> getTiedOperandsAsIntegerList();
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def FLOW_DispatchWorkgroupIDOp : FLOW_PureOp<"dispatch.workgroup.id", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{returns the index of the current workgroup in the grid}];
  let description = [{
    The global workgroup ID of the current workgroup in the range of
    `[0, flow.dispatch.workgroup.count)` along each dimension.

    Represented as a 3D grid classically written as XYZ.
    Corresponds to the `WorkgroupId` SPIR-V built-in and the `blockIdx` CUDA
    built-in variable.

    ```mlir
    %x = flow.dispatch.workgroup.id[0] : index
    %y = flow.dispatch.workgroup.id[1] : index
    %z = flow.dispatch.workgroup.id[2] : index
    ```
  }];

  let arguments = (ins IndexAttr:$dimension);
  let results = (outs FLOW_Dim:$result);

  let builders = [
    OpBuilder<(ins "unsigned":$dim),
    [{
      build($_builder, $_state, $_builder.getIndexType(), $_builder.getIndexAttr(dim));
    }]>,
  ];
  let assemblyFormat = "`[` $dimension `]` attr-dict `:` type($result)";

  let extraClassDeclaration = [{
    LogicalResult verify() {
      return verifyDispatchWorkgroupInfoOp(getOperation(), getDimension().getZExtValue());
    }
  }];
}

def FLOW_DispatchWorkgroupCountOp : FLOW_PureOp<"dispatch.workgroup.count", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{returns the total workgroup count of the grid}];
  let description = [{
    The total number of workgroups along each dimension in the dispatch grid.

    Represented as a 3D grid classically written as XYZ.
    Corresponds to the `NumWorkgroups` SPIR-V built-in and the `gridDim` CUDA
    built-in variable.

    ```mlir
    %x = flow.dispatch.workgroup.count[0] : index
    %y = flow.dispatch.workgroup.count[1] : index
    %z = flow.dispatch.workgroup.count[2] : index
    ```
  }];

  let arguments = (ins IndexAttr:$dimension);
  let results = (outs FLOW_Dim:$result);

  let builders = [
    OpBuilder<(ins "unsigned":$dim),
    [{
      build($_builder, $_state, $_builder.getIndexType(), $_builder.getIndexAttr(dim));
    }]>,
  ];
  let assemblyFormat = "`[` $dimension `]` attr-dict `:` type($result)";

  let extraClassDeclaration = [{
    LogicalResult verify() {
      return verifyDispatchWorkgroupInfoOp(getOperation(), getDimension().getZExtValue());
    }
  }];
}

def FLOW_DispatchWorkgroupSizeOp : FLOW_PureOp<"dispatch.workgroup.size", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{returns the size of each workgroup in invocations}];
  let description = [{
    The number of local invocations within the current workgroup along each
    dimension. Depending on backend this may map to the SIMT thread count or
    inner loop nest parameters.

    Workgroup sizes are not determined at the flow dialect level as they are
    dependent on the target backend determined when lowering into the HAL. It's
    still possible to use the symbolic workgroup size inside of dispatch
    executables as a placeholder for the resolved value once in the HAL.

    Represented as a 3D grid classically written as XYZ.
    Corresponds to the `WorkgroupSize` SPIR-V built-in and the `blockDim` CUDA
    built-in variable.

    ```mlir
    %x = flow.dispatch.workgroup.size[0] : index
    %y = flow.dispatch.workgroup.size[1] : index
    %z = flow.dispatch.workgroup.size[2] : index
    ```
  }];

  let arguments = (ins IndexAttr:$dimension);
  let results = (outs FLOW_Dim:$result);

  let builders = [
    OpBuilder<(ins "unsigned":$dim),
    [{
      build($_builder, $_state, $_builder.getIndexType(), $_builder.getIndexAttr(dim));
    }]>,
  ];

  let assemblyFormat = "`[` $dimension `]` attr-dict `:` type($result)";

  let extraClassDeclaration = [{
    LogicalResult verify() {
      return verifyDispatchWorkgroupInfoOp(getOperation(), getDimension().getZExtValue());
    }
  }];
}

def FLOW_DispatchTieShapeOp : FLOW_PureOp<"dispatch.tie_shape", [
  AllTypesMatch<["operand", "result"]>,
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
  Util_ShapeAwareOp,
]> {
  let summary = [{ties a runtime shape to a dispatch I/O argument}];
  let description = [{
    Metadata op used to tie a runtime-computed shape with dynamic dimensions to
    a dispatch input/output argument. All uses of the argument should use the
    pass-through result of this op to allow for SSA-based shape resolution.
  }];

  let arguments = (ins
    FLOW_DispatchTensor:$operand,
    FLOW_ShapeDynamicDims:$dynamic_dims
  );
  let results = (outs
    FLOW_DispatchTensor:$result
  );

  let assemblyFormat = [{
    $operand attr-dict
    `:` type($result) (`{` $dynamic_dims^ `}`)?
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getDynamicDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return getDynamicDims(); }
  }];

  let hasVerifier = 1;

  let hasFolder = 1;
}

def FLOW_DispatchTensorLoadOp : FLOW_PureOp<"dispatch.tensor.load", [
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
  let summary = [{loads a tensor from a dispatch input placeholder}];
  let description = [{
    Loads an input tensor or subtensor from an input placeholder. As each
    workgroup executes concurrently all workgroups will receive identical loaded
    results of regions that may overlap.
  }];

  let arguments = (ins
    FLOW_DispatchTensor:$source,
    FLOW_ShapeDynamicDims:$source_dims,
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
      unsigned sourceRank =
          getSource().getType().cast<DispatchTensorType>().asRankedTensorType().getRank();
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
        (IREE::Flow::DispatchTensorType sourceType,
         ArrayRef<OpFoldResult> mixedSizes);

    /// Returns the list of dimensions that are dropped if the
    /// !flow.dispatch.tensor.load is rank-reducing.
    llvm::SmallBitVector getDroppedDims();

    /// Return the result type as a `RankedTensorType`.
    RankedTensorType getType() {
      return getResult().getType().cast<RankedTensorType>();
    }

    /// Return the type of the source as `IREE::Flow::DispatchTensorType`.
    IREE::Flow::DispatchTensorType getSourceType() {
      return getSource().getType().cast<IREE::Flow::DispatchTensorType>();
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

def FLOW_DispatchTensorStoreOp : FLOW_Op<"dispatch.tensor.store", [
  AttrSizedOperandSegments,
  OffsetSizeAndStrideOpInterface,
  Util_ShapeAwareOp,
]> {
  let summary = [{stores a tensor into a dispatch output placeholder}];
  let description = [{
    Stores a tensor or subtensor into an output tensor placeholder. As each
    workgroup executes concurrently behavior is undefined if more than one
    workgroup stores into overlapping regions of the full output tensor.
  }];

  let arguments = (ins
    AnyRankedTensor:$value,
    FLOW_WritableDispatchTensor:$target,
    FLOW_ShapeDynamicDims:$target_dims,
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
           getTarget().getType().cast<DispatchTensorType>().asRankedTensorType().getRank();
      return {resultRank, resultRank, resultRank};
    }

    /// Return the type of the value type as a RankedTensorType.
    RankedTensorType getValueType() { return getValue().getType().cast<RankedTensorType>(); }

    /// Return the type of the target.
    IREE::Flow::DispatchTensorType getTargetType() {
      return getTarget().getType().cast<IREE::Flow::DispatchTensorType>();
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
    /// !flow.dispatch.tensor.load is rank-reducing.
    llvm::SmallBitVector getDroppedDims();

    /// Checks if the store covers the entire target. Note that
    /// this is best effort, especially in cases where the the shapes are dynamic.
    bool isStoreToWholeTarget();
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def FLOW_ReturnOp : FLOW_Op<"return", [Pure, ReturnLike, Terminator]> {
  let summary = [{return from a flow.dispatch_region}];
  let description = [{
    Returns the given values from the region and back to the host code.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let builders = [
    OpBuilder<(ins),
    [{
      build($_builder, $_state, std::nullopt);
    }]>,
  ];
}

} // OpGroupPartitionedRegionOps

//===----------------------------------------------------------------------===//
// Executables for outlined regions
//===----------------------------------------------------------------------===//

def OpGroupExecutableOps : OpDocGroup {
  let summary = "Executable ops";
  let description = "Executables for outlined regions.";
}

let opDocGroup = OpGroupExecutableOps in {

def FLOW_ExecutableOp : FLOW_Op<"executable", [
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"IREE::Flow::ExecutableEndOp">,
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
    // TODO(benvanik): add compatibility and versioning attributes.
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name
    attr-dict-with-keyword
    regions
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$name)>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return getBody().front(); }
    ::mlir::ModuleOp getInnerModule() {
      auto it = getBlock().getOps<::mlir::ModuleOp>();
      if (it.empty()) return {};
      return *it.begin();
    }
  }];

  let hasVerifier = 1;
}

def FLOW_ExecutableEndOp : FLOW_Op<"executable_end", [
  HasParent<"IREE::Flow::ExecutableOp">,
  Terminator,
]> {
  let summary = [{terminator pseudo-op for the executable op}];
  let assemblyFormat = "attr-dict";
}

def FLOW_ExecutableExportOp : FLOW_Op<"executable.export", [
  HasParent<"IREE::Flow::ExecutableOp">,
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
      "FlatSymbolRefAttr":$function_ref)>,
  ];

  let hasVerifier = 1;
}

} // OpGroupExecutableOps

//===----------------------------------------------------------------------===//
// Dispatch ops
//===----------------------------------------------------------------------===//

def OpGroupDispatchOps : OpDocGroup {
  let summary = "Dispatch ops";
  let description = "";
}

let opDocGroup = OpGroupDispatchOps in {

def FLOW_DispatchOp : FLOW_PureOp<"dispatch", [
  AttrSizedOperandSegments,
  FLOW_StreamableOp,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedOperandsIndexAndLength",
  ]>,
  Util_ShapeAwareOp,
]> {
  let summary = [{a dispatch of workgroups across a grid}];
  let description = [{
    Dispatches workgroups across an grid defined by the captured workload
    parameters carrying the information required to compute the workgroup count
    at runtime. The function for converting the workload into a 3D workgroup
    count is attached to the dispatch entry point and may contain
    arbitrary host logic.
  }];

  let arguments = (ins
    Variadic<FLOW_Dim>:$workload,
    SymbolRefArrayAttr:$entry_points,
    Variadic<AnyType>:$arguments,
    FLOW_ShapeDynamicDims:$argument_dims,
    FLOW_ShapeDynamicDims:$result_dims,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    custom<DispatchEntryPoints>($entry_points)
    (`[` $workload^ `]`)? ``
    `(` $arguments `)` attr-dict `:`
    custom<ShapedFunctionType>(ref($arguments),
                               type($arguments), $argument_dims,
                               type($results), $result_dims,
                               $tied_operands)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "ExecutableExportOp":$exportOp, "ValueRange":$workload,
      "TypeRange":$resultTypes, "ValueRange":$resultDims,
      "ValueRange":$arguments, "ValueRange":$argumentDims,
      "ArrayAttr":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins
      "SymbolRefAttr":$entryPoint, "ValueRange":$workload,
      "TypeRange":$resultTypes, "ValueRange":$resultDims,
      "ValueRange":$arguments, "ValueRange":$argumentDims,
      "ArrayAttr":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins
      "ExecutableExportOp":$exportOp, "ValueRange":$workload,
      "TypeRange":$resultTypes, "ValueRange":$resultDims,
      "ValueRange":$arguments, "ValueRange":$argumentDims,
      "ArrayRef<int64_t>":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        build($_builder, $_state, exportOp, workload,
              resultTypes, resultDims, arguments, argumentDims,
              $_builder.getIndexArrayAttr(tiedOperands),
              attributes);
      }]>,
    OpBuilder<(ins
      "SymbolRefAttr":$entryPoint, "ValueRange":$workload,
      "TypeRange":$resultTypes, "ValueRange":$resultDims,
      "ValueRange":$arguments, "ValueRange":$argumentDims,
      "ArrayRef<int64_t>":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        build($_builder, $_state, entryPoint, workload,
              resultTypes, resultDims, arguments, argumentDims,
              $_builder.getIndexArrayAttr(tiedOperands),
              attributes);
      }]>
  ];

  let extraClassDeclaration = [{
    FunctionType getEntryPointType();

    auto getEntryPointRefs() {
      return getEntryPoints().getAsRange<SymbolRefAttr>();
    }
    void forEachEntryPointAttr(std::function<void(SymbolRefAttr)> fn) {
      for (auto entryPointAttr : getEntryPointRefs()) fn(entryPointAttr);
    }

    // Returns a human-friendly string name for what is being dispatched.
    // May not be unique or a valid reference to an executable.
    std::string getEntryPointName();

    // StreamableOpInterface:
    bool isTransfer() { return false; }

    ValueRange getOperandDynamicDims(unsigned idx) {
      return IREE::Util::findVariadicDynamicDims(idx - getWorkload().size(), getArguments(), getArgumentDims());
    }
    ValueRange getResultDynamicDims(unsigned idx) {
      return IREE::Util::findVariadicDynamicDims(idx, getResults(), getResultDims());
    }
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

} // OpGroupDispatchOps

//===----------------------------------------------------------------------===//
// Streamable call ops
//===----------------------------------------------------------------------===//

def OpGroupStreamableCallOps : OpDocGroup {
  let summary = "Streamable call ops";
  let description = "";
}

let opDocGroup = OpGroupStreamableCallOps in {

def FLOW_FuncOp : FLOW_Op<"func", [
  CallableOpInterface,
  FunctionOpInterface,
  IsolatedFromAbove,
]> {
  let summary = [{streamable function declaration}];
  let description = [{
    Declares a function that can be called as an asynchronous streaming
    operation via `flow.call`. Today only external functions are allowed.
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
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$resAttrs
    )>,
  ];

  let extraClassDeclaration = [{
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<int64_t> tiedOperands = {},
                         ArrayRef<NamedAttribute> attrs = {},
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

def FLOW_CallOp : FLOW_Op<"call", [
  AttrSizedOperandSegments,
  CallOpInterface,
  FLOW_StreamableOp,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedOperandsIndexAndLength",
  ]>,
  Util_ShapeAwareOp,
]> {
  let summary = [{calls a streamable external host function}];
  let description = [{
    Calls a function taking/returning tensor values with stream semantics.
    Tensors have their shapes captured and may be tied to denote in-place
    operations. Asynchronous calls must have no side-effects.

    Note that returned tensors must have their shapes declared prior to the call
    as this is what allows the call to be made on the stream. If external host
    logic is required to compute the shape (avoid at all costs!) a separate
    func.call can be used outside of the stream to do so. If shapes are
    unknowable until the operation is performed it should be made as a normal
    asynchronous host call with 'coarse-fences' instead.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$arguments,
    FLOW_ShapeDynamicDims:$argument_dims,
    FLOW_ShapeDynamicDims:$result_dims,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    $callee
    `(` $arguments `)` attr-dict `:`
    custom<ShapedFunctionType>(ref($arguments),
                               type($arguments), $argument_dims,
                               type($results), $result_dims,
                               $tied_operands)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "SymbolRefAttr":$callee,
      "TypeRange":$resultTypes,
      "ValueRange":$arguments,
      "ArrayAttr":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins
      "SymbolRefAttr":$callee,
      "TypeRange":$resultTypes, "ValueRange":$resultDims,
      "ValueRange":$arguments, "ValueRange":$argumentDims,
      "ArrayAttr":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return getArguments().begin(); }
    operand_iterator arg_operand_end() { return getArguments().end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }

    // StreamableOpInterface:
    bool isTransfer() { return false; }

    ValueRange getOperandDynamicDims(unsigned idx) {
      return IREE::Util::findVariadicDynamicDims(idx, getArguments(), getArgumentDims());
    }
    ValueRange getResultDynamicDims(unsigned idx) {
      return IREE::Util::findVariadicDynamicDims(idx, getResults(), getResultDims());
    }

    /// Get the argument operands to the called function as a mutable range, this is
    /// required by the call interface.
    MutableOperandRange getArgOperandsMutable() {
      return getArgumentsMutable();
    }
  }];

  let hasVerifier = 1;
}

} // OpGroupStreamableCallOps

//===----------------------------------------------------------------------===//
// Tensor ops
//===----------------------------------------------------------------------===//

def OpGroupTensorOps : OpDocGroup {
  let summary = "Tensor ops";
  let description = "";
}

let opDocGroup = OpGroupTensorOps in {

// TODO(benvanik): make this behave like std.constant if not dynamic.
def FLOW_TensorConstantOp : FLOW_Op<"tensor.constant"> {
  let summary = [{tensor constant that can have dynamic dimensions}];
  let description = [{
    Allows specifying a constant where the return value can erase shape
    information. This operation is declared as having side effects and has no
    folder, so will not be optimized away by the compiler. The underlying shape
    information should be hidden from the compiler and resolved at runtime.

    ```mlir
    %c = flow.tensor.constant tensor<2x2xf32> -> tensor<?x?xf32>
    %res = math.absf %c : tensor<?x?xf32>
    ```
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{
    $value attr-dict `->` type($result)
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def FLOW_TensorTieShapeOp : FLOW_PureOp<"tensor.tie_shape", [
  AllTypesMatch<["operand", "result"]>,
  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
  Util_ShapeAwareOp,
]> {
  let summary = [{ties a runtime shape to a tensor value}];
  let description = [{
    Metadata op used to tie tensors with their runtime-computed dynamic
    dimensions. This only exists transiently in the IR as a witness to shape
    calculations and is removed during lowering.
  }];

  let arguments = (ins
    FLOW_Tensor:$operand,
    FLOW_ShapeDynamicDims:$dynamic_dims
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = [{
    $operand attr-dict
    `:` type($result) (`{` $dynamic_dims^ `}`)?
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getDynamicDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return getDynamicDims(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def FLOW_TensorReshapeOp : FLOW_PureOp<"tensor.reshape", [
  FLOW_StreamableOp,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
      "getTiedResult",
      "getTiedResultOperandIndex",
      "getTiedResultOperandIndices",
  ]>,
  Util_ShapeAwareOp,
]> {
  let summary = [{reshapes a tensor}];
  let description = [{
    Reshapes a tensor to a new shape without modifying the contents.
  }];

  let arguments = (ins
    FLOW_Tensor:$source,
    FLOW_ShapeDynamicDims:$source_dims,
    FLOW_ShapeDynamicDims:$result_dims
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = [{
    $source `:`
    type($source) (`{` $source_dims^ `}`)? `->`
    type($result) (`{` $result_dims^ `}`)?
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins
      "Type":$result_type, "Value":$source, "ValueRange":$target_dims),
    [{
      build($_builder, $_state,
          result_type,
          source,
          IREE::Util::buildDynamicDimsForValue($_state.location, source, $_builder),
          target_dims);
    }]>,
  ];

  let extraClassDeclaration = [{
    // StreamableOpInterface:
    bool isTransfer() { return true; }

    ValueRange getOperandDynamicDims(unsigned idx) { return getSourceDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultDims(); }
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def FLOW_TensorLoadOp : FLOW_PureOp<"tensor.load", [
  TypesMatchWith<"value type matches element type of target operand",
                  "source", "result",
                  "$_self.cast<ShapedType>().getElementType()">,
  AttrSizedOperandSegments,
  Util_ShapeAwareOp,
]> {
  let summary = [{loads a value from a tensor element}];
  let description = [{
    Returns the element at the given location from within the tensor.
  }];

  let arguments = (ins
    FLOW_Tensor:$source,
    FLOW_ShapeDynamicDims:$source_dims,
    Variadic<FLOW_Dim>:$indices
  );
  let results = (outs
    AnyTypeOf<[FLOW_PrimitiveType, AnyVector]>:$result
  );

  let assemblyFormat = [{
    $source (`[` $indices^ `]`)? `:`
    type($source) (`{` $source_dims^ `}`)?
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins
      "Type":$result_type, "Value":$source, CArg<"ValueRange", "{}">:$indices),
    [{
      build($_builder, $_state,
          result_type,
          source,
          IREE::Util::buildDynamicDimsForValue($_state.location, source, $_builder),
          indices);
    }]>,
  ];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getSourceDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return ValueRange{}; }
  }];

  let hasVerifier = 1;
  // TODO(benvanik): canonicalize to slice+load if dims are known.
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def FLOW_TensorStoreOp : FLOW_PureOp<"tensor.store", [
  AllTypesMatch<["target", "result"]>,
  TypesMatchWith<"value type matches element type of target operand",
                  "target", "value",
                  "$_self.cast<ShapedType>().getElementType()">,
  AttrSizedOperandSegments,
  Util_ShapeAwareOp,
]> {
  let summary = [{stores a value into a tensor element}];
  let description = [{
    Returns a tensor with the element at the given index set to the given value.
  }];

  let arguments = (ins
    AnyTypeOf<[FLOW_PrimitiveType, AnyVector]>:$value,
    FLOW_Tensor:$target,
    FLOW_ShapeDynamicDims:$target_dims,
    Variadic<FLOW_Dim>:$indices
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = [{
    $value `,` $target (`[` $indices^ `]`)? `:`
    type($target) (`{` $target_dims^ `}`)?
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$value, "Value":$target, CArg<"ValueRange", "{}">:$indices),
    [{
      build($_builder, $_state,
          target.getType(),
          value,
          target,
          IREE::Util::buildDynamicDimsForValue($_state.location, target, $_builder),
          indices);
    }]>,
  ];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return getTargetDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return getTargetDims(); }
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

def FLOW_TensorAllocaOp : FLOW_Op<"tensor.alloca", [
  Util_ShapeAwareOp,
  MemoryEffects<[MemAlloc]>,
]> {
  let summary = [{an empty tensor allocation with undefined contents}];
  let description = [{
    Returns a new transient tensor allocation with undefined contents.
    Subsequent writes must populate any ranges of the tensor that are later
    read. The resulting tensor may be long-lived and allocated as part of a
    dedicated allocation. Prefer using `flow.tensor.empty` whenever possible as
    this op disables nearly all allocation-related optimizations performed by
    the compiler. The presence of this op is often an indication of an improper
    lowering.
  }];

  let arguments = (ins
    FLOW_ShapeDynamicDims:$result_dims
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = [{
    `:` type($result) (`{` $result_dims^ `}`)?
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultDims(); }
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def FLOW_TensorEmptyOp : FLOW_PureOp<"tensor.empty", [
  FLOW_StreamableOp,
  Util_ShapeAwareOp,
]> {
  let summary = [{an empty tensor carrying metadata but no contents}];
  let description = [{
    Returns a tensor with undefined contents. Subsequent writes must populate
    any ranges of the tensor that are later read.
  }];

  let arguments = (ins
    FLOW_ShapeDynamicDims:$result_dims
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = [{
    `:` type($result) (`{` $result_dims^ `}`)?
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    // StreamableOpInterface:
    bool isTransfer() { return true; }

    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultDims(); }
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def FLOW_TensorSplatOp : FLOW_PureOp<"tensor.splat", [
  FLOW_StreamableOp,
  TypesMatchWith<"value type matches element type of result",
                  "result", "value",
                  "$_self.cast<ShapedType>().getElementType()">,
  Util_ShapeAwareOp,
]> {
  let summary = [{splats a value into a shaped tensor}];
  let description = [{
    Returns a tensor initialized to the given primitive value.
  }];

  let arguments = (ins
    FLOW_PrimitiveType:$value,
    FLOW_ShapeDynamicDims:$result_dims
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = [{
    $value `:` type($result) (`{` $result_dims^ `}`)?
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    // StreamableOpInterface:
    bool isTransfer() { return true; }

    ValueRange getOperandDynamicDims(unsigned idx) { return ValueRange{}; }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultDims(); }
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def FLOW_TensorCloneOp : FLOW_PureOp<"tensor.clone", [
  FLOW_StreamableOp,
  AllTypesMatch<["operand", "result"]>,
  Util_ShapeAwareOp,
]> {
  let summary = [{performs a full tensor clone operation}];
  let description = [{
    Clones the input tensor into an identical output tensor.
  }];

  let arguments = (ins
    FLOW_Tensor:$operand,
    FLOW_ShapeDynamicDims:$argument_dims
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = [{
    $operand `:` type($result) (`{` $argument_dims^ `}`)?
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins "Value":$operand),
    [{
      build($_builder, $_state,
          operand.getType(),
          operand,
          IREE::Util::buildDynamicDimsForValue($_state.location, operand, $_builder));
    }]>,
  ];

  let extraClassDeclaration = [{
    // StreamableOpInterface:
    bool isTransfer() { return true; }

    ValueRange getOperandDynamicDims(unsigned idx) { return getArgumentDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return getArgumentDims(); }
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def FLOW_TensorSliceOp : FLOW_PureOp<"tensor.slice", [
  FLOW_StreamableOp,
  AllRanksMatch<["source", "result"]>,
  AllElementTypesMatch<["source", "result"]>,
  AttrSizedOperandSegments,
  Util_ShapeAwareOp,
]> {
  let summary = [{slices out a subregion of a tensor}];
  let description = [{
    Clones a subregion of a tensor.
  }];

  let arguments = (ins
    FLOW_Tensor:$source,
    FLOW_ShapeDynamicDims:$source_dims,
    Variadic<FLOW_Dim>:$start_indices,
    Variadic<FLOW_Dim>:$lengths,
    FLOW_ShapeDynamicDims:$result_dims
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = [{
    $source `[` $start_indices `for` $lengths `]` `:`
    type($source) (`{` $source_dims^ `}`)? `->`
    type($result) (`{` $result_dims^ `}`)?
    attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    // StreamableOpInterface:
    bool isTransfer() { return true; }

    ValueRange getOperandDynamicDims(unsigned idx) { return getSourceDims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return getResultDims(); }
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def FLOW_TensorUpdateOp : FLOW_PureOp<"tensor.update", [
  FLOW_StreamableOp,
  AllRanksMatch<["update", "target", "result"]>,
  AllTypesMatch<["target", "result"]>,
  AllElementTypesMatch<["update", "target", "result"]>,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
      "getTiedResult",
      "getTiedResultOperandIndex",
      "getTiedResultOperandIndices",
  ]>,
  Util_ShapeAwareOp,
]> {
  let summary = [{updates a tensor with the contents of another tensor}];
  let description = [{
    Updates the target tensor with the contents of the update tensor at the
    given offset indices.
  }];

  let arguments = (ins
    FLOW_Tensor:$target,
    FLOW_ShapeDynamicDims:$target_dims,
    Variadic<FLOW_Dim>:$start_indices,
    FLOW_Tensor:$update,
    FLOW_ShapeDynamicDims:$update_dims
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = [{
    $update `,` $target `[` $start_indices `]` `:`
    type($update) (`{` $update_dims^ `}`)? `->`
    custom<ShapedTiedResult>(type($result), $target_dims)
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$target,
      "ValueRange":$start_indices,
      "Value":$update)>,
  ];

  let extraClassDeclaration = [{
    // StreamableOpInterface:
    bool isTransfer() { return true; }

    ValueRange getOperandDynamicDims(unsigned idx) {
      return idx == 0 ? getTargetDims() : getUpdateDims();
    }
    ValueRange getResultDynamicDims(unsigned idx) { return getTargetDims(); }
  }];

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def FLOW_TensorTraceOp : FLOW_Op<"tensor.trace", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<Util_ShapeAwareOp>,
]> {
  let summary = [{traces one or more tensor values at runtime}];
  let description = [{
    Traces out to a runtime trace sink (console, log file, etc) the given
    tensors. The key is arbitrary and can be used for identifying the set of
    values being traced.
  }];

  let arguments = (ins
    StrAttr:$key,
    Variadic<FLOW_Tensor>:$values,
    FLOW_ShapeDynamicDims:$value_dims
  );

  let assemblyFormat = [{
    $key `=` `[`
    custom<ShapedOperandList>($values, type($values), $value_dims)
    `]` attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins "StringRef":$key, "ValueRange":$values), [{
      build($_builder, $_state, key, values,
            IREE::Util::buildDynamicDimsForValues($_state.location, values, $_builder));
    }]>,
  ];

  let hasVerifier = 1;
}

//===---------------------------------------------------------------------===//
// Parameterization Ops
//===---------------------------------------------------------------------===//

class FLOW_DispatchWorkgroupCountOp<string mnemonic, list<Trait> traits = []> :
    FLOW_PureOp<mnemonic, traits> {
  let arguments = (ins Variadic<Index>:$operands);
  let results = (outs Index:$x, Index:$y, Index:$z);

  let assemblyFormat = [{
    attr-dict $operands
  }];
}

def FLOW_DispatchWorkgroupCountFromDagRootOp :
    FLOW_DispatchWorkgroupCountOp<"dispatch.workgroup_count_from_dag_root"> {
  let summary = [{
    workgroup count computed based on iteration range of the root of the DAG
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

def FLOW_DispatchWorkgroupCountFromSliceOp :
    FLOW_DispatchWorkgroupCountOp<"dispatch.workgroup_count_from_slice"> {
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

def FLOW_DispatchWorkloadOrdinalOp :
    FLOW_PureOp<"dispatch.workload.ordinal"> {
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
    `flow.dispatch.workgroups, i.e. the signature of the body of the op is not
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

} // OpGroupTensorOps

//===----------------------------------------------------------------------===//
// Collective communication ops
//===----------------------------------------------------------------------===//

def OpGroupCollectiveCommunicationOps : OpDocGroup {
  let summary = "Collective communication ops";
  let description = "";
}

let opDocGroup = OpGroupCollectiveCommunicationOps in {

def FLOW_ChannelDefaultOp : FLOW_PureOp<"channel.default", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = [{returns a default collective communication channel}];
  let description = [{
    Returns a channel initialized using the runtime environment.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$group
  );
  let results = (outs
    FLOW_Channel:$result
  );

  let assemblyFormat = [{
    ($group^)?
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def FLOW_ChannelSplitOp : FLOW_Op<"channel.split", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = [{splits a collective communication channel}];
  let description = [{
    Partitions the group associated with the given channel into disjoint
    subgroups for each unique value of color. Each new subgroup contains all
    participants of the same color and within each subgroup the key argument
    is used to define the rank order. When multiple participants in a group
    use the same key the tie will be broken using their rank in the parent
    group.
  }];

  let arguments = (ins
    FLOW_Channel:$channel,
    Index:$color,
    Index:$key
  );
  let results = (outs
    FLOW_Channel:$result
  );

  let assemblyFormat = [{
    $channel `,` $color `,` $key
    `:` type($channel) `->` type($result)
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
}

def FLOW_ChannelRankOp : FLOW_PureOp<"channel.rank", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{returns the rank of the local participant in the group}];
  let description = [{
    Returns the rank the channel represents as a participant in a collective
    group in `[0, count)`.
  }];

  let arguments = (ins
    FLOW_Channel:$channel
  );
  let results = (outs
    Index:$result
  );

  let assemblyFormat = [{
     $channel `:` type($result)
    attr-dict-with-keyword
  }];
}

def FLOW_ChannelCountOp : FLOW_PureOp<"channel.count", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{returns the total number of participants in the group}];
  let description = [{
    Returns the total participant count in the collective communicator group.
  }];

  let arguments = (ins
    FLOW_Channel:$channel
  );
  let results = (outs
    Index:$result
  );

  let assemblyFormat = [{
    $channel `:` type($result)
    attr-dict-with-keyword
  }];
}

def FLOW_CollectiveAllGatherOp : FLOW_Op<"collective.all_gather", [
  AllTypesMatch<["target", "result"]>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{performs all-gather operation}];
  let description = [{It gathers data from all ranks and concatenates them on the 0-th dimension.}];

  let arguments = (ins
    FLOW_CollectiveElementTypeAttr:$element_type,
    FLOW_Tensor:$target,
    FLOW_ShapeDynamicDims:$target_dims,
    FLOW_Tensor:$source,
    FLOW_Channel:$channel,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
  );
  let results = (outs
    FLOW_Tensor:$result
  );
  let assemblyFormat = [{
    $element_type `,` $target `,` $source `,` $channel `:`
    `(` type($target) `,` type($source) `,` type($channel) `)` `->`
    custom<ShapedTiedResult>(type($result), $target_dims, $tied_operands)
    attr-dict-with-keyword
  }];
  let builders = [
    OpBuilder<(ins
      "CollectiveElementTypeAttr":$element_type,
      "Value":$target,
      "Value":$source,
      "Value":$channel)>,
  ];
}

def FLOW_CollectiveAllReduceOp : FLOW_Op<"collective.all_reduce", [
  AllTypesMatch<["source", "target", "result"]>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{performs all-reduce operation}];
  let description = [{The operation reduces data across all the ranks in the channel.}];

  let arguments = (ins
    FLOW_CollectiveReductionOpAttr:$reduction_op,
    FLOW_CollectiveElementTypeAttr:$element_type,
    FLOW_Tensor:$target,
    FLOW_ShapeDynamicDims:$target_dims,
    FLOW_Tensor:$source,
    FLOW_Channel:$channel,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
  );
  let results = (outs
    FLOW_Tensor:$result
  );
  let assemblyFormat = [{
    $reduction_op `,` $element_type `,` $target `,` $source `,` $channel `:`
    `(` type($target) `,` type($source) `,` type($channel) `)` `->`
    custom<ShapedTiedResult>(type($result), $target_dims, $tied_operands)
    attr-dict-with-keyword
  }];
  let builders = [
    OpBuilder<(ins
      "CollectiveReductionOpAttr":$reduction_op,
      "CollectiveElementTypeAttr":$element_type,
      "Value":$target,
      "Value":$source,
      "Value":$channel)>,
  ];
}

def FLOW_CollectiveAllToAllOp : FLOW_Op<"collective.all_to_all", [
  AllTypesMatch<["target", "result"]>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{performs all-to-all operation}];
  let description = [{This operation mutually exchanges data acrosss all of the ranks in the channel.}];

  let arguments = (ins
    FLOW_CollectiveElementTypeAttr:$element_type,
    FLOW_Tensor:$target,
    FLOW_ShapeDynamicDims:$target_dims,
    FLOW_Tensor:$source,
    FLOW_Channel:$channel,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
  );
  let results = (outs
    FLOW_Tensor:$result
  );
  let assemblyFormat = [{
    $element_type `,` $target `,` $source `,` $channel `:`
    `(` type($target) `,` type($source) `,` type($channel) `)` `->`
    custom<ShapedTiedResult>(type($result), $target_dims, $tied_operands)
    attr-dict-with-keyword
  }];
  let builders = [
    OpBuilder<(ins
      "CollectiveElementTypeAttr":$element_type,
      "Value":$target,
      "Value":$source,
      "Value":$channel)>,
  ];
}

def FLOW_CollectiveReduceScatterOp : FLOW_Op<"collective.reduce_scatter", [
  AllTypesMatch<["target", "result"]>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{performs reduce and scatter operations}];
  let description = [{The operation reduces data across all the ranks in the channel and
    scatters the result to each rank.}];

  let arguments = (ins
    FLOW_CollectiveReductionOpAttr:$reduction_op,
    FLOW_CollectiveElementTypeAttr:$element_type,
    FLOW_Tensor:$target,
    FLOW_ShapeDynamicDims:$target_dims,
    FLOW_Tensor:$source,
    FLOW_Channel:$channel,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
  );
  let results = (outs
    FLOW_Tensor:$result
  );
  let assemblyFormat = [{
    $reduction_op `,` $element_type `,` $target `,` $source `,` $channel `:`
    `(` type($target) `,` type($source) `,` type($channel) `)` `->`
    custom<ShapedTiedResult>(type($result), $target_dims, $tied_operands)
    attr-dict-with-keyword
  }];
  let builders = [
    OpBuilder<(ins
      "CollectiveReductionOpAttr":$reduction_op,
      "CollectiveElementTypeAttr":$element_type,
      "Value":$target,
      "Value":$source,
      "Value":$channel)>,
  ];
}

def FLOW_CollectiveSendRecvOp : FLOW_Op<"collective.send_recv", [
  AllTypesMatch<["source", "target", "result"]>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedResult",
    "getTiedResultOperandIndex",
    "getTiedResultOperandIndices",
  ]>,
]> {
  let summary = [{performs a grouped send and receive operation}];
  let description = [{The operation sends data to the rank specificied by send
    and receives data from the rank specified by recv. If send is -1, this rank
    will not send any data. If recv is -1, this rank will not receive any data
    and the output will be all zeros.}];

  let arguments = (ins
    FLOW_CollectiveElementTypeAttr:$element_type,
    FLOW_Tensor:$target,
    FLOW_ShapeDynamicDims:$target_dims,
    FLOW_Tensor:$source,
    FLOW_Channel:$channel,
    Index:$send,
    Index:$recv,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
  );
  let results = (outs
    FLOW_Tensor:$result
  );
  let assemblyFormat = [{
    $element_type `,` $target `,` $source `,` $channel `,` $send `,` $recv `:`
    `(` type($target) `,` type($source) `,` type($channel) `,` type($send) `,` type($recv) `)` `->`
    custom<ShapedTiedResult>(type($result), $target_dims, $tied_operands)
    attr-dict-with-keyword
  }];
  let builders = [
    OpBuilder<(ins
      "CollectiveElementTypeAttr":$element_type,
      "Value":$target,
      "Value":$source,
      "Value":$channel,
      "Value":$send,
      "Value":$recv)>,
  ];
}

} // OpGroupCollectiveCommunicationOps

#endif  // IREE_DIALECT_FLOW_OPS
