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
include "iree/compiler/Dialect/Shape/IR/ShapeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class FLOW_PureOp<string mnemonic, list<OpTrait> traits = []> :
    FLOW_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

//===----------------------------------------------------------------------===//
// Partitioned regions
//===----------------------------------------------------------------------===//

def FLOW_DispatchWorkgroupsOp : FLOW_PureOp<"dispatch.workgroups", [
  IsolatedFromAbove,
  AttrSizedOperandSegments,
  SingleBlockImplicitTerminator<"IREE::Flow::ReturnOp">,
  DeclareOpInterfaceMethods<Util_ClosureOpInterface>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedOperandsIndexAndLength",
  ]>,
  DeclareOpInterfaceMethods<Shape_ShapeCarryingOpInterface>,
  Util_ShapeAwareOp,
]> {
  let summary = [{a dispatch of workgroups across an n-dimension grid}];
  let description = [{
    Dispatches some number of workgroups across an n-dimensional grid. The
    body region will be invoked for each workgroup with a unique
    `flow.dispatch.workgroup.id` in the range of
    `[0, flow.dispatch.workgroup.count)` (along each dimension).

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

    Because of the general nature of the op in this dialect the workgroup count
    provided to the `flow.dispatch.workgroups` op is in an abstract untiled
    domain. Unlike when lowering to the HAL dialect the number of dimensions is
    unbounded and does not yet have the workgroup size factored into it. As the
    dispatch is lowered the workgroup count range will be converted into a 3D
    XYZ grid space and divided up by the workgroup size chosen for particular
    target devices.
  }];

  let arguments = (ins
    Variadic<FLOW_Dim>:$workgroup_count,
    Variadic<AnyType>:$operands,
    FLOW_ShapeDynamicDims:$operand_dims,
    FLOW_ShapeDynamicDims:$result_dims,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    `[` $workgroup_count `]` ``
    `(` $operands `)` `:`
    custom<ShapedFunctionType>(ref($operands),
                               type($operands), $operand_dims,
                               type($results), $result_dims,
                               $tied_operands)
    attr-dict-with-keyword
    `=` `\n` ` ` ` ` ` `
    custom<DispatchWorkgroupBody>(ref(type($operands)),
                                  ref(type($results)),
                                  $body)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "ValueRange":$workgroupCount,
      "TypeRange":$resultTypes, "ValueRange":$resultDims,
      "ValueRange":$operands, "ValueRange":$operandDims,
      "ArrayRef<int64_t>":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    size_t getWorkgroupRank() { return workgroup_count().size(); }

    FunctionType getDispatchType() {
      return FunctionType::get(
          getContext(),
          llvm::to_vector<4>(llvm::map_range(
              operands(), [](Value value) { return value.getType(); })),
          getResultTypes());
    }

    /// Returns the index of the args() operand in the Operation operands list.
    unsigned mapArgOperandToOpOperand(unsigned i) { return i + workgroup_count().size(); };

    ValueRange getOperandDynamicDims(unsigned idx) {
      return IREE::Util::findVariadicDynamicDims(idx - workgroup_count().size(), operands(), operand_dims());
    }
    ValueRange getResultDynamicDims(unsigned idx) {
      return IREE::Util::findVariadicDynamicDims(idx, results(), result_dims());
    }
  }];

  let verifier = [{ return verifyDispatchWorkgroupsOp(*this); }];

  let hasCanonicalizer = 1;
}

def FLOW_DispatchWorkgroupRankOp : FLOW_PureOp<"dispatch.workgroup.rank", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{returns the rank of the workgroup dimensions}];
  let description = [{
    The number of workgroup dimensions used during dispatch, bounding the
    `flow.dispatch.workgroup.*` query functions.

    ```mlir
    %rank = flow.dispatch.workgroup.rank : index
    ```
  }];

  let arguments = (ins);
  let results = (outs FLOW_Dim:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let hasFolder = 1;
}

def FLOW_DispatchWorkgroupIDOp : FLOW_PureOp<"dispatch.workgroup.id", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{returns the index of the current workgroup in the grid}];
  let description = [{
    The global workgroup ID of the current workgroup in the range of
    `[0, flow.dispatch.workgroup.count)` along each dimension.

    Corresponds to the `WorkgroupId` SPIR-V built-in and the `blockIdx` CUDA
    built-in variable, only in the flow dialect the number of dimensions is not
    restricted to 3 (XYZ).

    ```mlir
    %x = flow.dispatch.workgroup.id[0] : index
    %y = flow.dispatch.workgroup.id[1] : index
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

  let verifier = [{ return verifyDispatchWorkgroupInfoOp(*this); }];
}

def FLOW_DispatchWorkgroupCountOp : FLOW_PureOp<"dispatch.workgroup.count", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{returns the total workgroup count of the grid}];
  let description = [{
    The total number of workgroups along each dimension in the dispatch grid.

    Corresponds to the `NumWorkgroups` SPIR-V built-in and the `gridDim` CUDA
    built-in variable, only in the flow dialect the number of dimensions is not
    restricted to 3 (XYZ).

    ```mlir
    %x = flow.dispatch.workgroup.count[0] : index
    %y = flow.dispatch.workgroup.count[1] : index
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

  let verifier = [{ return verifyDispatchWorkgroupInfoOp(*this); }];
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

    Corresponds to the `WorkgroupSize` SPIR-V built-in and the `blockDim` CUDA
    built-in variable, only in the flow dialect the number of dimensions is not
    restricted to 3 (XYZ).

    ```mlir
    %x = flow.dispatch.workgroup.size[0] : index
    %y = flow.dispatch.workgroup.size[1] : index
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

  let verifier = [{ return verifyDispatchWorkgroupInfoOp(*this); }];
}

def FLOW_DispatchShapeOp : FLOW_PureOp<"dispatch.shape", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = [{returns the shape of a dispatch region input/output tensor}];
  let description = [{
    Queries the shape of an input or output tensor of a
    `flow.dispatch.workgroups` region. The shape may have dynamic dimensions
    that will be resolved to runtime values.
  }];

  let arguments = (ins
    FLOW_DispatchTensor:$source
  );
  let results = (outs
    Shape_RankedShape:$result
  );

  let assemblyFormat = "$source `:` type($source) `->` type($result) attr-dict";

  let hasCanonicalizer = 1;
}

def FLOW_DispatchTieShapeOp : FLOW_PureOp<"dispatch.tie_shape"> {
  let summary = [{ties a runtime shape to a dispatch I/O argument}];
  let description = [{
    Metadata op used to tie a runtime-computed shape with dynamic dimensions to
    a dispatch input/output argument. All uses of the argument should use the
    pass-through result of this op to allow for SSA-based shape resolution.
  }];

  let arguments = (ins
    FLOW_DispatchTensor:$operand,
    Shape_RankedShape:$shape
  );
  let results = (outs
    FLOW_DispatchTensor:$result
  );

  // TODO(benvanik): figure out a way to make this look like shapex.tie_shape.
  let assemblyFormat = [{
    $operand `,` $shape attr-dict
    `:` `(` type($operand) `,` type($shape) `)` `->` type($result)
  }];

  let hasCanonicalizer = 1;
}

def FLOW_DispatchTensorLoadOp : FLOW_PureOp<"dispatch.tensor.load", [
    AttrSizedOperandSegments,
    OffsetSizeAndStrideOpInterface,
]> {
  let summary = [{loads a tensor from a dispatch input placeholder}];
  let description = [{
    Loads an input tensor or subtensor from an input placeholder. As each
    workgroup executes concurrently all workgroups will receive identical loaded
    results of regions that may overlap.
  }];

  let arguments = (ins
    FLOW_ReadableDispatchTensor:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    I64ArrayAttr:$static_offsets,
    I64ArrayAttr:$static_sizes,
    I64ArrayAttr:$static_strides
  );
  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    $source
    `,` `offsets` `=` custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    `,` `sizes` `=` custom<OperandsOrIntegersSizesList>($sizes, $static_sizes)
    `,` `strides` `=` custom<OperandsOrIntegersOffsetsOrStridesList>($strides, $static_strides)
    attr-dict `:` type($source) `->` type($result)
  }];

  let builders = [
    // Builder for tensor.load with empty offset, sizes and strides operands.
    // This is used to load an entire tensor.
    OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "RankedTensorType":$resultType, "Value":$source,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Builder for tensor.load with mixed static/dynamic opperands.
    OpBuilder<(ins "Value":$source, "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$sizes, "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];

  let extraClassDeclaration = [{
    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned sourceRank = source().getType().cast<DispatchTensorType>().asTensorType().getRank();
      return {sourceRank, sourceRank, sourceRank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    /// Returns the type of the result based on the sizes.
    static RankedTensorType inferResultType
        (IREE::Flow::DispatchTensorType sourceType,
         ArrayRef<OpFoldResult> mixedSizes);

    /// Returns the type of the result based on the sizes.
    static RankedTensorType inferRankReducedResultType
        (unsigned resultRank, IREE::Flow::DispatchTensorType sourceType,
         ArrayRef<OpFoldResult> mixedSizes);
}];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def FLOW_DispatchTensorStoreOp : FLOW_Op<"dispatch.tensor.store", [
    AttrSizedOperandSegments,
    OffsetSizeAndStrideOpInterface,
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
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    I64ArrayAttr:$static_offsets,
    I64ArrayAttr:$static_sizes,
    I64ArrayAttr:$static_strides
  );
  let results = (outs);

  let assemblyFormat = [{
    $value `,` $target
    `,` `offsets` `=` custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    `,` `sizes` `=` custom<OperandsOrIntegersSizesList>($sizes, $static_sizes)
    `,` `strides` `=` custom<OperandsOrIntegersOffsetsOrStridesList>($strides, $static_strides)
    attr-dict `:` type($value) `->` type($target)
  }];

  let builders = [
    // Builder for tensor.store with empty offset, sizes and strides operands.
    // This is used to store an entire tensor.
    OpBuilder<(ins "Value":$value, "Value":$target,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];

  let extraClassDeclaration = [{
    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned resultRank = target().getType().cast<DispatchTensorType>().asTensorType().getRank();
      return {resultRank, resultRank, resultRank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }
  }];

  let hasCanonicalizer = 1;
}

def FLOW_ReturnOp : FLOW_Op<"return", [NoSideEffect, ReturnLike, Terminator]> {
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
      build($_builder, $_state, llvm::None);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// Executables for outlined regions
//===----------------------------------------------------------------------===//

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
    OpBuilder<(ins "StringRef":$name)>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return body().front(); }

    ::mlir::ModuleOp getInnerModule() {
      return *getBlock().getOps<::mlir::ModuleOp>().begin();
    }
  }];

  let verifier = [{ return verifyExecutableOp(*this); }];
}

def FLOW_ExecutableEndOp : FLOW_Op<"executable_end", [
  HasParent<"IREE::Flow::ExecutableOp">,
  Terminator,
]> {
  let summary = [{terminator pseudo-op for the executable op}];
  let assemblyFormat = "attr-dict";
}

def FLOW_DispatchEntryOp : FLOW_Op<"dispatch.entry", [
  HasParent<"IREE::Flow::ExecutableOp">,
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
    FlatSymbolRefAttr:$function_ref,
    OptionalAttr<IndexAttr>:$workgroup_rank
  );

  let builders = [
    OpBuilder<(ins
      "StringRef":$sym_name,
      "FlatSymbolRefAttr":$function_ref,
      "IntegerAttr":$workgroup_rank)>,
  ];
}

//===----------------------------------------------------------------------===//
// Dispatch ops
//===----------------------------------------------------------------------===//

def FLOW_DispatchOp : FLOW_PureOp<"dispatch", [
  AttrSizedOperandSegments,
  FLOW_StreamableOp,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedOperandsIndexAndLength",
  ]>,
  DeclareOpInterfaceMethods<Shape_ShapeCarryingOpInterface>,
  Util_ShapeAwareOp,
]> {
  let summary = [{a dispatch of workgroups across an n-dimension grid}];
  let description = [{
    Dispatches workgroups across an n-dimensional grid defined by the specified
    workgroup count. The workgroup count may be dynamic and any dimension may be
    set to 0 to neuter the dispatch (no workgroup will execute).
  }];

  let arguments = (ins
    Variadic<FLOW_Dim>:$workgroup_count,
    SymbolRefAttr:$entry_point,
    Variadic<AnyType>:$operands,
    FLOW_ShapeDynamicDims:$operand_dims,
    FLOW_ShapeDynamicDims:$result_dims,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "DispatchEntryOp":$entryPoint, "ValueRange":$workgroupCount,
      "TypeRange":$resultTypes, "ValueRange":$resultDims,
      "ValueRange":$operands, "ValueRange":$operandDims,
      "ArrayAttr":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins
      "DispatchEntryOp":$entryPoint, "ValueRange":$workgroupCount,
      "TypeRange":$resultTypes, "ValueRange":$resultDims,
      "ValueRange":$operands, "ValueRange":$operandDims,
      "ArrayRef<int64_t>":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        build($_builder, $_state, entryPoint, workgroupCount,
              resultTypes, resultDims, operands, operandDims,
              $_builder.getIndexArrayAttr(tiedOperands),
              attributes);
      }]>
  ];

  let extraClassDeclaration = [{
    StringAttr executable();
    FunctionType getEntryPointType();

    // StreamableOpInterface:
    bool isTransfer() { return false; }

    ValueRange getOperandDynamicDims(unsigned idx) {
      return IREE::Util::findVariadicDynamicDims(idx - workgroup_count().size(), operands(), operand_dims());
    }
    ValueRange getResultDynamicDims(unsigned idx) {
      return IREE::Util::findVariadicDynamicDims(idx, results(), result_dims());
    }
  }];

  let assemblyFormat = [{
    $entry_point `[` $workgroup_count `]` ``
    `(` $operands `)` attr-dict `:`
    custom<ShapedFunctionType>(ref($operands),
                               type($operands), $operand_dims,
                               type($results), $result_dims,
                               $tied_operands)
  }];

  let verifier = [{ return verifyDispatchOp(*this); }];
}

//===----------------------------------------------------------------------===//
// Tensor ops
//===----------------------------------------------------------------------===//

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
    %res = "mhlo.abs"(%c) : (tensor<?x?xf32>) -> tensor<?x?xf32>
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

def FLOW_TensorReshapeOp : FLOW_PureOp<"tensor.reshape", [
  FLOW_StreamableOp,
  AllElementTypesMatch<["source", "result"]>,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
      "getTiedResult",
      "getTiedResultOperandIndex",
      "getTiedResultOperandIndices",
  ]>,
  DeclareOpInterfaceMethods<Shape_ShapeCarryingOpInterface>,
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
          Shape::buildOrFindDynamicDimsForValue($_state.location, source, $_builder),
          target_dims);
    }]>,
  ];

  let extraClassDeclaration = [{
    // StreamableOpInterface:
    bool isTransfer() { return true; }

    ValueRange getOperandDynamicDims(unsigned idx) { return source_dims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return result_dims(); }
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def FLOW_TensorLoadOp : FLOW_PureOp<"tensor.load", [
  TypesMatchWith<"value type matches element type of target operand",
                  "source", "result",
                  "$_self.cast<ShapedType>().getElementType()">,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<Shape_ShapeCarryingOpInterface>,
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
          Shape::buildOrFindDynamicDimsForValue($_state.location, source, $_builder),
          indices);
    }]>,
  ];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return source_dims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return ValueRange{}; }
  }];

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
  DeclareOpInterfaceMethods<Shape_ShapeCarryingOpInterface>,
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
          Shape::buildOrFindDynamicDimsForValue($_state.location, target, $_builder),
          indices);
    }]>,
  ];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) { return target_dims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return target_dims(); }
  }];

  let hasFolder = 1;
}

def FLOW_TensorSplatOp : FLOW_PureOp<"tensor.splat", [
  FLOW_StreamableOp,
  TypesMatchWith<"value type matches element type of result",
                  "result", "value",
                  "$_self.cast<ShapedType>().getElementType()">,
  DeclareOpInterfaceMethods<Shape_ShapeCarryingOpInterface>,
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
    ValueRange getResultDynamicDims(unsigned idx) { return result_dims(); }
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def FLOW_TensorCloneOp : FLOW_PureOp<"tensor.clone", [
  FLOW_StreamableOp,
  AllTypesMatch<["operand", "result"]>,
  DeclareOpInterfaceMethods<Shape_ShapeCarryingOpInterface>,
  Util_ShapeAwareOp,
]> {
  let summary = [{performs a full tensor clone operation}];
  let description = [{
    Clones the input tensor into an identical output tensor.
  }];

  let arguments = (ins
    FLOW_Tensor:$operand,
    FLOW_ShapeDynamicDims:$operand_dims
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = [{
    $operand `:` type($result) (`{` $operand_dims^ `}`)?
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins "Value":$operand),
    [{
      build($_builder, $_state,
          operand.getType(),
          operand,
          Shape::buildOrFindDynamicDimsForValue($_state.location, operand, $_builder));
    }]>,
  ];

  let extraClassDeclaration = [{
    // StreamableOpInterface:
    bool isTransfer() { return true; }

    ValueRange getOperandDynamicDims(unsigned idx) { return operand_dims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return operand_dims(); }
  }];

  // TODO(benvanik): canonicalize away entirely in most cases.
  let hasFolder = 1;
}

def FLOW_TensorSliceOp : FLOW_PureOp<"tensor.slice", [
  FLOW_StreamableOp,
  AllRanksMatch<["source", "result"]>,
  AllElementTypesMatch<["source", "result"]>,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<Shape_ShapeCarryingOpInterface>,
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

    ValueRange getOperandDynamicDims(unsigned idx) { return source_dims(); }
    ValueRange getResultDynamicDims(unsigned idx) { return result_dims(); }
  }];

  // TODO(benvanik): canonicalize multiple slices (traverse upward through ssa).
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
  DeclareOpInterfaceMethods<Shape_ShapeCarryingOpInterface>,
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
    FLOW_ShapeDynamicDims:$update_dims,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = [{
    $update `,` $target `[` $start_indices `]` `:`
    type($update) (`{` $update_dims^ `}`)? `->`
    custom<ShapedTiedResult>(type($result), $target_dims, $tied_operands)
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
      return idx == 0 ? target_dims() : update_dims();
    }
    ValueRange getResultDynamicDims(unsigned idx) { return target_dims(); }
  }];

  let verifier = [{ return verifyTensorUpdateOp(*this); }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def FLOW_TensorTraceOp : FLOW_Op<"tensor.trace", []> {
  let summary = [{trace value(s) operation}];
  let description = [{
    Traces out to a runtime trace sink (console, log file, etc) the given
    tensors and titles them with the given key. The key is informational only
    and useful for titling/marking specific sets of tensors for easier
    searching.
  }];

  let arguments = (ins
    StrAttr:$key,
    Variadic<FLOW_Tensor>:$operands
  );

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// Streams
//===----------------------------------------------------------------------===//

// TODO(#7277): remove when switched to streams (happens there now).
def FLOW_ExStreamFragmentOp : FLOW_PureOp<"ex.stream.fragment", [
  IsolatedFromAbove,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<Util_ClosureOpInterface>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface>,
  DeclareOpInterfaceMethods<Shape_ShapeCarryingOpInterface>,
  Util_ShapeAwareOp,
]> {
  let summary = [{experimental op for defining formed stream regions}];
  let description = [{
    Represents a region where all of the dispatches are meant to target the
    same execution stream. This will be replaced with a segmented version in the
    future that stitches the stream segments together.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    FLOW_ShapeDynamicDims:$operand_dims,
    FLOW_ShapeDynamicDims:$result_dims,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    `(` $operands `)` `:`
    custom<ShapedFunctionType>(ref($operands),
                               type($operands), $operand_dims,
                               type($results), $result_dims,
                               $tied_operands)
    attr-dict-with-keyword
    `=` `\n` ` ` ` ` ` `
    custom<StreamFragmentBody>(ref(type($operands)),
                               ref(type($results)),
                               ref($tied_operands),
                               $body)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "TypeRange":$resultTypes, "ValueRange":$resultDims,
      "ValueRange":$operands, "ValueRange":$operandDims,
      "ArrayRef<int64_t>":$tiedOperands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = [{
    ValueRange getOperandDynamicDims(unsigned idx) {
      return IREE::Util::findVariadicDynamicDims(idx, operands(), operand_dims());
    }
    ValueRange getResultDynamicDims(unsigned idx) {
      return IREE::Util::findVariadicDynamicDims(idx, results(), result_dims());
    }
  }];

  let verifier = [{ return verifyExStreamFragmentOp(*this); }];

  let hasCanonicalizer = 1;
}

#endif  // IREE_DIALECT_FLOW_OPS
