// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_INPUT_OPS_TD
#define IREE_DIALECTS_DIALECT_INPUT_OPS_TD

include "iree-dialects/Dialect/Input/InputDialect.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def IREEInput_NullOp : IREEInput_PureOp<"null"> {
  let summary = "a null value";
  let description = [{
    Initializes reference and variant types with a null value.
  }];

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Casts
//===----------------------------------------------------------------------===//

def IREEInput_TensorToBufferViewOp : IREEInput_PureOp<"cast.tensor_to_buffer_view"> {
  let summary = "Casts a tensor to a BufferView, capturing dynamic dims";
  let arguments = (ins
    IREEInput_Tensor:$source,
    IREEInput_ShapeDynamicDims:$source_dims
  );
  let results = (outs IREEInput_BufferViewType:$target);

  let assemblyFormat = [{
    $source `:` type($source) (`{` $source_dims^ `}`)? `->` type($target)
    attr-dict-with-keyword
  }];
}

def IREEInput_BufferViewToTensorOp : IREEInput_PureOp<"cast.buffer_view_to_tensor"> {
  let summary = "Casts a BufferView to a tensor, providing dynamic dims";
  let arguments = (ins
    IREEInput_BufferViewType:$source,
    IREEInput_ShapeDynamicDims:$target_dims
  );
  let results = (outs IREEInput_Tensor:$target);

  let assemblyFormat = [{
    $source `:` type($source) `->` type($target) (`{` $target_dims^ `}`)?
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// Global variables
//===----------------------------------------------------------------------===//

def IREEInput_GlobalOp : IREEInput_Op<"global", [
    Symbol,
  ]> {
  let summary = [{stateful global variable declaration}];
  let description = [{
    Declares a global variable that maintains its value across invocations.
    The value is tied to the execution context of the module and different
    contexts will have different global storage.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    UnitAttr:$is_mutable,
    OptionalAttr<FlatSymbolRefAttr>:$initializer,
    OptionalAttr<TypedAttrInterface>:$initial_value
  );

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    (`mutable` $is_mutable^)?
    $sym_name
    attr-dict
    (`initializer` `(` $initializer^ `)`)?
    custom<TypeOrAttr>($type, $initial_value)
  }];

  let builders = [
    OpBuilder<(ins
      "StringRef":$name,
      "bool":$isMutable,
      "Type":$type,
      "std::optional<TypedAttr>":$initialValue,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
    OpBuilder<(ins
      "StringRef":$name,
      "bool":$isMutable,
      "Type":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
  ];
}

def IREEInput_GlobalAddressOp : IREEInput_PureOp<"global.address"> {
  let summary = [{returns an address reference to a global}];
  let description = [{
    Returns the address of a global as a typed reference. Can be used with the
    global load and store indirect ops.
  }];

  let arguments = (ins
    IREEInput_GlobalRefAttr:$global
  );
  let results = (outs
    IREEInput_AnyGlobalPtr:$result
  );

  let assemblyFormat = [{
    $global attr-dict `:` type($result)
  }];
}

def IREEInput_GlobalLoadOp : IREEInput_Op<"global.load",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = [{loads a value from a global variable}];
  let description = [{
    Returns a copy of the global value.
  }];

  let arguments = (ins
    IREEInput_GlobalRefAttr:$global
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $global attr-dict `:` type($result)
  }];
}

def IREEInput_GlobalLoadIndirectOp : IREEInput_Op<"global.load.indirect"> {
  let summary = [{loads a value from a global variable}];
  let description = [{
    Returns a copy of the global value.
  }];

  let arguments = (ins
    IREEInput_AnyGlobalPtr:$global
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $global attr-dict `:` type($global) `->` type($result)
  }];
  let hasVerifier = 1;
}

def IREEInput_GlobalStoreOp : IREEInput_Op<"global.store",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = [{stores a value into a global variable}];
  let description = [{
    Stores a copy of the value into a global.
  }];

  let arguments = (ins
    AnyType:$value,
    IREEInput_GlobalRefAttr:$global
  );

  let assemblyFormat = [{
    $value `,` $global attr-dict `:` type($value)
  }];
}

def IREEInput_GlobalStoreIndirectOp : IREEInput_Op<"global.store.indirect"> {
  let summary = [{stores a value into a global variable}];
  let description = [{
    Stores a copy of the value into a global.
  }];

  let arguments = (ins
    AnyType:$value,
    IREEInput_AnyGlobalPtr:$global
  );

  let assemblyFormat = [{
    $value `,` $global attr-dict `:` type($value) `->` type($global)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Buffer Views
//===----------------------------------------------------------------------===//

def IREEInput_BufferViewRankOp : IREEInput_PureOp<"buffer_view.rank"> {
  let summary = [{buffer view rank query}];
  let description = [{
    Returns the rank of the buffer view.
  }];

  let arguments = (ins
    IREEInput_BufferViewType:$buffer_view
  );
  let results = (outs
    IREEInput_Dim:$result
  );

  let assemblyFormat = [{
    $buffer_view attr-dict `:` type($result)
  }];
}

def IREEInput_BufferViewDimOp : IREEInput_PureOp<"buffer_view.dim"> {
  let summary = [{buffer view dimension value query}];
  let description = [{
    Returns the value of the given dimension.
  }];

  let arguments = (ins
    IREEInput_BufferViewType:$buffer_view,
    IndexAttr:$index
  );
  let results = (outs
    IREEInput_Dim:$result
  );

  let assemblyFormat = [{
    $buffer_view `,` $index attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Mutable Lists
//===----------------------------------------------------------------------===//

def IREEInput_ListCreateOp : IREEInput_PureOp<
    "list.create", [MemoryEffects<[MemAlloc]>]> {
  let summary = [{creates a new empty list}];
  let description = [{
    Creates a new empty list with an optional initial capacity.
  }];

  let arguments = (ins
    Optional<Index>:$initial_capacity
  );
  let results = (outs
    IREEInput_AnyList:$result
  );

  let assemblyFormat = "($initial_capacity^)? attr-dict `:` type($result)";
}

def IREEInput_ListSizeOp : IREEInput_Op<"list.size", [MemoryEffects<[MemRead]>]> {
  let summary = [{the size of the list in elements}];
  let description = [{
    Returns the current size of the list in elements.
  }];

  let arguments = (ins
    IREEInput_AnyList:$list
  );
  let results = (outs
    Index:$result
  );

  let assemblyFormat = "operands attr-dict `:` type($list)";
}

def IREEInput_ListResizeOp : IREEInput_Op<"list.resize", [MemoryEffects<[MemWrite]>]> {
  let summary = [{resizes the list to a new count in elements}];
  let description = [{
    Resizes the list to contain `new_size` elements. This will either truncate
    the list if the existing size is greater than `new_size` or extend the list
    with the default list value of the element type.
  }];

  let arguments = (ins
    IREEInput_AnyList:$list,
    Index:$new_size
  );

  let assemblyFormat = "operands attr-dict `:` type($list)";
}

def IREEInput_ListGetOp : IREEInput_Op<"list.get", [MemoryEffects<[MemRead]>]> {
  let summary = [{element accessor}];
  let description = [{
    Returns the value of the element at the given index. Note that the value
    may be null if the element is null or the type does not match.
  }];

  let arguments = (ins
    IREEInput_AnyList:$list,
    Index:$index
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = "$list `[` $index `]` attr-dict `:` type($list) `->` type($result)";
}

def IREEInput_ListSetOp : IREEInput_Op<"list.set", [MemoryEffects<[MemWrite]>]> {
  let summary = [{element mutator}];
  let description = [{
    Sets the element at the given index to the new value.
  }];

  let arguments = (ins
    IREEInput_AnyList:$list,
    Index:$index,
    AnyType:$value
  );

  let assemblyFormat = "$list `[` $index `]` `,` $value attr-dict `:` type($list) `,` type($value)";
}

//===----------------------------------------------------------------------===//
// Tensor ops
//===----------------------------------------------------------------------===//

def IREEInput_TensorReshapeOp : IREEInput_PureOp<"tensor.reshape", [
    AllElementTypesMatch<["source", "result"]>,
    AttrSizedOperandSegments,
  ]> {
  let summary = [{reshapes a tensor}];
  let description = [{
    Reshapes a tensor to a new shape without modifying the contents.
  }];

  let arguments = (ins
    IREEInput_Tensor:$source,
    IREEInput_ShapeDynamicDims:$source_dims,
    IREEInput_ShapeDynamicDims:$result_dims
  );
  let results = (outs
    IREEInput_Tensor:$result
  );

  let assemblyFormat = [{
    $source `:`
    type($source) (`{` $source_dims^ `}`)? `->`
    type($result) (`{` $result_dims^ `}`)?
    attr-dict-with-keyword
  }];
}

def IREEInput_TensorLoadOp : IREEInput_PureOp<"tensor.load", [
    TypesMatchWith<"value type matches element type of target operand",
                   "source", "result",
                   "$_self.cast<ShapedType>().getElementType()">,
    AttrSizedOperandSegments,
  ]> {
  let summary = [{loads a value from a tensor element}];
  let description = [{
    Returns the element at the given location from within the tensor.
  }];

  let arguments = (ins
    IREEInput_Tensor:$source,
    IREEInput_ShapeDynamicDims:$source_dims,
    Variadic<IREEInput_Dim>:$indices
  );
  let results = (outs
    AnyTypeOf<[IREEInput_PrimitiveType, AnyVector]>:$result
  );

  let assemblyFormat = [{
    $source (`[` $indices^ `]`)? `:`
    type($source) (`{` $source_dims^ `}`)?
    attr-dict-with-keyword
  }];

}

def IREEInput_TensorStoreOp : IREEInput_PureOp<"tensor.store", [
    AllTypesMatch<["target", "result"]>,
    TypesMatchWith<"value type matches element type of target operand",
                   "target", "value",
                   "$_self.cast<ShapedType>().getElementType()">,
    AttrSizedOperandSegments,
  ]> {
  let summary = [{stores a value into a tensor element}];
  let description = [{
    Returns a tensor with the element at the given index set to the given value.
  }];

  let arguments = (ins
    AnyTypeOf<[IREEInput_PrimitiveType, AnyVector]>:$value,
    IREEInput_Tensor:$target,
    IREEInput_ShapeDynamicDims:$target_dims,
    Variadic<IREEInput_Dim>:$indices
  );
  let results = (outs
    IREEInput_Tensor:$result
  );

  let assemblyFormat = [{
    $value `,` $target (`[` $indices^ `]`)? `:`
    type($target) (`{` $target_dims^ `}`)?
    attr-dict-with-keyword
  }];
}

def IREEInput_TensorSplatOp : IREEInput_PureOp<"tensor.splat", [
    TypesMatchWith<"value type matches element type of result",
                   "result", "value",
                   "$_self.cast<ShapedType>().getElementType()">,
  ]> {
  let summary = [{splats a value into a shaped tensor}];
  let description = [{
    Returns a tensor initialized to the given primitive value.
  }];

  let arguments = (ins
    IREEInput_PrimitiveType:$value,
    IREEInput_ShapeDynamicDims:$result_dims
  );
  let results = (outs
    IREEInput_Tensor:$result
  );

  let assemblyFormat = [{
    $value `:` type($result) (`{` $result_dims^ `}`)?
    attr-dict-with-keyword
  }];
}

def IREEInput_TensorCloneOp : IREEInput_PureOp<"tensor.clone", [
    AllTypesMatch<["operand", "result"]>,
  ]> {
  let summary = [{performs a full tensor clone operation}];
  let description = [{
    Clones the input tensor into an identical output tensor.
  }];

  let arguments = (ins
    IREEInput_Tensor:$operand,
    IREEInput_ShapeDynamicDims:$operand_dims
  );
  let results = (outs
    IREEInput_Tensor:$result
  );

  let assemblyFormat = [{
    $operand `:` type($result) (`{` $operand_dims^ `}`)?
    attr-dict-with-keyword
  }];
}

def IREEInput_TensorSliceOp : IREEInput_PureOp<"tensor.slice", [
    AllRanksMatch<["source", "result"]>,
    AllElementTypesMatch<["source", "result"]>,
    AttrSizedOperandSegments,
  ]> {
  let summary = [{slices out a subregion of a tensor}];
  let description = [{
    Clones a subregion of a tensor.
  }];

  let arguments = (ins
    IREEInput_Tensor:$source,
    IREEInput_ShapeDynamicDims:$source_dims,
    Variadic<IREEInput_Dim>:$start_indices,
    Variadic<IREEInput_Dim>:$lengths,
    IREEInput_ShapeDynamicDims:$result_dims
  );
  let results = (outs
    IREEInput_Tensor:$result
  );

  let assemblyFormat = [{
    $source `[` $start_indices `for` $lengths `]` `:`
    type($source) (`{` $source_dims^ `}`)? `->`
    type($result) (`{` $result_dims^ `}`)?
    attr-dict-with-keyword
  }];
}

def IREEInput_TensorUpdateOp : IREEInput_PureOp<"tensor.update", [
    AllRanksMatch<["update", "target", "result"]>,
    AllTypesMatch<["target", "result"]>,
    AllElementTypesMatch<["update", "target", "result"]>,
    AttrSizedOperandSegments,
  ]> {
  let summary = [{updates a tensor with the contents of another tensor}];
  let description = [{
    Updates the target tensor with the contents of the update tensor at the
    given offset indices.
  }];

  let arguments = (ins
    IREEInput_Tensor:$target,
    IREEInput_ShapeDynamicDims:$target_dims,
    Variadic<IREEInput_Dim>:$start_indices,
    IREEInput_Tensor:$update,
    IREEInput_ShapeDynamicDims:$update_dims
  );
  let results = (outs
    IREEInput_Tensor:$result
  );

  let assemblyFormat = [{
    $update `,` $target `[` $start_indices `]` `:`
    type($update) (`{` $update_dims^ `}`)? `->`
    type($result) (`{` $target_dims^ `}`)?
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$target,
      "ValueRange":$start_indices,
      "Value":$update)>,
  ];
}

def IREEInput_TensorTraceOp : IREEInput_Op<"tensor.trace", []> {
  let summary = [{trace value(s) operation}];
  let description = [{
    Traces out to a runtime trace sink (console, log file, etc) the given
    tensors and titles them with the given key. The key is informational only
    and useful for titling/marking specific sets of tensors for easier
    searching.
  }];

  let arguments = (ins
    StrAttr:$key,
    Variadic<IREEInput_Tensor>:$operands
  );

  let assemblyFormat = "$key attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// Workgroup dispatch
// These ops allow both scheduling and accessing parameters of workgroup
// dispatches across an arbitrary nd-grid.
//===----------------------------------------------------------------------===//

// TODO: Define dispatch.workgroup op.

def IREEInput_DispatchWorkgroupIDOp : IREEInput_PureOp<"dispatch.workgroup.id"> {
  let summary = [{returns the index of the current workgroup in the grid}];
  let description = [{
    The global workgroup ID of the current workgroup in the range of
    `[0, iree_input.dispatch.workgroup.count)` along each dimension.

    Corresponds to the `WorkgroupId` SPIR-V built-in and the `blockIdx` CUDA
    built-in variable, only in the iree dialect the number of dimensions is not
    restricted to 3 (XYZ).

    ```mlir
    %x = iree_input.dispatch.workgroup.id[0] : index
    %y = iree_input.dispatch.workgroup.id[1] : index
    ```
  }];
  let arguments = (ins IndexAttr:$dimension);
  let results = (outs IREEInput_Dim:$result);
  let builders = [
    OpBuilder<(ins "unsigned":$dim),
    [{
      build($_builder, $_state, $_builder.getIndexType(), $_builder.getIndexAttr(dim));
    }]>,
  ];
  let assemblyFormat = "`[` $dimension `]` attr-dict `:` type($result)";
}

def IREEInput_DispatchWorkgroupCountOp : IREEInput_PureOp<"dispatch.workgroup.count"> {
  let summary = [{returns the total workgroup count of the grid}];
  let description = [{
    The total number of workgroups along each dimension in the dispatch grid.

    Corresponds to the `NumWorkgroups` SPIR-V built-in and the `gridDim` CUDA
    built-in variable, only in the iree dialect the number of dimensions is not
    restricted to 3 (XYZ).

    ```mlir
    %x = iree_input.dispatch.workgroup.count[0] : index
    %y = iree_input.dispatch.workgroup.count[1] : index
    ```
  }];
  let arguments = (ins IndexAttr:$dimension);
  let results = (outs IREEInput_Dim:$result);
  let builders = [
    OpBuilder<(ins "unsigned":$dim),
    [{
      build($_builder, $_state, $_builder.getIndexType(), $_builder.getIndexAttr(dim));
    }]>,
  ];
  let assemblyFormat = "`[` $dimension `]` attr-dict `:` type($result)";
}

def IREEInput_DispatchWorkgroupSizeOp : IREEInput_PureOp<"dispatch.workgroup.size"> {
  let summary = [{returns the size of each workgroup in invocations}];
  let description = [{
    The number of local invocations within the current workgroup along each
    dimension. Depending on backend this may map to the SIMT thread count or
    inner loop nest parameters.

    Workgroup sizes are not determined at the iree dialect level as they are
    dependent on the target backend determined when lowering into the HAL. It's
    still possible to use the symbolic workgroup size inside of dispatch
    executables as a placeholder for the resolved value once in the HAL.

    Corresponds to the `WorkgroupSize` SPIR-V built-in and the `blockDim` CUDA
    built-in variable, only in the iree dialect the number of dimensions is not
    restricted to 3 (XYZ).

    ```mlir
    %x = iree_input.dispatch.workgroup.size[0] : index
    %y = iree_input.dispatch.workgroup.size[1] : index
    ```
  }];
  let arguments = (ins IndexAttr:$dimension);
  let results = (outs IREEInput_Dim:$result);
  let builders = [
    OpBuilder<(ins "unsigned":$dim),
    [{
      build($_builder, $_state, $_builder.getIndexType(), $_builder.getIndexAttr(dim));
    }]>,
  ];
  let assemblyFormat = "`[` $dimension `]` attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Utilities
// Various utility ops needed for program completeness. These arguably should
// not be exclusive to IREE.
//===----------------------------------------------------------------------===//

def IREEInput_AlignOp : IREEInput_PureOp<"align", [
  SameOperandsAndResultType
  ]> {
  let summary = "Aligns up to a power-of-two alignment if required";
  let description = [{
     Aligns |value| up to the given power-of-two |alignment| if required.
  }];

  let arguments = (ins
    SignlessIntegerLike:$value,
    SignlessIntegerLike:$alignment
  );

  let results = (outs
    SignlessIntegerLike:$result
  );

  let assemblyFormat = [{
    $value `,` $alignment attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$value,
      "int64_t":$alignment
    ),
    [{
      build($_builder, $_state, value.getType(), value,
       $_builder.createOrFold<arith::ConstantIndexOp>($_state.location, alignment));
    }]>,
  ];
}

#endif // IREE_DIALECTS_DIALECT_INPUT_OPS_TD
