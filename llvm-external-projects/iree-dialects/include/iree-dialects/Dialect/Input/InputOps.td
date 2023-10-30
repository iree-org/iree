// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_INPUT_OPS_TD
#define IREE_DIALECTS_DIALECT_INPUT_OPS_TD

include "iree-dialects/Dialect/Input/InputDialect.td"
include "iree-dialects/Dialect/Input/InputInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Pseudo ops for conversion support
//===----------------------------------------------------------------------===//

def OpGroupPseudoOps : OpDocGroup {
  let summary = "Pseudo ops for conversion support";
  let description = "";
}

let opDocGroup = OpGroupPseudoOps in {

def IREEInput_TensorExportOp : IREEInput_PureOp<"tensor.export"> {
  let summary = "Exports a tensor to a Buffer(View), capturing dynamic dims";
  let arguments = (ins
    IREEInput_Tensor:$source,
    IREEInput_ShapeDynamicDims:$source_dims
  );
  let results = (outs
    AnyTypeOf<[IREEInput_BufferType, IREEInput_BufferViewType]>:$target
  );

  let assemblyFormat = [{
    $source `:` type($source) (`{` $source_dims^ `}`)? `->` type($target)
    attr-dict-with-keyword
  }];
}

def IREEInput_TensorImportOp : IREEInput_PureOp<"tensor.import"> {
  let summary = "Imports a Buffer(View) to a tensor, providing dynamic dims";
  let arguments = (ins
    AnyTypeOf<[IREEInput_BufferType, IREEInput_BufferViewType]>:$source,
    IREEInput_ShapeDynamicDims:$target_dims
  );
  let results = (outs IREEInput_Tensor:$target);

  let assemblyFormat = [{
    $source `:` type($source) `->` type($target) (`{` $target_dims^ `}`)?
    attr-dict-with-keyword
  }];
}

} // OpGroupPseudoOps

//===----------------------------------------------------------------------===//
// Global variables
//===----------------------------------------------------------------------===//

def OpGroupGlobalVariableOps : OpDocGroup {
  let summary = "Global variable ops";
  let description = "";
}

let opDocGroup = OpGroupGlobalVariableOps in {

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

} // OpGroupGlobalVariableOps

//===----------------------------------------------------------------------===//
// Buffers and Buffer Views
//===----------------------------------------------------------------------===//

def OpGroupBufferAndBufferViewOps : OpDocGroup {
  let summary = "Buffer and buffer view ops";
  let description = "";
}

let opDocGroup = OpGroupBufferAndBufferViewOps in {

def IREEInput_BufferSubspanOp : IREEInput_PureOp<"buffer.subspan", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = [{buffer subspan operation}];
  let description = [{
    Returns a reference to a subspan of the buffer.
  }];

  let arguments = (ins
    IREEInput_BufferType:$source_buffer,
    IREEInput_DeviceSize:$source_offset,
    IREEInput_DeviceSize:$length
  );
  let results = (outs
    IREEInput_BufferType:$result
  );

  let assemblyFormat = [{
    `<` $source_buffer `:` type($source_buffer) `>`
    `` `[` $source_offset `,` $length `]`
    `:` type($result)
    attr-dict-with-keyword
  }];
}

def IREEInput_BufferViewCreateOp : IREEInput_PureOp<"buffer_view.create"> {
  let summary = [{buffer view reference initializer}];
  let description = [{
    Creates a reference to a buffer with a particular shape and element type.
    The buffer is not copied and both the original and view references must be
    synchronized. This makes it easier to associate commonly-carried metadata
    along with the contents.
  }];

  let arguments = (ins
    IREEInput_BufferType:$source_buffer,
    IREEInput_DeviceSize:$source_offset,
    IREEInput_DeviceSize:$source_length,
    IREEInput_ElementType:$element_type,
    IREEInput_EncodingType:$encoding_type,
    IREEInput_Shape:$shape
  );
  let results = (outs
    IREEInput_BufferViewType:$result
  );

  let assemblyFormat = [{
    `buffer` `(` $source_buffer `:` type($source_buffer) `)`
    `` `[` $source_offset `,` $source_length `]`
    `shape` `(` `[` $shape `]` `)`
    `type` `(` $element_type `)`
    `encoding` `(` $encoding_type `)`
    `:` type($result)
    attr-dict-with-keyword
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "Value":$sourceBuffer,
      "Value":$sourceOffset,
      "Value":$sourceLength,
      "int32_t":$elementType,
      "int32_t":$encodingType,
      "ValueRange":$shape
    )>,
    OpBuilder<(ins
      "Value":$sourceBuffer,
      "Value":$sourceOffset,
      "Value":$sourceLength,
      "Value":$elementType,
      "Value":$encodingType,
      "ValueRange":$shape
    )>,
  ];
}

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

} // OpGroupBufferAndBufferViewOps

//===----------------------------------------------------------------------===//
// Byte buffers
//===----------------------------------------------------------------------===//

def OpGroupByteBufferOps : OpDocGroup {
  let summary = "Byte buffer ops";
  let description = "";
}

let opDocGroup = OpGroupByteBufferOps in {

def IREEInput_ByteBufferConstantOp : IREEInput_PureOp<"byte_buffer.constant", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = [{constant host-side byte buffer}];
  let description = [{
    Defines a compile-time byte buffer based on the given attribute value.
    The attribute will be serialized into the canonical IREE format for the
    chosen host target.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$name,
    StrAttr:$value,
    OptionalAttr<IndexAttr>:$alignment,
    OptionalAttr<StrAttr>:$mime_type
  );
  let results = (outs
    IREEInput_ByteBufferType:$result
  );

  let assemblyFormat = [{
    ($name^)? attr-dict `:` type($result) `=` $value
  }];

  let hasVerifier = 1;
}

} // OpGroupByteBufferOps

//===----------------------------------------------------------------------===//
// Mutable Lists
//===----------------------------------------------------------------------===//

def OpGroupMutableListOps : OpDocGroup {
  let summary = "Mutable list ops";
  let description = "";
}

let opDocGroup = OpGroupMutableListOps in {

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

} // OpGroupMutableListOps

//===----------------------------------------------------------------------===//
// Tensor ops
//===----------------------------------------------------------------------===//

def OpGroupTensorOps : OpDocGroup {
  let summary = "Tensor ops";
  let description = "";
}

let opDocGroup = OpGroupTensorOps in {

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

def IREEInput_TensorBitCastOp : IREEInput_PureOp<"tensor.bitcast", [
    AllElementTypesMatch<["source", "result"]>,
    AttrSizedOperandSegments,
  ]> {
  let summary = [{bitcasts a tensor}];
  let description = [{
    Bitcasts a tensor to a new shape without modifying the contents.
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
    DeclareOpInterfaceMethods<IREEInput_TiedOpInterface, [
      "getTiedResult",
      "getTiedResultOperandIndex",
      "getTiedResultOperandIndices",
    ]>,
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
    custom<ShapedTiedResult>(type($result), $target_dims)
    attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins
      "Value":$target,
      "ValueRange":$start_indices,
      "Value":$update)>,
  ];
}

def IREEInput_TensorTraceOp : IREEInput_Op<"tensor.trace", [
  AttrSizedOperandSegments,
]> {
  let summary = [{traces one or more tensor values at runtime}];
  let description = [{
    Traces out to a runtime trace sink (console, log file, etc) the given
    tensors. The key is arbitrary and can be used for identifying the set of
    values being traced.
  }];

  let arguments = (ins
    StrAttr:$key,
    Variadic<IREEInput_Tensor>:$values,
    IREEInput_ShapeDynamicDims:$value_dims
  );

  let assemblyFormat = [{
    $key `=` `[`
    custom<ShapedOperandList>($values, type($values), $value_dims)
    `]` attr-dict-with-keyword
  }];

  let builders = [
    OpBuilder<(ins "StringRef":$key, "ValueRange":$values)>,
  ];
}

} // OpGroupTensorOps

//===----------------------------------------------------------------------===//
// Workgroup dispatch
// These ops allow both scheduling and accessing parameters of workgroup
// dispatches across an arbitrary nd-grid.
//===----------------------------------------------------------------------===//

def OpGroupWorkgroupDispatchOps : OpDocGroup {
  let summary = "Workgroup dispatch ops";
  let description = "";
}

let opDocGroup = OpGroupWorkgroupDispatchOps in {

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

} // OpGroupWorkgroupDispatchOps

//===----------------------------------------------------------------------===//
// Utilities
// Various utility ops needed for program completeness. These arguably should
// not be exclusive to IREE.
//===----------------------------------------------------------------------===//

def OpGroupUtilityOps : OpDocGroup {
  let summary = "Utility ops";
  let description = "";
}

let opDocGroup = OpGroupUtilityOps in {

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

} // OpGroupUtilityOps

//===----------------------------------------------------------------------===//
// Executable Source
//===----------------------------------------------------------------------===//

def OpGroupExecutableSourceOps : OpDocGroup {
  let summary = "Executable source ops";
  let description = "";
}

let opDocGroup = OpGroupExecutableSourceOps in {

def IREEInput_ExecutableSourceOp : IREEInput_Op<"executable.source", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"IREE::Input::ExecutableSourceEndOp">,
    Symbol,
    SymbolTable,
  ]> {
  let summary = [{generic source contents of an executable op}];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    OptionalAttr<IREEInput_ExecutableObjectsAttr>:$objects
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name
    attr-dict-with-keyword
    ``
    regions
  }];
}

def IREEInput_ExecutableSourceEndOp : IREEInput_Op<"executable.source_end", [
    HasParent<"IREE::Input::ExecutableSourceOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the executable source op}];
  let assemblyFormat = "attr-dict";
}

def IREEInput_ExecutableExportOp : IREEInput_Op<"executable.export", [
    Symbol,
    HasParent<"IREE::Input::ExecutableSourceOp">,
    IsolatedFromAbove,
  ]> {
  let summary = [{executable entry point declaration}];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    OptionalAttr<IREEInput_OrdinalAttr>:$ordinal,
    IREEInput_PipelineLayoutAttr:$layout,
    OptionalAttr<IREEInput_WorkgroupSizeAttr>:$workgroup_size,
    OptionalAttr<IREEInput_SubgroupSizeAttr>:$subgroup_size,
    OptionalAttr<IndexAttr>:$workgroup_local_memory
  );

  let builders = [
    OpBuilder<(ins
      "StringAttr":$sym_name,
      "IntegerAttr":$ordinal,
      "IREE::Input::PipelineLayoutAttr":$layout,
      "ArrayAttr":$workgroup_size,
      "IntegerAttr":$subgroup_size,
      "IntegerAttr":$workgroup_local_memory
    ), [{
      build($_builder, $_state, nullptr, sym_name, ordinal, layout,
            workgroup_size, subgroup_size, workgroup_local_memory);
    }]>,
  ];

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name
    `ordinal` `(` $ordinal `)`
    `layout` `(` $layout `)`
    attr-dict-with-keyword
  }];

  // TODO(ezhulenev): Add verifier.
}

} // OpGroupExecutableSourceOps

//===----------------------------------------------------------------------===//
// Dispatch
//===----------------------------------------------------------------------===//

def OpGroupDispatchOps : OpDocGroup {
  let summary = "Dispatch ops";
  let description = "";
}

let opDocGroup = OpGroupDispatchOps in {

def IREEInput_DispatchOp : IREEInput_PureOp<"dispatch", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<IREEInput_TiedOpInterface, [
    "getTiedOperandsIndexAndLength",
  ]>,
]> {
  let summary = [{a dispatch of an executable across a grid}];

  let arguments = (ins
    Variadic<IREEInput_Dim>:$workload,
    SymbolRefAttr:$entry_point,
    Variadic<AnyType>:$arguments,
    IREEInput_ShapeDynamicDims:$argument_dims,
    IREEInput_ShapeDynamicDims:$result_dims,
    OptionalAttr<IREEInput_TiedOpStorageAttr>:$tied_operands
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    $entry_point
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
      }]>
  ];
}

} // OpGroupDispatchOps

//===----------------------------------------------------------------------===//
// Compiler hints
//===----------------------------------------------------------------------===//

def OpGroupCompilerHintOps : OpDocGroup {
  let summary = "Compiler hint ops";
  let description = "";
}

let opDocGroup = OpGroupCompilerHintOps in {

def IREEInput_OptimizationBarrierOp : IREEInput_Op<"optimization_barrier", [
  SameOperandsAndResultType,
]> {
  let summary = "Prevents compiler optimizations across a value.";
  let description = [{
    Wraps any operands in an unoptimizable identity to prevent its results from
    being folded. It will be dropped during the final step in compilation and
    has no effect at runtime.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    attr-dict
    ($operands^ `:` type($operands))?
  }];

  let builders = [
    OpBuilder<(ins
      "ValueRange":$operands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes
    )>,
  ];

  let hasVerifier = 1;
}

} // OpGroupCompilerHintOps

#endif // IREE_DIALECTS_DIALECT_INPUT_OPS_TD
