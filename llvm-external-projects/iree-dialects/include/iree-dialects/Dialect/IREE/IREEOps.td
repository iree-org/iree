// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREE_IREE_OPS_TD
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREE_IREE_OPS_TD

include "iree-dialects/Dialect/IREE/IREEDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def IREE_NullOp : IREE_PureOp<"null"> {
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

def IREE_TensorToBufferViewOp : IREE_PureOp<"cast.tensor_to_buffer_view"> {
  let summary = "Casts a tensor to a BufferView, capturing dynamic dims";
  let arguments = (ins
    IREE_Tensor:$source,
    IREE_ShapeDynamicDims:$source_dims
  );
  let results = (outs IREE_BufferViewType:$target);

  let assemblyFormat = [{
    $source `:` type($source) (`{` $source_dims^ `}`)? `->` type($target)
    attr-dict-with-keyword
  }];
}

def IREE_BufferViewToTensorOp : IREE_PureOp<"cast.buffer_view_to_tensor"> {
  let summary = "Casts a BufferView to a tensor, providing dynamic dims";
  let arguments = (ins
    IREE_BufferViewType:$source,
    IREE_ShapeDynamicDims:$target_dims
  );
  let results = (outs IREE_Tensor:$target);

  let assemblyFormat = [{
    $source `:` type($source) `->` type($target) (`{` $target_dims^ `}`)?
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// Global variables
//===----------------------------------------------------------------------===//

def IREE_GlobalOp : IREE_Op<"global", [
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
    OptionalAttr<AnyAttr>:$initial_value
  );

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    (`mutable` $is_mutable^)?
    $sym_name
    attr-dict
    (`initializer` `(` $initializer^ `)`):(``)?
    custom<TypeOrAttr>($type, $initial_value)
  }];
}

def IREE_GlobalAddressOp : IREE_PureOp<"global.address"> {
  let summary = [{returns an address reference to a global}];
  let description = [{
    Returns the address of a global as a typed reference. Can be used with the
    global load and store indirect ops.
  }];

  let arguments = (ins
    IREE_GlobalRefAttr:$global
  );
  let results = (outs
    IREE_AnyGlobalPtr:$result
  );

  let assemblyFormat = [{
    $global attr-dict `:` type($result)
  }];
}

def IREE_GlobalLoadOp : IREE_Op<"global.load"> {
  let summary = [{loads a value from a global variable}];
  let description = [{
    Returns a copy of the global value.
  }];

  let arguments = (ins
    IREE_GlobalRefAttr:$global
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $global attr-dict `:` type($result)
  }];
}

def IREE_GlobalLoadIndirectOp : IREE_Op<"global.load.indirect"> {
  let summary = [{loads a value from a global variable}];
  let description = [{
    Returns a copy of the global value.
  }];

  let arguments = (ins
    IREE_AnyGlobalPtr:$global
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = [{
    $global attr-dict `:` type($global) `->` type($result)
  }];
}

def IREE_GlobalStoreOp : IREE_Op<"global.store"> {
  let summary = [{stores a value into a global variable}];
  let description = [{
    Stores a copy of the value into a global.
  }];

  let arguments = (ins
    AnyType:$value,
    IREE_GlobalRefAttr:$global
  );

  let assemblyFormat = [{
    $value `,` $global attr-dict `:` type($value)
  }];
}

def IREE_GlobalStoreIndirectOp : IREE_Op<"global.store.indirect"> {
  let summary = [{stores a value into a global variable}];
  let description = [{
    Stores a copy of the value into a global.
  }];

  let arguments = (ins
    AnyType:$value,
    IREE_AnyGlobalPtr:$global
  );

  let assemblyFormat = [{
    $value `,` $global attr-dict `:` type($value) `->` type($global)
  }];
}

//===----------------------------------------------------------------------===//
// Buffer Views
//===----------------------------------------------------------------------===//

def IREE_BufferViewRankOp : IREE_PureOp<"buffer_view.rank"> {
  let summary = [{buffer view rank query}];
  let description = [{
    Returns the rank of the buffer view.
  }];

  let arguments = (ins
    IREE_BufferViewType:$buffer_view
  );
  let results = (outs
    IREE_Dim:$result
  );

  let assemblyFormat = [{
    $buffer_view attr-dict `:` type($result)
  }];
}

def IREE_BufferViewDimOp : IREE_PureOp<"buffer_view.dim"> {
  let summary = [{buffer view dimension value query}];
  let description = [{
    Returns the value of the given dimension.
  }];

  let arguments = (ins
    IREE_BufferViewType:$buffer_view,
    IndexAttr:$index
  );
  let results = (outs
    IREE_Dim:$result
  );

  let assemblyFormat = [{
    $buffer_view `,` $index attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Mutable Lists
//===----------------------------------------------------------------------===//

def IREE_ListCreateOp : IREE_PureOp<
    "list.create", [MemoryEffects<[MemAlloc]>]> {
  let summary = [{creates a new empty list}];
  let description = [{
    Creates a new empty list with an optional initial capacity.
  }];

  let arguments = (ins
    Optional<Index>:$initial_capacity
  );
  let results = (outs
    IREE_AnyList:$result
  );

  let assemblyFormat = "($initial_capacity^)? attr-dict `:` type($result)";
}

def IREE_ListSizeOp : IREE_Op<"list.size", [MemoryEffects<[MemRead]>]> {
  let summary = [{the size of the list in elements}];
  let description = [{
    Returns the current size of the list in elements.
  }];

  let arguments = (ins
    IREE_AnyList:$list
  );
  let results = (outs
    Index:$result
  );

  let assemblyFormat = "operands attr-dict `:` type($list)";
}

def IREE_ListResizeOp : IREE_Op<"list.resize", [MemoryEffects<[MemWrite]>]> {
  let summary = [{resizes the list to a new count in elements}];
  let description = [{
    Resizes the list to contain `new_size` elements. This will either truncate
    the list if the existing size is greater than `new_size` or extend the list
    with the default list value of the element type.
  }];

  let arguments = (ins
    IREE_AnyList:$list,
    Index:$new_size
  );

  let assemblyFormat = "operands attr-dict `:` type($list)";
}

def IREE_ListGetOp : IREE_Op<"list.get", [MemoryEffects<[MemRead]>]> {
  let summary = [{element accessor}];
  let description = [{
    Returns the value of the element at the given index. Note that the value
    may be null if the element is null or the type does not match.
  }];

  let arguments = (ins
    IREE_AnyList:$list,
    Index:$index
  );
  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = "$list `[` $index `]` attr-dict `:` type($list) `->` type($result)";
}

def IREE_ListSetOp : IREE_Op<"list.set", [MemoryEffects<[MemWrite]>]> {
  let summary = [{element mutator}];
  let description = [{
    Sets the element at the given index to the new value.
  }];

  let arguments = (ins
    IREE_AnyList:$list,
    Index:$index,
    AnyType:$value
  );

  let assemblyFormat = "$list `[` $index `]` `,` $value attr-dict `:` type($list) `,` type($value)";
}

//===----------------------------------------------------------------------===//
// Tensor ops
//===----------------------------------------------------------------------===//

def IREE_TensorReshapeOp : IREE_PureOp<"tensor.reshape", [
    AllElementTypesMatch<["source", "result"]>,
    AttrSizedOperandSegments,
  ]> {
  let summary = [{reshapes a tensor}];
  let description = [{
    Reshapes a tensor to a new shape without modifying the contents.
  }];

  let arguments = (ins
    IREE_Tensor:$source,
    IREE_ShapeDynamicDims:$source_dims,
    IREE_ShapeDynamicDims:$result_dims
  );
  let results = (outs
    IREE_Tensor:$result
  );

  let assemblyFormat = [{
    $source `:`
    type($source) (`{` $source_dims^ `}`)? `->`
    type($result) (`{` $result_dims^ `}`)?
    attr-dict-with-keyword
  }];
}

def IREE_TensorLoadOp : IREE_PureOp<"tensor.load", [
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
    IREE_Tensor:$source,
    IREE_ShapeDynamicDims:$source_dims,
    Variadic<IREE_Dim>:$indices
  );
  let results = (outs
    AnyTypeOf<[IREE_PrimitiveType, AnyVector]>:$result
  );

  let assemblyFormat = [{
    $source (`[` $indices^ `]`)? `:`
    type($source) (`{` $source_dims^ `}`)?
    attr-dict-with-keyword
  }];

}

def IREE_TensorStoreOp : IREE_PureOp<"tensor.store", [
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
    AnyTypeOf<[IREE_PrimitiveType, AnyVector]>:$value,
    IREE_Tensor:$target,
    IREE_ShapeDynamicDims:$target_dims,
    Variadic<IREE_Dim>:$indices
  );
  let results = (outs
    IREE_Tensor:$result
  );

  let assemblyFormat = [{
    $value `,` $target (`[` $indices^ `]`)? `:`
    type($target) (`{` $target_dims^ `}`)?
    attr-dict-with-keyword
  }];
}

def IREE_TensorSplatOp : IREE_PureOp<"tensor.splat", [
    TypesMatchWith<"value type matches element type of result",
                   "result", "value",
                   "$_self.cast<ShapedType>().getElementType()">,
  ]> {
  let summary = [{splats a value into a shaped tensor}];
  let description = [{
    Returns a tensor initialized to the given primitive value.
  }];

  let arguments = (ins
    IREE_PrimitiveType:$value,
    IREE_ShapeDynamicDims:$result_dims
  );
  let results = (outs
    IREE_Tensor:$result
  );

  let assemblyFormat = [{
    $value `:` type($result) (`{` $result_dims^ `}`)?
    attr-dict-with-keyword
  }];
}

def IREE_TensorCloneOp : IREE_PureOp<"tensor.clone", [
    AllTypesMatch<["operand", "result"]>,
  ]> {
  let summary = [{performs a full tensor clone operation}];
  let description = [{
    Clones the input tensor into an identical output tensor.
  }];

  let arguments = (ins
    IREE_Tensor:$operand,
    IREE_ShapeDynamicDims:$operand_dims
  );
  let results = (outs
    IREE_Tensor:$result
  );

  let assemblyFormat = [{
    $operand `:` type($result) (`{` $operand_dims^ `}`)?
    attr-dict-with-keyword
  }];
}

def IREE_TensorSliceOp : IREE_PureOp<"tensor.slice", [
    AllRanksMatch<["source", "result"]>,
    AllElementTypesMatch<["source", "result"]>,
    AttrSizedOperandSegments,
  ]> {
  let summary = [{slices out a subregion of a tensor}];
  let description = [{
    Clones a subregion of a tensor.
  }];

  let arguments = (ins
    IREE_Tensor:$source,
    IREE_ShapeDynamicDims:$source_dims,
    Variadic<IREE_Dim>:$start_indices,
    Variadic<IREE_Dim>:$lengths,
    IREE_ShapeDynamicDims:$result_dims
  );
  let results = (outs
    IREE_Tensor:$result
  );

  let assemblyFormat = [{
    $source `[` $start_indices `for` $lengths `]` `:`
    type($source) (`{` $source_dims^ `}`)? `->`
    type($result) (`{` $result_dims^ `}`)?
    attr-dict-with-keyword
  }];
}

def IREE_TensorUpdateOp : IREE_PureOp<"tensor.update", [
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
    IREE_Tensor:$target,
    IREE_ShapeDynamicDims:$target_dims,
    Variadic<IREE_Dim>:$start_indices,
    IREE_Tensor:$update,
    IREE_ShapeDynamicDims:$update_dims
  );
  let results = (outs
    IREE_Tensor:$result
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

def IREE_TensorTraceOp : IREE_Op<"tensor.trace", []> {
  let summary = [{trace value(s) operation}];
  let description = [{
    Traces out to a runtime trace sink (console, log file, etc) the given
    tensors and titles them with the given key. The key is informational only
    and useful for titling/marking specific sets of tensors for easier
    searching.
  }];

  let arguments = (ins
    StrAttr:$key,
    Variadic<IREE_Tensor>:$operands
  );

  let assemblyFormat = "$key attr-dict ($operands^ `:` type($operands))?";
}

#endif // IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREE_IREE_OPS_TD
