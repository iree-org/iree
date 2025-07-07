// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_UKERNEL_OPS
#define IREE_CODEGEN_DIALECT_UKERNEL_OPS

include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.td"
include "iree/compiler/Codegen/Interfaces/UKernelOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class IREECodegen_UKernelOp<string mnemonic, list<Trait> traits = []> :
  Op<IREECodegen_Dialect, mnemonic, !listconcat(traits,
    [DeclareOpInterfaceMethods<UKernelOpInterface,
        ["lowerToFunctionCall"]>,
     DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>])> {}

def IREECodegen_UKernelGenericOp :
    IREECodegen_UKernelOp<"ukernel.generic", [
      AttrSizedOperandSegments]> {
  let summary = [{Generic Microkernel operator.}];

  let description = [{
    Operation to wrap a computation forwarded to a microkernel.

    This operation is a generic representation of the DAG that is to be
    lowered into a micro-kernel. The name of the microkernel is specified
    as a `StrAttr`. The DAG to be forwarded is meant to be captured at
    tensor-level. The operation implements the `DestinationStyleOpInterface`
    so all tensors in the `outs` list must match the number and type of the
    results of the operation.
    After bufferization the tensor operands in `outs` are converted to
    a memref type. At the memref-level, the operands are expected to
    match directly into a function call with the arguments to the
    function call being the `ins`, `outs` and `other_operands`.

    The operands of `memref` type are expected to lower to this sequence of
    function arguments:
      `base_pointer, offset, stride0, stride1, stride2, ...`
    By default, all N strides are passed for each memref of rank N. This can be
    overridden by the optional `strided_dims` attribute: if set, its value
    is a list of dimensions for which strides are passed for each ShapedType
    operand, in the order that they appear as operands. Non-ShapedType operands
    are ignored in the strided_dims list.

    Note that `memref` semantics only guarantee that a `base_pointer + offset`
    represents the position to read from. So all micro-kernel implementations
    are expected to take at least a `base_pointer, offset` pair for each
    operand (input or output) of `memref` type. The `offset` has to be added to
    the `base_pointer` before dereferencing to read/write data. Also note that
    the `offset` is in number of elements.

    All other operands are expected to be scalar types.
    TODO: `vector` types can be supported as well, but needs better
    ABI specification.
  }];

  let arguments = (ins
    StrAttr:$u_kernel_fn_name,
    Variadic<AnyType>:$inputs,
    Variadic<AnyRankedTensorOrMemRefType>:$outputs,
    Variadic<AnyType>:$other_operands,
    OptionalAttr<DictionaryAttr>:$fn_def_attrs,
    OptionalAttr<IndexListArrayAttr>:$strided_dims);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    attr-dict $u_kernel_fn_name
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^  `:` type($outputs) `)`)?
    (`(` $other_operands^ `:` type($other_operands) `)`)?
    ( `fn_def_attrs` $fn_def_attrs^ )?
    (`strided_dims` `(` $strided_dims^ `)`)? (`->` type($results)^)?
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$result_types, "StringRef":$u_kernel_fn_name,
      "ValueRange":$inputs, "ValueRange":$outputs, "ValueRange":$other_operands,
      "DictionaryAttr":$fn_def_attrs, "int64_t":$num_strided_outer_dims)>,
    OpBuilder<(ins "TypeRange":$result_types, "StringRef":$u_kernel_fn_name,
      "ValueRange":$inputs, "ValueRange":$outputs, "ValueRange":$other_operands,
      "DictionaryAttr":$fn_def_attrs,
      "ArrayRef<SmallVector<int64_t>>":$strided_dims)>
  ];
  let extraClassDeclaration = [{
    // Get the list of strided dims for the operand at `operandIdx`.
    //   - Always returns and empty list for non ShapedType operands.
    //   - Returns {0, 1, ..., N-1} for ShapedType operands with rank N, when
    //     the `strided_dims` attribute is null.
    //   - Returns the corresponding dim list in `strided_dims` for ShapedType
    //     operands if `strided_dims` is not null.
    SmallVector<int64_t> getOperandStridedDims(int64_t operandIdx);
  }];
}

#endif // IREE_CODEGEN_DIALECT_UKERNEL_OPS
