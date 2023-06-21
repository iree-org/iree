// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_UKERNEL_OPS
#define IREE_CODEGEN_DIALECT_UKERNEL_OPS

include "iree/compiler/Codegen/Dialect/IREECodegenDialect.td"
include "iree/compiler/Codegen/Interfaces/UKernelOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

class IREECodegen_UKernelOp<string mnemonic, list<Trait> traits = []> : 
  Op<IREECodegen_Dialect, mnemonic, !listconcat(traits,
    [DeclareOpInterfaceMethods<UKernelOpInterface,
        ["lowerToFunctionCall"]>,
     DeclareOpInterfaceMethods<DestinationStyleOpInterface>])> {}
  
def IREECodegen_UKernelGenericOp :
    IREECodegen_UKernelOp<"ukernel.generic", [
      AttrSizedOperandSegments]> {
  let summary = "Generic Microkernel operator";

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
    overridden by the optional `strided_outer_dims` attribute: if set, its value
    is how many strides (started from the outer-most dimension) are passed for
    each memref. For memrefs of rank less than the value of
    `strided_outer_dims`, all strides are passed.

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
    OptionalAttr<IndexAttr>:$strided_outer_dims);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    attr-dict $u_kernel_fn_name
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^  `:` type($outputs) `)`)?
    (`(` $other_operands^ `:` type($other_operands) `)`)?
    ( `fn_def_attrs` $fn_def_attrs^ )?
    (`strided_outer_dims` `(` $strided_outer_dims^ `)`)? (`->` type($results)^)?
  }];
}

#endif // IREE_CODEGEN_DIALECT_UKERNEL_OPS
