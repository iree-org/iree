// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_MICRO_KERNEL_OPS
#define IREE_CODEGEN_DIALECT_MICRO_KERNEL_OPS

include "iree/compiler/Codegen/Dialect/IREECodegenDialect.td"
include "iree/compiler/Codegen/Interfaces/MicroKernelOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

class IREECodegen_MicroKernelOp<string mnemonic, list<Trait> traits = []> : 
  Op<IREECodegen_Dialect, mnemonic, !listconcat(traits,
    [DeclareOpInterfaceMethods<MicroKernelOpInterface,
        ["lowerToFunctionCall"]>,
     DeclareOpInterfaceMethods<DestinationStyleOpInterface>])> {}
  
def IREECodegen_GenericUKernelOp :
    IREECodegen_MicroKernelOp<"ukernel.generic", [
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
    
    The operands of `memref` type are expected to lower to the following
    argument sequence.
    - `memref<f32> lowers to base_ptr.
    - `memref<?xf32, <strides = [s0, 1], offsets = f>>
          lowers to (base_ptr, f, index s0)
    - `memref<?x?xf32, <strides = [s1, s0, 1], offsets = f>>
          lowers to (base_ptr, f, s1, s0)
    and so on....

    Innermost stride of `memref`s are expected to be 1.
    All other operands are expected to be scalar types.
    TODO: `vector` types can be supported as well, but needs better
    ABI specification.
  }];

  let arguments = (ins
    StrAttr:$micro_kernel_fn_name,
    Variadic<AnyType>:$inputs,
    Variadic<AnyRankedTensorOrMemRefType>:$outputs,
    Variadic<AnyType>:$other_operands);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    attr-dict $micro_kernel_fn_name
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^  `:` type($outputs) `)`)?
    (`(` $other_operands^ `:` type($other_operands) `)`)? (`->` type($results)^)?
  }];
}

def IREECodegen_Mmt4DUKernelOp :
    IREECodegen_MicroKernelOp<"ukernel.mmt4d", []>  {
  let summary = "Mmt4D Microkernel operator";

  let description = [{
    Operation to wrap a computation with root being a `linalg.mmt4d` 
    forwarded to a microkernel.

    This operation is meant to wrap a DAG whose is `linalg.mmt4d`-like
    operation and forwarded to a micro kernel. Unlike `ukernel.generic`,
    lowering into this operation does not fix the ABI of the micro kernel
    function. When lowering to a function, the function call is expected
    to match the ABI of the corresponding micro-kernel used.
  }];

  let arguments = (ins
    RankedTensorOrMemRefType<[AnyType], [4]>:$lhs,
    RankedTensorOrMemRefType<[AnyType], [4]>:$rhs,
    RankedTensorOrMemRefType<[AnyType], [4]>:$output,
    DefaultValuedAttr<BoolAttr, "false">:$accumulate);
  let results = (outs Optional<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict `lhs` `(` $lhs `:` type($lhs) `)`
    `rhs` `(` $rhs `:` type($rhs) `)`
    `outs` `(` $output `:` type($output) `)`
    `accumulate` `(` $accumulate `)` (`->` type($result)^)?
  }];
  let extraClassDeclaration = [{
    Type getLhsElementType() {
      return getLhs().getType().cast<ShapedType>().getElementType();
    }
    Type getRhsElementType() {
      return getRhs().getType().cast<ShapedType>().getElementType();
    }
    Type getOutputElementType() {
      return getOutput().getType().cast<ShapedType>().getElementType();
    }
  }];
}

#endif // IREE_CODEGEN_DIALECT_MICRO_KERNEL_OPS
