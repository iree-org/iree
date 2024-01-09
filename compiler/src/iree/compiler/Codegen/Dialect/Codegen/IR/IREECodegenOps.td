// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_IREECODEGENOPS
#define IREE_CODEGEN_DIALECT_IREECODEGENOPS

include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TensorTypeAttr : TypeAttrBase<"TensorType", "Tensor type attribute">;
  
def IREECodegen_QueryTileSizesOp :
    Op<IREECodegen_Dialect, "query_tile_sizes", [Pure]> {
  let summary = "Query tile sizes";

  let description = [{
    Query tile sizes
  }];

  let arguments = (ins TensorTypeAttr:$tensor_type);
  let results = (outs Variadic<Index>:$results);
  let assemblyFormat = [{
    attr-dict $tensor_type `->` type($results)
  }];
}

#endif // IREE_CODEGEN_DIALECT_IREECODEGENOPS
