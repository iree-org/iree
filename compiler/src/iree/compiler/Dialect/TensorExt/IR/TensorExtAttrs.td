// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_TENSOR_EXT_ATTRS
#define IREE_DIALECT_TENSOR_EXT_ATTRS

include "iree/compiler/Dialect/TensorExt/IR/TensorExtBase.td"
include "iree/compiler/Dialect/TensorExt/IR/TensorExtInterfaces.td"

def RaggedShapeAttr :
  AttrDef<IREETensorExt_Dialect, "RaggedShape", [
    DeclareAttrInterfaceMethods<IREETensorExt_SparseShapeAttrInterface>,
    DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface, [
      "getAffineMap",
      "isIdentity",
      "verifyLayout",
      "getStridesAndOffset"
    ]>]> {
  let mnemonic = "ragged_shape";

  let parameters = (ins "int64_t":$raggedRow);

  let assemblyFormat = "`<` $raggedRow `>`";

  let description = [{
    Attribute to encode that a shaped type is ragged shaped.

    if `raggedRow` is `n`, then the dimension `n` and `n+1` of the shaped type are
    "ragged" dimensions, i.e. the size of the `n+1` dimension of the shaped type
    varies. So a 2D ragged shaped type would have `raggedRow` as `0` where the size
    of dimension 1 is different as shown below.

    ```
    OOOO
    OOOOOO
    OOO
    OOOOO
    ```

    The shape is "logical" view of the ragged dimensions,
    i.e. the `n+1`-th dimension should always be dynamic. So the 2D shaped type
    above has a logical shape of `4x?`.
  }];
}

#endif // IREE_DIALECT_TENSOR_EXT_ATTRS
