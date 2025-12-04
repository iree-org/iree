// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXT_ATTRS
#define IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXT_ATTRS

include "iree/compiler/Dialect/TensorExt/IR/TensorExtBase.td"
include "iree/compiler/Dialect/TensorExt/IR/TensorExtInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Barrier Enums and Attributes
//===----------------------------------------------------------------------===//

// Transformation flags that control which operations can move through barriers
def IREETensorExt_TransformationFlag_None : I32BitEnumAttrCase<"None", 0x0000>;
def IREETensorExt_TransformationFlag_AllowExpand : I32BitEnumAttrCase<"AllowExpand", 0x0001>;
def IREETensorExt_TransformationFlag_AllowCollapse : I32BitEnumAttrCase<"AllowCollapse", 0x0002>;
def IREETensorExt_TransformationFlagBitfieldAttr :
    I32BitEnumAttr<"TransformationFlagBitfield", "valid TransformationFlag", [
      IREETensorExt_TransformationFlag_None,
      IREETensorExt_TransformationFlag_AllowExpand,
      IREETensorExt_TransformationFlag_AllowCollapse,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::TensorExt";
}

// Direction that operations can move through the barrier
def IREETensorExt_BarrierDirection_Up : I32EnumAttrCase<"Up", 0, "up">;
def IREETensorExt_BarrierDirection_Down : I32EnumAttrCase<"Down", 1, "down">;
def IREETensorExt_BarrierDirectionAttr :
    I32EnumAttr<"BarrierDirection", "valid BarrierDirection", [
      IREETensorExt_BarrierDirection_Up,
      IREETensorExt_BarrierDirection_Down,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::TensorExt";
}

//===----------------------------------------------------------------------===//
// RaggedShape Attribute
//===----------------------------------------------------------------------===//

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

    If `raggedRow` is `n`, then the dimension `n` and `n+1` of the shaped type are
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

#endif  // IREE_COMPILER_DIALECT_TENSOREXT_IR_TENSOREXT_ATTRS
