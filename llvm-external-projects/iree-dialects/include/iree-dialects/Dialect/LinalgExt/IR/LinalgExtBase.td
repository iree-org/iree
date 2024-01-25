// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_BASE
#define IREE_DIALECT_LINALGEXT_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def IREELinalgExt_Dialect : Dialect {
  let name = "iree_linalg_ext";
  let cppNamespace = "::mlir::iree_compiler::IREE::LinalgExt";
  let summary = [{
    IREE Linalg Extensions.
  }];
  let description = [{
    A dialect designed for experimenting with non-structured operations that
    cannot be represented efficiently/directly by the Linalg dialect.
  }];
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

class RankedTensorOrMemRefOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes,
      Or<[IsMemRefTypePred, And<[IsTensorTypePred, HasRankPred]>]>,
  "ranked tensor or memref", "::mlir::ShapedType">;

def AnyRankedTensorOrMemRefType : RankedTensorOrMemRefOf<[AnyType]>;

//===---------------------------------------------------------------------===//
// Data layout encoding attributes
//===---------------------------------------------------------------------===//

class IREELinalgExt_Attr<string name, list<Trait> traits = []>
  : AttrDef<IREELinalgExt_Dialect, name, traits>;

class IREELinalgExt_I32EnumAttr<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::iree_compiler::IREE::LinalgExt";
  let genSpecializedAttr = 0;
}

class IREELinalgExt_EnumAttr<EnumAttrInfo enumInfo, string name = "">
  : EnumAttr<IREELinalgExt_Dialect, enumInfo, name>;

def LHS : I32EnumAttrCase<"LHS", 0>;
def RHS : I32EnumAttrCase<"RHS", 1>;
def RESULT : I32EnumAttrCase<"RESULT", 2>;

def EncodingRole : IREELinalgExt_I32EnumAttr<"EncodingRole",
    "Describes the role of the tensor as an operand or a result of an operation.", [
      LHS,
      RHS,
      RESULT,
    ]>;

def EncodingRoleAttr :
  IREELinalgExt_EnumAttr<EncodingRole, "role">;

def EncodingAttr :
    IREELinalgExt_Attr<"Encoding"> {
  let mnemonic = "encoding";
  let summary = [{information to decide how to data-tile a tensor}];
   let description = [{
    This attribute describes the change in the layout for
    a given tensor to execute subsequent operations on
    the tiled layout. The encoding serves as a way to
    represent the change in the way the data is laid out in
    memory without changing the logical rank/extent of
    the tensor itself. When required, the encoding
    can be used to explicitly manifest the layout change
    through operations like pack/unpack.
  }];

  let assemblyFormat = "`<` struct(params) `>`";

  let parameters = (ins
    AttrParameter<"EncodingRoleAttr", "role of this tensor as an operand">:$role,
    AttrParameter<"ArrayAttr", "element types of the user's operands">:$element_types,
    OptionalParameter<"TypeAttr", "type of the original tensor type before padding">:$original_type,
    // TODO(#15466): generalize matmul_narrow_{M,N} into a list?
    OptionalParameter<"IntegerAttr", "optional M narrow dimension size (only for contraction op user_indexing_maps)">:$matmul_narrow_M,
    OptionalParameter<"IntegerAttr", "optional N narrow dimension size (only for contraction op user_indexing_maps)">:$matmul_narrow_N,
    OptionalParameter<"ArrayAttr", "Indexing maps of the operation using this tensor">:$user_indexing_maps
  );

  let genVerifyDecl = 0;
}

def RequiresSyncAttr : IREELinalgExt_Attr<"RequiresSync"> {
  let mnemonic = "requires_sync";
  let summary = [{A tensor encoding for tensors needing synchronization}];
   let description = [{
    A tensor encoding which is used to indicate that the buffer which this tensor
    will eventually bufferize to, needs to be synchronized between reads and
    writes, i.e. there needs to be explicit synchronization for Read-After-Write
    and Write-After-Read.
  }];
}

#endif // IREE_DIALECT_LINALGEXT_BASE
