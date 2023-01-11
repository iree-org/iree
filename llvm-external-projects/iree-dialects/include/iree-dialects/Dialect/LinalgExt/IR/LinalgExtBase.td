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
  let description = [{
    The `iree_linalg_ext` dialect is intended to experiment more support for
    non-structured operations, ie, can not be represented in Linalg operations.
  }];
  let hasCanonicalizer = 1;
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

// List of pre-defined data layout encoding attributes.
def GEMM_LHS
    : I32EnumAttrCase<"GEMM_LHS", 0>;
def GEMM_RESULT
    : I32EnumAttrCase<"GEMM_RESULT", 1>;
def GEMM_RHS
    : I32EnumAttrCase<"GEMM_RHS", 2>;
def GEMM_RHS_TRANSPOSE
    : I32EnumAttrCase<"GEMM_RHS_TRANSPOSE", 3>;

def TensorEncodingEnum
    : I32EnumAttr<"TensorEncoding",
                  "identifier for encoding used for the tensor",[
                    GEMM_LHS, GEMM_RESULT, GEMM_RHS, GEMM_RHS_TRANSPOSE
                  ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::LinalgExt";
  let genSpecializedAttr = 0;
}

def TensorEncodingAttr :
  EnumAttr<IREELinalgExt_Dialect, TensorEncodingEnum, ""> {
  let assemblyFormat = "``$value";
}

def IREELinalgExt_EncodingAttr : IREELinalgExt_Attr<"Encoding"> {
  let mnemonic = "encoding";
  let summary = [{tensor layout encoding}];
  let description = [{
    This attribute describes the change in the layout for
    a given tensor to execute subsequent operations on
    the tiled layout. The encoding serves as a way to
    represent the change in the way the data is laid out in
    memory without changing the logical rank/extent of
    the tensor itself. When required, the encoding
    can be used to explicitly manifest the layout change
    through operations like pack/unpack.

    Currently the encoding is just an enum that describes
    in an ad-hoc fashions the data layouts we initially care
    about. In fullness of time the encoding attribute can be
    made richer.
  }];

  let parameters = (ins
    AttrParameter<"IREE::LinalgExt::TensorEncodingAttr",
        "Tensor encoding to use for a tensor">:$encoding
  );

  let assemblyFormat = [{
    `<` `` $encoding `>`
  }];

  let builders = [
    AttrBuilder<(ins "TensorEncoding":$encoding)>
  ];
}


#endif // IREE_DIALECT_LINALGEXT_BASE
