// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VECTOREXT_BASE
#define IREE_DIALECT_VECTOREXT_BASE

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

include "iree-dialects/Dialect/VectorExt/IR/VectorExtInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def IREEVectorExt_Dialect : Dialect {
  let name = "iree_vector_ext";
  let cppNamespace = "::mlir::iree_compiler::IREE::VectorExt";
  let summary = [{
    IREE Vector Extensions.
  }];
  let description = [{
    A dialect designed for experimenting with vector operations
    beyond what is currently available in the Vector Dialect.
  }];
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerAttributes();
  }];
}

//===---------------------------------------------------------------------===//
// Vector layout attribute helpers
//===---------------------------------------------------------------------===//

class IREEVectorExt_Attr<string name, list<Trait> traits = []>
  : AttrDef<IREEVectorExt_Dialect, name, traits>;

class IREEVectorExt_I32EnumAttr<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::iree_compiler::IREE::VectorExt";
  let genSpecializedAttr = 0;
}

class IREEVectorExt_EnumAttr<EnumAttrInfo enumInfo, string name = "">
  : EnumAttr<IREEVectorExt_Dialect, enumInfo, name>;

#endif // IREE_DIALECT_VECTOREXT_BASE

