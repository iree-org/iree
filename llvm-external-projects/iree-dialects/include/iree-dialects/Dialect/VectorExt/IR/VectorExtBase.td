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
}

//===---------------------------------------------------------------------===//
// Vector layout attributes
//===---------------------------------------------------------------------===//

class IREEVectorExt_Attr<string name, list<Trait> traits = []>
  : AttrDef<IREEVectorExt_Dialect, name, traits>;

def PerDimLayoutAttr : IREEVectorExt_Attr<"PerDimLayout"> {
   let mnemonic = "per_dim_layout";
   let summary = [{high-dimensional vector register layout for a given vector dimension}];
   let description = [{
    This attribute describes the per dimension register layout for a given vector
    that could be prescribed by an operator such as matrix multiplication.
    This is a way to explicitly represent the layout in the IR
    when it is in the SIMD form prior to converting to the SIMT form so that
    we can reason about layouts, propagating layouts and layout conflicts.
   }];
   let parameters = (ins
     ArrayRefParameter<"std::string", "labels for the high dimensional layout dims">:$labels,
     ArrayRefParameter<"int64_t", "shapes for the high dimensional layout dims">:$shapes
   );
   let hasCustomAssemblyFormat = 1;
   let genVerifyDecl = 0;
}

def LayoutAttr : IREEVectorExt_Attr<"Layout"> {
  let mnemonic = "layout";
  let summary = [{high-dimensional vector register layout for a given vector}];
  let description = [{
    This contains a complete specification of the layout for a given vector,
    whereas the attribute above only specifies the per dimension layout.
  }];
  let parameters = (ins
    ArrayRefParameter<"PerDimLayoutAttr", "layout for each dimension of the vector">:$layouts
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 0;
}

#endif // IREE_DIALECT_VECTOREXT_BASE

