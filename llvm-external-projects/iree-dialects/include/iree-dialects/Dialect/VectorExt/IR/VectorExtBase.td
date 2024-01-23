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
}

//===---------------------------------------------------------------------===//
// Vector layout attributes
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

// Defines the batch dimensions for the original SIMD tensor.
// By convention, X is along rows and Y along columns.
def BATCHX : I32EnumAttrCase<"BATCHX", 0>;
def BATCHY : I32EnumAttrCase<"BATCHY", 1>;
// Defines the vector dimension.
def VECTORX : I32EnumAttrCase<"VECTORX", 2>;
def VECTORY : I32EnumAttrCase<"VECTORY", 3>;
def VECTORZ : I32EnumAttrCase<"VECTORZ", 4>;
// Defines the lane dimensions.
def LANEX : I32EnumAttrCase<"LANEX", 5>;
def LANEY : I32EnumAttrCase<"LANEY", 6>;
def LANEZ : I32EnumAttrCase<"LANEZ", 7>;

def LayoutDimension : IREEVectorExt_I32EnumAttr<"LayoutDimension",
    "Describes the dimension of the high-dimensional layout", [
      BATCHX,
      BATCHY,
      VECTORX,
      VECTORY,
      VECTORZ,
      LANEX,
      LANEY,
      LANEZ,
    ]>;

def LayoutDimensionAttr : IREEVectorExt_EnumAttr<LayoutDimension, "dimension">;

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
     ArrayRefParameter<"LayoutDimensionAttr", "labels for the high dimensional layout dims">:$labels,
     ArrayRefParameter<"int64_t", "shapes for the high dimensional layout dims">:$shapes
   );
   let assemblyFormat = "`<``[` $labels `]``,` `[` $shapes `]``>`";
   let genVerifyDecl = 0;
   let extraClassDeclaration = [{
      std::optional<int64_t> getShape(const LayoutDimension &dim);
      bool contains(const LayoutDimension &dim);
   }];
}

def LayoutAttr : IREEVectorExt_Attr<"Layout",
      [ DeclareAttrInterfaceMethods<VectorLayoutInterface> ]> {
  let mnemonic = "layout";
  let summary = [{high-dimensional vector register layout for a given vector}];
  let description = [{
    This contains a complete specification of the layout for a given vector,
    whereas the attribute above only specifies the per dimension layout.
  }];
  let parameters = (ins
    ArrayRefParameter<"PerDimLayoutAttr", "layout for each dimension of the vector">:$layouts
  );
  let assemblyFormat = "`<`$layouts`>`";
  let genVerifyDecl = 0;
  let extraClassDeclaration = [{
    std::optional<int64_t> getBatchDim(int64_t dim);
    PerDimLayoutAttr getDimLayout(int64_t dim) const;
  }];
}

#endif // IREE_DIALECT_VECTOREXT_BASE

