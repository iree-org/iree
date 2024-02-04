// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VECTOREXT_ATTRS
#define IREE_DIALECT_VECTOREXT_ATTRS

include "iree-dialects/Dialect/VectorExt/IR/VectorExtBase.td"

//===---------------------------------------------------------------------===//
// Vector layout attributes
//===---------------------------------------------------------------------===//

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

    // Returns the grid of lane ids. Assumes a valid layout.
    ::std::tuple<int64_t, int64_t, int64_t> getLaneGrid();
    PerDimLayoutAttr getDimLayout(int64_t dim) const;
  }];
}

#endif // IREE_DIALECT_VECTOREXT_ATTRS

