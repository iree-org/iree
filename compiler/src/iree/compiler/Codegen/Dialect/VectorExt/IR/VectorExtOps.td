// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VECTOREXT_OPS
#define IREE_DIALECT_VECTOREXT_OPS

include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtBase.td"
include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtAttrs.td"
include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class IREEVectorExt_PureOp<string mnemonic, list<Trait> traits = []> :
    Op<IREEVectorExt_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Layout ops.
//===----------------------------------------------------------------------===//

def IREEVectorExt_ToLayoutOp : IREEVectorExt_PureOp<"to_layout", [
  Pure,
  AllTypesMatch<["input", "output"]>
  ]> {
  let summary = "Layout conversion operator";
  let description = [{
    The layout conversion operator takes a shaped value and a layout and
    transforms the value to have that layout.

    If the "shared_memory_conversion" attribute is set, then this layout
    change has to be materialized through shared memory.
  }];
  let arguments = (ins
    AnyShaped:$input,
    VectorLayoutInterface:$layout,
    DefaultValuedAttr<UnitAttr, "false">:$shared_memory_conversion
  );
  let results = (outs
    AnyShaped:$output
  );
  let builders = [
    OpBuilder<(ins "Value":$input,
                   "VectorLayoutInterface":$layout,
                   CArg<"bool", "false">:$shared_memory_conversion), [{
      if (shared_memory_conversion) {
        build($_builder, $_state, input.getType(), input, layout, UnitAttr::get(input.getContext()));
      } else{
        build($_builder, $_state, input.getType(), input, layout);
      }
    }]>
  ];
  let extraClassDeclaration = [{
    bool hasTensorSemantics() {
      return isa<RankedTensorType>(getOutput().getType());
    }
  }];
  let assemblyFormat = "$input `to` $layout attr-dict `:` type($input)";
  let hasVerifier = 1;
}

def IREEVectorExt_ToSIMDOp : IREEVectorExt_PureOp<"to_simd",
    [SameOperandsAndResultElementType, Pure]> {
  let summary = "SIMT to SIMD conversion operation";
  let description = [{
    This operation is a temporary operation useful for source/target
    materializations when doing type conversions between distributed and not
    distributed vectors.
  }];
  let arguments = (ins
    AnyVector:$input
  );
  let results = (outs
    AnyVector:$output
  );
  let extraClassDeclaration = [{}];
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasFolder = 1;
}

def IREEVectorExt_ToSIMTOp : IREEVectorExt_PureOp<"to_simt",
    [SameOperandsAndResultElementType, Pure]> {
  let summary = "SIMD to SIMT conversion operation";
  let description = [{
    This operation is a temporary operation useful for source/target
    materializations when doing type conversions between distributed and not
    distributed vectors.
  }];
  let arguments = (ins
    AnyVector:$input
  );
  let results = (outs
    AnyVector:$output
  );
  let extraClassDeclaration = [{}];
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasFolder = 1;
}

#endif  // IREE_DIALECT_VECTOREXT_OPS
