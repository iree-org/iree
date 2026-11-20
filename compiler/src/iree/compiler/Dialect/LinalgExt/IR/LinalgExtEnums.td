// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_ENUMS
#define IREE_DIALECT_LINALGEXT_ENUMS

include "mlir/IR/EnumAttr.td"

// How `iree_linalg_ext.choose_qparams_affine` derives the quantization
// parameters of a block from the range of its values.
def IREELinalgExt_QuantizationMappingSymmetric
    : I32EnumAttrCase<"Symmetric", 0, "symmetric">;
def IREELinalgExt_QuantizationMappingAsymmetric
    : I32EnumAttrCase<"Asymmetric", 1, "asymmetric">;
def IREELinalgExt_QuantizationMappingAsymmetricMinError
    : I32EnumAttrCase<"AsymmetricMinError", 2, "asymmetric_min_error">;
def IREELinalgExt_QuantizationMappingSymmetricQmax
    : I32EnumAttrCase<"SymmetricQmax", 3, "symmetric_qmax">;

def IREELinalgExt_QuantizationMapping : I32EnumAttr<"QuantizationMapping",
    "mapping from the range of the real values to quantization parameters", [
      IREELinalgExt_QuantizationMappingSymmetric,
      IREELinalgExt_QuantizationMappingAsymmetric,
      IREELinalgExt_QuantizationMappingAsymmetricMinError,
      IREELinalgExt_QuantizationMappingSymmetricQmax,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::LinalgExt";
  let genSpecializedAttr = 0;
}

#endif // IREE_DIALECT_LINALGEXT_ENUMS
