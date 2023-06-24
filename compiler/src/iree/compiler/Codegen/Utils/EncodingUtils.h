// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_UTILS_ENCODINGUTILS_H_
#define IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_UTILS_ENCODINGUTILS_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

namespace mlir {
namespace iree_compiler {

// Enumeration of possible (LHS, RHS, Accumulator) type triples for the
// element types of the operands of a matmul-like. or linalg.mmt4d.
enum class MatmulType {
  F32F32F32,
  I8I8I32,
};

// Enumeration of the operands of a matmul-like operation such as linalg.matmul.
enum class MatmulOperandRole {
  LHS,
  RHS,
  RESULT,
};

// Constructs a MatmulType from separate operands element types, or returns
// std::nullopt if no MatmulType enumeration value would match.
std::optional<MatmulType>
getMatmulType(Type lhsElementType, Type rhsElementType, Type resultElementType);

// Helper to read the TensorEncoding from a TensorEncodingAttr on a TensorType.
// Return std::nullopt if the TensorType does not have a TensorEncodingAttr.
std::optional<IREE::LinalgExt::TensorEncoding>
getEncoding(RankedTensorType tensorType);

// Reads a MatmulType from a TensorEncoding, or returns std::nullopt if no
// MatmulType enumeration value would match.
std::optional<MatmulType>
getMatmulType(IREE::LinalgExt::TensorEncoding encoding);

// Reads a MatmulOperandRole from a TensorEncoding, or returns std::nullopt if
// no MatmulOperandRole enumeration value would match.
std::optional<MatmulOperandRole>
getMatmulOperandRole(IREE::LinalgExt::TensorEncoding encoding);

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_UTILS_ENCODINGUTILS_H_
