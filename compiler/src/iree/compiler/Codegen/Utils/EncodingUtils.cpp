// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/EncodingUtils.h"
#include <optional>

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace iree_compiler {

using IREE::LinalgExt::TensorEncoding;
using IREE::LinalgExt::TensorEncodingAttr;

std::optional<MatmulType> getMatmulType(Type lhsElementType,
                                        Type rhsElementType,
                                        Type resultElementType) {
  if (lhsElementType.isSignlessInteger(8) &&
      rhsElementType.isSignlessInteger(8) &&
      resultElementType.isSignlessInteger(32)) {
    return MatmulType::I8I8I32;
  }
  if (lhsElementType.isF32() && rhsElementType.isF32() &&
      resultElementType.isF32()) {
    return MatmulType::F32F32F32;
  }
  if (lhsElementType.isF16() && rhsElementType.isF16() &&
      resultElementType.isF32()) {
    return MatmulType::F16F16F32;
  }
  if (lhsElementType.isF16() && rhsElementType.isF16() &&
      resultElementType.isF16()) {
    return MatmulType::F16F16F16;
  }
  if (lhsElementType.isBF16() && rhsElementType.isBF16() &&
      resultElementType.isF32()) {
    return MatmulType::BF16BF16F32;
  }
  if (lhsElementType.isBF16() && rhsElementType.isBF16() &&
      resultElementType.isBF16()) {
    return MatmulType::BF16BF16BF16;
  }

  return std::nullopt;
}

std::optional<TensorEncoding> getEncoding(RankedTensorType tensorType) {
  auto encoding = tensorType.getEncoding();
  if (!encoding) {
    return std::nullopt;
  }
  auto tensorEncodingAttr = encoding.dyn_cast_or_null<TensorEncodingAttr>();
  if (!tensorEncodingAttr) {
    return std::nullopt;
  }
  return tensorEncodingAttr.getValue();
}

#define IREE_GETMATMULTYPE_CASE(TYPE)                                          \
  case TensorEncoding::MATMUL_##TYPE##_LHS:                                    \
  case TensorEncoding::MATMUL_##TYPE##_RHS:                                    \
  case TensorEncoding::MATMUL_##TYPE##_RESULT:                                 \
    return MatmulType::TYPE;

std::optional<MatmulType> getMatmulType(TensorEncoding encoding) {
  switch (encoding) {
    IREE_GETMATMULTYPE_CASE(F32F32F32)
    IREE_GETMATMULTYPE_CASE(I8I8I32)
    IREE_GETMATMULTYPE_CASE(F16F16F32)
    IREE_GETMATMULTYPE_CASE(F16F16F16)
    IREE_GETMATMULTYPE_CASE(BF16BF16F32)
    IREE_GETMATMULTYPE_CASE(BF16BF16BF16)
  default:
    return std::nullopt;
  }
}

#undef IREE_GETMATMULTYPE_CASE

#define IREE_GETMATMULOPERANDROLE_CASE(TYPE)                                   \
  case TensorEncoding::MATMUL_##TYPE##_LHS:                                    \
    return MatmulOperandRole::LHS;                                             \
  case TensorEncoding::MATMUL_##TYPE##_RHS:                                    \
    return MatmulOperandRole::RHS;                                             \
  case TensorEncoding::MATMUL_##TYPE##_RESULT:                                 \
    return MatmulOperandRole::RESULT;

std::optional<MatmulOperandRole> getMatmulOperandRole(TensorEncoding encoding) {
  switch (encoding) {
    IREE_GETMATMULOPERANDROLE_CASE(F32F32F32)
    IREE_GETMATMULOPERANDROLE_CASE(I8I8I32)
    IREE_GETMATMULOPERANDROLE_CASE(F16F16F32)
    IREE_GETMATMULOPERANDROLE_CASE(F16F16F16)
    IREE_GETMATMULOPERANDROLE_CASE(BF16BF16F32)
    IREE_GETMATMULOPERANDROLE_CASE(BF16BF16BF16)

  default:
    return std::nullopt;
  }
}

#undef IREE_GETMATMULOPERANDROLE_CASE

} // namespace iree_compiler
} // namespace mlir
