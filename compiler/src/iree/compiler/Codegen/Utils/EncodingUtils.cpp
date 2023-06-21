// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/EncodingUtils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace iree_compiler {

using IREE::LinalgExt::EncodingAttr;
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

  return std::nullopt;
}

std::optional<TensorEncoding> getEncoding(RankedTensorType tensorType) {
  auto encodingAttr =
      llvm::dyn_cast_if_present<EncodingAttr>(tensorType.getEncoding());
  if (!encodingAttr)
    return std::nullopt;
  return encodingAttr.getEncoding().getValue();
}

std::optional<MatmulType> getMatmulType(TensorEncoding encoding) {
  switch (encoding) {
  case TensorEncoding::MATMUL_F32F32F32_LHS:
  case TensorEncoding::MATMUL_F32F32F32_RHS:
  case TensorEncoding::MATMUL_F32F32F32_RESULT:
    return MatmulType::F32F32F32;
  case TensorEncoding::MATMUL_I8I8I32_LHS:
  case TensorEncoding::MATMUL_I8I8I32_RHS:
  case TensorEncoding::MATMUL_I8I8I32_RESULT:
    return MatmulType::I8I8I32;
  default:
    return std::nullopt;
  }
}

std::optional<MatmulOperandRole> getMatmulOperandRole(TensorEncoding encoding) {
  switch (encoding) {
  case TensorEncoding::MATMUL_F32F32F32_LHS:
  case TensorEncoding::MATMUL_I8I8I32_LHS:
    return MatmulOperandRole::LHS;
  case TensorEncoding::MATMUL_F32F32F32_RHS:
  case TensorEncoding::MATMUL_I8I8I32_RHS:
    return MatmulOperandRole::RHS;
  case TensorEncoding::MATMUL_F32F32F32_RESULT:
  case TensorEncoding::MATMUL_I8I8I32_RESULT:
    return MatmulOperandRole::RESULT;
  default:
    return std::nullopt;
  }
}

} // namespace iree_compiler
} // namespace mlir
