// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/EncodingInfo.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace iree_compiler {

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

}  // namespace iree_compiler
}  // namespace mlir
