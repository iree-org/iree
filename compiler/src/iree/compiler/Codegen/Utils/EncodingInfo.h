// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_UTILS_ENCODINGINFO_H_
#define IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_UTILS_ENCODINGINFO_H_

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

namespace mlir {
namespace iree_compiler {

enum class MatmulType {
  F32F32F32,
  I8I8I32,
};

std::optional<MatmulType> getMatmulType(Type lhsElementType,
                                        Type rhsElementType,
                                        Type resultElementType);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_UTILS_ENCODINGINFO_H_
