// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECTEXTENSIONS_H_
#define IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECTEXTENSIONS_H_

#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace linalg {
namespace transform {

void registerLinalgTransformDialectExtension(DialectRegistry &registry);

}  // namespace transform
}  // namespace linalg
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMDIALECTEXTENSIONS_H_
