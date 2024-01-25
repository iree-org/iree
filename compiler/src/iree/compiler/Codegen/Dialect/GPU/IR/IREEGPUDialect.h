// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUDIALECT_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

// clang-format off: must be included after all LLVM/MLIR headers
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler {} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUDIALECT_H_
