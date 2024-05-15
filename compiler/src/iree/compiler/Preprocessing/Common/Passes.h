// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PREPROCESSING_COMMON_PASSES_H_
#define IREE_COMPILER_PREPROCESSING_COMMON_PASSES_H_

#include <functional>

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::Preprocessing {

// Enum to represent target op types to be padded.
enum class PadTargetType {
  // Convolution Ops.
  ConvOp = 0,
  // Contraction-like Ops.
  ContractionOp = 1,
  // All ops (both convolution and contraction ops).
  All = 2,
};

enum class TransposeMatmulInput {
  // Transpose LHS input matrix.
  Lhs,
  // Transpose RHS input matrix.
  Rhs
};

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: keep

void registerCommonPreprocessingPasses();

} // namespace mlir::iree_compiler::Preprocessing

#endif // IREE_COMPILER_PREPROCESSING_COMMON_PASSES_H_
