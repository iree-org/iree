// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef STABLEHLO_IREE_CONVERSION_PREPROCESSING_PASSES_H_
#define STABLEHLO_IREE_CONVERSION_PREPROCESSING_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DECL
#include "stablehlo-iree/Conversion/Preprocessing/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerStableHLOPreprocessingPasses();

} // namespace mlir::iree_compiler::stablehlo

#endif // STABLEHLO_IREE_CONVERSION_PREPROCESSING_PASSES_H_
