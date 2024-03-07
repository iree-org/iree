// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_PREPROCESSING_PASSES_H_
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_PREPROCESSING_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DECL
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerStableHLOPreprocessingPasses();

} // namespace mlir::iree_compiler::stablehlo

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_PREPROCESSING_PASSES_H_
