// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_TOSA_INPUTCONVERSION_PASSES_H_
#define IREE_COMPILER_PLUGINS_INPUT_TOSA_INPUTCONVERSION_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// Performs input legalization for specific combination of input dialects.
void buildTOSAInputConversionPassPipeline(OpPassManager &passManager);

void registerTOSAConversionPassPipeline();

//------------------------------------------------------------------------------
// Conversions from TOSA into Linalg and other core IREE dialects
//------------------------------------------------------------------------------

// Set of patterns for materializing TOSA operations to linalg_ext.
void populateTosaToLinalgExtPatterns(RewritePatternSet *patterns);

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "compiler/plugins/input/TOSA/InputConversion/Passes.h.inc" // IWYU pragma: export

void registerTOSAConversionPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_PLUGINS_INPUT_TOSA_INPUTCONVERSION_PASSES_H_
