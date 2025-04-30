// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_PASSES_H_
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_PASSES_H_

#include "compiler/plugins/input/StableHLO/Conversion/PassDetail.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::stablehlo {

struct StableHloOptions : public PassPipelineOptions<StableHloOptions> {};

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

void buildStableHLOInputConversionPassPipeline(OpPassManager &passManager,
                                               const StableHloOptions &options);

// Performs input legalization on programs that may have originated from an XLA
// import (or made to interop with it).
void buildStableHLOXLAInputConversionPassPipeline(
    OpPassManager &passManager, const StableHloOptions &options);

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerStableHLOConversionPasses();

} // namespace mlir::iree_compiler::stablehlo

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_PASSES_H_
