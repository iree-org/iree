// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_CUSTOMCALLANDCOMPOSITEPREPROCESSING_H_
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_CUSTOMCALLANDCOMPOSITEPREPROCESSING_H_

#include "compiler/plugins/input/StableHLO/Conversion/TargetRegistry.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::stablehlo {

// For plugin preprocessing hooks: normalize raw VHLO before rewriting
// registered custom calls and composites. Unhandled operations remain available
// to later handlers or decomposition.
void buildCustomCallAndCompositePreprocessingPassPipeline(
    OpPassManager &passManager, const TargetRegistry &registry);

} // namespace mlir::iree_compiler::stablehlo

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_CUSTOMCALLANDCOMPOSITEPREPROCESSING_H_
