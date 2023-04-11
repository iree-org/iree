// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PREPROCESSING_PASSES_H_
#define IREE_COMPILER_PREPROCESSING_PASSES_H_

#include <functional>

#include "iree/compiler/Pipelines/Options.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

/// Adds a set of passes to the given pass manager that run preprocessing
/// passes specified in textual pass-pipeline format using
/// `iree-preprocessing-pass-pipeline`. This allows some user control
/// on the sequence of preprocessing passes to run after conversion from input
/// dialects like `mhlo`/`tosa` before running the core IREE compilation
/// pipelines (starting with the flow pipeline).
void buildPreprocessingPassPipeline(
    OpPassManager &passManager, const PreprocessingOptions &options,
    PipelineExtensions *pipelineExtensions = nullptr);

void registerPreprocessingPasses();

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_PREPROCESSING_PASSES_H_
