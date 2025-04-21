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
#include "mlir/Pass/PassOptions.h"

namespace mlir::iree_compiler::Preprocessing {

/// Placeholder struct for preprocessing pass pipeline options.
struct TransformOptions : public PassPipelineOptions<TransformOptions> {};

/// Adds a set of passes to the given pass manager that are run after input
/// conversion, but before any of the IREE compilation passes. There are many
/// ways preprocessing passes can be added. These options are exercised in the
/// following order
/// 1. Using Command line options : See `PreprocessingOptions` to see the order
///    of preference for different command line based preprocessing options.
/// 2. Extensions specified by plugins : Plugins can specify a preprocessing
///    pass pipeline to run.
void buildPreprocessingPassPipeline(
    OpPassManager &passManager, const PreprocessingOptions &options,
    PipelineExtensions *pipelineExtensions = nullptr);

void registerPreprocessingPasses();

} // namespace mlir::iree_compiler::Preprocessing

#endif // IREE_COMPILER_PREPROCESSING_PASSES_H_
