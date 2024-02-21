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

/// Creates a pass to convert linalg convolution ops into linalg.matmul ops
/// using im2col tranformation.
std::unique_ptr<Pass> createConvertConv2DToImg2ColPass();

/// A pass to convert convolutions to channels last and propagate.
std::unique_ptr<Pass> createConvertConvToChannelsLastPass();

/// Moves the body of the entire function into a single dispatch.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createMakeSingleDispatchForFunctionPass();

/// A pass to pad linalg ops to the next integer multiple of `paddingSize`.
std::unique_ptr<Pass> createPadLinalgOpsToIntegerMultiplePass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: keep

void registerCommonPreprocessingPasses();

} // namespace mlir::iree_compiler::Preprocessing

#endif // IREE_COMPILER_PREPROCESSING_COMMON_PASSES_H_
