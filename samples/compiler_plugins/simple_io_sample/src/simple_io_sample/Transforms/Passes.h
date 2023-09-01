// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SAMPLES_COMPILER_PLUGINS_SIMPLE_IO_SAMPLE_TRANSFORMS_PASSES_H_
#define IREE_SAMPLES_COMPILER_PLUGINS_SIMPLE_IO_SAMPLE_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::SimpleIO {

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeSimpleIOPass();

}  // namespace mlir::iree_compiler::IREE::SimpleIO

#endif  // IREE_SAMPLES_COMPILER_PLUGINS_SIMPLE_IO_SAMPLE_TRANSFORMS_PASSES_H_
