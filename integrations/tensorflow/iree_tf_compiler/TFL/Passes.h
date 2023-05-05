// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TFL_PASSES_H_
#define IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TFL_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_integrations {
namespace TFL {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// Create a single pipeline that will run all the needed IREE-specific TFL
// import passes in the right order.
void buildTFLImportPassPipeline(OpPassManager &pm);

//===----------------------------------------------------------------------===//
// IREE-specific passes for TFLite import
//===----------------------------------------------------------------------===//

// Retain functions used by tfl.call_once to avoid removal.
std::unique_ptr<OperationPass<ModuleOp>> createRetainCallOnceFuncsPass();

// Converts TFLite attributes that are useful to corresponding IREE attributes.
std::unique_ptr<OperationPass<ModuleOp>> createConvertModuleMetadataPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertFunctionMetadataPass();

// Lowers TFLite's global tensor operations to the Util dialect.
std::unique_ptr<OperationPass<ModuleOp>> createLowerGlobalTensorsPass();

// Strips all leftover TFLite-related attributes; none are needed by IREE.
std::unique_ptr<OperationPass<ModuleOp>> createStripModuleMetadataPass();
std::unique_ptr<OperationPass<func::FuncOp>> createStripFunctionMetadataPass();

// Validates whether any TFLite operations remain.
std::unique_ptr<OperationPass<func::FuncOp>> createVerifyFullyConvertedPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerTFLImportPassPipeline();

void registerAllPasses();

}  // namespace TFL
}  // namespace iree_integrations
}  // namespace mlir

#endif  // IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TFL_PASSES_H_
