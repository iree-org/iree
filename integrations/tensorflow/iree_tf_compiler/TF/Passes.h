// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TF_PASSES_H_
#define IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TF_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Create a single pipeline that will run all the needed IREE-specific TF import
// passes in the right order.
void buildTFImportPassPipeline(OpPassManager &pm, bool useTosa);
void registerTFImportPassPipeline();

//===----------------------------------------------------------------------===//
// IREE-specific Passes For TensorFlow Import
//===----------------------------------------------------------------------===//

// Converts the TF dialect to the XLA MHLO dialect.
std::unique_ptr<Pass> createConvertToMHLOPass();

// In a module tagged with `tf_saved_model.semantics`, lowers
// `tf_saved_model.global_variable`'s to `util.global`'s.
//
// This pass should be run before adopting the exports, which transitions to
// a module that does not have `tf_saved_model.semantics`.
std::unique_ptr<OperationPass<ModuleOp>> createLowerGlobalTensorsPass();

// In a module tagged with `tf_saved_model.semantics`, creates IREE ABI
// functions for any saved model exported functions.
std::unique_ptr<OperationPass<ModuleOp>> createSavedModelToIREEABIPass();

// Simplifies TensorFlow debug info for the purposes of making it easier to
// look at.
std::unique_ptr<OperationPass<ModuleOp>> createPrettifyDebugInfoPass();

// Push resource casts forward to better propagate resource related shapes.
std::unique_ptr<OperationPass<ModuleOp>> createPropagateResourceCastsPass();

// Strips tf.Assert ops.
std::unique_ptr<OperationPass<func::FuncOp>> createStripAssertsPass();

// Strips all TF-related attributes; none are needed by IREE.
std::unique_ptr<OperationPass<ModuleOp>> createStripModuleMetadataPass();
std::unique_ptr<OperationPass<func::FuncOp>> createStripFunctionMetadataPass();

// Validates whether any Tensorflow operations remain.
std::unique_ptr<OperationPass<func::FuncOp>> createVerifyFullyConvertedPass();

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

// Populates patterns for direct lowering TensorFlow ops that IREE manages
// (augmenting the standard MHLO lowerings).
void populateDirectLoweringPatterns(MLIRContext *context,
                                    RewritePatternSet &patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

inline void registerAllPasses() {
  registerTFImportPassPipeline();

  createConvertToMHLOPass();
  createLowerGlobalTensorsPass();
  createPrettifyDebugInfoPass();
  createPropagateResourceCastsPass();
  createSavedModelToIREEABIPass();
  createStripAssertsPass();
  createStripModuleMetadataPass();
  createStripFunctionMetadataPass();
  createVerifyFullyConvertedPass();
}

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir

#endif  // IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TF_PASSES_H_
