// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_PASSES_H
#define IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

namespace iree {
class IREEDialect;
}

namespace iree_compiler {
namespace IREE {
namespace PYDM {

class FuncOp;

/// References sources, either passed literally or by reference to a file.
/// One of `asmBlob` or `asmFilePath` should be populated.
struct SourceBundle {
  std::shared_ptr<std::string> asmBlob;
  Optional<std::string> asmFilePath;
};

/// Options for lowering to IREE.
struct LowerToIREEOptions {
  Optional<SourceBundle> linkRtlSource;
};

std::unique_ptr<OperationPass<ModuleOp>> createConvertIREEPyDMToIREEPass();
std::unique_ptr<OperationPass<FuncOp>> createLocalPropagateTypesPass();
std::unique_ptr<OperationPass<FuncOp>> createVariablesToSSAPass();
std::unique_ptr<OperationPass<>> createFixateWeakNumericPass();
std::unique_ptr<OperationPass<ModuleOp>>
createLinkIREEPyDMRTLPass(Optional<SourceBundle> linkRtlSourceBundle = None);
std::unique_ptr<OperationPass<ModuleOp>> createLowerIREEPyDMToRTLPass();

void buildPostImportPassPipeline(OpPassManager &passManager);
void buildLowerToIREEPassPipeline(OpPassManager &passManager,
                                  const LowerToIREEOptions &options);

/// Register all passes and pass pipelines.
void registerPasses();

} // namespace PYDM
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_PASSES_H
