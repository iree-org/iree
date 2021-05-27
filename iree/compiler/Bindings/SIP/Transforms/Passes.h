// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_BINDINGS_SIP_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_BINDINGS_SIP_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace SIP {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that setup a module for use
// with an IREE SIP-compatible runtime binding implementation (python, etc).
void buildTransformPassPipeline(OpPassManager &passManager);

void registerTransformPassPipeline();

//===----------------------------------------------------------------------===//
// SIP-compatible bindings support
//===----------------------------------------------------------------------===//

// Materializes reflection metadata on exported function arguments and results.
// This runs as close to the input processing as possible as it needs to
// annotate the ABI that the consumer is expecting to interop with.
std::unique_ptr<OperationPass<FuncOp>> createMaterializeReflectionAttrsPass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

inline void registerPasses() {
  registerTransformPassPipeline();
  createMaterializeReflectionAttrsPass();
}

}  // namespace SIP
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_BINDINGS_SIP_TRANSFORMS_PASSES_H_
