// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Passes.h"

#include "iree/compiler/Codegen/PassDetail.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

// NOTE: this runs on the top-level program module containing all
// hal.executable ops.
void buildVMVXLinkingPassPipeline(OpPassManager &passManager) {
  passManager.addPass(createVMVXLinkExecutablesPass());
}

}  // namespace iree_compiler
}  // namespace mlir
