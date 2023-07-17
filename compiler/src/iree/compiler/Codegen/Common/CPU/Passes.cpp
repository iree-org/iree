// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {

//===---------------------------------------------------------------------===//
// Register Common/CPU Passes
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Common/CPU/Passes.h.inc"
} // namespace

void registerCodegenCommonCPUPasses() {
  // Generated.
  registerPasses();
}
} // namespace iree_compiler
} // namespace mlir
