// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_WGSL_PASSES
#define IREE_CODEGEN_WGSL_PASSES

include "mlir/Pass/PassBase.td"

//===---------------------------------------------------------------------===//
// WGSL passes (keep alphabetical)
//===---------------------------------------------------------------------===//

def WGSLReplacePushConstants :
    InterfacePass<"iree-wgsl-replace-push-constants", "mlir::FunctionOpInterface"> {
  let summary =
      "Replaces push constant loads with binding loads for when using "
      "WGSL without push constant support";
  let constructor = "mlir::iree_compiler::createWGSLReplacePushConstantsPass()";
}

#endif // IREE_CODEGEN_WGSL_PASSES
