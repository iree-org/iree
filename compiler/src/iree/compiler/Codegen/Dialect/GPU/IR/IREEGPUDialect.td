// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_GPU_IREEGPU_DIALECT
#define IREE_CODEGEN_DIALECT_GPU_IREEGPU_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREE GPU Dialect
//===----------------------------------------------------------------------===//

def IREEGPU_Dialect : Dialect {
  let name = "iree_gpu";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let summary = [{
    A dialect representing attributes used by GPU focused IREE code generation.
  }];
  let description = [{
    This dialect provides operations and attributes to aid in code generation
    for GPU targets. The functionality in this dialect can be hardware specific,
    but is intended to be independent of the lowering target. Late lowerings to
    SPIR-V/LLVM are handled separately.
  }];
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
  }];
}

#endif // IREE_CODEGEN_DIALECT_GPU_IREEGPU_DIALECT
