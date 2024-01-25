// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_GPU_IREEGPU_DIALECT
#define IREE_CODEGEN_DIALECT_GPU_IREEGPU_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREE Codegen dialect
//===----------------------------------------------------------------------===//

def IREEGPU_Dialect : Dialect {
  let name = "iree_gpu";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let summary = [{
    A dialect representing attributes used by GPU focused IREE code generation.
  }];
  let description = [{
    This dialect is primarily meant to hold attributes that carry additional
    target specific information expanded based on executable target information.

    This information is only used by codegen to normalize the higher level
    target details across backends and devices. Late lowerings to SPIR-V/LLVM
    still rely on the information designed for those targets.
  }];
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
  }];
}

#endif // IREE_CODEGEN_DIALECT_GPU_IREEGPU_DIALECT
