// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_CPU_IREECPU_DIALECT
#define IREE_CODEGEN_DIALECT_CPU_IREECPU_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def IREECPU_Dialect : Dialect {
  let name = "iree_cpu";
  let cppNamespace = "::mlir::iree_compiler::IREE::CPU";

  let summary = [{
    A dialect for common functionality used by CPU focused IREE code generation.
  }];
  let description = [{
    This dialect provides operations and attributes to aid in code generation
    for CPU targets. The functionality in this dialect can be hardware specific,
    but is intended to be independent of the lowering target. Late lowerings to
    LLVM are handled separately.
  }];
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
  }];
}

#endif // IREE_CODEGEN_DIALECT_CPU_IREECPU_DIALECT
