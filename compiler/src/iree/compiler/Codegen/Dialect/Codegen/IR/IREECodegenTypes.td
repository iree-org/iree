// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_IREECODEGEN_TYPES
#define IREE_CODEGEN_DIALECT_IREECODEGEN_TYPES

include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.td"

def NullPointer
  : TypeDef<IREECodegen_Dialect, "NullPointer", []> {
  let summary = [{Pseudo null-pointer type. Lowers to a null pointer.}];
  let description = [{
    This is meant to be used only as arguments to microkernels.
  }];
  let mnemonic = "null_pointer";
}

#endif // IREE_CODEGEN_DIALECT_IREECODEGEN_TYPES
