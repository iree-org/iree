// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGPLUS_BASE
#define IREE_DIALECT_LINALGPLUS_BASE

include "mlir/IR/OpBase.td"

def LinalgPlus_Dialect : Dialect {
  let name = "linalg_plus";
  let cppNamespace = "::mlir::iree_compiler::linalg_plus";
  let description = [{
    The `linalg_plus` dialect is intended to experiment more support for
    non-structured operations, ie, can not be represented in Linalg operations.
  }];
}

#endif // IREE_DIALECT_LINALGPLUS_BASE
