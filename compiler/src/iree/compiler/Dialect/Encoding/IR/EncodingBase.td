// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_ENCODING_BASE
#define IREE_DIALECT_ENCODING_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def IREEEncoding_Dialect : Dialect {
  let name = "iree_encoding";
  let cppNamespace = "::mlir::iree_compiler::IREE::Encoding";
  let summary = [{
    Tensor encoding attributes and ops.
  }];
  let description = [{
    A dialect defining IREE tensor encoding attributes and related ops, used to
    implement data-tiling.
  }];
  let useDefaultAttributePrinterParser = 1;
}

#endif // IREE_DIALECT_ENCODING_BASE
