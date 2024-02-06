// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_INDEXING_IR_INDEXING_BASE
#define IREE_DIALECT_INDEXING_IR_INDEXING_BASE

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREE base dialect used for indexing arithmetic and analysis.
//===----------------------------------------------------------------------===//

def Indexing_Dialect : Dialect {
  let name = "indexing";
  let cppNamespace = "::mlir::iree_compiler::IREE::Indexing";

  let summary = [{
    A dialect used for supplementing and analyzing index arithmetic.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// General types and helpers
//===----------------------------------------------------------------------===//

def Indexing_IndexLike : AnyTypeOf<[I32, I64, Index]>;

#endif  // IREE_DIALECT_INDEXING_IR_INDEXING_BASE
