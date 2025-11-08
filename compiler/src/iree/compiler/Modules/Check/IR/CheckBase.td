// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_CHECK_DIALECT_CHECK_BASE
#define IREE_MODULES_CHECK_DIALECT_CHECK_BASE

include "iree/compiler/Dialect/Util/IR/UtilBase.td"

//===----------------------------------------------------------------------===//
// IREE Check dialect
//===----------------------------------------------------------------------===//

def CHECK_Dialect : Dialect {
  let name = "check";
  let cppNamespace = "::mlir::iree_compiler::IREE::Check";
  let summary = [{
    A dialect implementing test assertions for IREE modules.
  }];
}

//===----------------------------------------------------------------------===//
// CHECK op base classes
//===----------------------------------------------------------------------===//

class CHECK_Op<string mnemonic, list<Trait> traits = []> :
    Op<CHECK_Dialect, mnemonic, traits>;

#endif  // IREE_MODULES_CHECK_DIALECT_CHECK_BASE
