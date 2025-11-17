// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_HAL_LOADER_BASE
#define IREE_DIALECT_MODULES_HAL_LOADER_BASE

include "iree/compiler/Dialect/Util/IR/UtilBase.td"

//===----------------------------------------------------------------------===//
// IREE HAL inline executable loader dialect
//===----------------------------------------------------------------------===//

def HALLoader_Dialect : Dialect {
  let name = "hal_loader";
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL::Loader";

  let summary = [{
    IREE HAL inline executable loader runtime module dialect.
  }];
  let description = [{
    Low-level dialect for dynamically loading executables and dispatching work.
    Only operates synchronously, single-threaded, and on host-local buffers. Use
    the full HAL for all other cases.

    This dialect can be used alongside the full HAL but is intended for use in
    conjunction with the `hal_inline` dialect which also carries the same usage
    restrictions.

    See `hal_loader.imports.mlir` for the full list of exported functions.
  }];
}

//===----------------------------------------------------------------------===//
// HALLoader enums
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// HALLoader types
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// HALLoader op traits
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Base HALLoader op classes
//===----------------------------------------------------------------------===//

class HALLoader_Op<string mnemonic, list<Trait> traits = []> :
    Op<HALLoader_Dialect, mnemonic, !listconcat(traits, [])> {}

#endif  // IREE_DIALECT_MODULES_HAL_LOADER_BASE
