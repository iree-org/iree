// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_HAL_INLINE_BASE
#define IREE_DIALECT_MODULES_HAL_INLINE_BASE

include "iree/compiler/Dialect/Util/IR/UtilBase.td"

//===----------------------------------------------------------------------===//
// IREE HAL inline dialect
//===----------------------------------------------------------------------===//

def HALInline_Dialect : Dialect {
  let name = "hal_inline";
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL::Inline";

  let summary = [{
    IREE inline HAL interop runtime module dialect.
  }];
  let description = [{
    Low-level dialect for limited in-process ABI interop with the full HAL.
    Only operates synchronously, single-threaded, and on host-local buffers. Use
    the full HAL for all other cases.

    This dialect can be used alongside the full HAL but is intended for use in
    standalone configurations or paired with the `hal_loader` dialect which also
    carries the same usage restrictions.

    See `hal_inline.imports.mlir` for the full list of exported functions.
  }];
}

//===----------------------------------------------------------------------===//
// Base HALInline op classes
//===----------------------------------------------------------------------===//

class HALInline_Op<string mnemonic, list<Trait> traits = []> :
    Op<HALInline_Dialect, mnemonic, !listconcat(traits, [])> {}

#endif  // IREE_DIALECT_MODULES_HAL_INLINE_BASE
