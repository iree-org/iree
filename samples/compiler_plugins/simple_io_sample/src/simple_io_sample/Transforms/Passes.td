// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "mlir/Pass/PassBase.td"

#ifndef IREE_SIMPLEIO_PASSES
#define IREE_SIMPLEIO_PASSES

def LegalizeSimpleIO : Pass<"iree-simpleio-legalize", "mlir::ModuleOp"> {
  let summary = "Legalizes the simpleio sample ops";
  let constructor = [{
    ::mlir::iree_compiler::IREE::SimpleIO::createLegalizeSimpleIOPass()
  }];
}

#endif // IREE_SIMPLEIO_PASSES
