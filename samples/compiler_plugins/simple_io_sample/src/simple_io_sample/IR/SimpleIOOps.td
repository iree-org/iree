// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SIMPLE_IO_SAMPLE
#define SIMPLE_IO_SAMPLE

include "mlir/IR/OpBase.td"

def SimpleIO_Dialect : Dialect {
  let name = "simple_io";
  let cppNamespace = "::mlir::iree_compiler::IREE::SimpleIO";
}

class SimpleIO_Op<string mnemonic, list<Trait> traits = []> :
    Op<SimpleIO_Dialect, mnemonic, traits> {
}

def SimpleIO_PrintOp : SimpleIO_Op<"print", []> {
  let summary = [{Print}];
  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = [{
    attr-dict
  }];
}

#endif
