// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OOTEX_OPS
#define OOTEX_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Ootex_Dialect : Dialect {
  let name = "ootex";
  let cppNamespace = "::ootex";
  let summary = [{An out-of-tree dialect, owned by the plugin.}];
}

class Ootex_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ootex_Dialect, mnemonic, traits> {
}

def Ootex_MarkOp : Ootex_Op<"mark", []> {
  let summary = [{Marks the enclosing function for the plugin's pass.}];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = [{ attr-dict }];
}

#endif // OOTEX_OPS
