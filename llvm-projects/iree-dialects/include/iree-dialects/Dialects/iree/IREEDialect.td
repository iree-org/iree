// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_PROJECTS_IREE_DIALECTS_DIALECTS_IREE_IREE_DIALECT_TD
#define IREE_LLVM_PROJECTS_IREE_DIALECTS_DIALECTS_IREE_IREE_DIALECT_TD

include "mlir/IR/OpBase.td"

def IREE_Dialect : Dialect {
  let name = "iree";
  let summary = "Public ops/type/attributes legal for input to IREE's compiler";
  let description = [{
    IREE's compiler allows as input a number of common dialects. This dialect
    contains structural and unique ops that do not exist elsewhere or that IREE
    has an interest in maintaining as a stable set.

    The contents of this dialect often mirror various constructs in IREE's
    internal implementation. The focus here is on simplicity and stability
    over time. Generally, this dialect does not use "advanced" features and
    should be broadly source compatible over a range of LLVM versions. There
    are of course, limits, and source-compatibility is not guaranteed, since
    LLVM/MLIR's API surface is itself unstable.
  }];
  let cppNamespace = "::mlir::iree";
}

class IREE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<IREE_Dialect, mnemonic, traits>;

#endif // IREE_LLVM_PROJECTS_IREE_DIALECTS_DIALECTS_IREE_IREE_DIALECT_TD
