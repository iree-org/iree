// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_PROJECTS_IREE_DIALECTS_DIALECTS_IREE_PUBLIC_IREE_PUBLIC_OPS_TD
#define IREE_LLVM_PROJECTS_IREE_DIALECTS_DIALECTS_IREE_PUBLIC_IREE_PUBLIC_OPS_TD

include "iree-dialects/Dialects/iree_public/IREEPublicDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def IREEPublic_DummyOp : IREEPublic_Op<"dummy", [NoSideEffect]> {
  let summary = "Dummy op to be removed once dialect is populated properly.";
  let arguments = (ins);
  let results = (outs);
}

#endif // IREE_LLVM_PROJECTS_IREE_DIALECTS_DIALECTS_IREE_PUBLIC_IREE_PUBLIC_OPS_TD
