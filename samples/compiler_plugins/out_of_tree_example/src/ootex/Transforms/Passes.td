// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OOTEX_PASSES
#define OOTEX_PASSES

include "mlir/Pass/PassBase.td"

def AnnotateMarkedFunctions :
    Pass<"ootex-annotate-marked-functions", "mlir::ModuleOp"> {
  let summary = [{Annotates in-tree util.func ops holding an ootex.mark.}];
  let description = [{
    Reads the out-of-tree ootex dialect and writes to the in-tree util dialect,
    which is what a real plugin does: its own ops decide what happens to IREE's.
  }];
  let options = [
    Option<"tag", "tag", "std::string", [{"marked"}],
           "Attribute value written onto each marked function.">,
  ];
}

#endif // OOTEX_PASSES
