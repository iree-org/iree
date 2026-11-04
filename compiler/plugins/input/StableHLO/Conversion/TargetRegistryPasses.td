// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_TARGETREGISTRYPASSES
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_TARGETREGISTRYPASSES

include "mlir/Pass/PassBase.td"

// These passes require a caller-supplied registry. Keep their generated bases
// separate from Passes.td, whose passes are registered for textual construction.
def RewriteRegisteredCustomCallsAndCompositesPass :
    Pass<"iree-stablehlo-rewrite-registered-custom-calls-and-composites",
         "ModuleOp"> {
  let summary = "Rewrites registered StableHLO custom calls and composites";
  let description = [{
    Invokes registered handlers for custom-call targets and exact composite
    names and versions. Handlers replace operations they accept and leave IR
    unchanged when declining. Remaining composites retain their decomposition.

    Each pass owns a snapshot of the registry supplied at construction. Later
    registrations do not affect it.
  }];
  // Suppress generated parameterless factories. Only the generated base is
  // used; textual pipelines must construct a registry through a named pipeline.
  let constructor = [{
    createRewriteRegisteredCustomCallsAndCompositesPass(registry)
  }];
  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect",
  ];
}

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_TARGETREGISTRYPASSES
