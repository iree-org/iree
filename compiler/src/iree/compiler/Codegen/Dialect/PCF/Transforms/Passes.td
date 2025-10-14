// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_PCF_TRANSFORMS_PASSES
#define IREE_CODEGEN_DIALECT_PCF_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ConvertSRefToMemRefPass :
    Pass<"iree-pcf-convert-sref-to-memref", ""> {
  let summary = "Converts all `pcf.sref` types to memref.";
  let description = [{
    Propagates memref layouts and converts all `pcf.sref` types to memref.
  }];
}

def LowerStructuralPCFPass :
    Pass<"iree-pcf-lower-structural-pcf", ""> {
  let summary = "Lowers structured `pcf` ops to `cf` + `scf`.";
  let description = [{
    Lowers structured `pcf` ops to `cf` wrapped in `scf.execute_region` to allow
    for decoupling this lowering from `scf` lowerings.
  }];
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::PCF::PCFDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::cf::ControlFlowDialect",
  ];
}

def ResolveTokensPass :
    Pass<"iree-pcf-resolve-tokens", ""> {
  let summary = "Lowers `pcf.token` types.";
  let description = [{
    Lowers `pcf.token` types to coarser synchronization primitives. Custom
    token handling is expected to happen before this pass.
  }];
}

def ConvertForallToLoopsPass :
    Pass<"iree-pcf-convert-forall-to-loops", ""> {
  let summary = "Converts scf.forall ops to pcf.loop";
  let description = [{
    Primarily test pass for converting scf.forall with tensor results to
    pcf.loop ops. The underlying pattern for conversion is exposed separately
    with callbacks for mapping processor IDs to an execution scope. This avoids
    an extra interface for mapping_id -> scope.
  }];
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::PCF::PCFDialect"
  ];
}

#endif // IREE_CODEGEN_DIALECT_PCF_TRANSFORMS_PASSES
