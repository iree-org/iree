// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_PCF_TRANSFORMS_PASSES
#define IREE_CODEGEN_DIALECT_PCF_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ConvertForallToLoopsPass : Pass<"iree-pcf-convert-forall-to-loops", ""> {
  let summary = "Converts scf.forall ops to pcf.loop";
  let description = [{
    Test pass for converting `scf.forall` ops without mapping attributes to
    `pcf.loop` ops with sequential scope.

    The input is IR containing `scf.forall` ops with tensor results and
    `tensor.parallel_insert_slice` terminators. Only forall ops without
    mapping attributes are converted.

    The output replaces each matching `scf.forall` with a `pcf.loop`:
    - Iteration bounds come from the forall's upper/lower bounds and steps
    - `tensor.parallel_insert_slice` ops become `pcf.write_slice` ops
    - Shared output tensors become tied `pcf.sref` region arguments
    - The scope is set to `#pcf.sequential` for sequential execution

    The underlying conversion pattern is exposed separately via
    `convertForallToPCF()` with callbacks for mapping processor IDs to custom
    execution scopes.
  }];
  let dependentDialects = ["::mlir::iree_compiler::IREE::PCF::PCFDialect"];
}

#endif // IREE_CODEGEN_DIALECT_PCF_TRANSFORMS_PASSES
