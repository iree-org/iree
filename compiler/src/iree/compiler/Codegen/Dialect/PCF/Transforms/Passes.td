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

def FuseConsumersPass : Pass<"iree-pcf-fuse-consumers", ""> {
  let summary = "Fuses all consumers of pcf.generic/loop ops";
  let description = [{
    Test pass for fusing consumer operations into `pcf.generic` and `pcf.loop`
    ops.

    The input is IR containing PCF parallel ops with external consumers that
    implement `TilingInterface` or are `tensor.extract_slice` ops.

    The pass greedily fuses each tilable consumer into the producer PCF op by
    cloning the consumer into the PCF region, tiling it to match the iteration
    space, and replacing the external use with a new `pcf.write_slice`.

    Supported fusion scenarios:
    - Multiple `pcf.write_slice` producers for a single consumer value
    - Fusion along multiple operands with a single `pcf.write_slice` per operand
    - `tensor.extract_slice` by adding a condition based on the slice bounds.

    The underlying fusion patterns are exposed via `matchTilableConsumer()` and
    `fuseTilableConsumer()` for use in custom pipelines.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::affine::AffineDialect",
                           "::mlir::iree_compiler::IREE::PCF::PCFDialect"];
}

#endif // IREE_CODEGEN_DIALECT_PCF_TRANSFORMS_PASSES
