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

def TestFoldForallIntoPCFLoopPass : Pass<"iree-pcf-test-fold-forall-into-pcf-loop", ""> {
  let summary = "Test pass for foldForallIntoPCFLoop transform";
  let description = [{
    Test pass for folding `scf.forall` ops containing `pcf.loop` ops into a
    single `pcf.generic` operation.

    The input is IR containing `scf.forall` ops with `iree_codegen.local_mapping`
    attributes that contain a `pcf.loop` with `#pcf.sequential` scope as the
    last operation before the terminator.

    Structural requirements validated:
    - All ops in `scf.forall.in_parallel` are `tensor.parallel_insert_slice`.
    - All insert sources come from the same `pcf.loop` result.
    - All insert destinations are `scf.forall` shared_outs.
    - The `pcf.loop` is the last op before terminator in forall body.
    - The `pcf.loop` has a single count argument (linearized).
    - All `pcf.loop` region ref arg users are `pcf.write_slice` ops.
    - Ref args have `SyncOnReturnAttr` sync scope.

    The underlying transform is exposed via `foldForallIntoPCFLoop()` for use
    in custom pipelines with different matching criteria.
  }];
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::PCF::PCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::scf::SCFDialect"
  ];
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

def FusePCFWritesPass : Pass<"iree-pcf-fuse-pcf-writes", ""> {
  let summary = "Consolidates pcf.write_slice ops in loop bodies";
  let description = [{
    Test pass for composing `pcf.write_slice` operations with nested
    `scf.forall` producers.

    The input is IR containing `pcf.write_slice` ops where the source value is
    produced by an `scf.forall` with `tensor.parallel_insert_slice` in its
    terminator.

    The pass moves each matching `pcf.write_slice` inside the `scf.forall`
    body, composing the slice parameters:
    - New offsets: `write_offset + insert_offset * write_stride`
    - New sizes: from the `parallel_insert_slice`
    - New strides: `write_stride * insert_stride`

    This enables further lowering by ensuring writes happen at the granularity
    of the inner parallel loop rather than after the entire forall completes.

    The underlying pattern is exposed via `composeWriteSliceWithParallelInsert()`
    for use in custom pipelines.
  }];
  let dependentDialects = ["::mlir::iree_compiler::IREE::PCF::PCFDialect"];
}

def LowerStructuralPCFPass : Pass<"iree-pcf-lower-structural-pcf", ""> {
  let summary = "Lowers pcf.generic/loop to scf.execute_region/forall";
  let description = [{
    Lowers structured PCF parallel ops to SCF and CF ops. This is the final
    step of PCF lowering, converting the abstract parallel constructs to
    concrete control flow that can be further lowered to target-specific code.

    The input is expected to be IR where PCF ops no longer have tied results
    (after `iree-pcf-convert-sref-to-memref`). The scope attribute on each op
    determines how worker IDs and counts are materialized.

    Op conversions:
    - `pcf.generic` -> `scf.execute_region` with worker IDs from scope
    - `pcf.loop` -> serialized `scf.forall` with iteration bounds from count operands
    - `pcf.return` -> `scf.yield` (in generic) or `scf.forall.in_parallel` (in loop)
    - `pcf.branch_cond_return` -> `cf.cond_br` to a return block

    If `sync_on_return` is set on a parallel op, a barrier (determined by the
    scope) is inserted after the lowered op. This attribute is introduced when
    converting sref to memref to retain the required synchronization semantics
    of any tied results.
  }];
  let dependentDialects = ["::mlir::iree_compiler::IREE::PCF::PCFDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::cf::ControlFlowDialect"];
}

def ConvertSRefToMemRefPass : Pass<"iree-pcf-convert-sref-to-memref", ""> {
  let summary = "Converts pcf.sref types to concrete memref types";
  let description = [{
    Converts `pcf.sref` (shaped reference) types to `memref` types with
    appropriate layouts and memory spaces derived from the scope attribute.
    After this pass, there are no remaining results tied to PCF parallel ops.
    All `pcf.sref` types are converted to concrete `memref` types, preparing
    the IR for structural lowering of PCF ops to SCF/CF.

    The input is expected to be bufferized IR where tied inputs to parallel
    ops are already `memref` types. Layouts are derived from these parallel op
    tied inputs and propagated via dataflow analysis.

    Op conversions:
    - `pcf.alloc` -> `memref.alloc`
    - `pcf.write_slice` -> `memref.subview` + `memref.copy` or `vector.transfer_write`
    - `pcf.read_slice` -> `memref.subview` + `vector.transfer_read`
    - `pcf.generic`/`pcf.loop` region arguments become the tied memref inputs
  }];
  let dependentDialects = ["::mlir::iree_compiler::IREE::PCF::PCFDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::vector::VectorDialect"];
}

def ResolveTokensPass : Pass<"iree-pcf-resolve-tokens", ""> {
  let summary = "Resolves synchronization scopes on `pcf.sref` types.";
  let description = [{
    Resolves synchronization scopes attached to `pcf.sref` types by expanding
    them to their concrete representations. This pass should run before
    `iree-pcf-convert-sref-to-memref`.

    The input is IR containing `pcf.sref` types with sync scope attributes.
    The pass expands `pcf.sref<..., sync_scope>` types into a `pcf.sref`
    without sync scope plus any concrete types required by the sync scope
    attribute. For shaped refs with `sync_on_return` scope, the parent
    `pcf.generic` or `pcf.loop` op has its `sync_on_return` flag set to true,
    ensuring a barrier is inserted when the op is lowered.

    Write operations (`pcf.write_slice`) are updated to enqueue writes through
    the sync scope's interface methods.
  }];
}

#endif // IREE_CODEGEN_DIALECT_PCF_TRANSFORMS_PASSES
