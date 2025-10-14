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

//===---------------------------------------------------------------------===//
// [Test Passes]
//===---------------------------------------------------------------------===//

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

def FuseConsumersPass :
    Pass<"iree-pcf-fuse-consumers", ""> {
  let summary = "Fuses all consumers of pcf.generic/loop ops";
  let description = [{
    Primarily test pass for fusing tilable consumers of pcf.generic/loop ops.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::iree_compiler::IREE::PCF::PCFDialect"
  ];
}

def FusePCFWritesPass :
    Pass<"iree-pcf-fuse-pcf-writes", ""> {
  let summary = "Fuses pcf.write_slice with tensor.parallel_insert_slice from scf.forall";
  let description = [{
    Primarily test pass for composing pcf.write_slice operations where the
    source is produced by an scf.forall with tensor.parallel_insert_slice
    in its terminator. Composes the insert_slice with the write_slice.
  }];
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::PCF::PCFDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

#endif // IREE_CODEGEN_DIALECT_PCF_TRANSFORMS_PASSES
