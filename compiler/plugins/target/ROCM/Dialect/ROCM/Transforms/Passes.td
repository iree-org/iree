// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_TRANSFORMS_PASSES
#define IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ApplyBuiltinPDLPatternsPass : Pass<"iree-rocm-apply-builtin-pdl-patterns", ""> {
  let summary = "Parse a builtin file based on the target arch containing PDL patterns and apply them as patterns";
  let options = [
    ListOption<"targets", "targets", "std::string",
      "Name of the target(s) to use patterns from">,
  ];
  let dependentDialects = [
    "iree_compiler::IREE::Codegen::IREECodegenDialect",
    "iree_compiler::IREE::ROCM::ROCMDialect",
    "iree_compiler::IREE::Util::UtilDialect",
    "mlir::pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect",
  ];
}

#endif // IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_TRANSFORMS_PASSES
