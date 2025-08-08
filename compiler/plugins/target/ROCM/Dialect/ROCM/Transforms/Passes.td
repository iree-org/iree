// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_TRANSFORMS_PASSES
#define IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ApplyBuiltinPDLPatternsPass : InterfacePass<"iree-rocm-apply-builtin-pdl-patterns", "mlir::FunctionOpInterface"> {
  let summary = "Parse a builtin file based on the target arch containing PDL patterns and apply them as patterns";
  let options = [
    ListOption<"targets", "targets", "std::string",
      "Name of the target(s) to use patterns from">,
    Option<"enableSpecialization", "enable-specialization", "bool",
           /*defaultValue=*/"false",
           "Enable runtime specialization of dynamically shaped dispatches">,
    Option<"enableTensorUKernels", "enable-tensor-ukernels", "bool",
           /*defaultValue=*/"false", "Enable MLIR tensor-based ukernels">,
  ];
  let dependentDialects = [
    "iree_compiler::IREE::Codegen::IREECodegenDialect",
    "iree_compiler::IREE::GPU::IREEGPUDialect",
    "iree_compiler::IREE::ROCM::ROCMDialect",
    "iree_compiler::IREE::Util::UtilDialect",
    "mlir::pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect",
  ];
}


def ApplyBuiltinPDLPatternsDriverPass : Pass<"iree-rocm-apply-builtin-pdl-patterns-driver", "mlir::ModuleOp"> {
  let summary = "Applies builtin PDL patterns on each function-like op within a module";
  let description = [{
    Parses a builtin file containing specialization and/or PDL patterns based on the target arch and applies them as patterns.
  }];
  let options = [
    Option<"enableSpecialization", "enable-specialization", "bool",
           /*defaultValue=*/"false",
           "Enable runtime specialization of dynamically shaped dispatches">,
    Option<"enableTensorUkernels", "enable-tensor-ukernels", "bool",
           /*defaultValue=*/"false", "Enable MLIR-tensor-based ukernels">
  ];
  let dependentDialects = [
    // ROCDL, GPU, and Vector dialect dependencies are to make sure these dialects are loaded when parsing
    // the ukernels that contain them. The pass itself does not depend directly on these dialects.
    "iree_compiler::IREE::HAL::HALDialect",
    "iree_compiler::IREE::ROCM::ROCMDialect",
    "iree_compiler::IREE::Codegen::IREECodegenDialect",
    "mlir::gpu::GPUDialect",
    "mlir::pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect",
    "mlir::vector::VectorDialect",
    "ROCDL::ROCDLDialect",
  ];
}

#endif // IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_TRANSFORMS_PASSES
