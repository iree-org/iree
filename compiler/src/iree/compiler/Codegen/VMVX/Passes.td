// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_VMVX_PASSES
#define IREE_CODEGEN_VMVX_PASSES

include "mlir/Pass/PassBase.td"

//===---------------------------------------------------------------------===//
// VMVX passes (keep alphabetical)
//===---------------------------------------------------------------------===//

def VMVXAssignConstantOrdinalsPass :
    Pass<"iree-vmvx-assign-constant-ordinals", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Assigns executable constant ordinals across all VMVX variants.";
}

def VMVXSelectLoweringStrategyPass :
    Pass<"iree-vmvx-select-lowering-strategy", "ModuleOp"> {
  let summary =
      "Select a pipeline attribute for lowering the variant";
}

def VMVXLinkExecutablesPass :
    Pass<"iree-vmvx-link-executables", "mlir::ModuleOp"> {
  let summary = "Links VMVX HAL executables within the top-level program module.";
}

def VMVXLowerExecutableTargetPass :
    InterfacePass<"iree-vmvx-lower-executable-target", "mlir::FunctionOpInterface"> {
  let summary =
      "Lower executable target using a pipeline attribute";
}

def VMVXLowerLinalgMicrokernelsPass :
    Pass<"iree-vmvx-lower-linalg-microkernels", ""> {
  let summary =
      "Lowers linalg ops to the VMVX microkernel library";
  let options = [
    Option<"warnOnUnconverted", "warn-on-unconverted", "bool",
           /*default=*/"false",
           "Warns on any unconverted linalg ops which remain live">
  ];
}

def VMVXSplitReductionPass :
    InterfacePass<"iree-vmvx-split-reduction", "mlir::FunctionOpInterface"> {
  let summary = "Group single-precision sums into partial reductions";
  let description = [{
    With floating-point reassociation enabled, split single-input f32 sums
    with a static, divisible innermost reduction extent into contiguous chunks
    and combine their partial sums. The original initial value is used once.
    Dynamic parallel extents are preserved. Unsupported sums are left intact.
    This changes floating-point association and may change rounded results.
  }];
  let options = [
    Option<"reassociateFpReductions", "reassociate-fp-reductions", "bool", "false",
           "Allow reassociation of floating-point sum reductions.">,
    Option<"reductionChunkSize", "reduction-chunk-size", "int64_t", "64",
           "Chunk size for divisible reductions; must be at least two.">
  ];
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::linalg::LinalgDialect",
                           "mlir::tensor::TensorDialect"];
}

#endif // IREE_CODEGEN_VMVX_PASSES
