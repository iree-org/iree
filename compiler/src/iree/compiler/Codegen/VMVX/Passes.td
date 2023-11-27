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

def VMVXAssignConstantOrdinals :
    Pass<"iree-vmvx-assign-constant-ordinals", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Assigns executable constant ordinals across all VMVX variants.";
  let constructor = "mlir::iree_compiler::createVMVXAssignConstantOrdinalsPass()";
}

def VMVXSelectLoweringStrategy :
    Pass<"iree-vmvx-select-lowering-strategy",
         "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary =
      "Select a IREE::HAL::DispatchLoweringPassPipeline for lowering the variant";
  let constructor =
      "mlir::iree_compiler::createVMVXSelectLoweringStrategyPass()";
}

def VMVXLinkExecutables :
    Pass<"iree-vmvx-link-executables", "mlir::ModuleOp"> {
  let summary = "Links VMVX HAL executables within the top-level program module.";
  let constructor = "mlir::iree_compiler::createVMVXLinkExecutablesPass()";
}

def VMVXLowerExecutableTarget :
    Pass<"iree-vmvx-lower-executable-target",
         "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary =
      "Lower executable target using an IREE::HAL::DispatchLoweringPassPipeline";
  let constructor =
      "mlir::iree_compiler::createVMVXLowerExecutableTargetPass()";
}

def VMVXLowerLinalgMicrokernels :
    Pass<"iree-vmvx-lower-linalg-microkernels", ""> {
  let summary =
      "Lowers linalg ops to the VMVX microkernel library";
  let constructor = "mlir::iree_compiler::createVMVXLowerLinalgMicrokernelsPass()";
  let options = [
    Option<"warnOnUnconverted", "warn-on-unconverted", "bool",
           /*default=*/"false",
           "Warns on any unconverted linalg ops which remain live">
  ];
}

#endif // IREE_CODEGEN_VMVX_PASSES
