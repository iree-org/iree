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
      "Select a IREE::HAL::DispatchLoweringPassPipeline for lowering the variant";
}

def VMVXLinkExecutablesPass :
    Pass<"iree-vmvx-link-executables", "mlir::ModuleOp"> {
  let summary = "Links VMVX HAL executables within the top-level program module.";
}

def VMVXLowerExecutableTargetPass :
    InterfacePass<"iree-vmvx-lower-executable-target", "mlir::FunctionOpInterface"> {
  let summary =
      "Lower executable target using an IREE::HAL::DispatchLoweringPassPipeline";
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

#endif // IREE_CODEGEN_VMVX_PASSES
