// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_LLVMGPU_ROCDLPASSES
#define IREE_CODEGEN_LLVMGPU_ROCDLPASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ROCDL Passes (keep alphabetical)
//===----------------------------------------------------------------------===//

def ROCDLAnnotateKernelForTranslationPass : Pass<
    "iree-rocdl-annotate-kernel-for-translation", "LLVM::LLVMFuncOp"> {
  let summary = "Set function attributes before translating to LLVM IR";
  let dependentDialects = ["ROCDL::ROCDLDialect"];
}

def ROCDLBufferInstructionsOptimizationPass :
    InterfacePass<"iree-rocdl-buffer-instructions-optimization", "mlir::FunctionOpInterface"> {
  let summary = "Optimizations possible with buffer fat pointers";
  let description = [{
    Buffer fat pointers support out of bound access so we can make use of this
    feature for optimizations.

    This pass is a no-op on non-ROCDL targets.
  }];
}

def ROCDLConfigureBufferInstructionsPass :
    InterfacePass<"iree-rocdl-configure-buffer-instructions", "mlir::FunctionOpInterface"> {
  let summary = "Determine which subspans can be implemneted with buffer fat pointers";
  let description = [{
    Analyze hal.interface.binding.subspan operations within a dispatch to determine
    whether
    - The size of the binding itself, taken as a region of memory, is under 2 GB
     (the limit for indexing a buffer descriptor is uint32_t max, but a lot of
     intermediate code assumes indices are signed, se this limit is conservative) and
    - That the offset argument to the subspan is workgroup-uniform, since it will
     be folded into the base offset of the descriptor.

    Note that this pass only adds annotations to subspans - the descriptor will be
    constructed during bufferization based on annotations from this pass.

    This pass is a no-op on non-ROCDL targets.
  }];
  let dependentDialects = ["::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"];
}

def ROCDLLowerExecutableTargetPass : InterfacePass<
    "iree-rocdl-lower-executable-target", "mlir::FunctionOpInterface"> {
  let summary = "Lower an IREE hal.executable.variant op using a suitable "
                "pass pipeline";
}

def ROCDLSelectLoweringStrategyPass :
    Pass<"iree-rocdl-select-lowering-strategy", "ModuleOp"> {
  let summary = "Select a suitable lowering strategy for an IREE "
                "hal.executable.variant op";
}

#endif // IREE_CODEGEN_LLVMGPU_ROCDLPASSES
