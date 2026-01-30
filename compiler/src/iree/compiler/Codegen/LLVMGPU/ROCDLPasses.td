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

def ROCDLAddLDSAliasScopesPass : Pass<
    "iree-rocdl-add-lds-alias-scopes", "ModuleOp"> {
  let summary = "Add alias scope metadata to ROCDL LoadToLDS operations";
  let description = [{
    This pass adds LLVM alias scope metadata to ROCDL::LoadToLDSOp operations
    to enable the LLVM backend to distinguish between different LDS buffers
    used in double-buffered pipelining.

    Without alias scopes, the backend conservatively inserts s_waitcnt vmcnt(0)
    before every LDS access that might alias with a prior LoadToLDS operation.

    With alias scopes, the backend understands that LoadToLDS ops to different
    buffers don't alias, allowing it to eliminate redundant vmcnt(0) instructions
    and enable better overlap between global loads and LDS reads.
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "ROCDL::ROCDLDialect"
  ];
}

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

def ROCDLLoadToTransposeLoadPass :
    InterfacePass<"iree-rocdl-load-to-transpose-load", "mlir::FunctionOpInterface"> {
  let summary = "Lower vector.transfer_read to amdgpu.transpose_load when profitable";
  let description = [{
    This pass analyzes vector.transfer_read operations that are fed by
    iree_codegen.index_hint operations (with lane_constant and lane_increment
    attributes) and converts them to amdgpu.transpose_load operations when
    the access pattern matches the requirements for transpose load instructions.

    This pass relies on shape information from memrefs to verify and perform the
    transformation, so this pass should be run before any memref flattening,
    and before fully unrolling non-contiguous vector loads into scalar loads.

    This pass is ROCDL-specific as transpose_load is an AMD GPU feature
    (available on CDNA architectures like gfx950+).
  }];
  let dependentDialects = [
    "amdgpu::AMDGPUDialect",
    "::mlir::iree_compiler::IREE::Codegen::IREECodegenDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"
  ];
}

#endif // IREE_CODEGEN_LLVMGPU_ROCDLPASSES
