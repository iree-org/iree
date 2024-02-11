// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Pass/PassManager.h"

//===---------------------------------------------------------------------===//
// Include pass headers per target device
//===---------------------------------------------------------------------===//
#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/VMVX/Passes.h"
#include "iree/compiler/Codegen/WGSL/Passes.h"

namespace mlir::iree_compiler {

void registerCodegenPasses() {
  // Generated.
  registerCodegenCommonPasses();
  registerCodegenCommonCPUPasses();
  registerCodegenCommonGPUPasses();
  registerCodegenLLVMCPUPasses();
  registerCodegenLLVMGPUPasses();
  registerCodegenROCDLPasses();
  registerCodegenSPIRVPasses();
  registerCodegenVMVXPasses();
  registerCodegenWGSLPasses();
}

} // namespace mlir::iree_compiler
