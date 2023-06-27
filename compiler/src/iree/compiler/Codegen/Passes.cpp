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
#include "iree/compiler/Codegen/Common/CommonPasses.h"
#include "iree/compiler/Codegen/Common/GPU/CommonGPUPasses.h"
#include "iree/compiler/Codegen/LLVMCPU/LLVMCPUPasses.h"
#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUPasses.h"
#include "iree/compiler/Codegen/SPIRV/SPIRVPasses.h"
#include "iree/compiler/Codegen/VMVX/VMVXPasses.h"
#include "iree/compiler/Codegen/WGSL/WGSLPasses.h"

namespace mlir {
namespace iree_compiler {

void registerCodegenPasses() {
  // Generated.
  registerCodegenCommonPasses();
  registerCodegenCommonGPUPasses();
  registerCodegenLLVMCPUPasses();
  registerCodegenLLVMGPUPasses();
  registerCodegenSPIRVPasses();
  registerCodegenVMVXPasses();
  registerCodegenWGSLPasses();
}

} // namespace iree_compiler
} // namespace mlir
