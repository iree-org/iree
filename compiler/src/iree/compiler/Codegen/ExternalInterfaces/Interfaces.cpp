// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/ExternalInterfaces/Interfaces.h"

#include "iree/compiler/Codegen/ExternalInterfaces/CPUEncodingExternalModels.h"
#include "iree/compiler/Codegen/ExternalInterfaces/CodegenExternalModels.h"
#include "iree/compiler/Codegen/ExternalInterfaces/GPUEncodingExternalModels.h"

namespace mlir::iree_compiler {

void registerCodegenExternalInterfaces(DialectRegistry &registry) {
  IREE::Codegen::registerCodegenExternalModels(registry);
  IREE::CPU::registerCPUEncodingExternalModels(registry);
  IREE::GPU::registerGPUEncodingExternalModels(registry);
}

} // namespace mlir::iree_compiler
