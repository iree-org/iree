// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/ExternalInterfaces/Interfaces.h"

#include "iree/compiler/Codegen/Dialect/CPU/ExternalInterfaces/CPUPipelineExternalModels.h"

namespace mlir::iree_compiler {

void registerIREECPUExternalInterfaces(DialectRegistry &registry) {
  IREE::CPU::registerCPUPipelineExternalModels(registry);
}

} // namespace mlir::iree_compiler
