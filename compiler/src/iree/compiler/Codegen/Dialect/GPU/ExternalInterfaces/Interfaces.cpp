// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/ExternalInterfaces/Interfaces.h"

#include "iree/compiler/Codegen/Dialect/GPU/ExternalInterfaces/GPUScopeExternalModels.h"

namespace mlir::iree_compiler {

void registerIREEGPUExternalInterfaces(DialectRegistry &registry) {
  IREE::GPU::registerGPUScopeExternalModels(registry);
}

} // namespace mlir::iree_compiler
