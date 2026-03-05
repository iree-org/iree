// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Map/ExternalInterfaces/Interfaces.h"

#include "iree/compiler/Codegen/Dialect/Map/ExternalInterfaces/VectorLayoutInterfaceImpl.h"

namespace mlir::iree_compiler {

void registerIREEMapExternalInterfaces(DialectRegistry &registry) {
  IREE::Map::registerVectorLayoutInterfaceExternalModels(registry);
}

} // namespace mlir::iree_compiler
