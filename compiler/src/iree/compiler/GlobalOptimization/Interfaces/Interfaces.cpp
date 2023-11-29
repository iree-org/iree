// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Interfaces/Interfaces.h"
#include "iree/compiler/GlobalOptimization/Interfaces/HoistableTypeInterface.h"

namespace mlir {
namespace iree_compiler {

void registerGlobalOptimizationInterfaces(DialectRegistry &registry) {
  registerHoistableTypeInterfaces(registry);
}

} // namespace iree_compiler
} // namespace mlir
