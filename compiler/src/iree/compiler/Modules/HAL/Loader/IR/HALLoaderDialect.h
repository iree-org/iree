// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_MODULES_HAL_LOADER_IR_HALLOADERDIALECT_H_
#define IREE_COMPILER_MODULES_HAL_LOADER_IR_HALLOADERDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {
namespace Loader {

class HALLoaderDialect : public Dialect {
 public:
  explicit HALLoaderDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "hal_loader"; }
};

}  // namespace Loader
}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_MODULES_HAL_LOADER_IR_HALLOADERDIALECT_H_
