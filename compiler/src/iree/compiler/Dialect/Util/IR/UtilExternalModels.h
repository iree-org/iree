// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_IR_UTILEXTERNALMODELS_H_
#define IREE_COMPILER_DIALECT_UTIL_IR_UTILEXTERNALMODELS_H_

namespace mlir {
class DialectRegistry;

namespace iree_compiler {
namespace IREE {
namespace Util {

void registerUtilExternalModels(DialectRegistry& registry);

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_UTIL_IR_UTILEXTERNALMODELS_H_
