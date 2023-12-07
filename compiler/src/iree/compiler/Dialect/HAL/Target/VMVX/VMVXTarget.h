// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_VMVX_VMVXTARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_VMVX_VMVXTARGET_H_

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"

namespace mlir::iree_compiler::IREE::HAL {

// Registers the VMVX backends.
void registerVMVXTargetBackends();

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_VMVX_VMVXTARGET_H_
