// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VMVX/IR/VMVXTypes.h"

#include "llvm/ADT/StringExtras.h"

// Order matters:
#include "iree/compiler/Dialect/VMVX/IR/VMVXEnums.cpp.inc"

namespace mlir::iree_compiler::IREE::VMVX {

#include "iree/compiler/Dialect/VMVX/IR/VMVXOpInterfaces.cpp.inc"

} // namespace mlir::iree_compiler::IREE::VMVX
