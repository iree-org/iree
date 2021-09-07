// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_INTERFACES_H
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_INTERFACES_H

#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_pydm {

enum class BuiltinTypeCode;

}  // namespace iree_pydm
}  // namespace mlir

#include "iree-dialects/Dialect/IREEPyDM/IR/TypeInterfaces.h.inc"

#endif  // IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_INTERFACES_H
