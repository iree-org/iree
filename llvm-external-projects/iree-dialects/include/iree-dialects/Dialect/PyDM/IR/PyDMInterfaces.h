// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_PYDM_IR_PYDM_INTERFACES_H
#define IREE_DIALECTS_DIALECT_PYDM_IR_PYDM_INTERFACES_H

#include "iree-dialects/Dialect/PyDM/IR/Constants.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace PYDM {

enum class BuiltinTypeCode;

} // namespace PYDM
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#include "iree-dialects/Dialect/PyDM/IR/PyDMOpInterfaces.h.inc"
#include "iree-dialects/Dialect/PyDM/IR/PyDMTypeInterfaces.h.inc"

#endif // IREE_DIALECTS_DIALECT_PYDM_IR_PYDM_INTERFACES_H
