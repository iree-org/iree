// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VMVX_IR_VMVXTYPES_H_
#define IREE_COMPILER_DIALECT_VMVX_IR_VMVXTYPES_H_

#include <cstdint>

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#include "iree/compiler/Dialect/VMVX/IR/VMVXEnums.h.inc"  // IWYU pragma: export
#include "iree/compiler/Dialect/VMVX/IR/VMVXOpInterfaces.h.inc"  // IWYU pragma: export
// clang-format on

#endif  // IREE_COMPILER_DIALECT_VMVX_IR_VMVXTYPES_H_
