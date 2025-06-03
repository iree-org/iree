// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_ROCMDIALECT_H_
#define IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_ROCMDIALECT_H_

#include <mutex>

#include "iree/compiler/Utils/EmbeddedDataDirectory.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

// clang-format off: must be included after all LLVM/MLIR headers
#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMDialect.h.inc" // IWYU pragma: keep
// clang-format on

#endif // IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_ROCMDIALECT_H_
