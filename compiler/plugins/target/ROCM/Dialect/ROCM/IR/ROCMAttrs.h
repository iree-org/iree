// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_ROCMATTRS_H_
#define IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_ROCMATTRS_H_

#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

// clang-format off
#define GET_ATTRDEF_CLASSES
#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMAttrs.h.inc"
#undef GET_ATTRDEF_CLASSES
// clang-format on

#endif // IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_ROCMATTRS_H_
