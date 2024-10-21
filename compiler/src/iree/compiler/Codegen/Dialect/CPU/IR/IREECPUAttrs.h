// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUATTRS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUATTRS_H_

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/TypeSwitch.h"

// clang-format off
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUAttrs.h.inc"
#undef GET_ATTRDEF_CLASSES
// clang-format on

#endif // IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUATTRS_H_
