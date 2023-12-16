// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_MODULES_IO_PARAMETERS_TRANSFORMS_PASS_DETAIL_H_
#define IREE_COMPILER_MODULES_IO_PARAMETERS_TRANSFORMS_PASS_DETAIL_H_

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

#define GEN_PASS_CLASSES
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h.inc" // IWYU pragma: keep

} // namespace mlir::iree_compiler::IREE::IO::Parameters

#endif // IREE_COMPILER_MODULES_IO_PARAMETERS_TRANSFORMS_PASS_DETAIL_H_
