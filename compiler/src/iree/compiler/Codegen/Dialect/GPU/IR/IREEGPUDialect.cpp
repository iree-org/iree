// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.cpp.inc"

namespace mlir::iree_compiler::IREE::GPU {

void IREEGPUDialect::initialize() { registerAttributes(); }

} // namespace mlir::iree_compiler::IREE::GPU
