// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_TOOLS_INIT_TARGETS_H_
#define IREE_COMPILER_TOOLS_INIT_TARGETS_H_

namespace mlir::iree_compiler {

// This function should be called before creating any MLIRContext if one
// expects all the possible target backends to be available. Custom tools can
// select which targets they want to support by only registering those they
// need.
void registerHALTargetBackends();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_TOOLS_INIT_TARGETS_H_
