// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_REDUCER_DELTA_H
#define IREE_COMPILER_REDUCER_DELTA_H

#include "iree/compiler/Reducer/ChunkManager.h"
#include "iree/compiler/Reducer/Oracle.h"
#include "iree/compiler/Reducer/WorkItem.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {

using DeltaFunc = llvm::function_ref<void(ChunkManager &, WorkItem &)>;

void runDeltaPass(Oracle &oracle, WorkItem &root, DeltaFunc delta,
                  StringRef message);

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_REDUCER_DELTA_H
