// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_REDUCER_DELTA_STRATEGIES_H_
#define IREE_COMPILER_REDUCER_DELTA_STRATEGIES_H_

#include "iree/compiler/Reducer/Delta.h"

namespace mlir {
namespace iree_compiler {

void reduceLinalgOnTensorsDelta(Oracle &oracle, WorkItem &workItem);
void reduceFlowDispatchOperandToResultDelta(Oracle &oracle, WorkItem &workItem);
void reduceFlowDispatchResultBySplatDelta(Oracle &oracle, WorkItem &workItem);

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_REDUCER_DELTA_STRATEGIES_H_
