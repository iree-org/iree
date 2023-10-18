// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_REDUCER_DELTA_STRATEGIES_H_
#define IREE_COMPILER_REDUCER_DELTA_STRATEGIES_H_

#include "iree/compiler/Reducer/Framework/Delta.h"

namespace mlir::iree_compiler::Reducer {

void reduceLinalgOnTensorsDelta(ChunkManager &chunker, WorkItem &workItem);
void reduceFlowDispatchOperandToResultDelta(ChunkManager &chunker,
                                            WorkItem &workItem);
void reduceFlowDispatchResultBySplatDelta(ChunkManager &chunker,
                                          WorkItem &workItem);
void reduceOptimizationBarriersDelta(ChunkManager &chunker, WorkItem &workItem);

} // namespace mlir::iree_compiler::Reducer

#endif // IREE_COMPILER_REDUCER_DELTA_STRATEGIES_H_
