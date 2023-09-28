// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_REDUCER_DELTA_H
#define IREE_COMPILER_REDUCER_DELTA_H

#include "iree/compiler/Reducer/Framework/ChunkManager.h"
#include "iree/compiler/Reducer/Framework/Oracle.h"
#include "iree/compiler/Reducer/Framework/WorkItem.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::Reducer {

using DeltaFunc = llvm::function_ref<void(ChunkManager &, WorkItem &)>;

class Delta {
public:
  Delta(Oracle &oracle, WorkItem &root) : oracle(oracle), root(root) {}

  void runDeltaPass(DeltaFunc delta, StringRef message);

private:
  FailureOr<WorkItem> checkChunk(Chunk maybeInterestingChunk,
                                 DeltaFunc deltaFunc,
                                 ArrayRef<Chunk> maybeInterestingChunks,
                                 DenseSet<Chunk> &uninterestingChunks);

  bool increaseGranuality(SmallVector<Chunk> &chunks);

  Oracle &oracle;
  WorkItem &root;
};

} // namespace mlir::iree_compiler::Reducer

#endif // IREE_COMPILER_REDUCER_DELTA_H
