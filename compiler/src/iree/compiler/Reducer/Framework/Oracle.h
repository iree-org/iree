// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_REDUCER_ORACLE_H
#define IREE_COMPILER_REDUCER_ORACLE_H

#include "iree/compiler/Reducer/Framework/ChunkManager.h"
#include "iree/compiler/Reducer/Framework/WorkItem.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::Reducer {

class Oracle {
public:
  Oracle(StringRef testScript, bool useBytecode)
      : testScript(testScript), useBytecode(useBytecode) {}

  bool isInteresting(WorkItem &workItem);

private:
  StringRef testScript;
  bool useBytecode;
};

} // namespace mlir::iree_compiler::Reducer

#endif // IREE_COMPILER_REDUCER_ORACLE_H
