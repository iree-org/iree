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

namespace mlir {
namespace iree_compiler {

class Oracle {
public:
  Oracle(StringRef testScript, llvm::raw_ostream &debugOs)
      : testScript(testScript), debugOs(debugOs) {}

  bool isInteresting(WorkItem &workItem);

private:
  StringRef testScript;
  llvm::raw_ostream &debugOs;
};

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_REDUCER_ORACLE_H
