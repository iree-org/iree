// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/Utils/ConversionUtils.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

static void emitLegalizationErrors(Location loc,
                                   const DenseSet<Operation *> &illegalOps) {
  // Print op errors for each of the illegal ops that still remain.
  llvm::MapVector<StringRef, int> opNameCounts;
  for (Operation *illegalOp : illegalOps) {
    StringRef opName = illegalOp->getName().getStringRef();
    opNameCounts[opName]++;
    illegalOp->emitOpError() << ": illegal op still exists";
  }

  std::vector<std::string> errorMessages;
  errorMessages.reserve(opNameCounts.size());
  for (const auto &opInfo : opNameCounts) {
    errorMessages.push_back(
        llvm::formatv("\t{0} (count: {1})", opInfo.first, opInfo.second));
  }
  emitError(loc) << "The following illegal operations still remain: \n"
                 << llvm::join(errorMessages, "\n") << "\n";
}

LogicalResult verifyAllOperationsAreLegal(Operation *op,
                                          const ConversionTarget &target) {
  // We don't just use applyPartialConversion with no patterns because this pass
  // shouldn't alter the IR at all (including via folding or canonicalizations
  // that dialect conversion does automatically).
  DenseSet<Operation *> illegalOps;
  op->walk([&](Operation *op) {
    if (!target.isLegal(op)) {
      illegalOps.insert(op);
    }
  });
  if (illegalOps.empty()) return success();
  emitLegalizationErrors(op->getLoc(), illegalOps);
  return failure();
}

}  // namespace iree_compiler
}  // namespace mlir
