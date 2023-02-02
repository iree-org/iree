// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/TracingUtils.h"

#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_compiler {

#if IREE_ENABLE_COMPILER_TRACING && \
    IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

namespace {
thread_local llvm::SmallVector<iree_zone_id_t, 8> passTraceZonesStack;
}  // namespace

static void prettyPrintOpBreadcrumb(Operation *op, llvm::raw_ostream &os) {
  auto parentOp = op->getParentOp();
  if (parentOp) {
    prettyPrintOpBreadcrumb(parentOp, os);
    os << " > ";
  }
  os << op->getName();
  if (auto symbolOp = dyn_cast<SymbolOpInterface>(op)) {
    os << " @" << symbolOp.getName();
  }
}

void PassTracing::runBeforePass(Pass *pass, Operation *op) {
  IREE_TRACE_ZONE_BEGIN_EXTERNAL(z0, __FILE__, strlen(__FILE__), __LINE__,
                                 pass->getName().data(), pass->getName().size(),
                                 NULL, 0);
  passTraceZonesStack.push_back(z0);

  std::string breadcrumbStorage;
  llvm::raw_string_ostream os(breadcrumbStorage);
  prettyPrintOpBreadcrumb(op, os);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, os.str().data());
}
void PassTracing::runAfterPass(Pass *pass, Operation *op) {
  IREE_TRACE_ZONE_END(passTraceZonesStack.back());
  passTraceZonesStack.pop_back();
}
void PassTracing::runAfterPassFailed(Pass *pass, Operation *op) {
  IREE_TRACE_ZONE_END(passTraceZonesStack.back());
  passTraceZonesStack.pop_back();
}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

}  // namespace iree_compiler
}  // namespace mlir
