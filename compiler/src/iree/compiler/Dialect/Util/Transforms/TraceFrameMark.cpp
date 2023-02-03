// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/tracing.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

namespace {

// Warning: the 'name' used must be null-terminated and the same underlying
// string data must be used with matching calls to Begin and End
//
// These patterns should both work, for example:
//   createTraceFrameMarkBeginPass("Foo");
//   createTraceFrameMarkEndPass("Foo");
//
//   const char phaseName[] = "Foo\0";
//   createTraceFrameMarkBeginPass(phaseName);
//   createTraceFrameMarkEndPass(phaseName);
//
// This will *not* work:
//   std::string phaseNameBegin = "Foo";
//   std::string phaseNameEnd = "Foo";
//   createTraceFrameMarkBeginPass(phaseNameBegin);
//   createTraceFrameMarkEndPass(phaseNameEnd);

class TraceFrameMarkBeginPass
    : public TraceFrameMarkBeginBase<TraceFrameMarkBeginPass> {
 public:
  TraceFrameMarkBeginPass() = default;
  TraceFrameMarkBeginPass(llvm::StringRef name) { this->name = name; }

  void runOnOperation() override {
    // Always mark the top level (unnamed) frame.
    IREE_TRACE_FRAME_MARK();

    if (!name.empty()) {
      IREE_TRACE_FRAME_MARK_BEGIN_NAMED(name.data());
    }
  }
};

class TraceFrameMarkEndPass
    : public TraceFrameMarkEndBase<TraceFrameMarkEndPass> {
 public:
  TraceFrameMarkEndPass() = default;
  TraceFrameMarkEndPass(llvm::StringRef name) { this->name = name; }

  void runOnOperation() override {
    if (!name.empty()) {
      IREE_TRACE_FRAME_MARK_END_NAMED(name.data());
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createTraceFrameMarkBeginPass(
    llvm::StringRef name) {
  return std::make_unique<TraceFrameMarkBeginPass>(name);
}

std::unique_ptr<OperationPass<ModuleOp>> createTraceFrameMarkEndPass(
    llvm::StringRef name) {
  return std::make_unique<TraceFrameMarkEndPass>(name);
}

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
