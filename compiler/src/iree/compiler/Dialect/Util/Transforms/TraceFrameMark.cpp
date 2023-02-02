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

class TraceFrameMarkBeginPass
    : public TraceFrameMarkBeginBase<TraceFrameMarkBeginPass> {
 public:
  TraceFrameMarkBeginPass() = default;
  TraceFrameMarkBeginPass(TraceFrameName name) { this->name = name; }

  void runOnOperation() override {
    // Always mark the top level (unnamed) frame.
    IREE_TRACE_FRAME_MARK();

    switch (name) {
      case TraceFrameName::None:
        break;
      case TraceFrameName::Input:
        IREE_TRACE_FRAME_MARK_BEGIN_NAMED("Input");
        break;
      case TraceFrameName::ABI:
        IREE_TRACE_FRAME_MARK_BEGIN_NAMED("ABI");
        break;
      case TraceFrameName::Flow:
        IREE_TRACE_FRAME_MARK_BEGIN_NAMED("Flow");
        break;
      case TraceFrameName::Stream:
        IREE_TRACE_FRAME_MARK_BEGIN_NAMED("Stream");
        break;
      case TraceFrameName::HAL:
        IREE_TRACE_FRAME_MARK_BEGIN_NAMED("HAL");
        break;
      case TraceFrameName::VM:
        IREE_TRACE_FRAME_MARK_BEGIN_NAMED("VM");
        break;
    }
  }
};

class TraceFrameMarkEndPass
    : public TraceFrameMarkEndBase<TraceFrameMarkEndPass> {
 public:
  TraceFrameMarkEndPass() = default;
  TraceFrameMarkEndPass(TraceFrameName name) { this->name = name; }

  void runOnOperation() override {
    switch (name) {
      case TraceFrameName::None:
        break;
      case TraceFrameName::Input:
        IREE_TRACE_FRAME_MARK_END_NAMED("Input");
        break;
      case TraceFrameName::ABI:
        IREE_TRACE_FRAME_MARK_END_NAMED("ABI");
        break;
      case TraceFrameName::Flow:
        IREE_TRACE_FRAME_MARK_END_NAMED("Flow");
        break;
      case TraceFrameName::Stream:
        IREE_TRACE_FRAME_MARK_END_NAMED("Stream");
        break;
      case TraceFrameName::HAL:
        IREE_TRACE_FRAME_MARK_END_NAMED("HAL");
        break;
      case TraceFrameName::VM:
        IREE_TRACE_FRAME_MARK_END_NAMED("VM");
        break;
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createTraceFrameMarkBeginPass(
    TraceFrameName name) {
  return std::make_unique<TraceFrameMarkBeginPass>(name);
}

std::unique_ptr<OperationPass<ModuleOp>> createTraceFrameMarkEndPass(
    TraceFrameName name) {
  return std::make_unique<TraceFrameMarkEndPass>(name);
}

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
