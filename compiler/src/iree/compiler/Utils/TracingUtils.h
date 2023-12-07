// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_TRACINGUTILS_H_
#define IREE_COMPILER_UTILS_TRACINGUTILS_H_

#include "iree/base/api.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"

namespace mlir::iree_compiler {

// Instruments passes using IREE's runtime tracing support.
//
// Usage:
//   passManager.addInstrumentation(std::make_unique<PassTracing>());
struct PassTracing : public PassInstrumentation {
  PassTracing() {}
  ~PassTracing() override = default;

#if IREE_ENABLE_COMPILER_TRACING &&                                            \
    IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  void runBeforePass(Pass *pass, Operation *op) override;
  void runAfterPass(Pass *pass, Operation *op) override;
  void runAfterPassFailed(Pass *pass, Operation *op) override;
#endif // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
};

#if IREE_ENABLE_COMPILER_TRACING &&                                            \
    IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

enum {
  IREE_TRACING_COMPILER_MESSAGE_LEVEL_ERROR = 0xFF0000u,
  IREE_TRACING_COMPILER_MESSAGE_LEVEL_WARNING = 0xFFFF00u,
  IREE_TRACING_COMPILER_MESSAGE_LEVEL_INFO = 0xFFFFFFu,
  IREE_TRACING_COMPILER_MESSAGE_LEVEL_VERBOSE = 0xC0C0C0u,
  IREE_TRACING_COMPILER_MESSAGE_LEVEL_DEBUG = 0x00FF00u,
};

// Fork of IREE_TRACE_MESSAGE_DYNAMIC, taking std::string (or llvm::StringRef).
#define IREE_COMPILER_TRACE_MESSAGE_DYNAMIC(level, value_string)               \
  ___tracy_emit_messageC(value_string.data(), value_string.size(),             \
                         IREE_TRACING_COMPILER_MESSAGE_LEVEL_##level, 0)

// Adds a pass to |passManager| that marks the beginning of a named frame.
// * |frameName| must be a null-terminated string
// * |frameName| must use the same underlying storage as the name used with
//   IREE_TRACE_ADD_END_FRAME_PASS
#define IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, frameName)                \
  passManager.addPass(createTraceFrameMarkBeginPass(frameName));

// Adds a pass to |passManager| that marks the end of a named frame.
// * |frameName| must be a null-terminated string
// * |frameName| must use the same underlying storage as the name used with
//   IREE_TRACE_ADD_BEGIN_FRAME_PASS
#define IREE_TRACE_ADD_END_FRAME_PASS(passManager, frameName)                  \
  passManager.addPass(createTraceFrameMarkEndPass(frameName));

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createTraceFrameMarkBeginPass(llvm::StringRef name = "");
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createTraceFrameMarkEndPass(llvm::StringRef name = "");

#else
#define IREE_COMPILER_TRACE_MESSAGE_DYNAMIC(level, value_string)
#define IREE_TRACE_ADD_BEGIN_FRAME_PASS(passManager, frameName)
#define IREE_TRACE_ADD_END_FRAME_PASS(passManager, frameName)
#endif // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_TRACINGUTILS_H_
