// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_API_INTERNAL_DIAGNOSTICS_H
#define IREE_COMPILER_API_INTERNAL_DIAGNOSTICS_H

#include <functional>
#include <string_view>

#include "mlir/IR/Diagnostics.h"

namespace mlir::iree_compiler::embed {

/// Provides a diagnostic handler callback which uses heuristics to format
/// a message and invoke an additional callback with the formatted diagnostic.
/// This is an alternative to the use of a SourceMgrDiagnosticHandler in the
/// case where similar functionality is desirable but the diagnostics should
/// be captured and returned in some manner of API (versus streaming to
/// stderr or a stream).
///
/// It is not a pure drop in replacement for SourceMgrDiagnosticHandler because
/// that class relies on various low level properties of the stream to enable
/// color, extract source lines, etc.
class FormattingDiagnosticHandler {
 public:
  using Callback = std::function<void(DiagnosticSeverity severity,
                                      std::string_view message)>;

  FormattingDiagnosticHandler(MLIRContext *ctx, Callback callback);
  ~FormattingDiagnosticHandler();

  LogicalResult emit(Diagnostic &diag);

 private:
  DiagnosticEngine::HandlerID handlerID;
  MLIRContext *ctx;
  Callback callback;
};

}  // namespace mlir::iree_compiler::embed

#endif  // IREE_COMPILER_API_INTERNAL_DIAGNOSTICS_H
