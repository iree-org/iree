// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_TRANSFORMS_PASS_DETAIL_H_
#define IREE_COMPILER_DIALECT_STREAM_TRANSFORMS_PASS_DETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Stream {

// TODO(benvanik): find a way to share this with IREEVM.h w/o circular deps.
// Defines the output format of a dump pass.
enum class DumpOutputFormat {
  // Dumping disabled.
  None = 0,
  // Human-readable pretty printing.
  Pretty = 1,
  // Pretty printing with additional information that can result in large dumps.
  Verbose = 2,
  // Comma separated values for throwing into Sheets.
  CSV = 3,
  // JSON format for better structure and data exchange.
  JSON = 4,
};

#define GEN_PASS_CLASSES
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc" // IWYU pragma: keep

} // namespace mlir::iree_compiler::IREE::Stream

#endif // IREE_COMPILER_DIALECT_STREAM_TRANSFORMS_PASS_DETAIL_H_
