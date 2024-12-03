// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PASSES_H_

#include <optional>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
class OpBuilder;
class Type;
class Value;
} // namespace mlir

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DECL_ANNOTATEOPORDINALSPASS
#define GEN_PASS_DECL_APPLYPATTERNSPASS
#define GEN_PASS_DECL_COMBINEINITIALIZERSPASS
#define GEN_PASS_DECL_DROPCOMPILERHINTSPASS
#define GEN_PASS_DECL_DUMPMODULEPASS
// Has un-tablegen-able options (a pass pipeline)
// #define GEN_PASS_DECL_FIXEDPOINTITERATORPASS
#define GEN_PASS_DECL_FOLDGLOBALSPASS
#define GEN_PASS_DECL_FUSEGLOBALSPASS
#define GEN_PASS_DECL_HOISTINTOGLOBALSPASS
#define GEN_PASS_DECL_IPOPASS
#define GEN_PASS_DECL_IMPORTRESOURCESPASS
#define GEN_PASS_DECL_OPTIMIZEINTARITHMETICPASS
#define GEN_PASS_DECL_PROPAGATESUBRANGESPASS
#define GEN_PASS_DECL_SIMPLIFYGLOBALACCESSESPASS
#define GEN_PASS_DECL_STRIPANDSPLATCONSTANTSPASS
#define GEN_PASS_DECL_STRIPDEBUGOPSPASS
#define GEN_PASS_DECL_TESTCONVERSIONPASS
#define GEN_PASS_DECL_TESTFLOATRANGEANALYSISPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

// Kept for compatibility
std::unique_ptr<Pass> createDumpModulePass(std::string path);

std::unique_ptr<OperationPass<void>>
createFixedPointIteratorPass(OpPassManager pipeline);

// Expression hoisting.
struct ExprHoistingOptions {
  using RegisterDialectsFn = std::function<void(DialectRegistry &)>;

  // Hook to register extra dependent dialects needed for types implementing
  // the `HoistableTypeInterace`.
  std::optional<RegisterDialectsFn> registerDependentDialectsFn = std::nullopt;

  // Threshold for controlling the maximum allowed increase in the stored size
  // of a single global as a result of hoisting.
  int64_t maxSizeIncreaseThreshold = 2147483647;
};
std::unique_ptr<Pass>
createHoistIntoGlobalsPass(const ExprHoistingOptions &options);

// Register all Passes
void registerTransformPasses();

} // namespace mlir::iree_compiler::IREE::Util

#endif // IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PASSES_H_
