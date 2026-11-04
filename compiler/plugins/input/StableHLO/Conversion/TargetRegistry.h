// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_TARGETREGISTRY_H_
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_TARGETREGISTRY_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

// Handlers receive the original operation with all its operands and regions.
// Check applicability before changing IR, including helper symbols and
// attributes. Success must replace the matched operation. Failure must leave
// the IR unchanged for later handlers or decomposition.
//
// Make all IR changes through the supplied rewriter so the rewrite driver and
// provenance listener can observe them.
// Captured handler state must support concurrent calls from pass clones.
class TargetRegistry {
public:
  using CustomCallHandler = std::function<LogicalResult(
      mlir::stablehlo::CustomCallOp, PatternRewriter &)>;
  using CompositeHandler = std::function<LogicalResult(
      mlir::stablehlo::CompositeOp, PatternRewriter &)>;
  using DialectRegistration = std::function<void(DialectRegistry &)>;

  // Duplicate keys fail without replacing the existing handler.
  // Custom calls with effects, tokens, futures or aliases are left unchanged.
  LogicalResult addCustomCall(StringRef name, CustomCallHandler handler,
                              DialectRegistration registerDialects = {});
  // Registers a handler for this exact composite name and semantic version.
  // Unregistered versions remain available for decomposition. A custom call's
  // api_version identifies its calling interface; custom-call handlers must
  // check which interfaces they support.
  LogicalResult addComposite(StringRef name, int32_t version,
                             CompositeHandler handler,
                             DialectRegistration registerDialects = {});

  const CustomCallHandler *lookupCustomCall(StringRef name) const;
  const CompositeHandler *lookupComposite(StringRef name,
                                          int32_t version) const;
  void registerDependentDialects(DialectRegistry &dialects) const;

private:
  template <typename Handler>
  struct Entry {
    Handler handler;
    DialectRegistration registerDialects;
  };
  llvm::StringMap<Entry<CustomCallHandler>> customCalls;
  // Versions of each composite name.
  llvm::StringMap<std::map<int32_t, Entry<CompositeHandler>>> composites;
};

// Patterns and passes copy the registry, so later registrations cannot affect
// an already constructed pipeline.
//
// If a handler declines a composite, it remains available for later handlers.
// The standard input conversion pipeline decomposes any remaining composites.
// To retain one, the pipeline must exclude it from decomposition and provide a
// downstream consumer.
void populateTargetRewritePatterns(RewritePatternSet &patterns,
                                   const TargetRegistry &registry);
// The pass requires an explicit registry and cannot be parsed directly from
// text. Use a named pipeline that constructs the registry for textual callers.
std::unique_ptr<Pass> createRewriteRegisteredCustomCallsAndCompositesPass(
    const TargetRegistry &registry);

} // namespace mlir::iree_compiler::stablehlo

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_TARGETREGISTRY_H_
