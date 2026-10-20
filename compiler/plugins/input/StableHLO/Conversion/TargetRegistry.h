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

// Handlers receive the original operation, including all operands and regions.
// Decomposition fallback requires its complete signature and unchanged IR.
// Validate applicability before changing IR, including helper symbols and
// attributes. Success replaces the operation; failure makes no IR changes.
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
void populateTargetRewritePatterns(RewritePatternSet &patterns,
                                   const TargetRegistry &registry);
std::unique_ptr<Pass>
createRewriteRegisteredTargetsPass(const TargetRegistry &registry);

} // namespace mlir::iree_compiler::stablehlo

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_TARGETREGISTRY_H_
