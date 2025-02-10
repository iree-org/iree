// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------- ConvertUnsupportedFloatArithPass.cpp ----------------===//
//
//   Emulate arith and vector floating point operations that use float types
//   which are unspported on a target by inserting extf/truncf pairs around all
//   such operations in order to produce arithmetic that can be performed while
//   preserving the original rounding behavior.
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTUNSUPPORTEDFLOATARITHPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct ConvertUnsupportedFloatArithPass final
    : public impl::ConvertUnsupportedFloatArithPassBase<
          ConvertUnsupportedFloatArithPass> {
  void runOnOperation() override;
  using Base::Base;
};

} // namespace
//

static LogicalResult ParseFromOption(MLIRContext *ctx,
                                     ArrayRef<std::string> sourceTypeStrs,
                                     std::string targetTypeStr,
                                     SmallVectorImpl<Type> &sourceTypes,
                                     Type &targetType) {

  std::optional<FloatType> maybeTargetType =
      arith::parseFloatType(ctx, targetTypeStr);
  if (!maybeTargetType) {
    emitError(UnknownLoc::get(ctx), "could not map target type '" +
                                        targetTypeStr +
                                        "' to a known floating-point type");
    return failure();
  }
  targetType = *maybeTargetType;
  for (StringRef sourceTypeStr : sourceTypeStrs) {
    std::optional<FloatType> maybeSourceType =
        arith::parseFloatType(ctx, sourceTypeStr);
    if (!maybeSourceType) {
      emitError(UnknownLoc::get(ctx), "could not map source type '" +
                                          sourceTypeStr +
                                          "' to a known floating-point type");
      return failure();
    }
    sourceTypes.push_back(*maybeSourceType);
  }

  return success();
}

void ConvertUnsupportedFloatArithPass::runOnOperation() {

  MLIRContext *context = &getContext();
  Operation *op = getOperation();
  SmallVector<Type> sourceTypes;
  Type targetType;

  if (failed(ParseFromOption(context, sourceTypeStrs, targetTypeStr,
                             sourceTypes, targetType))) {
    return signalPassFailure();
  }

  if (sourceTypes.empty()) {
    (void)emitOptionalWarning(
        std::nullopt,
        "no source types specified, float emulation will do nothing");
    return signalPassFailure();
  }

  if (llvm::is_contained(sourceTypes, targetType)) {
    emitError(UnknownLoc::get(context),
              "target type cannot be an unsupported source type");
    return signalPassFailure();
  }

  TypeConverter converter;
  arith::populateEmulateUnsupportedFloatsConversions(converter, sourceTypes,
                                                     targetType);
  RewritePatternSet patterns(context);
  arith::populateEmulateUnsupportedFloatsPatterns(patterns, converter);
  ConversionTarget target(*context);
  arith::populateEmulateUnsupportedFloatsLegality(target, converter);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace mlir::iree_compiler
