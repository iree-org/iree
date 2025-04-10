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
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-convert-unsupported-float-arith"

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

// Populates source and target conversion types based on the target
// architecture.
// TODO(pashu123): Refine the patterns based on the target arch.
static void populateSourceAndTargetType(MLIRContext *ctx, Operation *op,
                                        SmallVectorImpl<Type> &sourceTypes,
                                        Type &targetType) {
  auto gpuAttr = getGPUTargetAttr(op);
  if (!gpuAttr) {
    return;
  }
  StringRef chipset = gpuAttr.getArch();
  FailureOr<amdgpu::Chipset> maybeChipset = amdgpu::Chipset::parse(chipset);
  if (failed(maybeChipset)) {
    LLVM_DEBUG(llvm::dbgs() << "Invalid chip name");
    return;
  }
  // Add source and target conversion types for gfx94{*} series.
  if (maybeChipset->majorVersion == 9 && maybeChipset->minorVersion == 4) {
    sourceTypes.insert(sourceTypes.end(), {Float8E4M3FNUZType::get(ctx),
                                           Float8E5M2FNUZType::get(ctx)});
    targetType = Float32Type::get(ctx);
  }
  return;
}

void ConvertUnsupportedFloatArithPass::runOnOperation() {
  MLIRContext *context = &getContext();
  FunctionOpInterface funcOp = getOperation();
  SmallVector<Type> sourceTypes;
  Type targetType = nullptr;

  populateSourceAndTargetType(context, funcOp, sourceTypes, targetType);

  if (sourceTypes.empty() || !targetType) {
    LLVM_DEBUG(llvm::dbgs() << "no source or target type specified, float "
                               "emulation will do nothing\n");
    return;
  }

  if (llvm::is_contained(sourceTypes, targetType)) {
    funcOp->emitError() << " target type cannot be an unsupported source type";
    return signalPassFailure();
  }

  TypeConverter converter;
  arith::populateEmulateUnsupportedFloatsConversions(converter, sourceTypes,
                                                     targetType);
  RewritePatternSet patterns(context);
  arith::populateEmulateUnsupportedFloatsPatterns(patterns, converter);
  ConversionTarget target(*context);
  arith::populateEmulateUnsupportedFloatsLegality(target, converter);

  if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
