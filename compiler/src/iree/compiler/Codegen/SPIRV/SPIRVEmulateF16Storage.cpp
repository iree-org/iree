// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPIRVEMULATEF16STORAGEPASS
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"

namespace {

static bool hasHalfElementType(Type type) {
  while (auto shaped = dyn_cast<ShapedType>(type)) {
    type = shaped.getElementType();
  }
  return type.isF16();
}

static bool hasHalfValues(Operation *function) {
  return function
      ->walk([](Operation *op) {
        if (llvm::any_of(op->getOperandTypes(), hasHalfElementType) ||
            llvm::any_of(op->getResultTypes(), hasHalfElementType)) {
          return WalkResult::interrupt();
        }
        for (Region &region : op->getRegions()) {
          for (Block &block : region) {
            if (llvm::any_of(block.getArgumentTypes(), hasHalfElementType)) {
              return WalkResult::interrupt();
            }
          }
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static bool hasNarrowStorage(Operation *function) {
  auto isNarrowMemref = [](Type type) {
    auto memref = dyn_cast<MemRefType>(type);
    return memref && memref.getElementType().isSignlessInteger() &&
           memref.getElementTypeBitWidth() < 32;
  };
  return function
      ->walk([isNarrowMemref](Operation *op) {
        if (llvm::any_of(op->getOperandTypes(), isNarrowMemref) ||
            llvm::any_of(op->getResultTypes(), isNarrowMemref)) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

// The common arithmetic pass preserves each f16 rounding boundary with exact
// software conversions. Retyping those values as integer storage afterwards
// also preserves bitcasts, including NaN payloads, without half-pack
// intrinsics. The narrow-integer pass packs the two-byte values into words.
static OpPassManager buildHalfEmulationPipeline(StringRef anchor,
                                                bool emulateHalf) {
  OpPassManager pipeline(anchor);
  if (emulateHalf) {
    ConvertUnsupportedFloatArithPassOptions arithmetic;
    arithmetic.includeF16 = true;
    arithmetic.includeMath = true;
    pipeline.addPass(createConvertUnsupportedFloatArithPass(arithmetic));
    ConvertUnsupportedFloatToIntBuffersPassOptions storage;
    storage.includeBf16 = false;
    storage.includeF8E5M2 = false;
    storage.includeF8E4M3FN = false;
    storage.includeF8E5M2FNUZ = false;
    storage.includeF8E4M3FNUZ = false;
    storage.includeF8E8M0FNU = false;
    storage.includeF16 = true;
    pipeline.addPass(createConvertUnsupportedFloatToIntBuffersPass(storage));
  }
  // Use word storage even when 16-bit memory is available without 16-bit
  // arithmetic. This keeps loads and bit operations in supported register
  // types.
  EmulateNarrowTypePassOptions words;
  words.loadStoreBitwidth = 32;
  pipeline.addPass(createEmulateNarrowTypePass(words));
  pipeline.addPass(createVectorTransferLoweringPass());
  return pipeline;
}

struct SPIRVEmulateF16StoragePass final
    : impl::SPIRVEmulateF16StoragePassBase<SPIRVEmulateF16StoragePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    buildHalfEmulationPipeline("func.func", /*emulateHalf=*/true)
        .getDependentDialects(registry);
  }

  void runOnOperation() override {
    FunctionOpInterface function = getOperation();
    IREE::GPU::TargetAttr target = getGPUTargetAttr(function);
    if (!target) {
      return;
    }
    if (supportsNativeSPIRVF16(target)) {
      return;
    }
    auto compute = target.getWgp().getCompute().getValue();
    bool nativeStorage =
        IREE::GPU::bitEnumContainsAll(target.getWgp().getStorage().getValue(),
                                      IREE::GPU::StorageBitwidths::B16);
    bool emulateHalf = hasHalfValues(function);
    bool nativeInteger = IREE::GPU::bitEnumContainsAll(
        compute, IREE::GPU::ComputeBitwidths::Int16);
    if (!emulateHalf &&
        ((nativeInteger && nativeStorage) || !hasNarrowStorage(function))) {
      return;
    }
    OpPassManager pipeline = buildHalfEmulationPipeline(
        function->getName().getStringRef(), emulateHalf);
    if (failed(runPipeline(pipeline, function))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
