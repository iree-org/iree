// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- EraseHALDescriptorTypeFromMemRef -----------------------------------===//
// Patterns and pass to erase #hal.descriptor_type from MemRef memory space.
// The purpose of these utilities is just to make transitioning easier--right
// now converting to LLVM still has lots of underlying assumption over numeric
// memory spaces, and some pattern does not support memory space other than 0.
//===----------------------------------------------------------------------===//

#include <memory>

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-erase-hal-descriptor-type"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ERASEHALDESCRIPTORTYPEFROMMEMREFPASS
#define GEN_PASS_DEF_CONVERTHALDESCRIPTORTYPETOGPUADDRESSSPACEPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Returns true if the given `type` is considered as legal.
static bool isLegalType(Type type) {
  if (auto memRefType = llvm::dyn_cast<BaseMemRefType>(type)) {
    Attribute spaceAttr = memRefType.getMemorySpace();
    return !spaceAttr || !llvm::isa<IREE::HAL::DescriptorTypeAttr>(spaceAttr);
  }
  return true;
}

struct CastToFatBufferAlways
    : public OpRewritePattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp op,
                                PatternRewriter &rewriter) const override {
    auto memRefType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!memRefType)
      return failure();
    auto addrSpace = dyn_cast_if_present<amdgpu::AddressSpaceAttr>(
        memRefType.getMemorySpace());
    if (!addrSpace)
      return failure();
    MemRefType::Builder asGlobal(memRefType);
    asGlobal.setMemorySpace(
        gpu::AddressSpaceAttr::get(op.getContext(), gpu::AddressSpace::Global));
    rewriter.modifyOpInPlace(
        op, [&]() { op.getResult().setType((MemRefType)(asGlobal)); });
    rewriter.setInsertionPointAfter(op);
    auto asFatBuf = rewriter.create<amdgpu::FatRawBufferCastOp>(
        op.getLoc(), op.getResult(), /*validBytes=*/Value{},
        /*cacheSwizzleStride=*/Value{}, /*boundsCheck=*/true,
        /*resetOffset=*/true);
    Value newDesc = asFatBuf;
    // Un-reset dynamic offsets
    if (asFatBuf.getType() != memRefType) {
      newDesc =
          rewriter.create<memref::CastOp>(op.getLoc(), memRefType, newDesc);
    }
    rewriter.replaceAllUsesExcept(op, newDesc, asFatBuf);
    return success();
  }
};

struct EraseHALDescriptorTypeFromMemRefPass final
    : impl::EraseHALDescriptorTypeFromMemRefPassBase<
          EraseHALDescriptorTypeFromMemRefPass> {
  void runOnOperation() override {
    AttrTypeReplacer replacer;
    replacer.addReplacement(
        [](BaseMemRefType memRefType) -> std::optional<BaseMemRefType> {
          if (isLegalType(memRefType))
            return std::nullopt;

          // Erase the #hal.descriptor_type memory space.
          if (auto rankedType = llvm::dyn_cast<MemRefType>(memRefType)) {
            return MemRefType::get(memRefType.getShape(),
                                   memRefType.getElementType(),
                                   rankedType.getLayout());
          }
          return UnrankedMemRefType::get(memRefType.getElementType(),
                                         /*memorySpace=*/0);
        });

    Operation *op = getOperation();

    replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/false,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
  }
};

struct ConvertHALDescriptorTypeToGPUAddressSpacePass final
    : impl::ConvertHALDescriptorTypeToGPUAddressSpacePassBase<
          ConvertHALDescriptorTypeToGPUAddressSpacePass> {
  void runOnOperation() override {
    AttrTypeReplacer replacer;
    replacer.addReplacement(
        [](BaseMemRefType memRefType) -> std::optional<BaseMemRefType> {
          if (isLegalType(memRefType))
            return std::nullopt;

          // NOTE THIS IS COMMENTED OUT FOR A QUICK TEST DO NOT MERGE
          // Attribute globalSpace = gpu::AddressSpaceAttr::get(
          //    memRefType.getContext(), gpu::AddressSpace::Global);
          Attribute globalSpace = amdgpu::AddressSpaceAttr::get(
              memRefType.getContext(), amdgpu::AddressSpace::FatRawBuffer);

          // Erase the #hal.descriptor_type memory space.
          if (auto rankedType = llvm::dyn_cast<MemRefType>(memRefType)) {
            return MemRefType::get(memRefType.getShape(),
                                   memRefType.getElementType(),
                                   rankedType.getLayout(), globalSpace);
          }
          return UnrankedMemRefType::get(memRefType.getElementType(),
                                         /*memorySpace=*/globalSpace);
        });

    Operation *op = getOperation();

    replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/false,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

    RewritePatternSet patterns(&getContext());
    patterns.add<CastToFatBufferAlways>(op->getContext());
    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
} // namespace mlir::iree_compiler
