// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.cpp.inc" // IWYU pragma: keep
// clang-format on

using namespace mlir;
using namespace mlir::iree_compiler::IREE::Codegen;
namespace IREE = mlir::iree_compiler::IREE;

//===----------------------------------------------------------------------===//
// ExtractStridedMetadataOp
//===----------------------------------------------------------------------===//

/// The number and type of the results are inferred from the
/// shape of the source.
LogicalResult ExtractStridedMetadataOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ExtractStridedMetadataOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto sourceType = llvm::dyn_cast<MemRefType>(adaptor.getSource().getType());
  if (!sourceType)
    return failure();

  unsigned sourceRank = sourceType.getRank();
  IndexType indexType = IndexType::get(context);
  auto memrefType =
      MemRefType::get({}, sourceType.getElementType(),
                      MemRefLayoutAttrInterface{}, sourceType.getMemorySpace());
  // Base.
  inferredReturnTypes.push_back(memrefType);
  // Offset.
  inferredReturnTypes.push_back(indexType);
  // Sizes and strides.
  for (unsigned i = 0; i < sourceRank * 2; ++i)
    inferredReturnTypes.push_back(indexType);
  return success();
}

void ExtractStridedMetadataOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getBaseBuffer(), "base_buffer");
  setNameFn(getOffset(), "offset");
  // For multi-result to work properly with pretty names and packed syntax `x:3`
  // we can only give a pretty name to the first value in the pack.
  if (!getSizes().empty()) {
    setNameFn(getSizes().front(), "sizes");
    setNameFn(getStrides().front(), "strides");
  }
}

//===----------------------------------------------------------------------===//
// LoadFromBufferOp
//===----------------------------------------------------------------------===//

LogicalResult LoadFromBufferOp::verify() {
  RankedTensorType tensorType = getTensor().getType();
  MemRefType memrefType = getBuffer().getType();
  if (failed(verifyCompatibleShape(tensorType.getShape(),
                                   memrefType.getShape())) ||
      tensorType.getElementType() != memrefType.getElementType()) {
    return emitOpError("buffer and tensor shapes must be compatible and "
                       "element types must match");
  }
  return success();
}

void LoadFromBufferOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getBufferMutable(),
                       SideEffects::DefaultResource::get());
}

LogicalResult LoadFromBufferOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointAfterValue(getBuffer());
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0] = memref::getMixedSizes(b, getLoc(), getBuffer());
  return success();
}

//===----------------------------------------------------------------------===//
// StoreToBufferOp
//===----------------------------------------------------------------------===//

LogicalResult StoreToBufferOp::verify() {
  RankedTensorType tensorType = getTensor().getType();
  MemRefType memrefType = getBuffer().getType();
  if (failed(verifyCompatibleShape(tensorType.getShape(),
                                   memrefType.getShape())) ||
      tensorType.getElementType() != memrefType.getElementType()) {
    return emitOpError("tensor and buffer shapes must be compatible and "
                       "element types must match");
  }
  return success();
}

void StoreToBufferOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getBufferMutable(),
                       SideEffects::DefaultResource::get());
}
