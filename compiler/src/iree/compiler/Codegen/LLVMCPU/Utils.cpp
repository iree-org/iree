// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Utils.h"

#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "iree-llvmcpu-utils"

constexpr char kMaxStackAllocationSizeAttrName[] = "max_stack_allocation_size";
constexpr char kNativeVectorSizeAttrName[] = "native_vector_size";

namespace mlir::iree_compiler {

static llvm::cl::opt<int32_t> clMaxAllowedNumberOfNativeVectors(
    "iree-llvmcpu-max-allowed-number-of-native-vectors",
    llvm::cl::desc("ratio used to compute the max allowed vector size"),
    llvm::cl::init(512));

std::optional<int64_t>
getConfigMaxStackAllocationSize(DictionaryAttr targetConfig) {
  auto attr = targetConfig.getAs<IntegerAttr>(kMaxStackAllocationSizeAttrName);
  if (attr) {
    return attr.getInt();
  }
  return std::nullopt;
}
void addConfigMaxStackAllocationSize(MLIRContext *context,
                                     int64_t maxStackAllocationSize,
                                     SmallVectorImpl<NamedAttribute> &config) {
  config.emplace_back(
      StringAttr::get(context, kMaxStackAllocationSizeAttrName),
      IntegerAttr::get(IntegerType::get(context, 64), maxStackAllocationSize));
}

std::optional<int64_t> getConfigNativeVectorSize(DictionaryAttr targetConfig) {
  auto attr = targetConfig.getAs<IntegerAttr>(kNativeVectorSizeAttrName);
  if (attr) {
    return attr.getInt();
  }
  return std::nullopt;
}
void addConfigNativeVectorSize(MLIRContext *context, int64_t nativeVectorSize,
                               SmallVectorImpl<NamedAttribute> &config) {
  config.emplace_back(
      StringAttr::get(context, kNativeVectorSizeAttrName),
      IntegerAttr::get(IntegerType::get(context, 64), nativeVectorSize));
}

bool preferIntrinsicsOverAsm(DictionaryAttr targetConfig) {
  auto intrinsicsAttr =
      targetConfig.getAs<BoolAttr>("prefer_intrinsics_over_asm");
  return intrinsicsAttr && intrinsicsAttr.getValue();
}

bool hasAVX2Feature(DictionaryAttr targetConfig) {
  return hasFeature(targetConfig, "+avx2");
}

bool hasAVX512fFeature(DictionaryAttr targetConfig) {
  return hasFeature(targetConfig, "+avx512f");
}

bool hasVFeature(DictionaryAttr targetConfig) {
  return hasFeature(targetConfig, "+v");
}

bool hasZve32xFeature(DictionaryAttr targetConfig) {
  return hasFeature(targetConfig, "+zve32x");
}

bool hasZve32fFeature(DictionaryAttr targetConfig) {
  return hasFeature(targetConfig, "+zve32f");
}

bool hasZve64xFeature(DictionaryAttr targetConfig) {
  return hasFeature(targetConfig, "+zve64x");
}

bool hasAnyVFeature(DictionaryAttr targetConfig) {
  return hasVFeature(targetConfig) || hasZve32xFeature(targetConfig) ||
         hasZve32fFeature(targetConfig) || hasZve64xFeature(targetConfig) ||
         hasFeature(targetConfig, "+zve64f") ||
         hasFeature(targetConfig, "+zve64d");
}

bool hasAnySVEFeature(DictionaryAttr targetConfig) {
  return hasFeature(targetConfig, "+sve") ||
         hasFeature(targetConfig, "+sve2") || hasFeature(targetConfig, "+v9a");
}

bool hasSMEFeature(DictionaryAttr targetConfig) {
  return hasFeature(targetConfig, "+sme");
}

bool hasI8mmFeature(DictionaryAttr targetConfig) {
  return hasFeature(targetConfig, "+i8mm");
}

bool isLinalgGeneric2DTranspose(linalg::GenericOp genericOp) {
  // Check op has 2 dimensions.
  if (genericOp.getNumLoops() != 2)
    return false;

  // Check op has single input and output.
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
    return false;

  // Check all iterators are parallel.
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops())
    return false;

  // Check that the two indexing maps are a permutation of each other.
  SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
  bool isTranspose =
      (indexingMaps[0].isPermutation() && indexingMaps[1].isIdentity()) ||
      (indexingMaps[1].isPermutation() && indexingMaps[0].isIdentity());
  if (!isTranspose)
    return false;

  // Make sure the region only contains a yield op.
  Block &body = genericOp.getRegion().front();
  if (!llvm::hasSingleElement(body))
    return false;

  auto yieldOp = cast<linalg::YieldOp>(body.getTerminator());

  // The yield op should return the block argument corresponding to the input.
  auto yieldArg = dyn_cast<BlockArgument>(yieldOp.getValues()[0]);
  if (!yieldArg || yieldArg.getArgNumber() != 0 || yieldArg.getOwner() != &body)
    return false;

  return true;
}

bool mayHaveUndefinedBehaviorInMasking(Operation *op) {
  // Those operations will be lowered to division or related instructions,
  // and they might result in divide-by-zero.
  if (isa<mlir::arith::RemSIOp, mlir::arith::RemUIOp, mlir::arith::DivSIOp,
          mlir::arith::DivUIOp, mlir::arith::CeilDivSIOp,
          mlir::arith::CeilDivUIOp, mlir::arith::FloorDivSIOp,
          mlir::arith::DivFOp, mlir::arith::RemFOp>(op)) {
    return true;
  }
  return false;
}

int64_t getMaxVectorSizeForLargeVectorCheck(
    IREE::HAL::ExecutableTargetAttr targetAttr) {
  // Use 64 bits as target hardware vector size if the native_vector_size is not
  // present.
  int64_t maxVectorSizeInBytes = 8;
  if (targetAttr) {
    std::optional<int64_t> nativeVectorSizeAttr =
        getConfigNativeVectorSize(targetAttr.getConfiguration());
    if (nativeVectorSizeAttr) {
      maxVectorSizeInBytes = nativeVectorSizeAttr.value();
    }
  }
  maxVectorSizeInBytes *= clMaxAllowedNumberOfNativeVectors;
  return maxVectorSizeInBytes;
}

} // namespace mlir::iree_compiler
