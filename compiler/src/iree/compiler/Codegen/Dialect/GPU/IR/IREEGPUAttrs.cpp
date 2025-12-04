// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/DerivedConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPUTileSwizzleUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"
#include "iree/compiler/Utils/Indexing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.cpp.inc"

#define DEBUG_TYPE "iree-gpu-attrs"

namespace mlir::iree_compiler::IREE::GPU {

using ::mlir::iree_compiler::IREE::Codegen::TileSwizzle;

//===----------------------------------------------------------------------===//
// MMA intrinsics semantics: shapes, layouts, operand element types.
//===----------------------------------------------------------------------===//

static LogicalResult verifyMmaIndexingMaps(ArrayRef<AffineMap> maps) {
  return linalg::inferContractionDims(maps);
}

static int getBlockSize(MMAIntrinsic /*intrinsic*/) {
  // Not supporting any block size other than 1 at the moment.
  return 1;
}

static uint32_t getArchID(MMAIntrinsic intrinsic) {
  return static_cast<int>(intrinsic) & 0xFF00;
}

static bool is_AMD_MFMA(MMAIntrinsic intrinsic) {
  return getArchID(intrinsic) >= 0x1000 && getArchID(intrinsic) <= 0x17FF;
}

static bool is_AMD_WMMA(MMAIntrinsic intrinsic) {
  return getArchID(intrinsic) >= 0x1800 && getArchID(intrinsic) <= 0x1FFF;
}

static bool is_AMD(MMAIntrinsic intrinsic) {
  return is_AMD_MFMA(intrinsic) || is_AMD_WMMA(intrinsic);
}

int64_t getIntrinsicSubgroupSize(ScaledMMAIntrinsic intrinsic) {
  switch (intrinsic) {
  case ScaledMMAIntrinsic::MFMA_SCALE_F32_16x16x128_B32:
  case ScaledMMAIntrinsic::MFMA_SCALE_F32_32x32x64_B32:
    return 64;
  }
  assert(false &&
         "all cases should've been handled in ScaledMMA::getBlockSize()");
  return 0;
}

int64_t getIntrinsicSubgroupSize(MMAIntrinsic intrinsic) {
  // Not using Wave64 at all at the moment, so the only place where the
  // subgroup size is 64 is on CDNA* architectures.
  return is_AMD_MFMA(intrinsic) ? 64 : 32;
}

static std::tuple<Type, Type, Type> getABCElementTypes(MLIRContext *context,
                                                       MMAIntrinsic intrinsic) {
  Type f8E4M3FNUZ = Float8E4M3FNUZType::get(context);
  Type f8E5M2FNUZ = Float8E5M2FNUZType::get(context);
  Type f8E4M3FN = Float8E4M3FNType::get(context);
  Type f8E5M2 = Float8E5M2Type::get(context);
  Type f16 = Float16Type::get(context);
  Type bf16 = BFloat16Type::get(context);
  Type f32 = Float32Type::get(context);
  Type f64 = Float64Type::get(context);
  Type i8 = IntegerType::get(context, 8);
  Type i32 = IntegerType::get(context, 32);
  switch (intrinsic) {
  case MMAIntrinsic::MFMA_F64_16x16x4_F64:
    return {f64, f64, f64};
  case MMAIntrinsic::MFMA_F32_16x16x4_F32:
  case MMAIntrinsic::WMMA_F32_16x16x4_F32:
    return {f32, f32, f32};
  case MMAIntrinsic::MFMA_F32_16x16x16_F16:
  case MMAIntrinsic::MFMA_F32_32x32x8_F16:
  case MMAIntrinsic::MFMA_F32_16x16x32_F16:
  case MMAIntrinsic::MFMA_F32_32x32x16_F16:
  case MMAIntrinsic::WMMAR3_F32_16x16x16_F16:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F16:
  case MMAIntrinsic::NV_WMMA_F32_16x16x16_F16:
  case MMAIntrinsic::WMMA_F32_16x16x32_F16:
    return {f16, f16, f32};
  case MMAIntrinsic::WMMAR3_F16_16x16x16_F16:
  case MMAIntrinsic::WMMAR4_F16_16x16x16_F16:
  case MMAIntrinsic::WMMA_F16_16x16x32_F16:
  case MMAIntrinsic::NV_WMMA_F16_16x16x16_F16:
    return {f16, f16, f16};
  case MMAIntrinsic::MFMA_F32_16x16x8_BF16:
  case MMAIntrinsic::MFMA_F32_32x32x4_BF16:
  case MMAIntrinsic::MFMA_F32_16x16x16_BF16:
  case MMAIntrinsic::MFMA_F32_32x32x8_BF16:
  case MMAIntrinsic::MFMA_F32_16x16x32_BF16:
  case MMAIntrinsic::MFMA_F32_32x32x16_BF16:
  case MMAIntrinsic::WMMAR3_F32_16x16x16_BF16:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_BF16:
  case MMAIntrinsic::WMMA_F32_16x16x32_BF16:
    return {bf16, bf16, f32};
  case MMAIntrinsic::WMMAR3_BF16_16x16x16_BF16:
  case MMAIntrinsic::WMMAR4_BF16_16x16x16_BF16:
  case MMAIntrinsic::WMMA_BF16_16x16x32_BF16:
    return {bf16, bf16, bf16};
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FNUZ:
    return {f8E4M3FNUZ, f8E4M3FNUZ, f32};
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2FNUZ:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E5M2FNUZ:
    return {f8E5M2FNUZ, f8E5M2FNUZ, f32};
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ:
    return {f8E4M3FNUZ, f8E5M2FNUZ, f32};
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ:
    return {f8E5M2FNUZ, f8E4M3FNUZ, f32};
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E5M2:
  case MMAIntrinsic::MFMA_F32_16x16x128_F8E5M2:
  case MMAIntrinsic::MFMA_F32_32x32x64_F8E5M2:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F8E5M2:
  case MMAIntrinsic::WMMA_F32_16x16x64_F8E5M2:
  case MMAIntrinsic::WMMA_F32_16x16x128_F8E5M2:
    return {f8E5M2, f8E5M2, f32};
  case MMAIntrinsic::WMMA_F16_16x16x64_F8E5M2:
  case MMAIntrinsic::WMMA_F16_16x16x128_F8E5M2:
    return {f8E5M2, f8E5M2, f16};
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_16x16x128_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_32x32x64_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::WMMA_F32_16x16x64_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::WMMA_F32_16x16x128_F8E5M2_F8E4M3FN:
    return {f8E5M2, f8E4M3FN, f32};
  case MMAIntrinsic::WMMA_F16_16x16x64_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::WMMA_F16_16x16x128_F8E5M2_F8E4M3FN:
    return {f8E5M2, f8E4M3FN, f16};
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_16x16x128_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_32x32x64_F8E4M3FN:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F8E4M3FN:
  case MMAIntrinsic::WMMA_F32_16x16x64_F8E4M3FN:
  case MMAIntrinsic::WMMA_F32_16x16x128_F8E4M3FN:
    return {f8E4M3FN, f8E4M3FN, f32};
  case MMAIntrinsic::WMMA_F16_16x16x64_F8E4M3FN:
  case MMAIntrinsic::WMMA_F16_16x16x128_F8E4M3FN:
    return {f8E4M3FN, f8E4M3FN, f16};
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::MFMA_F32_16x16x128_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::MFMA_F32_32x32x64_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::WMMA_F32_16x16x64_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::WMMA_F32_16x16x128_F8E4M3FN_F8E5M2:
    return {f8E4M3FN, f8E5M2, f32};
  case MMAIntrinsic::WMMA_F16_16x16x64_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::WMMA_F16_16x16x128_F8E4M3FN_F8E5M2:
    return {f8E4M3FN, f8E5M2, f16};
  case MMAIntrinsic::MFMA_I32_16x16x16_I8:
  case MMAIntrinsic::MFMA_I32_32x32x8_I8:
  case MMAIntrinsic::MFMA_I32_16x16x32_I8:
  case MMAIntrinsic::MFMA_I32_32x32x16_I8:
  case MMAIntrinsic::MFMA_I32_16x16x64_I8:
  case MMAIntrinsic::MFMA_I32_32x32x32_I8:
  case MMAIntrinsic::WMMAR3_I32_16x16x16_I8:
  case MMAIntrinsic::WMMAR4_I32_16x16x16_I8:
  case MMAIntrinsic::WMMA_I32_16x16x64_I8:
    return {i8, i8, i32};
  }
  assert(false && "unexpected enum value");
  return {};
}

/// Returns the MNK shape for an intrinsic without an implemented concrete
/// layout.
static std::tuple<int64_t, int64_t, int64_t>
getUnsupportedMNKShape(MMAIntrinsic intrinsic) {
  switch (intrinsic) {
  case MMAIntrinsic::NV_WMMA_F32_16x16x16_F16:
  case MMAIntrinsic::NV_WMMA_F16_16x16x16_F16:
    return {16, 16, 16};
  default:
    assert(false && "unexpected enum value");
    return {};
  }
  return {};
}

MMASingleSubgroupLayout getSingleSubgroupLayout(MMAIntrinsic intrinsic,
                                                int operandIndex) {
  auto mfmaLhs16xK = [](int64_t k) -> MMASingleSubgroupLayout {
    assert(k % 4 == 0 && "doesn't support blocked MFMAs");
    return {/*outer=*/{1, 1}, /*thread=*/{16, 4}, /*tstrides=*/{1, 16},
            /*element=*/{1, k / 4}};
  };
  auto mfmaRhsKx16 = [](int64_t k) -> MMASingleSubgroupLayout {
    assert(k % 4 == 0 && "doesn't support blocked MFMAs");
    return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
            /*element=*/{k / 4, 1}};
  };

  auto mfmaLhs32xK = [](int64_t k) -> MMASingleSubgroupLayout {
    assert(k % 2 == 0 && "doesn't support blocked MFMAs");
    return {/*outer=*/{1, 1}, /*thread=*/{32, 2}, /*tstrides=*/{1, 32},
            /*element=*/{1, k / 2}};
  };
  auto mfmaRhsKx32 = [](int64_t k) -> MMASingleSubgroupLayout {
    assert(k % 2 == 0 && "doesn't support blocked MFMAs");
    return {/*outer=*/{1, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
            /*element=*/{k / 2, 1}};
  };

  const MMASingleSubgroupLayout mfmaAcc16x16 = {
      /*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
      /*element=*/{4, 1}};
  const MMASingleSubgroupLayout mfmaAcc32x32 = {
      /*outer=*/{4, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
      /*element=*/{4, 1}};

  // Note: For gfx12, we specify here that, for example with K=16, lane 0 takes
  // A[0, 0..7] and that lane 16 takes A[0, 8..15]. The hardware will internally
  // bounce between the low halves and high halves of lanes every two registers
  // - that is, the value used for A[0, 4] comes out of lane 16's first register
  // in an F16 or BF16 computation. This is noted here in case someone starts
  // chasing some unusual rounding failure or is confused by why the tiling in
  // the manual doesn't *technically* match the below.
  auto gfx12Wmma16xK = [](int64_t k) -> MMASingleSubgroupLayout {
    return {/*outer=*/{1, 1}, /*thread=*/{16, 2}, /*tstrides=*/{1, 16},
            /*element=*/{1, k / 2}};
  };
  auto gfx12WmmaKx16 = [](int64_t k) -> MMASingleSubgroupLayout {
    return {/*outer=*/{1, 1}, /*thread=*/{2, 16}, /*tstrides=*/{16, 1},
            /*element=*/{k / 2, 1}};
  };
  const MMASingleSubgroupLayout gfx12WmmaAcc16x16 = gfx12WmmaKx16(16);

  switch (intrinsic) {
  case MMAIntrinsic::MFMA_F32_16x16x4_F32:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return mfmaLhs16xK(4);
    case kMMAOperandRhs:
      return mfmaRhsKx16(4);
    case kMMAOperandAcc:
      return mfmaAcc16x16;
    }
  // Note: the returned layout for f64 differs than for other MFMAs.
  case MMAIntrinsic::MFMA_F64_16x16x4_F64:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return mfmaLhs16xK(4);
    case kMMAOperandRhs:
      return mfmaRhsKx16(4);
    case kMMAOperandAcc:
      return {/*outer=*/{4, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{1, 1}};
    }
  case MMAIntrinsic::MFMA_F32_16x16x8_BF16: {
    switch (operandIndex) {
    case kMMAOperandLhs:
      return mfmaLhs16xK(8);
    case kMMAOperandRhs:
      return mfmaRhsKx16(8);
    case kMMAOperandAcc:
      return mfmaAcc16x16;
    }
  }
  case MMAIntrinsic::MFMA_F32_32x32x4_BF16:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return mfmaLhs32xK(4);
    case kMMAOperandRhs:
      return mfmaRhsKx32(4);
    case kMMAOperandAcc:
      return mfmaAcc32x32;
    }
  case MMAIntrinsic::MFMA_I32_16x16x16_I8:
  case MMAIntrinsic::MFMA_F32_16x16x16_F16:
  case MMAIntrinsic::MFMA_F32_16x16x16_BF16:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return mfmaLhs16xK(16);
    case kMMAOperandRhs:
      return mfmaRhsKx16(16);
    case kMMAOperandAcc:
      return mfmaAcc16x16;
    }
  case MMAIntrinsic::MFMA_I32_32x32x8_I8:
  case MMAIntrinsic::MFMA_F32_32x32x8_F16:
  case MMAIntrinsic::MFMA_F32_32x32x8_BF16:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return mfmaLhs32xK(8);
    case kMMAOperandRhs:
      return mfmaRhsKx32(8);
    case kMMAOperandAcc:
      return mfmaAcc32x32;
    }
  case MMAIntrinsic::MFMA_F32_16x16x32_F16:
  case MMAIntrinsic::MFMA_F32_16x16x32_BF16:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2FNUZ:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::MFMA_I32_16x16x32_I8:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return mfmaLhs16xK(32);
    case kMMAOperandRhs:
      return mfmaRhsKx16(32);
    case kMMAOperandAcc:
      return mfmaAcc16x16;
    }
  case MMAIntrinsic::MFMA_F32_32x32x16_F16:
  case MMAIntrinsic::MFMA_F32_32x32x16_BF16:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E5M2FNUZ:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E5M2:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::MFMA_I32_32x32x16_I8:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return mfmaLhs32xK(16);
    case kMMAOperandRhs:
      return mfmaRhsKx32(16);
    case kMMAOperandAcc:
      return mfmaAcc32x32;
    }
  case MMAIntrinsic::MFMA_I32_16x16x64_I8:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return mfmaLhs16xK(64);
    case kMMAOperandRhs:
      return mfmaRhsKx16(64);
    case kMMAOperandAcc:
      return mfmaAcc16x16;
    }
  case MMAIntrinsic::MFMA_I32_32x32x32_I8:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return mfmaLhs32xK(32);
    case kMMAOperandRhs:
      return mfmaRhsKx32(32);
    case kMMAOperandAcc:
      return mfmaAcc32x32;
    }
  case MMAIntrinsic::MFMA_F32_16x16x128_F8E5M2:
  case MMAIntrinsic::MFMA_F32_16x16x128_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_16x16x128_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_16x16x128_F8E4M3FN_F8E5M2:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return mfmaLhs16xK(128);
    case kMMAOperandRhs:
      return mfmaRhsKx16(128);
    case kMMAOperandAcc:
      return mfmaAcc16x16;
    }
  case MMAIntrinsic::MFMA_F32_32x32x64_F8E5M2:
  case MMAIntrinsic::MFMA_F32_32x32x64_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_32x32x64_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_32x32x64_F8E4M3FN_F8E5M2:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return mfmaLhs32xK(64);
    case kMMAOperandRhs:
      return mfmaRhsKx32(64);
    case kMMAOperandAcc:
      return mfmaAcc32x32;
    }

  case MMAIntrinsic::WMMAR3_F32_16x16x16_F16:
  case MMAIntrinsic::WMMAR3_F32_16x16x16_BF16:
  case MMAIntrinsic::WMMAR3_I32_16x16x16_I8:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return {/*outer=*/{1, 1}, /*thread=*/{16, 1}, /*strides=*/{1, 0},
              /*element=*/{1, 16}};
    case kMMAOperandRhs:
      return {/*outer=*/{1, 1}, /*thread=*/{1, 16}, /*tstrides=*/{0, 1},
              /*element=*/{16, 1}};
    case kMMAOperandAcc:
      return {/*outer=*/{8, 1}, /*thread=*/{2, 16}, /*tstrides=*/{16, 1},
              /*element=*/{1, 1}};
    }
  case MMAIntrinsic::WMMAR3_F16_16x16x16_F16:
  case MMAIntrinsic::WMMAR3_BF16_16x16x16_BF16:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return {/*outer=*/{1, 1}, /*thread=*/{16, 1}, /*strides=*/{1, 0},
              /*element=*/{1, 16}};
    case kMMAOperandRhs:
      return {/*outer=*/{1, 1}, /*thread=*/{1, 16}, /*tstrides=*/{0, 1},
              /*element=*/{16, 1}};
    case kMMAOperandAcc:
      return {/*outer=*/{16, 1}, /*thread=*/{1, 16}, /*tstrides=*/{0, 1},
              /*element=*/{1, 1}};
    }
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F16:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_BF16:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F8E5M2:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F8E4M3FN:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::WMMAR4_I32_16x16x16_I8:
  case MMAIntrinsic::WMMAR4_F16_16x16x16_F16:
  case MMAIntrinsic::WMMAR4_BF16_16x16x16_BF16:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return gfx12Wmma16xK(16);
    case kMMAOperandRhs:
      return gfx12WmmaKx16(16);
    case kMMAOperandAcc:
      return gfx12WmmaAcc16x16;
    }
  case MMAIntrinsic::WMMA_F32_16x16x4_F32:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return gfx12Wmma16xK(4);
    case kMMAOperandRhs:
      return gfx12WmmaKx16(4);
    case kMMAOperandAcc:
      return gfx12WmmaAcc16x16;
    }
  case MMAIntrinsic::WMMA_F32_16x16x32_F16:
  case MMAIntrinsic::WMMA_F32_16x16x32_BF16:
  case MMAIntrinsic::WMMA_F16_16x16x32_F16:
  case MMAIntrinsic::WMMA_BF16_16x16x32_BF16:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return gfx12Wmma16xK(32);
    case kMMAOperandRhs:
      return gfx12WmmaKx16(32);
    case kMMAOperandAcc:
      return gfx12WmmaAcc16x16;
    }
  case MMAIntrinsic::WMMA_F32_16x16x64_F8E4M3FN:
  case MMAIntrinsic::WMMA_F32_16x16x64_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::WMMA_F32_16x16x64_F8E5M2:
  case MMAIntrinsic::WMMA_F32_16x16x64_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::WMMA_F16_16x16x64_F8E4M3FN:
  case MMAIntrinsic::WMMA_F16_16x16x64_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::WMMA_F16_16x16x64_F8E5M2:
  case MMAIntrinsic::WMMA_F16_16x16x64_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::WMMA_I32_16x16x64_I8:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return gfx12Wmma16xK(64);
    case kMMAOperandRhs:
      return gfx12WmmaKx16(64);
    case kMMAOperandAcc:
      return gfx12WmmaAcc16x16;
    }
  case MMAIntrinsic::WMMA_F32_16x16x128_F8E5M2:
  case MMAIntrinsic::WMMA_F32_16x16x128_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::WMMA_F32_16x16x128_F8E4M3FN:
  case MMAIntrinsic::WMMA_F32_16x16x128_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::WMMA_F16_16x16x128_F8E5M2:
  case MMAIntrinsic::WMMA_F16_16x16x128_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::WMMA_F16_16x16x128_F8E4M3FN:
  case MMAIntrinsic::WMMA_F16_16x16x128_F8E4M3FN_F8E5M2:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return gfx12Wmma16xK(128);
    case kMMAOperandRhs:
      return gfx12WmmaKx16(128);
    case kMMAOperandAcc:
      return gfx12WmmaAcc16x16;
    }
  case MMAIntrinsic::NV_WMMA_F32_16x16x16_F16:
  case MMAIntrinsic::NV_WMMA_F16_16x16x16_F16:
    return {};
  }
  assert(false && "unexpected enum value");
  return {};
}

MMASingleSubgroupLayout getSingleSubgroupLayout(MMAIntrinsic intrinsic,
                                                int operandIndex,
                                                bool colMajor) {
  MMASingleSubgroupLayout baseLayout =
      getSingleSubgroupLayout(intrinsic, operandIndex);
  assert(baseLayout.element.size() == 2 && "expected 2d layout");
  if (colMajor) {
    std::swap(baseLayout.element[0], baseLayout.element[1]);
    std::swap(baseLayout.thread[0], baseLayout.thread[1]);
    std::swap(baseLayout.outer[0], baseLayout.outer[1]);
    std::swap(baseLayout.tstrides[0], baseLayout.tstrides[1]);
  }
  return baseLayout;
}

MMASingleSubgroupLayout
getSingleSubgroupLayout(VirtualMMAIntrinsic virtualIntrinsic, int operandIndex,
                        bool colMajor) {
  MMASingleSubgroupLayout baseLayout =
      getSingleSubgroupLayout(virtualIntrinsic, operandIndex);
  assert(baseLayout.element.size() == 2 && "expected 2d layout");
  if (colMajor) {
    std::swap(baseLayout.element[0], baseLayout.element[1]);
    std::swap(baseLayout.thread[0], baseLayout.thread[1]);
    std::swap(baseLayout.outer[0], baseLayout.outer[1]);
    std::swap(baseLayout.tstrides[0], baseLayout.tstrides[1]);
  }
  return baseLayout;
}

// Struct describing the shape of a MMA operation, but not the detailed layout.
struct OpaqueMmaLayout {
  int64_t mSize = 0;
  int64_t nSize = 0;
  int64_t kSize = 0;
  Type aType;
  Type bType;
  Type cType;
};

static std::tuple<int64_t, int64_t, int64_t>
getMNKShapeFromIntrinsic(MMAIntrinsic intrinsic) {
  if (is_AMD(intrinsic)) {
    auto lhs = getSingleSubgroupLayout(intrinsic, kMMAOperandLhs);
    auto rhs = getSingleSubgroupLayout(intrinsic, kMMAOperandRhs);
    return {lhs.outer[0] * lhs.thread[0] * lhs.element[0],
            rhs.outer[1] * rhs.thread[1] * rhs.element[1],
            lhs.outer[1] * lhs.thread[1] * lhs.element[1]};
  }
  return getUnsupportedMNKShape(intrinsic);
}

int64_t getMSize(MMAIntrinsic intrinsic) {
  return std::get<0>(getMNKShapeFromIntrinsic(intrinsic));
}
int64_t getNSize(MMAIntrinsic intrinsic) {
  return std::get<1>(getMNKShapeFromIntrinsic(intrinsic));
}
int64_t getKSize(MMAIntrinsic intrinsic) {
  return std::get<2>(getMNKShapeFromIntrinsic(intrinsic));
}

static OpaqueMmaLayout getOpaqueMMALayout(MLIRContext *context,
                                          MMAIntrinsic intrinsic) {
  OpaqueMmaLayout o;
  std::tie(o.aType, o.bType, o.cType) = getABCElementTypes(context, intrinsic);
  std::tie(o.mSize, o.nSize, o.kSize) = getMNKShapeFromIntrinsic(intrinsic);
  return o;
}

MMASingleSubgroupLayout
getSingleSubgroupLayout(IREE::Codegen::InnerTileDescAttrInterface mmaKind,
                        int operandIndex) {
  if (auto mmaAttr = dyn_cast<MMAAttr>(mmaKind)) {
    // |colMajor| indicates that the accumulator layout should be returned
    // column major.
    return IREE::GPU::getSingleSubgroupLayout(
        mmaAttr.getIntrinsic(), operandIndex,
        operandIndex == kMMAOperandAcc && mmaAttr.getColMajor());
  }
  if (auto vmmaAttr = dyn_cast<VirtualMMAAttr>(mmaKind)) {
    return IREE::GPU::getSingleSubgroupLayout(
        vmmaAttr.getIntrinsic(), operandIndex,
        operandIndex == kMMAOperandAcc && vmmaAttr.getColMajor());
  }
  assert(false && "unhandled MMA Interface type.");
  return {};
}

void IREE::GPU::InnerTiledSemanticsAttr::getTileTypes(
    IREE::Codegen::InnerTileDescAttrInterface kind,
    llvm::SmallVectorImpl<mlir::VectorType> &result) const {
  if (getDistributed()) {
    kind.getDistributedTileTypes(result);
  } else {
    kind.getUndistributedTileTypes(result);
  }
}

//===----------------------------------------------------------------------===//
// MMA Attributes
//===----------------------------------------------------------------------===//

MMAAttr MMAAttr::get(MLIRContext *context, MMAIntrinsic type) {
  return Base::get(context, type, /*colMajor=*/false);
}

int64_t MMAAttr::getExpectedNumInputs() const { return 2; }

int64_t MMAAttr::getExpectedNumOutputs() const { return 1; }

LogicalResult MMAAttr::verifyIndexingMaps(ArrayRef<AffineMap> maps) const {
  return verifyMmaIndexingMaps(maps);
}

void MMAAttr::getUndistributedTileTypes(
    SmallVectorImpl<VectorType> &result) const {
  MLIRContext *ctx = getContext();
  OpaqueMmaLayout o = getOpaqueMMALayout(ctx, getIntrinsic());
  result.assign({VectorType::get({o.mSize, o.kSize}, o.aType),
                 VectorType::get({o.kSize, o.nSize}, o.bType),
                 VectorType::get({o.mSize, o.nSize}, o.cType)});
}

template <typename MMAIntrinsicType>
static VectorType getThreadVectorType(MLIRContext *context,
                                      MMAIntrinsicType intrinsic,
                                      int operandIndex) {
  auto o = getOpaqueMMALayout(context, intrinsic);
  auto s = getSingleSubgroupLayout(intrinsic, operandIndex);
  Type elemType = isIntrinsicLhs<MMAIntrinsicType>(operandIndex)   ? o.aType
                  : isIntrinsicRhs<MMAIntrinsicType>(operandIndex) ? o.bType
                                                                   : o.cType;
  return VectorType::get(
      {s.element[0] * s.element[1] * s.outer[0] * s.outer[1]}, elemType);
}

void MMAAttr::getDistributedTileTypes(
    SmallVectorImpl<VectorType> &result) const {
  MLIRContext *context = getContext();
  MMAIntrinsic intrinsic = getIntrinsic();
  result.assign({getThreadVectorType(context, intrinsic, kMMAOperandLhs),
                 getThreadVectorType(context, intrinsic, kMMAOperandRhs),
                 getThreadVectorType(context, intrinsic, kMMAOperandAcc)});
}

std::optional<SmallVector<int64_t, 2>>
MMAAttr::getUndistributedTileDimExpansion(int64_t operandIndex,
                                          int64_t dim) const {
  assert(operandIndex <= 2 && "invalid operand index");
  assert(dim < 2 && "pre-expansion inner tiles all have two elements");
  MMASingleSubgroupLayout layout =
      getSingleSubgroupLayout(*this, static_cast<int>(operandIndex));
  if (layout.outer[dim] > 1) {
    return SmallVector<int64_t, 2>{layout.outer[dim],
                                   layout.element[dim] * layout.thread[dim]};
  }
  return std::nullopt;
}

int64_t MMAAttr::getBlockSize() const {
  return IREE::GPU::getBlockSize(getIntrinsic());
}

int64_t MMAAttr::getSubgroupSize() const {
  return getIntrinsicSubgroupSize(getIntrinsic());
}

Attribute MMAAttr::getDistributionMappingKind() const {
  // Explicit distribution currently unsupported for NV intrinsics.
  MMAIntrinsic intrinsic = getIntrinsic();
  if (intrinsic == MMAIntrinsic::NV_WMMA_F16_16x16x16_F16 ||
      intrinsic == MMAIntrinsic::NV_WMMA_F32_16x16x16_F16) {
    return Attribute();
  }
  return IREE::GPU::LaneIdAttr::get(getContext(), 0);
}

OpFoldResult MMAAttr::getDistributionWorkerCount(OpBuilder &, Location,
                                                 Operation *) const {
  if (!getDistributionMappingKind())
    return OpFoldResult();
  return getAsIndexOpFoldResult(getContext(), getSubgroupSize());
}

// Get virtual intrinsics that is composed/based on queried op.
SmallVector<VirtualMMAIntrinsic> MMAAttr::getVirtualIntrinsics() const {
  switch (getIntrinsic()) {
  case MMAIntrinsic::MFMA_F32_16x16x16_F16:
    return {VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F16};
  case MMAIntrinsic::MFMA_F32_32x32x8_F16:
    return {VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F16};
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ:
    return {VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ};
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FNUZ:
    return {VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F8E4M3FNUZ};
  default:
    return {};
  }
}

static Value createMmaOp(OpBuilder &builder, Location loc,
                         MMAIntrinsic intrinsic, Type resultType, Value lhs,
                         Value rhs, Value acc, bool colMajor = false) {
  auto getVecOrSingleElem = [&](Value vec) -> Value {
    bool one = cast<VectorType>(vec.getType()).getNumElements() == 1;
    return one ? vector::ExtractOp::create(builder, loc, vec, 0) : vec;
  };
  auto layout = getOpaqueMMALayout(builder.getContext(), intrinsic);
  if (is_AMD_MFMA(intrinsic)) {
    // MFMA intrinsics want single-element operands of element type, not vector.
    lhs = getVecOrSingleElem(lhs);
    rhs = getVecOrSingleElem(rhs);

    // Because the thread layout of the lhs and rhs are transpositions of one
    // another for all MFMA variants, to produce a column major result we can
    // simply swap the operands to the MFMA.
    if (colMajor) {
      std::swap(lhs, rhs);
    }
    return amdgpu::MFMAOp::create(builder, loc, resultType, layout.mSize,
                                  layout.nSize, layout.kSize,
                                  getBlockSize(intrinsic), lhs, rhs, acc)
        .getResult();
  }
  if (is_AMD_WMMA(intrinsic)) {
    return amdgpu::WMMAOp::create(builder, loc, resultType, layout.mSize,
                                  layout.nSize, layout.kSize, lhs, rhs, acc)
        .getResult();
  }
  return {};
}

// Generates amdgpu.mfma/wmma operation on the given inputs for this attribute
// type.
LogicalResult
MMAAttr::buildUnderlyingOperations(OpBuilder &builder, Location loc,
                                   ValueRange inputs, ValueRange outputs,
                                   SmallVectorImpl<Value> &results) const {
  if (inputs.size() != 2) {
    return failure();
  }
  if (outputs.size() != 1) {
    return failure();
  }
  SmallVector<VectorType> threadTypes;
  getDistributedTileTypes(threadTypes);
  if (!llvm::equal(threadTypes,
                   llvm::concat<Type>(inputs.getTypes(), outputs.getTypes()))) {
    return failure();
  }

  if (Value value =
          createMmaOp(builder, loc, getIntrinsic(), outputs[0].getType(),
                      inputs[0], inputs[1], outputs[0], getColMajor())) {
    results.push_back(value);
    return success();
  }
  return failure();
}

static LogicalResult populateCanonicalOffsetsSizesAndStrides(
    OpBuilder &builder, Location loc, Value laneId,
    ArrayRef<int64_t> permutation, MMASingleSubgroupLayout subgroupLayout,
    SmallVectorImpl<OpFoldResult> &canonicalOffsets,
    SmallVectorImpl<OpFoldResult> &canonicalSizes,
    SmallVectorImpl<OpFoldResult> &canonicalStrides) {
  SmallVector<int64_t> rankReducedShape;

  for (auto [outer, thread, element] :
       llvm::zip_equal(subgroupLayout.outer, subgroupLayout.thread,
                       subgroupLayout.element)) {
    if (outer != 1) {
      rankReducedShape.push_back(outer);
    }
    rankReducedShape.push_back(thread * element);
  }

  if (permutation.size() != rankReducedShape.size()) {
    return failure();
  }

  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  Value cZero = arith::ConstantIndexOp::create(builder, loc, 0);
  canonicalStrides.append(rankReducedShape.size(), one);

  SmallVector<Value> vtids;
  SmallVector<int64_t> vtidBasis;
  SmallVector<size_t> dimToVtid;
  if (failed(basisFromSizesStrides(subgroupLayout.thread,
                                   subgroupLayout.tstrides, vtidBasis,
                                   dimToVtid))) {
    return failure();
  }
  auto splitLaneId = affine::AffineDelinearizeIndexOp::create(
      builder, loc, laneId, vtidBasis, /*hasOuterBound=*/false);

  // Each thread grabs `element` contiguous data, so the vtid needs to be
  // multiplied by `element` to get the next bunch of data.
  // vtid: virtual thread id
  // tid: lane id
  // vtid = ((tid floordiv stride_i) mod size_i) * element_i.
  //
  // Instead of computing those maps, we use one big `delinearize` expression
  // in order to prevent unwanted "simplifications" on affine maps that
  // worsen the generated code quality.
  for (auto [splitResultIdx, element] :
       llvm::zip_equal(dimToVtid, subgroupLayout.element)) {
    Value vtid = splitLaneId.getResult(splitResultIdx);
    int64_t vtidLen = vtidBasis[splitResultIdx - 1];
    if (element != 1) {
      vtid = affine::AffineLinearizeIndexOp::create(
          builder, loc, ValueRange{vtid, cZero},
          ArrayRef<int64_t>{vtidLen, element},
          /*disjoint=*/true);
    }
    vtids.push_back(vtid);
  }

  int64_t idx = 0;
  for (auto [element, outer] :
       llvm::zip_equal(subgroupLayout.element, subgroupLayout.outer)) {
    if (outer != 1) {
      canonicalSizes.push_back(builder.getIndexAttr(outer));
      canonicalOffsets.push_back(zero);
    }
    canonicalSizes.push_back(builder.getIndexAttr(element));
    canonicalOffsets.push_back(vtids[idx++]);
  }
  canonicalOffsets.assign(applyPermutation(canonicalOffsets, permutation));
  canonicalSizes.assign(applyPermutation(canonicalSizes, permutation));
  return success();
}

LogicalResult MMAAttr::populateOperandOffsetsSizesStrides(
    OpBuilder &builder, Location loc, uint32_t operandIndex, Value laneId,
    ArrayRef<int64_t> permutation, SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &sizes,
    SmallVectorImpl<OpFoldResult> &strides) const {
  assert(operandIndex <= 2 && "Must index valid MMA operand");
  MMASingleSubgroupLayout subgroupLayout =
      getSingleSubgroupLayout(getIntrinsic(), operandIndex,
                              operandIndex == kMMAOperandAcc && getColMajor());
  SmallVector<OpFoldResult> canonicalOffsets;
  SmallVector<OpFoldResult> canonicalSizes;
  if (failed(populateCanonicalOffsetsSizesAndStrides(
          builder, loc, laneId, permutation, subgroupLayout, canonicalOffsets,
          canonicalSizes, strides))) {
    return failure();
  }
  offsets.append(canonicalOffsets);
  sizes.append(canonicalSizes);

  return success();
}

//===----------------------------------------------------------------------===//
// DataTiledMMA Attributes
//===----------------------------------------------------------------------===//

int64_t DataTiledMMAAttr::getExpectedNumInputs() const { return 2; }

int64_t DataTiledMMAAttr::getExpectedNumOutputs() const { return 1; }

LogicalResult
DataTiledMMAAttr::verifyIndexingMaps(ArrayRef<AffineMap> maps) const {
  return verifyMmaIndexingMaps(maps);
}

int64_t DataTiledMMAAttr::getSubgroupSize() const {
  return getIntrinsicSubgroupSize(getIntrinsic());
}

int64_t DataTiledMMAAttr::getFlatWorkgroupSize() const {
  return getSubgroupSize() * getSubgroupsM() * getSubgroupsN() *
         getSubgroupsK();
}

/// Increment the mutable vector `indices` to traverse the index space below
/// `sizes`, with the last dimension moving fastest, or returns false if that
/// index space was exhausted.
static bool incrementIndices(MutableArrayRef<int64_t> indices,
                             ArrayRef<int64_t> sizes) {
  for (int i = indices.size() - 1; i >= 0; --i) {
    if (++indices[i] == sizes[i]) {
      indices[i] = 0;
    } else {
      return true; // Found an index that we could increment without wrapping.
    }
  }
  return false; // All indices wrapped around.
}

/// Flattens the input vector `value` to 1-D if the rank is greater than 1. Note
/// that it returns the value directly if it is a 0-D vector.
static Value flattenVector(OpBuilder &builder, Location loc, Value value) {
  Type type = value.getType();
  VectorType vectorType = dyn_cast<VectorType>(type);
  assert(vectorType);
  if (vectorType.getRank() <= 1) {
    return value;
  }
  auto flatVectorType = VectorType::get({vectorType.getNumElements()},
                                        vectorType.getElementType());
  return vector::ShapeCastOp::create(builder, loc, flatVectorType, value);
}

/// Returns intrinsic-level slices tiling the input multi-MMA-level tile
/// `value`.
static SmallVector<Value>
distributeMmaFragmentToIntrinsics(OpBuilder &builder, Location loc, Value value,
                                  const TileSwizzle &swizzle) {
  auto internalShape = sliceSwizzledShape(swizzle, [](TileSwizzle::Dim dim) {
    return dim.kind == TileSwizzle::Dim::Kind::Internal;
  });
  auto crossIntrinsicShape =
      sliceSwizzledShape(swizzle, [](TileSwizzle::Dim dim) {
        return dim.kind == TileSwizzle::Dim::Kind::CrossIntrinsic;
      });
  LDBG() << "crossIntrinsicShape: " << llvm::interleaved(crossIntrinsicShape);
  int rank = internalShape.size();
  SmallVector<int64_t> indices(rank, 0);
  SmallVector<int64_t> strides(rank, 1);
  SmallVector<Value> distributedValues;
  do {
    Value extract = vector::ExtractStridedSliceOp::create(
        builder, loc, value, indices, internalShape, strides);
    distributedValues.push_back(flattenVector(builder, loc, extract));
  } while (incrementIndices(indices, crossIntrinsicShape));
  return distributedValues;
}

LogicalResult DataTiledMMAAttr::buildUnderlyingOperations(
    OpBuilder &builder, Location loc, ValueRange inputs, ValueRange outputs,
    SmallVectorImpl<Value> &results) const {
  // Validation. Similar to MMAAttr::buildMmaOperation.
  if (inputs.size() != 2) {
    return failure();
  }
  if (outputs.size() != 1) {
    return failure();
  }
  SmallVector<VectorType> regTypes;
  getDistributedTileTypes(regTypes);
  if (!llvm::equal(regTypes,
                   llvm::concat<Type>(inputs.getTypes(), outputs.getTypes()))) {
    return failure();
  }

  // Prepare Lhs/Rhs/Acc operand slices to feed the intrinsic.
  TileSwizzle lhsSwizzle = getSwizzle(*this, kMMAOperandLhs);
  LDBG() << "DataTiledMMAAttr::buildMmaOperation";
  LDBG() << "    lhsSwizzle: " << lhsSwizzle;
  SmallVector<Value> intrinsicsLhs =
      distributeMmaFragmentToIntrinsics(builder, loc, inputs[0], lhsSwizzle);

  TileSwizzle rhsSwizzle = getSwizzle(*this, kMMAOperandRhs);
  LDBG() << "DataTiledMMAAttr::buildMmaOperation";
  LDBG() << "    rhsSwizzle: " << rhsSwizzle;
  SmallVector<Value> intrinsicsRhs =
      distributeMmaFragmentToIntrinsics(builder, loc, inputs[1], rhsSwizzle);

  TileSwizzle accSwizzle = getSwizzle(*this, kMMAOperandAcc);
  LDBG() << "DataTiledMMAAttr::buildMmaOperation";
  LDBG() << "    accSwizzle: " << accSwizzle;

  SmallVector<Value> intrinsicsAcc =
      distributeMmaFragmentToIntrinsics(builder, loc, outputs[0], accSwizzle);

  MMAIntrinsic intrinsic = getIntrinsic();
  VectorType intrinCType =
      getThreadVectorType(builder.getContext(), intrinsic, kMMAOperandAcc);

  // Loop over the 3 unroll_{m,n,k} dimensions to create the intrinsics.
  for (int mu = 0; mu < getIntrinsicsM(); ++mu) {
    for (int nu = 0; nu < getIntrinsicsN(); ++nu) {
      for (int ku = 0; ku < getIntrinsicsK(); ++ku) {
        Value lhs = intrinsicsLhs[mu * getIntrinsicsK() + ku];
        Value rhs = intrinsicsRhs[nu * getIntrinsicsK() + ku];
        Value &acc = intrinsicsAcc[mu * getIntrinsicsN() + nu];
        acc = createMmaOp(builder, loc, intrinsic, intrinCType, lhs, rhs, acc);
      }
    }
  }

  // Insert the results into the destination accumulator.
  SmallVector<int64_t> accCrossIntrinsicShape =
      sliceSwizzledShape(accSwizzle, [](TileSwizzle::Dim dim) {
        return dim.kind == TileSwizzle::Dim::Kind::CrossIntrinsic;
      });
  SmallVector<int64_t> accInternalShape =
      sliceSwizzledShape(accSwizzle, [](TileSwizzle::Dim dim) {
        return dim.kind == TileSwizzle::Dim::Kind::Internal;
      });

  LDBG() << "accCrossIntrinsicShape: "
         << llvm::interleaved(accCrossIntrinsicShape);
  LDBG() << "accInternalShape: " << llvm::interleaved(accInternalShape);
  int dstRank = accCrossIntrinsicShape.size();
  SmallVector<int64_t> strides(dstRank, 1);
  SmallVector<int64_t> indices(dstRank, 0);
  Value acc = outputs[0];
  for (Value intrAcc : intrinsicsAcc) {
    auto expandedAcc = vector::ShapeCastOp::create(
        builder, loc,
        VectorType::get(
            accInternalShape,
            cast<VectorType>(outputs[0].getType()).getElementType()),
        intrAcc);
    acc = vector::InsertStridedSliceOp::create(builder, loc, expandedAcc, acc,
                                               indices, strides);
    incrementIndices(indices, accCrossIntrinsicShape);
  }
  results.push_back(acc);
  return success();
}

TileSwizzle DataTiledMMAAttr::getTileSwizzle(unsigned operandIndex) const {
  return getSwizzle(*this, operandIndex);
}

IREE::Codegen::TileMxNxKxKb DataTiledMMAAttr::getTileMNKKb() const {
  IREE::Codegen::TileMxNxKxKb innerTile;
  std::tie(innerTile.M, innerTile.N, innerTile.K) =
      getMNKShapeFromIntrinsic(getIntrinsic());
  innerTile.M *= getIntrinsicsM() * getSubgroupsM();
  innerTile.N *= getIntrinsicsN() * getSubgroupsN();
  innerTile.K *= getIntrinsicsK() * getSubgroupsK();
  return innerTile;
}

void DataTiledMMAAttr::getElementTypes(SmallVectorImpl<Type> &result) const {
  auto [a, b, c] = IREE::GPU::getABCElementTypes(getContext(), getIntrinsic());
  result.assign({a, b, c});
}

//===----------------------------------------------------------------------===//
// VirtualMMA Attributes
//===----------------------------------------------------------------------===//

VirtualMMAAttr VirtualMMAAttr::get(MLIRContext *context,
                                   VirtualMMAIntrinsic type) {
  return Base::get(context, type, /*colMajor=*/false);
}

static std::tuple<int64_t, int64_t, int64_t>
getMNKShape(VirtualMMAIntrinsic type) {
  // V(Virtual)MFMA instructions which have 2 mfma instructions interleaved
  // along the k dimension.
  switch (type) {
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ:
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F16:
    return {16, 16, 32};
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F8E4M3FNUZ:
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F16:
    return {32, 32, 16};
  }
  assert(false && "unhandled virtual mma layout type.");
  return {};
}

static std::tuple<Type, Type, Type>
getABCElementTypes(MLIRContext *context, VirtualMMAIntrinsic type) {
  Type f8E4M3FNUZ = Float8E4M3FNUZType::get(context);
  Type f16 = Float16Type::get(context);
  Type f32 = Float32Type::get(context);

  switch (type) {
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ:
    return {f8E4M3FNUZ, f8E4M3FNUZ, f32};
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F8E4M3FNUZ:
    return {f8E4M3FNUZ, f8E4M3FNUZ, f32};
  // V(Virtual)MFMA instructions which have 2 mfma instructions interleaved
  // along the k dimension.
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F16:
    return {f16, f16, f32};
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F16:
    return {f16, f16, f32};
  }
  assert(false && "unhandled virtual mma layout type.");
  return {};
}

static OpaqueMmaLayout getOpaqueMMALayout(MLIRContext *context,
                                          VirtualMMAIntrinsic intrinsic) {
  OpaqueMmaLayout o;
  std::tie(o.aType, o.bType, o.cType) = getABCElementTypes(context, intrinsic);
  auto lhs = getSingleSubgroupLayout(intrinsic, kMMAOperandLhs);
  auto rhs = getSingleSubgroupLayout(intrinsic, kMMAOperandRhs);
  o.mSize = lhs.outer[0] * lhs.thread[0] * lhs.element[0];
  o.kSize = lhs.outer[1] * lhs.thread[1] * lhs.element[1];
  o.nSize = rhs.outer[1] * rhs.thread[1] * rhs.element[1];
  return o;
}

int64_t VirtualMMAAttr::getExpectedNumInputs() const { return 2; }

int64_t VirtualMMAAttr::getExpectedNumOutputs() const { return 1; }

LogicalResult
VirtualMMAAttr::verifyIndexingMaps(ArrayRef<AffineMap> maps) const {
  return verifyMmaIndexingMaps(maps);
}

void VirtualMMAAttr::getUndistributedTileTypes(
    SmallVectorImpl<VectorType> &result) const {
  MLIRContext *ctx = getContext();
  OpaqueMmaLayout o = getOpaqueMMALayout(ctx, getIntrinsic());
  result.assign({VectorType::get({o.mSize, o.kSize}, o.aType),
                 VectorType::get({o.kSize, o.nSize}, o.bType),
                 VectorType::get({o.mSize, o.nSize}, o.cType)});
}

void VirtualMMAAttr::getDistributedTileTypes(
    SmallVectorImpl<VectorType> &result) const {
  MLIRContext *context = getContext();
  VirtualMMAIntrinsic intrinsic = getIntrinsic();
  result.assign({getThreadVectorType(context, intrinsic, kMMAOperandLhs),
                 getThreadVectorType(context, intrinsic, kMMAOperandRhs),
                 getThreadVectorType(context, intrinsic, kMMAOperandAcc)});
}

int64_t VirtualMMAAttr::getSubgroupSize() const {
  switch (getIntrinsic()) {
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ:
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F16:
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F8E4M3FNUZ:
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F16: {
    return 64;
  }
  }
  assert(false && "unhandled virtual mma layout type.");
  return 0;
}

Attribute VirtualMMAAttr::getDistributionMappingKind() const {
  return IREE::GPU::LaneIdAttr::get(getContext(), 0);
}

OpFoldResult VirtualMMAAttr::getDistributionWorkerCount(OpBuilder &, Location,
                                                        Operation *) const {
  return getAsIndexOpFoldResult(getContext(), getSubgroupSize());
}

LogicalResult VirtualMMAAttr::populateOperandOffsetsSizesStrides(
    OpBuilder &builder, Location loc, uint32_t operandIndex, Value laneId,
    ArrayRef<int64_t> permutation, SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &sizes,
    SmallVectorImpl<OpFoldResult> &strides) const {
  assert(operandIndex <= 2 && "Must index valid MMA operand");
  MMASingleSubgroupLayout subgroupLayout =
      getSingleSubgroupLayout(getIntrinsic(), operandIndex,
                              operandIndex == kMMAOperandAcc && getColMajor());
  SmallVector<OpFoldResult> canonicalOffsets;
  SmallVector<OpFoldResult> canonicalSizes;
  if (failed(populateCanonicalOffsetsSizesAndStrides(
          builder, loc, laneId, permutation, subgroupLayout, canonicalOffsets,
          canonicalSizes, strides))) {
    return failure();
  }
  offsets.append(canonicalOffsets);
  sizes.append(canonicalSizes);

  return success();
}

int64_t VirtualMMAAttr::getIntrinsicsK() const {
  switch (getIntrinsic()) {
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F16:
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F16: {
    return 2;
  }
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ:
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F8E4M3FNUZ: {
    return 1;
  }
  }
  assert(false && "unhandled virtual mma layout type.");
  return 0;
}

// Generates amdgpu.mfma/wmma operation on the given inputs for this attribute
// type.
LogicalResult VirtualMMAAttr::buildUnderlyingOperations(
    OpBuilder &builder, Location loc, ValueRange inputs, ValueRange outputs,
    SmallVectorImpl<Value> &results) const {
  if (inputs.size() != 2) {
    return failure();
  }
  if (outputs.size() != 1) {
    return failure();
  }
  SmallVector<VectorType> threadTypes;
  getDistributedTileTypes(threadTypes);
  if (!llvm::equal(threadTypes,
                   llvm::concat<Type>(inputs.getTypes(), outputs.getTypes()))) {
    return failure();
  }

  switch (getIntrinsic()) {
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ:
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F16:
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F8E4M3FNUZ:
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F16: {
    // Generate mfma's for K with unrolled kernels.
    const int64_t unrollKFactor = getIntrinsicsK();
    auto [m, n, k] = getMNKShape();
    // Compute actual/native intrinsic's K size.
    int64_t nativeKSize = k / unrollKFactor;

    auto [aType, bType, cType] = getABCVectorTypes();
    if (aType.getShape()[0] != bType.getShape()[0]) {
      // Currently only support case where lhs and rhs
      // has same vectorWidth.
      return failure();
    }
    int64_t vectorWidth = aType.getShape()[0] / unrollKFactor;
    Value acc = outputs[0];
    for (int i = 0; i < unrollKFactor; i++) {
      int64_t offset = vectorWidth * i;
      Value sliced_lhs = vector::ExtractStridedSliceOp::create(
          builder, loc, inputs[0], ArrayRef<int64_t>{offset},
          ArrayRef<int64_t>{vectorWidth}, ArrayRef<int64_t>{1});
      Value sliced_rhs = vector::ExtractStridedSliceOp::create(
          builder, loc, inputs[1], ArrayRef<int64_t>{offset},
          ArrayRef<int64_t>{vectorWidth}, ArrayRef<int64_t>{1});
      if (getColMajor()) {
        std::swap(sliced_lhs, sliced_rhs);
      }
      acc = amdgpu::MFMAOp::create(builder, loc, outputs[0].getType(), m, n,
                                   nativeKSize, getBlockSize(), sliced_lhs,
                                   sliced_rhs, acc)
                .getResult();
    }
    results.push_back(acc);
    return success();
  }
  }
  return failure();
}

int64_t VirtualMMAAttr::getBlockSize() const {
  switch (getIntrinsic()) {
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ:
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F16:
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F8E4M3FNUZ:
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F16: {
    return 1;
  }
  }
  assert(false && "unhandled virtual mma layout type.");
  return 0;
}

MMASingleSubgroupLayout getSingleSubgroupLayout(VirtualMMAIntrinsic intrinsic,
                                                int operandIndex) {
  switch (intrinsic) {
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F16:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return {/*outer=*/{1, 1}, /*thread=*/{16, 4}, /*tstrides=*/{1, 16},
              /*element=*/{1, 8}};
    case kMMAOperandRhs:
      return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{8, 1}};
    case kMMAOperandAcc:
      return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{4, 1}};
    }
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return {/*outer=*/{1, 2}, /*thread=*/{16, 4}, /*tstrides=*/{1, 16},
              /*element=*/{1, 4}};
    case kMMAOperandRhs:
      return {/*outer=*/{2, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{4, 1}};
    case kMMAOperandAcc:
      return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{4, 1}};
    }
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F16:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return {/*outer=*/{1, 1}, /*thread=*/{32, 2}, /*tstrides=*/{1, 32},
              /*element=*/{1, 8}};
    case kMMAOperandRhs:
      return {/*outer=*/{1, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
              /*element=*/{8, 1}};
    case kMMAOperandAcc:
      return {/*outer=*/{4, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
              /*element=*/{4, 1}};
    }
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F8E4M3FNUZ:
    switch (operandIndex) {
    case kMMAOperandLhs:
      return {/*outer=*/{1, 2}, /*thread=*/{32, 2}, /*tstrides=*/{1, 32},
              /*element=*/{1, 4}};
    case kMMAOperandRhs:
      return {/*outer=*/{2, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
              /*element=*/{4, 1}};
    case kMMAOperandAcc:
      return {/*outer=*/{4, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
              /*element=*/{4, 1}};
    }
  }
  assert(false && "unhandled virtual mma layout type.");
  return {};
}

//===----------------------------------------------------------------------===//
// ScaledMMA Attributes
//===----------------------------------------------------------------------===//

int64_t ScaledMMAAttr::getExpectedNumInputs() const { return 4; }

int64_t ScaledMMAAttr::getExpectedNumOutputs() const { return 1; }

MMASingleSubgroupLayout getSingleSubgroupLayout(ScaledMMAIntrinsic intrinsic,
                                                int64_t operandIndex) {
  const MMASingleSubgroupLayout mfmaAcc16x16 = {
      /*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
      /*element=*/{4, 1}};
  const MMASingleSubgroupLayout mfmaAcc32x32 = {
      /*outer=*/{4, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
      /*element=*/{4, 1}};

  switch (intrinsic) {
  case ScaledMMAIntrinsic::MFMA_SCALE_F32_16x16x128_B32:
    switch (operandIndex) {
    case kScaledMMAOperandLhs:
      return {/*outer=*/{1, 1, 1}, /*thread=*/{16, 4, 1},
              /*tstrides=*/{1, 16, 1},
              /*element=*/{1, 1, 32}};
    case kScaledMMAOperandRhs:
      return {/*outer=*/{1, 1, 1}, /*thread=*/{4, 1, 16},
              /*tstrides=*/{16, 1, 1},
              /*element=*/{1, 32, 1}};
    case kScaledMMAOperandLhsScale:
      return {/*outer=*/{1, 1}, /*thread=*/{16, 4}, /*tstrides=*/{1, 16},
              /*element=*/{1, 1}};
    case kScaledMMAOperandRhsScale:
      return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{1, 1}};
    case kScaledMMAOperandAcc:
      return mfmaAcc16x16;
    }
  case ScaledMMAIntrinsic::MFMA_SCALE_F32_32x32x64_B32:
    switch (operandIndex) {
    case kScaledMMAOperandLhs:
      return {/*outer=*/{1, 1, 1}, /*thread=*/{32, 2, 1},
              /*tstrides=*/{1, 32, 1},
              /*element=*/{1, 1, 32}};
    case kScaledMMAOperandRhs:
      return {/*outer=*/{1, 1, 1}, /*thread=*/{2, 1, 32},
              /*tstrides=*/{32, 1, 1},
              /*element=*/{1, 32, 1}};
    case kScaledMMAOperandLhsScale:
      return {/*outer=*/{1, 1}, /*thread=*/{32, 2}, /*tstrides=*/{1, 32},
              /*element=*/{1, 1}};
    case kScaledMMAOperandRhsScale:
      return {/*outer=*/{1, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
              /*element=*/{1, 1}};
    case kScaledMMAOperandAcc:
      return mfmaAcc32x32;
    }
  }
  assert(false && "Unhandled scaled MMA intrinsic");
  return {};
}

int64_t ScaledMMAAttr::getBlockSize() const {
  switch (getIntrinsic()) {
  case ScaledMMAIntrinsic::MFMA_SCALE_F32_16x16x128_B32:
  case ScaledMMAIntrinsic::MFMA_SCALE_F32_32x32x64_B32:
    return 32;
  }
  assert(false &&
         "all cases should've been handled in ScaledMMA::getBlockSize()");
  return 0;
}

MMASingleSubgroupLayout getSingleSubgroupLayout(ScaledMMAIntrinsic intrinsic,
                                                int64_t operandIndex,
                                                bool isAccColMajor) {
  MMASingleSubgroupLayout baseLayout =
      getSingleSubgroupLayout(intrinsic, operandIndex);
  if (operandIndex == kScaledMMAOperandAcc && isAccColMajor) {
    std::swap(baseLayout.outer[0], baseLayout.outer[1]);
    std::swap(baseLayout.thread[0], baseLayout.thread[1]);
    std::swap(baseLayout.tstrides[0], baseLayout.tstrides[1]);
    std::swap(baseLayout.element[0], baseLayout.element[1]);
  }
  return baseLayout;
}

int64_t ScaledMMAAttr::getSubgroupSize() const {
  return getIntrinsicSubgroupSize(getIntrinsic());
}

SmallVector<Type> ScaledMMAAttr::getSupportedInputTypes(MLIRContext *ctx) {
  return {Float8E8M0FNUType::get(ctx),  Float8E5M2Type::get(ctx),
          Float8E5M2FNUZType::get(ctx), Float8E4M3FNType::get(ctx),
          Float8E4M3FNUZType::get(ctx), Float4E2M1FNType::get(ctx)};
}

SmallVector<Type> ScaledMMAAttr::getSupportedOutputTypes(MLIRContext *ctx) {
  return {Float32Type::get(ctx)};
}

LogicalResult
ScaledMMAAttr::verifyIndexingMaps(ArrayRef<AffineMap> maps) const {
  if (failed(verifyMmaIndexingMaps({maps[kScaledMMAOperandLhs],
                                    maps[kScaledMMAOperandRhs],
                                    maps[kScaledMMAOperandAcc]}))) {
    return failure();
  }

  SmallVector<llvm::SmallDenseSet<AffineExpr, 4>> resExprs(
      maps.size(), llvm::SmallDenseSet<AffineExpr, 4>());
  for (auto [set, map] : llvm::zip_equal(resExprs, maps)) {
    set.insert_range(map.getResults());
  }
  // Note: the below conditions are a best guess and may be too strict.

  // Check LHS scales aren't using indexes that LHS isn't.
  if (llvm::any_of(resExprs[kScaledMMAOperandLhsScale], [&](auto e) {
        return !resExprs[kScaledMMAOperandLhs].contains(e);
      })) {
    return failure();
  }
  // Check RHS scales aren't using indexes that RHS isn't.
  if (llvm::any_of(resExprs[kScaledMMAOperandRhsScale], [&](auto e) {
        return !resExprs[kScaledMMAOperandRhs].contains(e);
      })) {
    return failure();
  }
  return success();
}

void ScaledMMAAttr::getUndistributedTileTypes(
    SmallVectorImpl<VectorType> &results) const {
  MMASingleSubgroupLayout lhsLayout =
      getSingleSubgroupLayout(getIntrinsic(), kScaledMMAOperandLhs);
  MMASingleSubgroupLayout rhsLayout =
      getSingleSubgroupLayout(getIntrinsic(), kScaledMMAOperandRhs);

  int64_t blockSize = getBlockSize();
  int64_t m = lhsLayout.outer[0] * lhsLayout.thread[0] * lhsLayout.element[0];
  int64_t kScale =
      lhsLayout.outer[1] * lhsLayout.thread[1] * lhsLayout.element[1];
  [[maybe_unused]] int64_t layoutBlockSize =
      lhsLayout.outer[2] * lhsLayout.thread[2] * lhsLayout.element[2];
  assert(blockSize == layoutBlockSize &&
         "expected block size to be set up correctly");
  int64_t n = rhsLayout.outer[2] * rhsLayout.thread[2] * rhsLayout.element[2];

  Type lhsType = getLhsElemType();
  Type rhsType = getRhsElemType();
  Type accType = getAccElemType();
  Type scaleType = Float8E8M0FNUType::get(getContext());

  results.push_back(VectorType::get({m, kScale, blockSize}, lhsType));
  results.push_back(VectorType::get({kScale, blockSize, n}, rhsType));
  results.push_back(VectorType::get({m, kScale}, scaleType));
  results.push_back(VectorType::get({kScale, n}, scaleType));
  results.push_back(VectorType::get({m, n}, accType));
}

void ScaledMMAAttr::getDistributedTileTypes(
    SmallVectorImpl<VectorType> &results) const {
  Type lhsType = getLhsElemType();
  Type rhsType = getRhsElemType();
  Type accType = getAccElemType();
  Type scaleType = Float8E8M0FNUType::get(getContext());

  std::array<Type, 5> argTypes = {lhsType, rhsType, scaleType, scaleType,
                                  accType};
  for (auto [opIndex, type] : llvm::enumerate(argTypes)) {
    MMASingleSubgroupLayout layout =
        getSingleSubgroupLayout(getIntrinsic(), opIndex);
    int64_t outer = ShapedType::getNumElements(layout.outer);
    int64_t element = ShapedType::getNumElements(layout.element);
    results.push_back(VectorType::get({outer * element}, type));
  }
}

std::optional<SmallVector<int64_t, 2>>
ScaledMMAAttr::getUndistributedTileDimExpansion(int64_t operandIndex,
                                                int64_t dim) const {
  assert(operandIndex <= kScaledMMAOperandAcc && "invalid operand index");
  MMASingleSubgroupLayout layout =
      getSingleSubgroupLayout(getIntrinsic(), operandIndex, getColMajor());
  if (layout.outer[dim] > 1) {
    return SmallVector<int64_t, 2>{layout.outer[dim],
                                   layout.element[dim] * layout.thread[dim]};
  }
  return std::nullopt;
}

Attribute ScaledMMAAttr::getDistributionMappingKind() const {
  return IREE::GPU::LaneIdAttr::get(getContext(), 0);
}

OpFoldResult ScaledMMAAttr::getDistributionWorkerCount(OpBuilder &, Location,
                                                       Operation *) const {
  return getAsIndexOpFoldResult(getContext(), getSubgroupSize());
}

LogicalResult ScaledMMAAttr::populateOperandOffsetsSizesStrides(
    OpBuilder &builder, Location loc, uint32_t operandIndex, Value laneId,
    ArrayRef<int64_t> permutation, SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &sizes,
    SmallVectorImpl<OpFoldResult> &strides) const {
  assert(operandIndex <= kScaledMMAOperandAcc && "Scaled MFMA has 5 operands");

  MMASingleSubgroupLayout subgroupLayout =
      getSingleSubgroupLayout(getIntrinsic(), operandIndex, getColMajor());

  SmallVector<OpFoldResult> canonicalOffsets;
  SmallVector<OpFoldResult> canonicalSizes;
  if (failed(populateCanonicalOffsetsSizesAndStrides(
          builder, loc, laneId, permutation, subgroupLayout, canonicalOffsets,
          canonicalSizes, strides))) {
    return failure();
  }
  offsets.append(canonicalOffsets);
  sizes.append(canonicalSizes);

  return success();
}

LogicalResult ScaledMMAAttr::buildUnderlyingOperations(
    OpBuilder &builder, Location loc, ValueRange inputs, ValueRange outputs,
    SmallVectorImpl<Value> &results) const {
  if (inputs.size() != 4) {
    return failure();
  }
  if (outputs.size() != 1) {
    return failure();
  }
  SmallVector<VectorType> threadTypes;
  getDistributedTileTypes(threadTypes);
  if (!llvm::equal(threadTypes,
                   llvm::concat<Type>(inputs.getTypes(), outputs.getTypes()))) {
    return failure();
  }

  SmallVector<VectorType> subgroupTypes;
  getUndistributedTileTypes(subgroupTypes);

  // Note: the scales argument is given as a vector of 4
  // scales + a constant selector to say which byte in the vector is to be used.
  // This'll allow better value reuse, hence why we extend to 4-vectors
  // instead of clamping to scalars here.

  FloatType f8E8M0 = builder.getF8E8M0Type();
  Value zeroScales = arith::ConstantOp::create(
      builder, loc,
      SplatElementsAttr::get(
          VectorType::get({getScalesVectorSize()}, f8E8M0),
          llvm::APFloat::getSmallest(f8E8M0.getFloatSemantics())));
  auto padScales = [&](Value scales) {
    Value scale = vector::ExtractOp::create(builder, loc, scales, 0);
    Value padded = vector::InsertOp::create(builder, loc, scale, zeroScales, 0);
    return padded;
  };

  Value lhs = inputs[kScaledMMAOperandLhs];
  Value rhs = inputs[kScaledMMAOperandRhs];
  Value lhsScales = padScales(inputs[kScaledMMAOperandLhsScale]);
  Value rhsScales = padScales(inputs[kScaledMMAOperandRhsScale]);
  Value acc = outputs[0];

  ArrayRef<int64_t> lhsShape = subgroupTypes[kScaledMMAOperandLhs].getShape();
  ArrayRef<int64_t> rhsShape = subgroupTypes[kScaledMMAOperandRhs].getShape();
  int64_t m = lhsShape[0];
  // We use m x [k / kPerBlock] x blockSize as the LHS pre-distribution shape
  // since this makes the higher-level tiling clearer.
  int64_t k = lhsShape[1] * lhsShape[2];
  int64_t n = rhsShape[2];

  // Since the LHS and RHS layouts are both {M,N}xK, we can get a column-major
  // result just by swapping the LHS and RHS.
  if (getColMajor()) {
    std::swap(lhs, rhs);
    std::swap(lhsScales, rhsScales);
    std::swap(n, m);
  }

  Value result =
      amdgpu::ScaledMFMAOp::create(builder, loc, m, n, k, lhs, rhs, acc,
                                   lhsScales, rhsScales, /*scalesIdxA=*/0,
                                   /*scalesIdxB=*/0);
  results.push_back(result);
  return success();
}

//===----------------------------------------------------------------------===//
// DataTiledScaledMMA Attributes
//===----------------------------------------------------------------------===//

TileSwizzle
DataTiledScaledMMAAttr::getTileSwizzle(unsigned operandIndex) const {
  return getSwizzle(*this, operandIndex);
}

static std::tuple<int64_t, int64_t, int64_t, int64_t>
getMNKKbShapeFromScaledIntrinsic(ScaledMMAIntrinsic intrinsic) {
  MMASingleSubgroupLayout lhs =
      getSingleSubgroupLayout(intrinsic, kScaledMMAOperandLhs);
  MMASingleSubgroupLayout rhs =
      getSingleSubgroupLayout(intrinsic, kScaledMMAOperandRhs);
  int64_t m = lhs.outer[0] * lhs.thread[0] * lhs.element[0];
  int64_t n = rhs.outer[2] * rhs.thread[2] * rhs.element[2];
  int64_t k = lhs.outer[1] * lhs.thread[1] * lhs.element[1];
  int64_t kB = lhs.outer[2] * lhs.thread[2] * lhs.element[2];
  return {m, n, k, kB};
}

int64_t getMSize(ScaledMMAIntrinsic intrinsic) {
  return std::get<0>(getMNKKbShapeFromScaledIntrinsic(intrinsic));
}
int64_t getNSize(ScaledMMAIntrinsic intrinsic) {
  return std::get<1>(getMNKKbShapeFromScaledIntrinsic(intrinsic));
}
int64_t getKSize(ScaledMMAIntrinsic intrinsic) {
  return std::get<2>(getMNKKbShapeFromScaledIntrinsic(intrinsic));
}
int64_t getKbSize(ScaledMMAIntrinsic intrinsic) {
  return std::get<3>(getMNKKbShapeFromScaledIntrinsic(intrinsic));
}

IREE::Codegen::TileMxNxKxKb DataTiledScaledMMAAttr::getTileMNKKb() const {
  IREE::Codegen::TileMxNxKxKb innerTile;
  std::tie(innerTile.M, innerTile.N, innerTile.K, innerTile.KB) =
      getMNKKbShapeFromScaledIntrinsic(getIntrinsic());
  innerTile.M *= getIntrinsicsM() * getSubgroupsM();
  innerTile.N *= getIntrinsicsN() * getSubgroupsN();
  innerTile.K *= getIntrinsicsK() * getSubgroupsK();
  return innerTile;
}

void DataTiledScaledMMAAttr::getElementTypes(
    SmallVectorImpl<Type> &result) const {
  result.push_back(getLhsElemType());
  result.push_back(getRhsElemType());
  result.push_back(Float8E8M0FNUType::get(getContext()));
  result.push_back(Float8E8M0FNUType::get(getContext()));
  result.push_back(getAccElemType());
  return;
}

static Value createScaledMmaOp(OpBuilder &builder, Location loc,
                               ScaledMMAIntrinsic intrinsic, Type resultType,
                               Value lhs, Value rhs, Value lhsScales,
                               Value rhsScales, Value acc) {
  FloatType f8E8M0 = builder.getF8E8M0Type();
  Value zeroScales = arith::ConstantOp::create(
      builder, loc,
      SplatElementsAttr::get(
          VectorType::get({4}, f8E8M0),
          llvm::APFloat::getSmallest(f8E8M0.getFloatSemantics())));
  auto padScales = [&](Value scales) {
    Value scale = vector::ExtractOp::create(builder, loc, scales, 0);
    Value padded = vector::InsertOp::create(builder, loc, scale, zeroScales, 0);
    return padded;
  };

  lhsScales = padScales(lhsScales);
  rhsScales = padScales(rhsScales);

  int64_t m = getMSize(intrinsic);
  // We use m x [k / kPerBlock] x blockSize as the LHS pre-distribution shape
  // since this makes the higher-level tiling clearer.
  int64_t k = getKSize(intrinsic) * getKbSize(intrinsic);
  int64_t n = getNSize(intrinsic);

  return amdgpu::ScaledMFMAOp::create(builder, loc, m, n, k, lhs, rhs, acc,
                                      lhsScales, rhsScales, /*scalesIdxA=*/0,
                                      /*scalesIdxB=*/0);
}

LogicalResult DataTiledScaledMMAAttr::buildUnderlyingOperations(
    OpBuilder &builder, Location loc, ValueRange inputs, ValueRange outputs,
    SmallVectorImpl<Value> &results) const {
  // Validation. Similar to MMAAttr::buildMmaOperation.
  if (inputs.size() != 4) {
    return failure();
  }
  if (outputs.size() != 1) {
    return failure();
  }
  SmallVector<VectorType> regTypes;
  getDistributedTileTypes(regTypes);
  if (!llvm::equal(regTypes,
                   llvm::concat<Type>(inputs.getTypes(), outputs.getTypes()))) {
    return failure();
  }

  // Prepare Lhs/Rhs/Acc operand slices to feed the intrinsic.
  const unsigned lhsIdx = 0;
  const unsigned rhsIdx = 1;
  const unsigned lhsScalesIdx = 2;
  const unsigned rhsScalesIdx = 3;
  const unsigned accIdx = 4;
  TileSwizzle lhsSwizzle = getSwizzle(*this, lhsIdx);
  LDBG() << "DataTiledScaledMMAAttr::buildMmaOperation";
  LDBG() << "    lhsSwizzle: " << lhsSwizzle;
  SmallVector<Value> intrinsicsLhs =
      distributeMmaFragmentToIntrinsics(builder, loc, inputs[0], lhsSwizzle);

  TileSwizzle rhsSwizzle = getSwizzle(*this, rhsIdx);
  LDBG() << "DataTiledScaledMMAAttr::buildMmaOperation";
  LDBG() << "    rhsSwizzle: " << rhsSwizzle;
  SmallVector<Value> intrinsicsRhs =
      distributeMmaFragmentToIntrinsics(builder, loc, inputs[1], rhsSwizzle);

  TileSwizzle lhsScalesSwizzle = getSwizzle(*this, lhsScalesIdx);
  LDBG() << "DataTiledScaledMMAAttr::buildMmaOperation";
  LDBG() << "    lhsScalesSwizzle: " << lhsScalesSwizzle;
  SmallVector<Value> intrinsicsLhsScales = distributeMmaFragmentToIntrinsics(
      builder, loc, inputs[2], lhsScalesSwizzle);

  TileSwizzle rhsScalesSwizzle = getSwizzle(*this, rhsScalesIdx);
  LDBG() << "DataTiledScaledMMAAttr::buildMmaOperation";
  LDBG() << "    rhsScalesSwizzle: " << rhsScalesSwizzle;
  SmallVector<Value> intrinsicsRhsScales = distributeMmaFragmentToIntrinsics(
      builder, loc, inputs[3], rhsScalesSwizzle);

  TileSwizzle accSwizzle = getSwizzle(*this, accIdx);
  LDBG() << "DataTiledScaledMMAAttr::buildMmaOperation";
  LDBG() << "    accSwizzle: " << accSwizzle;

  SmallVector<Value> intrinsicsAcc =
      distributeMmaFragmentToIntrinsics(builder, loc, outputs[0], accSwizzle);

  ScaledMMAIntrinsic intrinsic = getIntrinsic();
  auto intrinCType = cast<VectorType>(intrinsicsAcc.front().getType());

  // Loop over the 3 unroll_{m,n,k} dimensions to create the intrinsics.
  for (int64_t mu = 0; mu < getIntrinsicsM(); ++mu) {
    for (int64_t nu = 0; nu < getIntrinsicsN(); ++nu) {
      for (int64_t ku = 0; ku < getIntrinsicsK(); ++ku) {
        Value lhs = intrinsicsLhs[mu * getIntrinsicsK() + ku];
        Value rhs = intrinsicsRhs[nu * getIntrinsicsK() + ku];
        Value lhsScales = intrinsicsLhsScales[mu * getIntrinsicsK() + ku];
        Value rhsScales = intrinsicsRhsScales[nu * getIntrinsicsK() + ku];
        Value &acc = intrinsicsAcc[mu * getIntrinsicsN() + nu];
        acc = createScaledMmaOp(builder, loc, intrinsic, intrinCType, lhs, rhs,
                                lhsScales, rhsScales, acc);
      }
    }
  }

  // Insert the results into the destination accumulator.
  SmallVector<int64_t> accCrossIntrinsicShape =
      sliceSwizzledShape(accSwizzle, [](TileSwizzle::Dim dim) {
        return dim.kind == TileSwizzle::Dim::Kind::CrossIntrinsic;
      });
  SmallVector<int64_t> accInternalShape =
      sliceSwizzledShape(accSwizzle, [](TileSwizzle::Dim dim) {
        return dim.kind == TileSwizzle::Dim::Kind::Internal;
      });

  LDBG() << "accCrossIntrinsicShape: "
         << llvm::interleaved(accCrossIntrinsicShape);
  LDBG() << "accInternalShape: " << llvm::interleaved(accInternalShape);
  size_t dstRank = accCrossIntrinsicShape.size();
  SmallVector<int64_t> strides(dstRank, 1);
  SmallVector<int64_t> indices(dstRank, 0);
  Value acc = outputs[0];
  for (Value intrAcc : intrinsicsAcc) {
    auto expandedAcc = vector::ShapeCastOp::create(
        builder, loc,
        VectorType::get(
            accInternalShape,
            cast<VectorType>(outputs[0].getType()).getElementType()),
        intrAcc);
    acc = vector::InsertStridedSliceOp::create(builder, loc, expandedAcc, acc,
                                               indices, strides);
    incrementIndices(indices, accCrossIntrinsicShape);
  }
  results.push_back(acc);
  return success();
}

int64_t DataTiledScaledMMAAttr::getExpectedNumInputs() const { return 4; }

int64_t DataTiledScaledMMAAttr::getExpectedNumOutputs() const { return 1; }

int64_t DataTiledScaledMMAAttr::getSubgroupSize() const {
  return getIntrinsicSubgroupSize(getIntrinsic());
}

int64_t DataTiledScaledMMAAttr::getFlatWorkgroupSize() const {
  return getSubgroupSize() * getSubgroupsM() * getSubgroupsN() *
         getSubgroupsK();
}

LogicalResult
DataTiledScaledMMAAttr::verifyIndexingMaps(ArrayRef<AffineMap> maps) const {
  return IREE::LinalgExt::inferScaledContractionDims(maps);
}

//===----------------------------------------------------------------------===//
// Target Attributes
//===----------------------------------------------------------------------===//

std::optional<int> TargetAttr::getCUDAComputeCapability() const {
  StringRef arch = getArch();
  if (!arch.starts_with("sm_"))
    return false;
  APInt version;
  if (arch.substr(3).getAsInteger(10, version)) {
    return false;
  }
  return version.getZExtValue();
}

bool TargetAttr::supportsTF32InputMMAOps() const {
  // TODO: scan the list of MMA ops to decude after plumbing through support
  // for NVIDIA TensorCore MMA ops.
  if (auto cc = getCUDAComputeCapability())
    return cc >= 80;
  return false;
}

bool TargetAttr::supportsSyncMMAOps() const {
  if (auto cc = getCUDAComputeCapability())
    return cc >= 80;
  return false;
}

std::array<int64_t, 3> TargetAttr::getMaximumWorkgroupCount() const {
  DenseI32ArrayAttr maxWgpCount = getWgp().getMaxWorkgroupCounts();
  assert(maxWgpCount.size() <= 3 && "expected only workgroup count for x,y,z");
  std::array<int64_t, 3> maxWorkgroupCount = {
      ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic};
  for (auto [index, value] : llvm::enumerate(maxWgpCount.asArrayRef())) {
    maxWorkgroupCount[index] = value;
  }
  return maxWorkgroupCount;
}

//===----------------------------------------------------------------------===//
// Lowering Config Attributes
//===----------------------------------------------------------------------===//

constexpr StringLiteral kWorkgroupLevelName = "workgroup";
constexpr StringLiteral kPartialReductionLevelName = "partial_reduction";
constexpr StringLiteral kReductionLevelName = "reduction";
constexpr StringLiteral kSerialLevelName = "serial";
constexpr StringLiteral kThreadLevelName = "thread";
constexpr StringLiteral kSubgroupLevelName = "subgroup";
constexpr StringLiteral kLaneLevelName = "lane";

StringRef getTilingLevelName(GPU::TilingLevel level) {
  switch (level) {
  case GPU::TilingLevel::Workgroup:
    return kWorkgroupLevelName;
  case GPU::TilingLevel::PartialReduction:
    return kPartialReductionLevelName;
  case GPU::TilingLevel::Reduction:
    return kReductionLevelName;
  case GPU::TilingLevel::Serial:
    return kSerialLevelName;
  case GPU::TilingLevel::Thread:
    return kThreadLevelName;
  case GPU::TilingLevel::Subgroup:
    return kSubgroupLevelName;
  case GPU::TilingLevel::Lane:
    return kLaneLevelName;
  }
  assert(false && "Unknown tiling level");
  return StringRef();
}

static SmallVector<int64_t> getIntegerVector(ArrayAttr array) {
  if (!array || !llvm::all_of(array.getValue(), llvm::IsaPred<IntegerAttr>)) {
    return {};
  }
  return llvm::map_to_vector(array.getValue(), [](Attribute s) -> int64_t {
    return cast<IntegerAttr>(s).getInt();
  });
}

static SmallVector<int64_t> getTileSizes(DictionaryAttr config,
                                         GPU::TilingLevel level) {
  return getIntegerVector(config.getAs<ArrayAttr>(getTilingLevelName(level)));
}

SmallVector<int64_t> LoweringConfigAttr::getWorkgroupTileSizes() const {
  return getTileSizes(getAttributes(), GPU::TilingLevel::Workgroup);
}

SmallVector<int64_t>
LoweringConfigAttr::getStaticTilingLevelSizes(unsigned level,
                                              Operation *op) const {
  if (level > llvm::to_underlying(GPU::TilingLevel::Lane)) {
    return {};
  }
  return getTileSizes(getAttributes(), static_cast<GPU::TilingLevel>(level));
}

SmallVector<OpFoldResult>
LoweringConfigAttr::getTilingLevelSizes(OpBuilder &b, unsigned level,
                                        Operation *op) const {
  if (level > llvm::to_underlying(GPU::TilingLevel::Lane)) {
    return {};
  }
  SmallVector<int64_t> sizes =
      getTileSizes(getAttributes(), static_cast<GPU::TilingLevel>(level));
  return llvm::map_to_vector(
      sizes, [&](int64_t s) -> OpFoldResult { return b.getIndexAttr(s); });
}

bool LoweringConfigAttr::hasTilingLevel(unsigned level) const {
  if (level > llvm::to_underlying(GPU::TilingLevel::Lane)) {
    return false;
  }
  return !getTileSizes(getAttributes(), static_cast<GPU::TilingLevel>(level))
              .empty();
}

bool LoweringConfigAttr::hasWorkgroupTilingLevel() const {
  return !getWorkgroupTileSizes().empty();
}

constexpr StringLiteral kLoweringStrategyName = "lowering_strategy";

std::optional<StringRef> LoweringConfigAttr::getLoweringStrategy() const {
  if (auto name = getAttributes().getAs<StringAttr>(kLoweringStrategyName)) {
    return name.strref();
  }
  return std::nullopt;
}
constexpr StringLiteral kWorkgroupReorderingStrategyName =
    "workgroup_reordering_strategy";

::mlir::iree_compiler::IREE::Codegen::WorkgroupReorderingAttrInterface
LoweringConfigAttr::getWorkgroupReorderingStrategy() const {
  if (auto attr = getAttributes()
                      .getAs<::mlir::iree_compiler::IREE::Codegen::
                                 WorkgroupReorderingAttrInterface>(
                          kWorkgroupReorderingStrategyName)) {
    return attr;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Conditional Transpose Workgroup Reordering
//===----------------------------------------------------------------------===//

static std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>,
                  SmallVector<OpFoldResult>>
getLoopBounds(ArrayRef<Range> loopRanges,
              ArrayRef<OpFoldResult> givenTileSizes) {
  SmallVector<OpFoldResult> lbs, ubs, steps;
  for (auto [loopRange, givenTileSize] :
       llvm::zip_equal(loopRanges, givenTileSizes)) {
    // No loop if the tile size is 0.
    if (isZeroInteger(givenTileSize))
      continue;
    lbs.push_back(loopRange.offset);
    ubs.push_back(loopRange.size);
    steps.push_back(givenTileSize);
  }
  return {lbs, ubs, steps};
}

static Value computeNumTiles(OpBuilder &b, Location loc, OpFoldResult size,
                             OpFoldResult tileSize) {
  AffineExpr s0, s1;
  bindSymbols(b.getContext(), s0, s1);
  AffineExpr ceilDivExpr = s0.ceilDiv(s1);
  OpFoldResult numTiles = affine::makeComposedFoldedAffineApply(
      b, loc, ceilDivExpr, {size, tileSize});
  return getValueOrCreateConstantIntOp(b, loc, numTiles);
}

static AffineExpr getMulExpr(OpBuilder &b) {
  AffineExpr s0, s1;
  bindSymbols(b.getContext(), s0, s1);
  return s0 * s1;
}

static AffineExpr getSubExpr(OpBuilder &b) {
  AffineExpr s0, s1;
  bindSymbols(b.getContext(), s0, s1);
  return s0 - s1;
}

/// Computes the total number of different Tile loads accross all XCDs per
/// iteration for the pingpong matmul kernel. For each iteration, we distribute
/// tile loads to invidiual CUs along the X axis of the RHS.
static Value computeNumTileLoads(OpBuilder &b, Location loc, OpFoldResult size,
                                 OpFoldResult tileSize, Value numXCDs,
                                 Value numCUs) {
  Value numTiles = computeNumTiles(b, loc, size, tileSize);
  AffineExpr mulExpr = getMulExpr(b);
  Value totalCUs =
      getValueOrCreateConstantIntOp(b, loc,
                                    affine::makeComposedFoldedAffineApply(
                                        b, loc, mulExpr, {numXCDs, numCUs}));
  auto numX = arith::MinUIOp::create(b, loc, numTiles, totalCUs)->getResult(0);

  AffineExpr s0, s1;
  bindSymbols(b.getContext(), s0, s1);
  AffineExpr numTilesExpr =
      s0.ceilDiv(s1) + s1; // totalCUs.ceildiv(numX) + numX
  OpFoldResult numTileLoads = affine::makeComposedFoldedAffineApply(
      b, loc, numTilesExpr, {totalCUs, numX});
  return getValueOrCreateConstantIndexOp(b, loc, numTileLoads);
}

/// Retrieves the condition for transposed ordering. Transposed order is enabled
/// if the number of tile loads is at least 20% less than in regular order.
static Value getCondition(OpBuilder &b, Location loc,
                          ArrayRef<OpFoldResult> lbs,
                          ArrayRef<OpFoldResult> ubs,
                          ArrayRef<OpFoldResult> steps, Value numXCDs,
                          Value numCUs) {
  assert(ubs.size() == 2 && steps.size() == 2 && "rank must be 2");

  AffineExpr subExpr = getSubExpr(b);
  OpFoldResult size0 =
      affine::makeComposedFoldedAffineApply(b, loc, subExpr, {ubs[0], lbs[0]});
  OpFoldResult size1 =
      affine::makeComposedFoldedAffineApply(b, loc, subExpr, {ubs[1], lbs[1]});

  Value transposedOrder =
      computeNumTileLoads(b, loc, size0, steps[0], numXCDs, numCUs);
  Value defaultOrder =
      computeNumTileLoads(b, loc, size1, steps[1], numXCDs, numCUs);

  AffineExpr s0, s1, s2;
  bindSymbols(b.getContext(), s0, s1, s2);
  AffineExpr transposedOrderBumpExpr = (s0 * s1).floorDiv(s2);
  Value c4 = arith::ConstantIntOp::create(b, loc, 4, 32);
  Value c5 = arith::ConstantIntOp::create(b, loc, 5, 32);

  Value defaultOrderTolerance = getValueOrCreateConstantIntOp(
      b, loc,
      affine::makeComposedFoldedAffineApply(b, loc, transposedOrderBumpExpr,
                                            {defaultOrder, c4, c5}));
  return mlir::arith::CmpIOp::create(b, loc, mlir::arith::CmpIPredicate::ult,
                                     transposedOrder, defaultOrderTolerance);
}
/// Swap values based on `pred`.
static void swapIf(OpBuilder &b, Location loc, OpFoldResult pred,
                   SmallVector<OpFoldResult> &values,
                   ArrayRef<size_t> ids = {0, 1}) {

  assert(ids.size() == 2 && "Can only swap between 2 indices");
  Value v0 = getValueOrCreateConstantIndexOp(b, loc, values[ids[0]]);
  Value v1 = getValueOrCreateConstantIndexOp(b, loc, values[ids[1]]);
  Value predVal = getValueOrCreateConstantIndexOp(b, loc, pred);
  values[ids[0]] =
      mlir::arith::SelectOp::create(b, loc, predVal, v1, v0).getResult();
  values[ids[1]] =
      mlir::arith::SelectOp::create(b, loc, predVal, v0, v1).getResult();
}

static std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>,
                  SmallVector<size_t>>
computeOffsetAndSize(ArrayRef<Range> loopRanges,
                     ArrayRef<OpFoldResult> givenTileSizes,
                     ArrayRef<Value> ivs) {
  SmallVector<size_t> ids;
  SmallVector<OpFoldResult> offsets, sizes;

  offsets.reserve(loopRanges.size());
  sizes.reserve(loopRanges.size());
  const Value *ivIt = ivs.begin();

  for (auto [loopRange, givenTileSize] :
       llvm::zip_equal(loopRanges, givenTileSizes)) {
    if (isZeroInteger(givenTileSize)) {
      offsets.push_back(loopRange.offset);
      sizes.push_back(loopRange.size);
    } else {
      offsets.push_back(*ivIt++);
      sizes.push_back(givenTileSize);
      ids.push_back(sizes.size() - 1);
    }
  }
  return {offsets, sizes, ids};
}

FailureOr<mlir::scf::SCFTilingOptions::CustomLoopHeaderInfo>
ConditionalTransposeAttr::generateLoopHeaderFn(
    OpBuilder &builder, Location loc, ArrayRef<Range> loopRanges,
    ArrayRef<OpFoldResult> givenTileSizes,
    ValueRange destinationTensors) const {
  scf::ForallOp forallOp;
  // Get loop bounds.
  SmallVector<OpFoldResult> lbs, ubs, steps;
  std::tie(lbs, ubs, steps) = getLoopBounds(loopRanges, givenTileSizes);

  if (lbs.size() != 2) {
    return emitError(loc) << "conditional_transpose only supports rank 2";
  }

  SmallVector<Attribute> deviceMappingAttribute;
  deviceMappingAttribute.push_back(IREE::Codegen::WorkgroupMappingAttr::get(
      builder.getContext(), IREE::Codegen::symbolizeWorkgroupId(1).value()));
  deviceMappingAttribute.push_back(IREE::Codegen::WorkgroupMappingAttr::get(
      builder.getContext(), IREE::Codegen::symbolizeWorkgroupId(0).value()));

  std::optional<ArrayAttr> mappingAttr =
      builder.getArrayAttr(deviceMappingAttribute);

  // Compute condition for transpose.
  Value numCUsVal = arith::ConstantIndexOp::create(builder, loc, getNumCUs());
  Value numXCDs = arith::ConstantIndexOp::create(builder, loc, getNumXCDs());
  Value cond = getCondition(builder, loc, lbs, ubs, steps, numXCDs, numCUsVal);

  // Apply condition on loop bounds.
  swapIf(builder, loc, cond, lbs);
  swapIf(builder, loc, cond, ubs);
  swapIf(builder, loc, cond, steps);

  forallOp = scf::ForallOp::create(builder, loc, lbs, ubs, steps,
                                   destinationTensors, mappingAttr);
  builder.setInsertionPoint(forallOp.getTerminator());

  SmallVector<Value> ivs = forallOp.getInductionVars();
  SmallVector<OpFoldResult> offsets, sizes;
  SmallVector<size_t> ids;
  std::tie(offsets, sizes, ids) =
      computeOffsetAndSize(loopRanges, givenTileSizes, ivs);

  // Apply condition on offsets.
  swapIf(builder, loc, cond, offsets, ids);

  ValueRange innerDestinationTensors = forallOp.getRegionOutArgs();
  return mlir::scf::SCFTilingOptions::CustomLoopHeaderInfo{
      {cast<LoopLikeOpInterface>(forallOp.getOperation())},
      offsets,
      sizes,
      innerDestinationTensors};
}

LogicalResult ConditionalTransposeAttr::generateLoopTerminatorFn(
    OpBuilder &builder, Location loc, ArrayRef<LoopLikeOpInterface> loops,
    ValueRange tiledResults, ArrayRef<SmallVector<OpFoldResult>> resultOffsets,
    ArrayRef<SmallVector<OpFoldResult>> resultSizes,
    ValueRange destinationTensors) const {
  assert(loops.size() == 1 && "Expected loop count should be 1.");
  LoopLikeOpInterface loop = loops.front();
  scf::ForallOp *forallOp = dyn_cast<scf::ForallOp>(&loop);
  if (!forallOp) {
    return emitError(loc) << "Only scf.forall op are supported";
  }

  builder.setInsertionPointToEnd(forallOp->getTerminator().getBody());

  for (auto [tiledValue, destinationTensor, resultOffset, resultSize] :
       llvm::zip_equal(tiledResults, destinationTensors, resultOffsets,
                       resultSizes)) {
    SmallVector<OpFoldResult> resultStride(resultOffset.size(),
                                           builder.getIndexAttr(1));

    tensor::ParallelInsertSliceOp::create(builder, loc, tiledValue,
                                          destinationTensor, resultOffset,
                                          resultSize, resultStride);
  }
  return success();
}

LogicalResult
ConditionalTransposeAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 int64_t nbXcds, int64_t nbCus) {
  if (nbXcds == 0 || nbCus == 0) {
    return emitError()
           << "conditional_transpose must have non-zeros parameters";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DerivedThreadConfigAttr
//===----------------------------------------------------------------------===//

SmallVector<int64_t>
DerivedThreadConfigAttr::getStaticTilingLevelSizes(unsigned level,
                                                   Operation *op) const {
  if (level != llvm::to_underlying(GPU::TilingLevel::Thread)) {
    return {};
  }
  return deriveThreadTileSizes(op);
}

SmallVector<OpFoldResult>
DerivedThreadConfigAttr::getTilingLevelSizes(OpBuilder &b, unsigned level,
                                             Operation *op) const {
  if (level > llvm::to_underlying(GPU::TilingLevel::Thread)) {
    return {};
  }
  SmallVector<int64_t> sizes = deriveThreadTileSizes(op);
  return llvm::map_to_vector(
      sizes, [&](int64_t s) -> OpFoldResult { return b.getIndexAttr(s); });
}

bool DerivedThreadConfigAttr::hasTilingLevel(unsigned level) const {
  return level == llvm::to_underlying(GPU::TilingLevel::Thread);
}

//===----------------------------------------------------------------------===//
// UseGlobalLoadDMAAttr
//===----------------------------------------------------------------------===//

SmallVector<int64_t>
UseGlobalLoadDMAAttr::getStaticTilingLevelSizes(unsigned level,
                                                Operation *op) const {
  if (level == llvm::to_underlying(GPU::TilingLevel::Subgroup)) {
    // Subgroup tile sizes are derived from translation_info, not stored here.
    return {};
  }
  if (level == llvm::to_underlying(GPU::TilingLevel::Thread)) {
    return globalLoadDMATileSizes(op);
  }
  return {};
}

SmallVector<OpFoldResult>
UseGlobalLoadDMAAttr::getTilingLevelSizes(OpBuilder &b, unsigned level,
                                          Operation *op) const {
  if (level == llvm::to_underlying(GPU::TilingLevel::Subgroup)) {
    // Subgroup tile sizes are derived from translation_info, not stored here.
    return {};
  }
  if (level == llvm::to_underlying(GPU::TilingLevel::Thread)) {
    SmallVector<int64_t> sizes = globalLoadDMATileSizes(op);
    return getAsIndexOpFoldResult(b.getContext(), sizes);
  }
  return {};
}

bool UseGlobalLoadDMAAttr::hasTilingLevel(unsigned level) const {
  // Subgroup level is not stored in this attribute anymore.
  return level == llvm::to_underlying(GPU::TilingLevel::Thread);
}

//===----------------------------------------------------------------------===//
// PromoteWithCacheSwizzleAttr
//===----------------------------------------------------------------------===//

Value PromoteWithCacheSwizzleAttr::promoteOperand(
    mlir::OpBuilder &builder, mlir::OpOperand &operand) const {
  return cacheSwizzlePromotionImpl(builder, operand, getCopyConfig());
}

//===----------------------------------------------------------------------===//
// LaneIdAttr
//===----------------------------------------------------------------------===//

int64_t LaneIdAttr::getMappingId() const { return getDim(); }

bool LaneIdAttr::isLinearMapping() const { return true; }

int64_t LaneIdAttr::getRelativeIndex() const { return getDim(); }

//===----------------------------------------------------------------------===//
// GPU Pipeline Options
//===----------------------------------------------------------------------===//

GPUPipelineOptionsAttr GPUPipelineOptionsAttr::get(
    MLIRContext *context, unsigned prefetchNumStages,
    bool noReduceSharedMemoryBankConflicts, bool useIgemmConvolution,
    std::optional<ReorderWorkgroupsStrategy> reorderWorkgroupsStrategy) {
  auto strategyAttr = ReorderWorkgroupsStrategyAttr();
  if (reorderWorkgroupsStrategy) {
    strategyAttr =
        ReorderWorkgroupsStrategyAttr::get(context, *reorderWorkgroupsStrategy);
  }
  Builder b(context);
  std::optional<int64_t> prefetchOpt;
  if (prefetchNumStages > 0) {
    prefetchOpt = prefetchNumStages;
  }
  return Base::get(context, prefetchOpt,
                   b.getBoolAttr(noReduceSharedMemoryBankConflicts),
                   b.getBoolAttr(useIgemmConvolution), strategyAttr);
}

//===----------------------------------------------------------------------===//
// Attribute Registration
//===----------------------------------------------------------------------===//

void IREEGPUDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::GPU
