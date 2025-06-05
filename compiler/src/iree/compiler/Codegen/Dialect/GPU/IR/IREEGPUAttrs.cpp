// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include <numeric>

#include "iree/compiler/Codegen/Dialect/GPU/IR/DerivedConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPUTileSwizzleUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"
#include "iree/compiler/Utils/Indexing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"

#define DEBUG_TYPE "iree-gpu-attrs"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.cpp.inc"

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

static int64_t getIntrinsicSubgroupSize(MMAIntrinsic intrinsic) {
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
    return {f32, f32, f32};
  case MMAIntrinsic::MFMA_F32_16x16x16_F16:
  case MMAIntrinsic::MFMA_F32_32x32x8_F16:
  case MMAIntrinsic::MFMA_F32_16x16x32_F16:
  case MMAIntrinsic::MFMA_F32_32x32x16_F16:
  case MMAIntrinsic::WMMAR3_F32_16x16x16_F16:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F16:
  case MMAIntrinsic::NV_WMMA_F32_16x16x16_F16:
    return {f16, f16, f32};
  case MMAIntrinsic::WMMAR3_F16_16x16x16_F16:
  case MMAIntrinsic::WMMAR4_F16_16x16x16_F16:
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
    return {bf16, bf16, f32};
  case MMAIntrinsic::WMMAR3_BF16_16x16x16_BF16:
  case MMAIntrinsic::WMMAR4_BF16_16x16x16_BF16:
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
    return {f8E5M2, f8E5M2, f32};
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_16x16x128_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_32x32x64_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F8E5M2_F8E4M3FN:
    return {f8E5M2, f8E4M3FN, f32};
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_16x16x128_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_32x32x64_F8E4M3FN:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F8E4M3FN:
    return {f8E4M3FN, f8E4M3FN, f32};
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::MFMA_F32_16x16x128_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::MFMA_F32_32x32x64_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F8E4M3FN_F8E5M2:
    return {f8E4M3FN, f8E5M2, f32};
  case MMAIntrinsic::MFMA_I32_16x16x16_I8:
  case MMAIntrinsic::MFMA_I32_32x32x8_I8:
  case MMAIntrinsic::MFMA_I32_16x16x32_I8:
  case MMAIntrinsic::MFMA_I32_32x32x16_I8:
  case MMAIntrinsic::MFMA_I32_16x16x64_I8:
  case MMAIntrinsic::MFMA_I32_32x32x32_I8:
  case MMAIntrinsic::WMMAR3_I32_16x16x16_I8:
  case MMAIntrinsic::WMMAR4_I32_16x16x16_I8:
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
                                                MMAFragment fragment) {
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

  switch (intrinsic) {
  case MMAIntrinsic::MFMA_F32_16x16x4_F32:
    switch (fragment) {
    case MMAFragment::Lhs:
      return mfmaLhs16xK(4);
    case MMAFragment::Rhs:
      return mfmaRhsKx16(4);
    case MMAFragment::Acc:
      return mfmaAcc16x16;
    }
  // Note: the returned layout for f64 differs than for other MFMAs
  case MMAIntrinsic::MFMA_F64_16x16x4_F64:
    switch (fragment) {
    case MMAFragment::Lhs:
      return mfmaLhs16xK(4);
    case MMAFragment::Rhs:
      return mfmaRhsKx16(4);
    case MMAFragment::Acc:
      return {/*outer=*/{4, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{1, 1}};
    }
  case MMAIntrinsic::MFMA_F32_16x16x8_BF16: {
    switch (fragment) {
    case MMAFragment::Lhs:
      return mfmaLhs16xK(8);
    case MMAFragment::Rhs:
      return mfmaRhsKx16(8);
    case MMAFragment::Acc:
      return mfmaAcc16x16;
    }
  }
  case MMAIntrinsic::MFMA_F32_32x32x4_BF16:
    switch (fragment) {
    case MMAFragment::Lhs:
      return mfmaLhs32xK(4);
    case MMAFragment::Rhs:
      return mfmaRhsKx32(4);
    case MMAFragment::Acc:
      return mfmaAcc32x32;
    }
  case MMAIntrinsic::MFMA_I32_16x16x16_I8:
  case MMAIntrinsic::MFMA_F32_16x16x16_F16:
  case MMAIntrinsic::MFMA_F32_16x16x16_BF16:
    switch (fragment) {
    case MMAFragment::Lhs:
      return mfmaLhs16xK(16);
    case MMAFragment::Rhs:
      return mfmaRhsKx16(16);
    case MMAFragment::Acc:
      return mfmaAcc16x16;
    }
  case MMAIntrinsic::MFMA_I32_32x32x8_I8:
  case MMAIntrinsic::MFMA_F32_32x32x8_F16:
  case MMAIntrinsic::MFMA_F32_32x32x8_BF16:
    switch (fragment) {
    case MMAFragment::Lhs:
      return mfmaLhs32xK(8);
    case MMAFragment::Rhs:
      return mfmaRhsKx32(8);
    case MMAFragment::Acc:
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
    switch (fragment) {
    case MMAFragment::Lhs:
      return mfmaLhs16xK(32);
    case MMAFragment::Rhs:
      return mfmaRhsKx16(32);
    case MMAFragment::Acc:
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
    switch (fragment) {
    case MMAFragment::Lhs:
      return mfmaLhs32xK(16);
    case MMAFragment::Rhs:
      return mfmaRhsKx32(16);
    case MMAFragment::Acc:
      return mfmaAcc32x32;
    }
  case MMAIntrinsic::MFMA_I32_16x16x64_I8:
    switch (fragment) {
    case MMAFragment::Lhs:
      return mfmaLhs16xK(64);
    case MMAFragment::Rhs:
      return mfmaRhsKx16(64);
    case MMAFragment::Acc:
      return mfmaAcc16x16;
    }
  case MMAIntrinsic::MFMA_I32_32x32x32_I8:
    switch (fragment) {
    case MMAFragment::Lhs:
      return mfmaLhs32xK(32);
    case MMAFragment::Rhs:
      return mfmaRhsKx32(32);
    case MMAFragment::Acc:
      return mfmaAcc32x32;
    }
  case MMAIntrinsic::MFMA_F32_16x16x128_F8E5M2:
  case MMAIntrinsic::MFMA_F32_16x16x128_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_16x16x128_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_16x16x128_F8E4M3FN_F8E5M2:
    switch (fragment) {
    case MMAFragment::Lhs:
      return mfmaLhs16xK(128);
    case MMAFragment::Rhs:
      return mfmaRhsKx16(128);
    case MMAFragment::Acc:
      return mfmaAcc16x16;
    }
  case MMAIntrinsic::MFMA_F32_32x32x64_F8E5M2:
  case MMAIntrinsic::MFMA_F32_32x32x64_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_32x32x64_F8E4M3FN:
  case MMAIntrinsic::MFMA_F32_32x32x64_F8E4M3FN_F8E5M2:
    switch (fragment) {
    case MMAFragment::Lhs:
      return mfmaLhs32xK(64);
    case MMAFragment::Rhs:
      return mfmaRhsKx32(64);
    case MMAFragment::Acc:
      return mfmaAcc32x32;
    }

  case MMAIntrinsic::WMMAR3_F32_16x16x16_F16:
  case MMAIntrinsic::WMMAR3_F32_16x16x16_BF16:
  case MMAIntrinsic::WMMAR3_I32_16x16x16_I8:
    switch (fragment) {
    case MMAFragment::Lhs:
      return {/*outer=*/{1, 1}, /*thread=*/{16, 1}, /*strides=*/{1, 0},
              /*element=*/{1, 16}};
    case MMAFragment::Rhs:
      return {/*outer=*/{1, 1}, /*thread=*/{1, 16}, /*tstrides=*/{0, 1},
              /*element=*/{16, 1}};
    case MMAFragment::Acc:
      return {/*outer=*/{8, 1}, /*thread=*/{2, 16}, /*tstrides=*/{16, 1},
              /*element=*/{1, 1}};
    }
  case MMAIntrinsic::WMMAR3_F16_16x16x16_F16:
  case MMAIntrinsic::WMMAR3_BF16_16x16x16_BF16:
    switch (fragment) {
    case MMAFragment::Lhs:
      return {/*outer=*/{1, 1}, /*thread=*/{16, 1}, /*strides=*/{1, 0},
              /*element=*/{1, 16}};
    case MMAFragment::Rhs:
      return {/*outer=*/{1, 1}, /*thread=*/{1, 16}, /*tstrides=*/{0, 1},
              /*element=*/{16, 1}};
    case MMAFragment::Acc:
      return {/*outer=*/{16, 1}, /*thread=*/{1, 16}, /*tstrides=*/{0, 1},
              /*element=*/{1, 1}};
    }
  // Note: We specify here that, for examplee, lane 0 takes A[0, 0..7] and that
  // lane 16 takes A[0, 8..15]. The hardware will internally bounce between
  // the low halves and high halves of lanes every two registers - that is,
  // the value used for A[0, 4] comes out of lane 16's first register in an F16
  // or BF16 computation. This is noted here in case someone starts chasing
  // some unusual rounding failure or is confused by why the tiling in the
  // manual doesn't *technically* match the below.
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F16:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_BF16:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F8E5M2:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F8E5M2_F8E4M3FN:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F8E4M3FN:
  case MMAIntrinsic::WMMAR4_F32_16x16x16_F8E4M3FN_F8E5M2:
  case MMAIntrinsic::WMMAR4_I32_16x16x16_I8:
    switch (fragment) {
    case MMAFragment::Lhs:
      return {/*outer=*/{1, 1}, /*thread=*/{16, 2}, /*strides=*/{1, 16},
              /*element=*/{1, 8}};
    case MMAFragment::Rhs:
      return {/*outer=*/{1, 1}, /*thread=*/{2, 16}, /*tstrides=*/{16, 1},
              /*element=*/{8, 1}};
    case MMAFragment::Acc:
      return {/*outer=*/{1, 1}, /*thread=*/{2, 16}, /*tstrides=*/{16, 1},
              /*element=*/{8, 1}};
    }
  case MMAIntrinsic::WMMAR4_F16_16x16x16_F16:
  case MMAIntrinsic::WMMAR4_BF16_16x16x16_BF16:
    switch (fragment) {
    case MMAFragment::Lhs:
      return {/*outer=*/{1, 1}, /*thread=*/{16, 2}, /*strides=*/{1, 16},
              /*element=*/{1, 8}};
    case MMAFragment::Rhs:
      return {/*outer=*/{1, 1}, /*thread=*/{2, 16}, /*tstrides=*/{16, 1},
              /*element=*/{8, 1}};
    case MMAFragment::Acc:
      return {/*outer=*/{1, 1}, /*thread=*/{2, 16}, /*tstrides=*/{16, 1},
              /*element=*/{8, 1}};
    }
  case MMAIntrinsic::NV_WMMA_F32_16x16x16_F16:
  case MMAIntrinsic::NV_WMMA_F16_16x16x16_F16:
    return {};
  }
  assert(false && "unexpected enum value");
  return {};
}

MMASingleSubgroupLayout getSingleSubgroupLayout(MMAIntrinsic intrinsic,
                                                MMAFragment fragment,
                                                bool colMajor) {
  MMASingleSubgroupLayout baseLayout =
      getSingleSubgroupLayout(intrinsic, fragment);
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
    auto lhs = getSingleSubgroupLayout(intrinsic, MMAFragment::Lhs);
    auto rhs = getSingleSubgroupLayout(intrinsic, MMAFragment::Rhs);
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

MMASingleSubgroupLayout getSingleSubgroupLayout(MmaInterfaceAttr mmaKind,
                                                MMAFragment fragment) {
  if (auto mmaAttr = dyn_cast<MMAAttr>(mmaKind)) {
    // |colMajor| indicates that the accumulator layout should be returned
    // column major.
    return getSingleSubgroupLayout(mmaAttr.getIntrinsic(), fragment,
                                   fragment == MMAFragment::Acc &&
                                       mmaAttr.getColMajor());
  }
  if (auto vmmaAttr = dyn_cast<VirtualMMAAttr>(mmaKind)) {
    return getSingleSubgroupLayout(vmmaAttr.getIntrinsic(), fragment);
  }
  assert(false && "unhandled MMA Interface type.");
  return {};
}

//===----------------------------------------------------------------------===//
// MMA Attributes
//===----------------------------------------------------------------------===//

MMAAttr MMAAttr::get(MLIRContext *context, MMAIntrinsic type) {
  return Base::get(context, type, /*colMajor=*/false);
}

int64_t MMAAttr::getExpectedNumInputs() const { return 2; }

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
                                      MMAFragment fragment) {
  auto o = getOpaqueMMALayout(context, intrinsic);
  auto s = getSingleSubgroupLayout(intrinsic, fragment);
  Type elemType = (fragment == MMAFragment::Lhs)   ? o.aType
                  : (fragment == MMAFragment::Rhs) ? o.bType
                                                   : o.cType;
  return VectorType::get(
      {s.element[0] * s.element[1] * s.outer[0] * s.outer[1]}, elemType);
}

void MMAAttr::getDistributedTileTypes(
    SmallVectorImpl<VectorType> &result) const {
  MLIRContext *context = getContext();
  MMAIntrinsic intrinsic = getIntrinsic();
  result.assign({getThreadVectorType(context, intrinsic, MMAFragment::Lhs),
                 getThreadVectorType(context, intrinsic, MMAFragment::Rhs),
                 getThreadVectorType(context, intrinsic, MMAFragment::Acc)});
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
    bool one = llvm::cast<VectorType>(vec.getType()).getNumElements() == 1;
    return one ? builder.create<vector::ExtractOp>(loc, vec, 0) : vec;
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
    return builder
        .create<amdgpu::MFMAOp>(loc, resultType, layout.mSize, layout.nSize,
                                layout.kSize, getBlockSize(intrinsic), lhs, rhs,
                                acc)
        .getResult();
  }
  if (is_AMD_WMMA(intrinsic)) {
    return builder.create<amdgpu::WMMAOp>(loc, resultType, lhs, rhs, acc)
        .getResult();
  }
  return {};
}

// Generates amdgpu.mfma/wmma operation on the given inputs for this attribute
// type.
FailureOr<Value> MMAAttr::buildMmaOperation(OpBuilder &builder, Location loc,
                                            Type resultType, Value lhs,
                                            Value rhs, Value acc) const {
  auto [aType, bType, cType] = getABCVectorTypes();
  if (aType != lhs.getType() || bType != rhs.getType() ||
      cType != acc.getType()) {
    return failure();
  }
  // Fail if the result type does not match with the expected return type of
  // the intrinsic. We expect the caller to handle type conversions externally.
  if (cType != resultType) {
    return failure();
  }
  if (Value value = createMmaOp(builder, loc, getIntrinsic(), resultType, lhs,
                                rhs, acc, getColMajor())) {
    return value;
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
  Value cZero = builder.create<arith::ConstantIndexOp>(loc, 0);
  canonicalStrides.append(rankReducedShape.size(), one);

  SmallVector<Value> vtids;
  SmallVector<int64_t> vtidBasis;
  SmallVector<size_t> dimToVtid;
  if (failed(basisFromSizesStrides(subgroupLayout.thread,
                                   subgroupLayout.tstrides, vtidBasis,
                                   dimToVtid))) {
    return failure();
  }
  auto splitLaneId = builder.create<affine::AffineDelinearizeIndexOp>(
      loc, laneId, vtidBasis, /*hasOuterBound=*/false);

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
      vtid = builder.create<affine::AffineLinearizeIndexOp>(
          loc, ValueRange{vtid, cZero}, ArrayRef<int64_t>{vtidLen, element},
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
  auto fragment = static_cast<IREE::GPU::MMAFragment>(operandIndex);
  MMASingleSubgroupLayout subgroupLayout = getSingleSubgroupLayout(
      getIntrinsic(), fragment, fragment == MMAFragment::Acc && getColMajor());
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

/// Returns the swizzled tile shape, but with dim sizes overwritten with 1 if
/// `predicate` returns false.
static SmallVector<int64_t>
sliceSwizzledShape(const TileSwizzle &swizzle,
                   std::function<bool(TileSwizzle::Dim)> predicate) {
  SmallVector<int64_t> shape;
  for (TileSwizzle::ExpandShapeDimVectorType e : swizzle.expandShape) {
    for (TileSwizzle::Dim d : e) {
      shape.push_back(predicate(d) ? d.size : 1);
    }
  }
  applyPermutationToVector(shape, swizzle.permutation);
  return shape;
}

int64_t DataTiledMMAAttr::getExpectedNumInputs() const { return 2; }

LogicalResult
DataTiledMMAAttr::verifyIndexingMaps(ArrayRef<AffineMap> maps) const {
  return verifyMmaIndexingMaps(maps);
}

void DataTiledMMAAttr::getUndistributedTileTypes(
    SmallVectorImpl<VectorType> &result) const {
  MLIRContext *ctx = getContext();
  OpaqueMmaLayout o = getOpaqueMMALayout(ctx, getIntrinsic());
  int64_t m = o.mSize * getIntrinsicsM() * getSubgroupsM();
  int64_t n = o.nSize * getIntrinsicsN() * getSubgroupsN();
  int64_t k = o.kSize * getIntrinsicsK();
  result.push_back(VectorType::get({m, k}, o.aType));
  result.push_back(VectorType::get({k, n}, o.bType));
  result.push_back(VectorType::get({m, n}, o.cType));
}

void DataTiledMMAAttr::getDistributedTileTypes(
    SmallVectorImpl<VectorType> &result) const {
  auto [A, B, C] = getABCElementTypes();
  auto getShape = [=](MMAFragment fragment) {
    return sliceSwizzledShape(
        getSwizzle(*this, fragment), [](TileSwizzle::Dim d) {
          return d.kind != TileSwizzle::Dim::Kind::CrossThread;
        });
  };
  result.assign({VectorType::get(getShape(MMAFragment::Lhs), A),
                 VectorType::get(getShape(MMAFragment::Rhs), B),
                 VectorType::get(getShape(MMAFragment::Acc), C)});
}

int64_t DataTiledMMAAttr::getSubgroupSize() const {
  return getIntrinsicSubgroupSize(getIntrinsic());
}

Attribute DataTiledMMAAttr::getDistributionMappingKind() const {
  return gpu::GPUThreadMappingAttr::get(getContext(),
                                        gpu::MappingId::LinearDim0);
}

LogicalResult DataTiledMMAAttr::populateOperandOffsetsSizesStrides(
    OpBuilder &builder, Location loc, uint32_t operandIndex, Value threadId,
    ArrayRef<int64_t> permutation, SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &sizes,
    SmallVectorImpl<OpFoldResult> &strides) const {
  assert(operandIndex <= 2 && "Must index valid MMA operand");
  auto fragment = static_cast<IREE::GPU::MMAFragment>(operandIndex);
  TileSwizzle swizzle = getSwizzle(*this, fragment);

  LLVM_DEBUG({
    DBGS() << "DataTiledMMAAttr::populateOperandOffsetsSizesStrides\n";
    DBGS() << "    fragment: " << llvm::to_underlying(fragment) << "\n";
    DBGS() << "    swizzle: " << swizzle << "\n";
  });

  MLIRContext *ctx = builder.getContext();
  SmallVector<OpFoldResult> tileSizes = getAsIndexOpFoldResult(
      ctx, sliceSwizzledShape(swizzle, [](TileSwizzle::Dim d) {
        return d.kind != TileSwizzle::Dim::Kind::CrossThread;
      }));

  // Most of the rest of this function is the computation of the offsets.
  // The basic idea is to delinearize the threadId over the basis of
  // cross-thread dimensions. These cross-thread dimensions may be either
  // the intrinsic's own, or they may come from expansion to multiple subgroups.
  // Normally, that distinction is irrelevant here: we just delinearize the
  // thread-id over all cross-thread dimensions.
  //
  // There is one case that makes things more complicated, encountered so far
  // only on RDNA3. That is when some intrinsic has multiple (so far, 2) threads
  // reading the same data. This redundancy is not encoded in the TileSwizzle
  // structures that we are using here. Instead, in that case, the thread grid
  // (as encoded in the TileSwizzle) is smaller than the subgroup size. In that
  // case, there is an implied thread-distribution-only dimension along which
  // multiple threads read exactly the same data.
  // So we need to distinguish layoutThreadSizes vs. distributionThreadSizes.
  SmallVector<int64_t> layoutThreadSizes =
      sliceSwizzledShape(swizzle, [](TileSwizzle::Dim d) {
        return d.kind == TileSwizzle::Dim::Kind::CrossThread;
      });
  // In layoutThreadSizes, intrinsic level dimensions are mixed with expansion
  // to multiple subgroups, so in order to tell if there are additional
  // distribution-only thread dimensions, we need to get back to the intrinsic.
  TileSwizzle intrinsicSwizzle = getIntrinsicSwizzle(getIntrinsic(), fragment);

  SmallVector<int64_t> intrinsicLayoutThreadSizes =
      sliceSwizzledShape(intrinsicSwizzle, [](TileSwizzle::Dim d) {
        return d.kind == TileSwizzle::Dim::Kind::CrossThread;
      });
  int64_t intrinsicLayoutThreadBound =
      ShapedType::getNumElements(intrinsicLayoutThreadSizes);
  SmallVector<int64_t> distributionThreadSizes = layoutThreadSizes;
  int distributionOnlyDimIdx =
      distributionThreadSizes.size() - intrinsicLayoutThreadSizes.size();
  // Now we are able to tell if there is an extra distribution-only dimension.
  bool hasDistributionOnlyDim = intrinsicLayoutThreadBound < getSubgroupSize();
  if (hasDistributionOnlyDim) {
    // Insert the extra distribution-only dimension. This will need to be paired
    // below with erasing the corresponding dim out of the delinearized indices.
    distributionThreadSizes.insert(
        distributionThreadSizes.begin() + distributionOnlyDimIdx,
        getSubgroupSize() / intrinsicLayoutThreadBound);
  }

  // Obtain the offsets from delinearization along the distributionThreadSizes.
  // Use a delinearize without outer bound and throw away its initial result
  // to get clamping behavior.
  SmallVector<OpFoldResult> tileOffsets =
      builder
          .create<affine::AffineDelinearizeIndexOp>(
              loc, getValueOrCreateConstantIndexOp(builder, loc, threadId),
              distributionThreadSizes, /*hasOuterBound=*/false)
          ->getResults()
          .drop_front();

  if (hasDistributionOnlyDim) {
    // Erase the delinearized index that corresponds to the extra distribution
    // dimension that we had inserted above. This is what causes multiple
    // threads (which only differed in the index being discarded here) to read
    // exactly the same data.
    tileOffsets.erase(tileOffsets.begin() + distributionOnlyDimIdx);
  }

  // Strides are trivial: each slice is contiguous along the *expanded* dims
  // even if it may not be contiguous in the flattened layout.
  SmallVector<OpFoldResult> tileStrides(tileSizes.size(),
                                        builder.getIndexAttr(1));

  offsets.append(tileOffsets);
  sizes.append(tileSizes);
  strides.append(tileStrides);

  return success();
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
  VectorType vectorType = llvm::dyn_cast<VectorType>(type);
  assert(vectorType);
  if (vectorType.getRank() <= 1) {
    return value;
  }
  auto flatVectorType = VectorType::get({vectorType.getNumElements()},
                                        vectorType.getElementType());
  return builder.create<vector::ShapeCastOp>(loc, flatVectorType, value);
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
  LDBG("crossIntrinsicShape: " << llvm::interleaved(crossIntrinsicShape));
  int rank = internalShape.size();
  SmallVector<int64_t> indices(rank, 0);
  SmallVector<int64_t> strides(rank, 1);
  SmallVector<Value> distributedValues;
  do {
    Value extract = builder.create<vector::ExtractStridedSliceOp>(
        loc, value, indices, internalShape, strides);
    distributedValues.push_back(flattenVector(builder, loc, extract));
  } while (incrementIndices(indices, crossIntrinsicShape));
  return distributedValues;
}

FailureOr<Value> DataTiledMMAAttr::buildMmaOperation(OpBuilder &builder,
                                                     Location loc,
                                                     Type resultType, Value lhs,
                                                     Value rhs,
                                                     Value acc) const {
  // Validation. Similar to MMAAttr::buildMmaOperation.
  auto [aType, bType, cType] = getABCVectorTypes();
  if (aType != lhs.getType() || bType != rhs.getType() ||
      cType != acc.getType()) {
    return failure();
  }
  // Fail if the result type does not match with the expected return type of
  // the intrinsic. We expect the caller to handle type conversions externally.
  if (cType != resultType) {
    return failure();
  }

  // Prepare Lhs/Rhs/Acc operand slices to feed the intrinsic.
  TileSwizzle lhsSwizzle = getSwizzle(*this, MMAFragment::Lhs);
  LDBG("DataTiledMMAAttr::buildMmaOperation");
  LDBG("    lhsSwizzle: " << lhsSwizzle);
  SmallVector<Value> intrinsicsLhs =
      distributeMmaFragmentToIntrinsics(builder, loc, lhs, lhsSwizzle);

  TileSwizzle rhsSwizzle = getSwizzle(*this, MMAFragment::Rhs);
  LDBG("DataTiledMMAAttr::buildMmaOperation");
  LDBG("    rhsSwizzle: " << rhsSwizzle);
  SmallVector<Value> intrinsicsRhs =
      distributeMmaFragmentToIntrinsics(builder, loc, rhs, rhsSwizzle);

  TileSwizzle accSwizzle = getSwizzle(*this, MMAFragment::Acc);
  LDBG("DataTiledMMAAttr::buildMmaOperation");
  LDBG("    accSwizzle: " << accSwizzle);

  SmallVector<Value> intrinsicsAcc =
      distributeMmaFragmentToIntrinsics(builder, loc, acc, accSwizzle);

  MMAIntrinsic intrinsic = getIntrinsic();
  VectorType intrinCType =
      getThreadVectorType(builder.getContext(), intrinsic, MMAFragment::Acc);

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

  LDBG("accCrossIntrinsicShape: " << llvm::interleaved(accCrossIntrinsicShape));
  LDBG("accInternalShape: " << llvm::interleaved(accInternalShape));
  int dstRank = accCrossIntrinsicShape.size();
  SmallVector<int64_t> strides(dstRank, 1);
  SmallVector<int64_t> indices(dstRank, 0);
  for (Value intrAcc : intrinsicsAcc) {
    auto expandedAcc = builder.create<vector::ShapeCastOp>(
        loc, VectorType::get(accInternalShape, cType.getElementType()),
        intrAcc);
    acc = builder.create<vector::InsertStridedSliceOp>(loc, expandedAcc, acc,
                                                       indices, strides);
    incrementIndices(indices, accCrossIntrinsicShape);
  }
  return acc;
}

//===----------------------------------------------------------------------===//
// VirtualMMA Attributes
//===----------------------------------------------------------------------===//

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
  auto lhs = getSingleSubgroupLayout(intrinsic, MMAFragment::Lhs);
  auto rhs = getSingleSubgroupLayout(intrinsic, MMAFragment::Rhs);
  o.mSize = lhs.outer[0] * lhs.thread[0] * lhs.element[0];
  o.kSize = lhs.outer[1] * lhs.thread[1] * lhs.element[1];
  o.nSize = rhs.outer[1] * rhs.thread[1] * rhs.element[1];
  return o;
}

int64_t VirtualMMAAttr::getExpectedNumInputs() const { return 2; }

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
  result.assign({getThreadVectorType(context, intrinsic, MMAFragment::Lhs),
                 getThreadVectorType(context, intrinsic, MMAFragment::Rhs),
                 getThreadVectorType(context, intrinsic, MMAFragment::Acc)});
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

LogicalResult VirtualMMAAttr::populateOperandOffsetsSizesStrides(
    OpBuilder &builder, Location loc, uint32_t operandIndex, Value laneId,
    ArrayRef<int64_t> permutation, SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &sizes,
    SmallVectorImpl<OpFoldResult> &strides) const {
  assert(operandIndex <= 2 && "Must index valid MMA operand");
  auto fragment = static_cast<IREE::GPU::MMAFragment>(operandIndex);
  MMASingleSubgroupLayout subgroupLayout =
      getSingleSubgroupLayout(getIntrinsic(), fragment);
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
FailureOr<Value> VirtualMMAAttr::buildMmaOperation(OpBuilder &builder,
                                                   Location loc,
                                                   Type resultType, Value lhs,
                                                   Value rhs, Value acc) const {
  auto [aType, bType, cType] = getABCVectorTypes();
  if (aType != lhs.getType() || bType != rhs.getType() ||
      cType != acc.getType()) {
    return failure();
  }
  // Fail if the result type does not match with the expected return type of
  // the intrinsic. We expect the caller to handle type conversions externally.
  if (cType != resultType) {
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
    for (int i = 0; i < unrollKFactor; i++) {
      int64_t offset = vectorWidth * i;
      Value sliced_lhs = builder.create<vector::ExtractStridedSliceOp>(
          loc, lhs, ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{vectorWidth},
          ArrayRef<int64_t>{1});
      Value sliced_rhs = builder.create<vector::ExtractStridedSliceOp>(
          loc, rhs, ArrayRef<int64_t>{offset}, ArrayRef<int64_t>{vectorWidth},
          ArrayRef<int64_t>{1});
      acc = builder
                .create<amdgpu::MFMAOp>(loc, resultType, m, n, nativeKSize,
                                        getBlockSize(), sliced_lhs, sliced_rhs,
                                        acc)
                .getResult();
    }
    return acc;
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
                                                MMAFragment fragment) {
  switch (intrinsic) {
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F16:
    switch (fragment) {
    case MMAFragment::Lhs:
      return {/*outer=*/{1, 1}, /*thread=*/{16, 4}, /*tstrides=*/{1, 16},
              /*element=*/{1, 8}};
    case MMAFragment::Rhs:
      return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{8, 1}};
    case MMAFragment::Acc:
      return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{4, 1}};
    }
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ:
    switch (fragment) {
    case MMAFragment::Lhs:
      return {/*outer=*/{1, 2}, /*thread=*/{16, 4}, /*tstrides=*/{1, 16},
              /*element=*/{1, 4}};
    case MMAFragment::Rhs:
      return {/*outer=*/{2, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{4, 1}};
    case MMAFragment::Acc:
      return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{4, 1}};
    }
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F16:
    switch (fragment) {
    case MMAFragment::Lhs:
      return {/*outer=*/{1, 1}, /*thread=*/{32, 2}, /*tstrides=*/{1, 32},
              /*element=*/{1, 8}};
    case MMAFragment::Rhs:
      return {/*outer=*/{1, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
              /*element=*/{8, 1}};
    case MMAFragment::Acc:
      return {/*outer=*/{4, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
              /*element=*/{4, 1}};
    }
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F8E4M3FNUZ:
    switch (fragment) {
    case MMAFragment::Lhs:
      return {/*outer=*/{1, 2}, /*thread=*/{32, 2}, /*tstrides=*/{1, 32},
              /*element=*/{1, 4}};
    case MMAFragment::Rhs:
      return {/*outer=*/{2, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
              /*element=*/{4, 1}};
    case MMAFragment::Acc:
      return {/*outer=*/{4, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
              /*element=*/{4, 1}};
    }
  }
  assert(false && "unhandled virtual mma layout type.");
  return {};
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

//===----------------------------------------------------------------------===//
// Lowering Config Attributes
//===----------------------------------------------------------------------===//

constexpr StringLiteral kWorkgroupLevelName = "workgroup";
constexpr StringLiteral kPartialReductionLevelName = "partial_reduction";
constexpr StringLiteral kReductionLevelName = "reduction";
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
  if (level != llvm::to_underlying(GPU::TilingLevel::Thread)) {
    return {};
  }
  return globalLoadDMATileSizes(op);
}

SmallVector<OpFoldResult>
UseGlobalLoadDMAAttr::getTilingLevelSizes(OpBuilder &b, unsigned level,
                                          Operation *op) const {
  if (level > llvm::to_underlying(GPU::TilingLevel::Subgroup)) {
    return {};
  }
  SmallVector<int64_t> sizes = globalLoadDMATileSizes(op);
  return getAsIndexOpFoldResult(b.getContext(), sizes);
}

bool UseGlobalLoadDMAAttr::hasTilingLevel(unsigned level) const {
  return level == llvm::to_underlying(GPU::TilingLevel::Subgroup);
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
    MLIRContext *context, bool prefetchSharedMemory,
    bool noReduceSharedMemoryBankConflicts, bool useIgemmConvolution,
    std::optional<ReorderWorkgroupsStrategy> reorderWorkgroupsStrategy) {
  auto strategyAttr = ReorderWorkgroupsStrategyAttr();
  if (reorderWorkgroupsStrategy) {
    strategyAttr =
        ReorderWorkgroupsStrategyAttr::get(context, *reorderWorkgroupsStrategy);
  }
  Builder b(context);
  return Base::get(context, b.getBoolAttr(prefetchSharedMemory),
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
