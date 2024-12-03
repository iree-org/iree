// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include <cfloat>
#include <numeric>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/DerivedConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPUTileSwizzleUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
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

using ::mlir::iree_compiler::IREE::Codegen::MaterializeEncodingInfo;
using ::mlir::iree_compiler::IREE::Codegen::TileMxNxK;
using ::mlir::iree_compiler::IREE::Codegen::TileSwizzle;

//===----------------------------------------------------------------------===//
// MMA intrinsics semantics: shapes, layouts, operand element types.
//===----------------------------------------------------------------------===//

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

static int64_t getIntrinsicSubgroupSize(MMAIntrinsic intrinsic) {
  // Not using Wave64 at all at the moment, so the only place where the
  // subgroup size is 64 is on CDNA* architectures.
  return is_AMD_MFMA(intrinsic) ? 64 : 32;
}

static std::tuple<Type, Type, Type> getABCElementTypes(MLIRContext *context,
                                                       MMAIntrinsic intrinsic) {
  Type f8E4M3FNUZ = Float8E4M3FNUZType::get(context);
  Type f8E5M2FNUZ = Float8E5M2FNUZType::get(context);
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
  case MMAIntrinsic::WMMA_F32_16x16x16_F16:
  case MMAIntrinsic::NV_WMMA_F32_16x16x16_F16:
    return {f16, f16, f32};
  case MMAIntrinsic::WMMA_F16_16x16x16_F16:
  case MMAIntrinsic::NV_WMMA_F16_16x16x16_F16:
    return {f16, f16, f16};
  case MMAIntrinsic::MFMA_F32_16x16x8_BF16:
  case MMAIntrinsic::MFMA_F32_32x32x4_BF16:
  case MMAIntrinsic::MFMA_F32_16x16x16_BF16:
  case MMAIntrinsic::MFMA_F32_32x32x8_BF16:
  case MMAIntrinsic::WMMA_F32_16x16x16_BF16:
    return {bf16, bf16, f32};
  case MMAIntrinsic::WMMA_BF16_16x16x16_BF16:
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
  case MMAIntrinsic::MFMA_I32_16x16x16_I8:
  case MMAIntrinsic::MFMA_I32_32x32x8_I8:
  case MMAIntrinsic::MFMA_I32_16x16x32_I8:
  case MMAIntrinsic::MFMA_I32_32x32x16_I8:
  case MMAIntrinsic::WMMA_I32_16x16x16_I8:
    return {i8, i8, i32};
  }
  assert(false && "unexpected enum value");
  return {};
}

MMASingleSubgroupLayout getSingleSubgroupLayout(MMAIntrinsic intrinsic,
                                                MMAFragment fragment) {
  switch (intrinsic) {
  case MMAIntrinsic::MFMA_F32_16x16x4_F32:
    switch (fragment) {
    case MMAFragment::Lhs:
      return {/*outer=*/{1, 1}, /*thread=*/{16, 4}, /*tstrides=*/{1, 16},
              /*element=*/{1, 1}};
    case MMAFragment::Rhs:
      return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{1, 1}};
    case MMAFragment::Acc:
      return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{4, 1}};
    }
  case MMAIntrinsic::MFMA_F64_16x16x4_F64:
    switch (fragment) {
    case MMAFragment::Lhs:
      return {/*outer=*/{1, 1}, /*thread=*/{16, 4}, /*tstrides=*/{1, 16},
              /*element=*/{1, 1}};
    case MMAFragment::Rhs:
      return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{1, 1}};
    case MMAFragment::Acc:
      return {/*outer=*/{4, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{1, 1}};
    }
  case MMAIntrinsic::MFMA_F32_16x16x8_BF16: {
    switch (fragment) {
    case MMAFragment::Lhs:
      return {/*outer=*/{1, 1}, /*thread=*/{16, 4}, /*tstrides=*/{1, 16},
              /*element=*/{1, 2}};
    case MMAFragment::Rhs:
      return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{2, 1}};
    case MMAFragment::Acc:
      return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{4, 1}};
    }
  }
  case MMAIntrinsic::MFMA_F32_32x32x4_BF16:
    switch (fragment) {
    case MMAFragment::Lhs:
      return {/*outer=*/{1, 1}, /*thread=*/{32, 2}, /*tstrides=*/{1, 32},
              /*element=*/{1, 2}};
    case MMAFragment::Rhs:
      return {/*outer=*/{1, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
              /*element=*/{2, 1}};
    case MMAFragment::Acc:
      return {/*outer=*/{4, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
              /*element=*/{4, 1}};
    }
  case MMAIntrinsic::MFMA_I32_16x16x16_I8:
  case MMAIntrinsic::MFMA_F32_16x16x16_F16:
  case MMAIntrinsic::MFMA_F32_16x16x16_BF16:
    switch (fragment) {
    case MMAFragment::Lhs:
      return {/*outer=*/{1, 1}, /*thread=*/{16, 4}, /*tstrides=*/{1, 16},
              /*element=*/{1, 4}};
    case MMAFragment::Rhs:
      return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{4, 1}};
    case MMAFragment::Acc:
      return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*tstrides=*/{16, 1},
              /*element=*/{4, 1}};
    }
  case MMAIntrinsic::MFMA_I32_32x32x8_I8:
  case MMAIntrinsic::MFMA_F32_32x32x8_F16:
  case MMAIntrinsic::MFMA_F32_32x32x8_BF16:
    switch (fragment) {
    case MMAFragment::Lhs:
      return {/*outer=*/{1, 1}, /*thread=*/{32, 2}, /*tstrides=*/{1, 32},
              /*element=*/{1, 4}};
    case MMAFragment::Rhs:
      return {/*outer=*/{1, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
              /*element=*/{4, 1}};
    case MMAFragment::Acc:
      return {/*outer=*/{4, 1}, /*thread=*/{2, 32}, /*tstrides=*/{32, 1},
              /*element=*/{4, 1}};
    }
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2FNUZ:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_I32_16x16x32_I8:
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
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E5M2FNUZ:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ:
  case MMAIntrinsic::MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_I32_32x32x16_I8:
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
  case MMAIntrinsic::WMMA_F32_16x16x16_F16:
  case MMAIntrinsic::WMMA_F32_16x16x16_BF16:
  case MMAIntrinsic::WMMA_I32_16x16x16_I8:
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
  case MMAIntrinsic::WMMA_F16_16x16x16_F16:
  case MMAIntrinsic::WMMA_BF16_16x16x16_BF16:
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
  case MMAIntrinsic::NV_WMMA_F32_16x16x16_F16:
  case MMAIntrinsic::NV_WMMA_F16_16x16x16_F16:
    return {};
  }
  assert(false && "unexpected enum value");
  return {};
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

template <typename MMAIntrinsicType>
static OpaqueMmaLayout getOpaqueMMALayout(MLIRContext *context,
                                          MMAIntrinsicType intrinsic) {
  OpaqueMmaLayout o;
  std::tie(o.aType, o.bType, o.cType) = getABCElementTypes(context, intrinsic);
  auto lhs = getSingleSubgroupLayout(intrinsic, MMAFragment::Lhs);
  auto rhs = getSingleSubgroupLayout(intrinsic, MMAFragment::Rhs);
  o.mSize = lhs.outer[0] * lhs.thread[0] * lhs.element[0];
  o.kSize = lhs.outer[1] * lhs.thread[1] * lhs.element[1];
  o.nSize = rhs.outer[1] * rhs.thread[1] * rhs.element[1];
  return o;
}

MMASingleSubgroupLayout getSingleSubgroupLayout(MmaInterfaceAttr mmaKind,
                                                MMAFragment fragment) {
  if (auto mmaAttr = dyn_cast<MMAAttr>(mmaKind)) {
    return getSingleSubgroupLayout(mmaAttr.getIntrinsic().getValue(), fragment);
  }
  if (auto vmmaAttr = dyn_cast<VirtualMMAAttr>(mmaKind)) {
    return getSingleSubgroupLayout(vmmaAttr.getIntrinsic().getValue(),
                                   fragment);
  }
  assert(false && "unhandled MMA Interface type.");
  return {};
}

//===----------------------------------------------------------------------===//
// MMA Attributes
//===----------------------------------------------------------------------===//

Attribute MMAAttr::parse(AsmParser &p, Type type) {
  if (failed(p.parseLess()))
    return {};

  FailureOr<MMAIntrinsicAttr> mmaIntrinsic =
      FieldParser<MMAIntrinsicAttr>::parse(p);
  if (failed(mmaIntrinsic)) {
    p.emitError(p.getCurrentLocation(), "failed to parse mfma type identifier");
    return {};
  }

  if (failed(p.parseGreater()))
    return {};

  return get(p.getContext(), mmaIntrinsic->getValue());
}

void MMAAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  os << stringifyMMAIntrinsic(getIntrinsic().getValue());
  os << ">";
}

MMAAttr MMAAttr::get(MLIRContext *context, MMAIntrinsic type) {
  auto layout = getOpaqueMMALayout(context, type);
  return Base::get(context, MMAIntrinsicAttr::get(context, type), layout.mSize,
                   layout.nSize, layout.kSize, layout.aType, layout.bType,
                   layout.cType);
}

std::tuple<Type, Type, Type> MMAAttr::getABCElementTypes() const {
  return {getAType(), getBType(), getCType()};
}

std::tuple<int64_t, int64_t, int64_t> MMAAttr::getMNKShape() const {
  return {getMSize(), getNSize(), getKSize()};
}

template <typename MMAIntrinsicType>
static VectorType getVectorType(MLIRContext *context,
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

std::tuple<VectorType, VectorType, VectorType>
MMAAttr::getABCVectorTypes() const {
  MLIRContext *context = getContext();
  MMAIntrinsic intrinsic = getIntrinsic().getValue();
  VectorType aVecType = getVectorType(context, intrinsic, MMAFragment::Lhs);
  VectorType bVecType = getVectorType(context, intrinsic, MMAFragment::Rhs);
  VectorType cVecType = getVectorType(context, intrinsic, MMAFragment::Acc);
  return {aVecType, bVecType, cVecType};
}

int64_t MMAAttr::getBlockSize() const {
  return IREE::GPU::getBlockSize(getIntrinsic().getValue());
}

int64_t MMAAttr::getSubgroupSize() const {
  return getIntrinsicSubgroupSize(getIntrinsic().getValue());
}

FailureOr<IREE::GPU::MMAScope> MMAAttr::getMmaScope() const {
  return IREE::GPU::MMAScope::Subgroup;
}

// Get virtual intrinsics that is composed/based on queried op.
SmallVector<VirtualMMAIntrinsic> MMAAttr::getVirtualIntrinsics() const {
  switch (getIntrinsic().getValue()) {
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
                         Value rhs, Value acc) {
  auto getVecOrSingleElem = [&](Value vec) -> Value {
    bool one = llvm::cast<VectorType>(vec.getType()).getNumElements() == 1;
    return one ? builder.create<vector::ExtractOp>(loc, vec, 0) : vec;
  };
  auto layout = getOpaqueMMALayout(builder.getContext(), intrinsic);
  if (is_AMD_MFMA(intrinsic)) {
    // MFMA intrinsics want single-element operands of element type, not vector.
    lhs = getVecOrSingleElem(lhs);
    rhs = getVecOrSingleElem(rhs);
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
  if (Value value = createMmaOp(builder, loc, getIntrinsic().getValue(),
                                resultType, lhs, rhs, acc)) {
    return value;
  }
  return failure();
}

static LogicalResult populateCanonicalOffsetsSizesAndStrides(
    OpBuilder &builder, Location loc, Value laneId,
    ArrayRef<int64_t> permutation, MMASingleSubgroupLayout subgroupLayout,
    SmallVector<OpFoldResult> &canonicalOffsets,
    SmallVector<OpFoldResult> &canonicalSizes,
    SmallVector<OpFoldResult> &canonicalStrides) {
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
  canonicalStrides.append(rankReducedShape.size(), one);

  // Each thread grabs `element` contiguous data, so the vtid needs to be
  // multiplied by `element` to get the next bunch of data.
  // vtid: virtual thread id
  // tid: lane id
  // vtid = ((tid floordiv stride_i) mod size_i) * element_i.
  SmallVector<OpFoldResult> vtids;
  for (auto [dimSize, dimStride, element] :
       llvm::zip_equal(subgroupLayout.thread, subgroupLayout.tstrides,
                       subgroupLayout.element)) {
    if (dimSize == 1) {
      vtids.push_back(zero);
      continue;
    }

    // ((tid floordiv stride) mod size) * element.
    AffineExpr tidExpr = builder.getAffineDimExpr(0);
    AffineMap vtidMap = AffineMap::get(
        /*dims=*/1, /*syms=*/0,
        (tidExpr.floorDiv(dimStride) % dimSize) * element);
    Value vtid = builder.create<affine::AffineApplyOp>(loc, vtidMap, laneId);
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
  applyPermutationToVector(canonicalOffsets, permutation);
  applyPermutationToVector(canonicalSizes, permutation);
  return success();
}

LogicalResult MMAAttr::populateOperandOffsetsSizesStrides(
    OpBuilder &builder, Location loc, IREE::GPU::MMAFragment fragment,
    Value laneId, ArrayRef<int64_t> permutation,
    SmallVector<OpFoldResult> &offsets, SmallVector<OpFoldResult> &sizes,
    SmallVector<OpFoldResult> &strides) const {

  MMASingleSubgroupLayout subgroupLayout =
      getSingleSubgroupLayout(getIntrinsic().getValue(), fragment);
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

std::tuple<Type, Type, Type> DataTiledMMAAttr::getABCElementTypes() const {
  MLIRContext *ctx = getContext();
  auto opaqueLayout = getOpaqueMMALayout(ctx, getIntrinsic().getValue());
  return {opaqueLayout.aType, opaqueLayout.bType, opaqueLayout.cType};
}

std::tuple<int64_t, int64_t, int64_t> DataTiledMMAAttr::getMNKShape() const {
  MLIRContext *ctx = getContext();
  auto opaqueLayout = getOpaqueMMALayout(ctx, getIntrinsic().getValue());
  return {opaqueLayout.mSize * getUnrollM() * getSubgroupsM(),
          opaqueLayout.nSize * getUnrollN() * getSubgroupsN(),
          opaqueLayout.kSize * getUnrollK()};
}

std::tuple<VectorType, VectorType, VectorType>
DataTiledMMAAttr::getABCVectorTypes() const {
  auto [A, B, C] = getABCElementTypes();
  auto getShape = [=](MMAFragment fragment) {
    return sliceSwizzledShape(
        getSwizzle(*this, fragment), [](TileSwizzle::Dim d) {
          return d.kind != TileSwizzle::Dim::Kind::CrossThread;
        });
  };
  return {VectorType::get(getShape(MMAFragment::Lhs), A),
          VectorType::get(getShape(MMAFragment::Rhs), B),
          VectorType::get(getShape(MMAFragment::Acc), C)};
}

int64_t DataTiledMMAAttr::getSubgroupSize() const {
  return getIntrinsicSubgroupSize(getIntrinsic().getValue());
}

FailureOr<IREE::GPU::MMAScope> DataTiledMMAAttr::getMmaScope() const {
  return IREE::GPU::MMAScope::Workgroup;
}

LogicalResult DataTiledMMAAttr::populateOperandOffsetsSizesStrides(
    OpBuilder &builder, Location loc, IREE::GPU::MMAFragment fragment,
    Value threadId, ArrayRef<int64_t> permutation,
    SmallVector<OpFoldResult> &offsets, SmallVector<OpFoldResult> &sizes,
    SmallVector<OpFoldResult> &strides) const {
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
  TileSwizzle intrinsicSwizzle =
      getIntrinsicSwizzle(getIntrinsic().getValue(), fragment);
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
  LLVM_DEBUG({
    DBGS() << "crossIntrinsicShape: ";
    llvm::interleaveComma(crossIntrinsicShape, llvm::dbgs());
    llvm::dbgs() << "\n";
  });
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
  LLVM_DEBUG({
    DBGS() << "DataTiledMMAAttr::buildMmaOperation\n";
    DBGS() << "    lhsSwizzle: " << lhsSwizzle << "\n";
  });
  SmallVector<Value> intrinsicsLhs =
      distributeMmaFragmentToIntrinsics(builder, loc, lhs, lhsSwizzle);

  TileSwizzle rhsSwizzle = getSwizzle(*this, MMAFragment::Rhs);
  LLVM_DEBUG({
    DBGS() << "DataTiledMMAAttr::buildMmaOperation\n";
    DBGS() << "    rhsSwizzle: " << rhsSwizzle << "\n";
  });
  SmallVector<Value> intrinsicsRhs =
      distributeMmaFragmentToIntrinsics(builder, loc, rhs, rhsSwizzle);

  TileSwizzle accSwizzle = getSwizzle(*this, MMAFragment::Acc);
  LLVM_DEBUG({
    DBGS() << "DataTiledMMAAttr::buildMmaOperation\n";
    DBGS() << "    accSwizzle: " << accSwizzle << "\n";
  });

  SmallVector<Value> intrinsicsAcc =
      distributeMmaFragmentToIntrinsics(builder, loc, acc, accSwizzle);

  MMAIntrinsic intrinsic = getIntrinsic().getValue();
  VectorType intrinCType =
      getVectorType(builder.getContext(), intrinsic, MMAFragment::Acc);

  // Loop over the 3 unroll_{m,n,k} dimensions to create the intrinsics.
  for (int mu = 0; mu < getUnrollM(); ++mu) {
    for (int nu = 0; nu < getUnrollN(); ++nu) {
      for (int ku = 0; ku < getUnrollK(); ++ku) {
        Value lhs = intrinsicsLhs[mu * getUnrollK() + ku];
        Value rhs = intrinsicsRhs[nu * getUnrollK() + ku];
        Value &acc = intrinsicsAcc[mu * getUnrollN() + nu];
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

  LLVM_DEBUG({
    DBGS() << "accCrossIntrinsicShape: ";
    llvm::interleaveComma(accCrossIntrinsicShape, llvm::dbgs());
    llvm::dbgs() << "\n";
    DBGS() << "accInternalShape: ";
    llvm::interleaveComma(accInternalShape, llvm::dbgs());
    llvm::dbgs() << "\n";
  });
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

VirtualMMAAttr VirtualMMAAttr::get(MLIRContext *context,
                                   VirtualMMAIntrinsic type) {
  auto intrinsicAttr = VirtualMMAIntrinsicAttr::get(context, type);
  return VirtualMMAAttr::get(context, intrinsicAttr);
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

std::tuple<Type, Type, Type> VirtualMMAAttr::getABCElementTypes() const {
  MLIRContext *ctx = getContext();
  auto opaqueLayout = getOpaqueMMALayout(ctx, getIntrinsic().getValue());
  return {opaqueLayout.aType, opaqueLayout.bType, opaqueLayout.cType};
}

std::tuple<VectorType, VectorType, VectorType>
VirtualMMAAttr::getABCVectorTypes() const {
  MLIRContext *context = getContext();
  VirtualMMAIntrinsic intrinsic = getIntrinsic().getValue();
  VectorType aVecType = getVectorType(context, intrinsic, MMAFragment::Lhs);
  VectorType bVecType = getVectorType(context, intrinsic, MMAFragment::Rhs);
  VectorType cVecType = getVectorType(context, intrinsic, MMAFragment::Acc);
  return {aVecType, bVecType, cVecType};
}

std::tuple<int64_t, int64_t, int64_t> VirtualMMAAttr::getMNKShape() const {
  MLIRContext *ctx = getContext();
  auto opaqueLayout = getOpaqueMMALayout(ctx, getIntrinsic().getValue());
  return {opaqueLayout.mSize, opaqueLayout.nSize, opaqueLayout.kSize};
}

int64_t VirtualMMAAttr::getSubgroupSize() const {
  switch (getIntrinsic().getValue()) {
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

FailureOr<IREE::GPU::MMAScope> VirtualMMAAttr::getMmaScope() const {
  return IREE::GPU::MMAScope::Subgroup;
}

LogicalResult VirtualMMAAttr::populateOperandOffsetsSizesStrides(
    OpBuilder &builder, Location loc, IREE::GPU::MMAFragment fragment,
    Value laneId, ArrayRef<int64_t> permutation,
    SmallVector<OpFoldResult> &offsets, SmallVector<OpFoldResult> &sizes,
    SmallVector<OpFoldResult> &strides) const {

  MMASingleSubgroupLayout subgroupLayout =
      getSingleSubgroupLayout(getIntrinsic().getValue(), fragment);
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

int64_t VirtualMMAAttr::getUnrollK() const {
  switch (getIntrinsic().getValue()) {
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
  switch (getIntrinsic().getValue()) {
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ:
  case VirtualMMAIntrinsic::VMFMA_F32_16x16x32_F16:
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F8E4M3FNUZ:
  case VirtualMMAIntrinsic::VMFMA_F32_32x32x16_F16: {
    // Generate mfma's for K with unrolled kernels.
    const int64_t unrollKFactor = getUnrollK();
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
  switch (getIntrinsic().getValue()) {
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
// iree_gpu.gpu_encoding_layout
//===----------------------------------------------------------------------===//

static MMAAttr chooseIntrinsicMMAAttr(TypeRange eTypes, TargetWgpAttr wgp) {
  MMAAttr candidateMma;
  for (MMAAttr mma : wgp.getMma()) {
    // Filter out intrinsics that don't match the element types of this matmul.
    auto [et0, et1, et2] = mma.getABCElementTypes();
    if (et0 != eTypes[0] || et1 != eTypes[1] || et2 != eTypes[2]) {
      continue;
    }
    // If multiple intrinsics are available for the given element types, we have
    // to make a choice. On CDNA3, there may be an intrinsic with larger M/N and
    // smaller K, which would optimize power, and an intrinsic with larger K,
    // which would optimize performance when power is not the bottleneck.
    // Currently we just choose the intrinsic maximizing K, but that can be
    // revisited later.
    if (candidateMma && candidateMma.getKSize() > mma.getKSize()) {
      continue;
    }
    candidateMma = mma;
  }
  return candidateMma;
}

static DataTiledMMAAttr
chooseDataTiledMMAAttr(TypeRange eTypes, TargetAttr target,
                       Encoding::EncodingAttr encoding) {
  if (!target) {
    return {};
  }
  MLIRContext *ctx = target.getContext();
  IREE::GPU::TargetWgpAttr wgp = target.getWgp();
  if (!wgp.getMaxLoadInstructionBits() || !wgp.getVgprSpaceBits() ||
      !wgp.getSimdsPerWgp()) {
    // Missing workgroup parameters: data tiling not supported on this target.
    return {};
  }

  //
  // Step 1: select a MMAIntrinsic.
  //
  MMAAttr intrinsicMma = chooseIntrinsicMMAAttr(eTypes, wgp);
  if (!intrinsicMma) {
    return {};
  }

  //
  // Step 2: Select the unrolling factors for the generic case where there is no
  //         narrow dimension.
  //

  auto sizeInBits = [](VectorType type) -> int {
    return type.getElementTypeBitWidth() * type.getNumElements();
  };

  auto [intrinsicA, intrinsicB, intrinsicC] = intrinsicMma.getABCVectorTypes();
  // The unrollK factor serves to allow loads from the A and B matrices to use
  // the target ISA's vector loads. For instance, if the ISA has 128-bit loads
  // and each intrinsic consumes only 32 bits from A and B, then we want to set
  // unrollK=4 to turn 4 separate 32-bit loads into one 128-bit load.
  int intrinsicLoadBits =
      std::min(sizeInBits(intrinsicA), sizeInBits(intrinsicB));
  const int unrollK =
      std::max(1, *wgp.getMaxLoadInstructionBits() / intrinsicLoadBits);

  // The total amount of unrolling along the M and N dimensions is normally
  // limited only by the number of available registers, since larger M and N
  // yields higher arithmetic intensity. Here, we do not yet distinguish between
  // plain unrolling (more instructions on each thread) and
  // unrolling-to-subgroups (more threads), since expanding to more subgroups
  // correspondingly divides the available register space between this many
  // subgroups, making it cancel out of the equation here.
  //
  // We need to solve for two variables here, unroll_m and unroll_n, constrained
  // by one quadratic equation expressing that the A, B and C tiles must fit in
  // VGPR space. Since we have only 1 constraint for two variables, we
  // self-impose a second constraint for now: that the unrolling shape should be
  // square, i.e. unrollM == unrollN.
  // TODO(#18850): that is suboptimal for narrow cases.
  //
  // Now we have only one variable, call it x, to solve for.

  // The register space taken is:
  //     A-tile: x * unrollK * sizeInBits(intrinsicA)
  //     B-tile: x * unrollK * sizeInBits(intrinsicB)
  //     C-tile: x^2 * sizeInBits(intrinsicC)
  // So the equation to solve is:
  //       x^2 * sizeInBits(intrinsicC)
  //     + x   * unrollK * (sizeInBits(intrinsicA) + sizeInBits(intrinsicB))
  //    == wgp.getVgprSpaceBits()
  float c2 = sizeInBits(intrinsicC);
  float c1 = unrollK * (sizeInBits(intrinsicA) + sizeInBits(intrinsicB));
  float c0 = -*wgp.getVgprSpaceBits(); // negative by construction.
  // Now the equation to solve is: c2 * x^2 + c1 * x + c0 == 0.
  float discriminant = c1 * c1 - 4 * c0 * c2; // positive, because c0 < 0.
  // x = unique positive solution.
  float x = (-c1 + std::sqrt(discriminant)) / (2 * c2);

#ifndef NDEBUG
  // Self-check quadratic solver. 10 epsilon is just a crude upper bound;
  // In practice, cancellation results in check == 0 in current cases.
  float check = c2 * x * x + c1 * x + c0;
  assert(std::abs(check) < 10 * FLT_EPSILON * std::abs(c0));
#endif

  // Now, looking geometrically at our unrolling space along the M and N
  // dimensions, we solve the following problem in the (M,N)-plane: approximate
  // a square of side length `x`, by a rectangle of side lengths `totalUnrollM`
  // and `totalUnrollN`, under the constraints:
  // 1. totalUnrollM * totalUnrollN <= x * x
  //    * Reason: by construction of x, any larger area would exceed the
  //      wgp.getVgprSpaceBits() budget.
  // 2. totalUnrollM and totalUnrollN are powers of 2.
  //    * Reason: that is a self-imposed constraint for now to avoid prematurely
  //      entering excessing fine-tuning of unrolling factors. Also, since below
  //      we will put all the unroll-to-subgroups in the N dimension, that
  //      requires totalUnrollN to be a multiple of wgp.getSimdsPerWgp(),
  //      which is typically a power of 2, specifically 4.
  //      TODO(#18851): we will not always put all the unroll-to-subgroups on N.
  // 3. totalUnrollN >= totalUnrollM.
  //    * Reason: Just like the previous constraint, that is also motivated by
  //      the code below currently putting all the unroll-to-subgroups in the N
  //      dimension, which requires a sufficiently large totalUnrollN.
  //      TODO(#18851): we will not always put all the unroll-to-subgroups on N.
  //
  // Set totalUnrollN = round x to nearest power of two, break ties away from 0
  // per specification of std::round.
  int totalUnrollN = std::exp2(std::round(std::log2(x)));
  // Based on above constraint 1:
  float unroundedMaxTotalUnrollM = x * x / totalUnrollN;
  int totalUnrollM = std::exp2(std::floor(std::log2(unroundedMaxTotalUnrollM)));

  // Now we introduce unroll-to-subgroups. It doesn't change the overall tile
  // size, as it increases the number of subgroups but correspondingly decreases
  // the number of registers available to each subgroups. In other words, the
  // overall tile size determined above only needed to be concerned with the
  // overall number of registers, not with how they are split between subgroups.
  //
  // For now for simplicity we put all the unroll-to-subgroups in the N
  // dimension. TODO(#18851): revisit that.
  //
  // That does simplify the below adjustments for narrow M/N, as we don't need
  // to think about unroll-to-subgroups when making the narrowing adjustment.
  int subgroupsM = 1;
  int subgroupsN = *wgp.getSimdsPerWgp();
  int unrollM = totalUnrollM / subgroupsM;
  int unrollN = totalUnrollN / subgroupsN;

  //
  // Step 3: Adjust the unrolling factors when there is a narrow dimension.
  // TODO(#18850): dealing with narrow cases as a fix-up is suboptimal.
  //
  IREE::Encoding::MatmulNarrowDim narrowDim =
      IREE::Encoding::getMatmulNarrowDim(encoding);
  if (narrowDim.isM()) {
    unrollM = std::min(unrollM, static_cast<int>(llvm::divideCeil(
                                    narrowDim.size, intrinsicMma.getMSize())));
  }
  if (narrowDim.isN()) {
    std::swap(unrollM, unrollN);
    std::swap(subgroupsM, subgroupsN);
    assert(subgroupsN == 1);
    unrollN = std::min(unrollN, static_cast<int>(llvm::divideCeil(
                                    narrowDim.size, intrinsicMma.getNSize())));
  }

  return DataTiledMMAAttr::get(ctx, intrinsicMma.getIntrinsic(), unrollM,
                               subgroupsM, unrollN, subgroupsN, unrollK);
}

MaterializeEncodingInfo
GPUEncodingLayoutAttr::getEncodingInfo(RankedTensorType type) const {
  auto encoding =
      llvm::dyn_cast_or_null<IREE::Encoding::EncodingAttr>(type.getEncoding());

  MaterializeEncodingInfo info;
  if (!encoding) {
    return info;
  }

  DataTiledMMAAttr mma = chooseDataTiledMMAAttr(encoding.getElementTypesArray(),
                                                getTargetAttr(), encoding);
  if (!mma) {
    return info;
  }

  // Map the matmul TileMxNxK to an actual tile shape for the tensor at hand,
  // based on its operand index in the matmul.
  TileMxNxK innerTile;
  std::tie(innerTile.M, innerTile.N, innerTile.K) = mma.getMNKShape();
  info = getEncodingInfoForMatmul(encoding, innerTile);
  auto fragment =
      static_cast<IREE::GPU::MMAFragment>(encoding.getOperandIndex().getInt());
  info.swizzle = getSwizzle(mma, fragment);
  return info;
}

static Operation *lowerContractionOpToMultiMmaOp(OpBuilder &builder,
                                                 linalg::LinalgOp linalgOp,
                                                 ValueRange operands,
                                                 TargetAttr targetAttr) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return nullptr;
  }
  FailureOr<linalg::ContractionDimensions> contractionDims =
      linalg::inferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    return nullptr;
  }

  auto inputs = linalgOp.getDpsInputOperands();
  auto outputs = linalgOp.getDpsInits();

  auto lhsType = cast<RankedTensorType>(inputs[0]->get().getType());
  auto rhsType = cast<RankedTensorType>(inputs[1]->get().getType());
  auto resultType = cast<RankedTensorType>(outputs[0].getType());
  auto lhsEncoding = IREE::Encoding::getEncodingAttr(lhsType);
  auto rhsEncoding = IREE::Encoding::getEncodingAttr(rhsType);
  auto resultEncoding = IREE::Encoding::getEncodingAttr(resultType);
  if (!lhsEncoding || !rhsEncoding || !resultEncoding) {
    return nullptr;
  }

  if (lhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_LHS ||
      rhsEncoding.getOperandIndex().getValue() != IREE::Encoding::MATMUL_RHS ||
      resultEncoding.getOperandIndex().getValue() !=
          IREE::Encoding::MATMUL_RESULT) {
    return nullptr;
  }

  IREE::GPU::DataTiledMMAAttr mma = chooseDataTiledMMAAttr(
      resultEncoding.getElementTypesArray(), targetAttr, resultEncoding);
  if (!mma) {
    LDBG("can't find supported Mma intrinsic");
    return nullptr;
  }
  LDBG("Target MMA: " << mma);

  MLIRContext *ctx = builder.getContext();
  SmallVector<AffineExpr> lhsExprs, rhsExprs, accExprs;
  int baseIdx = contractionDims->batch.empty() ? 0 : 1;
  if (baseIdx) {
    AffineExpr bExpr = builder.getAffineDimExpr(0);
    lhsExprs.push_back(bExpr);
    rhsExprs.push_back(bExpr);
    accExprs.push_back(bExpr);
  }
  AffineExpr mExpr = builder.getAffineDimExpr(baseIdx + 0);
  AffineExpr nExpr = builder.getAffineDimExpr(baseIdx + 1);
  AffineExpr kExpr = builder.getAffineDimExpr(baseIdx + 2);

  // The outer dims are all in row-major order after relayout.
  lhsExprs.append({mExpr, kExpr});
  rhsExprs.append({nExpr, kExpr});
  accExprs.append({mExpr, nExpr});
  int64_t numDims = baseIdx + 3;
  auto lhsMap = AffineMap::get(numDims, 0, lhsExprs, ctx);
  auto rhsMap = AffineMap::get(numDims, 0, rhsExprs, ctx);
  auto accMap = AffineMap::get(numDims, 0, accExprs, ctx);

  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();

  Location loc = linalgOp.getLoc();
  Operation *mmaOp = builder.create<MultiMmaOp>(
      loc, operands[0], operands[1], operands[2],
      ArrayRef<AffineMap>{lhsMap, rhsMap, accMap}, iteratorTypes, mma);
  return mmaOp;
}

Operation *GPUEncodingLayoutAttr::lowerOp(OpBuilder &b, Operation *op,
                                          TypeRange convertedResTypes,
                                          ValueRange convertedOperands) const {
  auto linalgOp = llvm::dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return nullptr;
  }
  return lowerContractionOpToMultiMmaOp(b, linalgOp, convertedOperands,
                                        getTargetAttr());
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

static StringRef getTilingLevelName(GPU::TilingLevel level) {
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
  return StringAttr();
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
