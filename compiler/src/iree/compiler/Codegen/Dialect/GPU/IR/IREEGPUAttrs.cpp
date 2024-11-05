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
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
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

using LayoutDimension = mlir::iree_compiler::IREE::VectorExt::LayoutDimension;
using LayoutDimensionAttr =
    mlir::iree_compiler::IREE::VectorExt::LayoutDimensionAttr;
using VectorLayoutInterface =
    mlir::iree_compiler::IREE::VectorExt::VectorLayoutInterface;
using PerDimLayoutAttr = mlir::iree_compiler::IREE::VectorExt::PerDimLayoutAttr;
using LayoutAttr = mlir::iree_compiler::IREE::VectorExt::LayoutAttr;
using NestedLayoutAttr = mlir::iree_compiler::IREE::VectorExt::NestedLayoutAttr;

namespace mlir::iree_compiler::IREE::GPU {

namespace {
// Struct containing abstract MMA shape and type information.
struct OpaqueMmaLayout {
  int64_t mSize;
  int64_t nSize;
  int64_t kSize;
  Type aType;
  Type bType;
  Type cType;
};

// Struct containing concrete MMA shape, type, and layout information.
struct ConcreteMmaLayout {
  OpaqueMmaLayout base;
  PerDimLayoutAttr aMLayout;
  PerDimLayoutAttr aKLayout;
  PerDimLayoutAttr bKLayout;
  PerDimLayoutAttr bNLayout;
  PerDimLayoutAttr cMLayout;
  PerDimLayoutAttr cNLayout;
};
} // namespace

//===----------------------------------------------------------------------===//
// #iree_gpu.mma_vector_layout
//===----------------------------------------------------------------------===//

static PerDimLayoutAttr getBatchedPerDimLayoutAttr(LayoutDimensionAttr batchDim,
                                                   PerDimLayoutAttr baseLayout,
                                                   int64_t problemSize,
                                                   int64_t fragmentDimSize) {
  assert(problemSize % fragmentDimSize == 0 &&
         "invalid layout fragment for problem size");

  SmallVector<LayoutDimensionAttr, 3> dimAttrs(baseLayout.getLabels());
  dimAttrs.insert(dimAttrs.begin(), batchDim);

  SmallVector<int64_t, 3> shapes(baseLayout.getShapes());
  shapes.insert(shapes.begin(), problemSize / fragmentDimSize);
  auto layout =
      PerDimLayoutAttr::get(baseLayout.getContext(), dimAttrs, shapes);
  return layout;
}

// Get the batched layout attributes for the given fragment layouts, indexing
// map, and problem shape. The canonical fragment map is used to compare against
// the problem map |indexingMap|. For example, for mma fragment B (RHS):
//
// indexingMap = affine_map<(d0, d1, d2) -> (d1, d2) # Transposed B
// fragmentMap = affine_map<(d0, d1, d2) -> (d2, d1)
// problemShape = [32, 64]
// fragmentSize = [16, 8]
// fragmentLayouts = [kLayout, nLayout]
//
// Gives batched layout
//
// Dim0 Layout = [BATCHX, nLayoutLabels], [8, nLayoutShape]
// Dim1 Layout = [BATCHY, kLayoutLabels], [2, kLayoutShape]
static LayoutAttr
getBatchedLayoutAttr(AffineMap indexingMap, AffineMap fragmentMap,
                     ArrayRef<int64_t> problemShape,
                     ArrayRef<int64_t> fragmentSize,
                     ArrayRef<PerDimLayoutAttr> fragmentLayouts) {
  // Current distribution to MFMA operations does not support batched
  // contractions so that is reflected here.
  assert(indexingMap.getNumResults() == 2 &&
         "invalid indexing map to non-batched simple contraction");

  LayoutDimensionAttr batchX = LayoutDimensionAttr::get(
      indexingMap.getContext(), LayoutDimension::BATCHX);
  LayoutDimensionAttr batchY = LayoutDimensionAttr::get(
      indexingMap.getContext(), LayoutDimension::BATCHY);

  SmallVector<PerDimLayoutAttr, 2> perDimAttrs;
  for (auto [expr, batchType] :
       llvm::zip_equal(indexingMap.getResults(),
                       SmallVector<LayoutDimensionAttr, 2>{batchX, batchY})) {
    auto maybeResultPosition = fragmentMap.getResultPosition(expr);
    assert(maybeResultPosition && "fragment map and problem map mismatch");
    int64_t idx = *maybeResultPosition;
    perDimAttrs.push_back(getBatchedPerDimLayoutAttr(
        batchType, fragmentLayouts[idx], problemShape[idx], fragmentSize[idx]));
  }

  return LayoutAttr::get(indexingMap.getContext(), perDimAttrs);
}

static FailureOr<std::tuple<VectorLayoutInterface, VectorLayoutInterface,
                            VectorLayoutInterface>>
getContractionLayout(vector::ContractionOp contract, ConcreteMmaLayout layout) {
  MLIRContext *context = contract.getContext();
  FailureOr<linalg::ContractionDimensions> maybeContractionDims =
      linalg::inferContractionDims(contract.getIndexingMapsArray());
  if (failed(maybeContractionDims)) {
    return failure();
  }
  auto contractionDims = *maybeContractionDims;
  // TODO: Relax this condition to strictly alignment requirements.
  if (contractionDims.k.size() != 1 || contractionDims.m.size() != 1 ||
      contractionDims.n.size() != 1) {
    return failure();
  }
  // TODO: Support batched contractions.
  if (contractionDims.batch.size() > 0) {
    return failure();
  }
  unsigned mDim = contractionDims.m[0];
  unsigned nDim = contractionDims.n[0];
  unsigned kDim = contractionDims.k[0];

  SmallVector<int64_t> iterationBounds;
  contract.getIterationBounds(iterationBounds);

  int64_t problemMSize = iterationBounds[mDim];
  int64_t problemNSize = iterationBounds[nDim];
  int64_t problemKSize = iterationBounds[kDim];

  int64_t mSize = layout.base.mSize;
  int64_t nSize = layout.base.nSize;
  int64_t kSize = layout.base.kSize;

  // The problem size currently must be strictly aligned to the size of the mma.
  // This is expected to succeed assuming the correct [masked] vector size was
  // set at strategy configuration time (for this mma).
  if (problemMSize % mSize != 0 || problemNSize % nSize ||
      problemKSize % kSize) {
    return failure();
  }

  LayoutAttr aLayout = getBatchedLayoutAttr(
      contract.getIndexingMapsArray()[0],
      AffineMap::getMultiDimMapWithTargets(3, {mDim, kDim}, context),
      {problemMSize, problemKSize}, {mSize, kSize},
      {layout.aMLayout, layout.aKLayout});
  LayoutAttr bLayout = getBatchedLayoutAttr(
      contract.getIndexingMapsArray()[1],
      AffineMap::getMultiDimMapWithTargets(3, {kDim, nDim}, context),
      {problemKSize, problemNSize}, {kSize, nSize},
      {layout.bKLayout, layout.bNLayout});
  LayoutAttr cLayout = getBatchedLayoutAttr(
      contract.getIndexingMapsArray()[2],
      AffineMap::getMultiDimMapWithTargets(3, {mDim, nDim}, context),
      {problemMSize, problemNSize}, {mSize, nSize},
      {layout.cMLayout, layout.cNLayout});

  return std::make_tuple<VectorLayoutInterface, VectorLayoutInterface,
                         VectorLayoutInterface>(aLayout, bLayout, cLayout);
}

//===----------------------------------------------------------------------===//
// Layout Attribute Building Helpers
//===----------------------------------------------------------------------===//

static OpaqueMmaLayout getOpaqueMFMALayout(MLIRContext *context,
                                           MMAIntrinsic type) {
  Type f8E4M3FNUZ = Float8E4M3FNUZType::get(context);
  Type f8E5M2FNUZ = Float8E5M2FNUZType::get(context);
  Type f16 = Float16Type::get(context);
  Type bf16 = BFloat16Type::get(context);
  Type f32 = Float32Type::get(context);

  Type i8 = IntegerType::get(context, 8);
  Type i32 = IntegerType::get(context, 32);

  switch (type) {
  case MMAIntrinsic::MFMA_F32_16x16x4_F32: {
    return OpaqueMmaLayout{16, 16, 4, f32, f32, f32};
  }
  case MMAIntrinsic::MFMA_F32_16x16x16_F16: {
    return OpaqueMmaLayout{16, 16, 16, f16, f16, f32};
  }
  case MMAIntrinsic::MFMA_F32_32x32x8_F16: {
    return OpaqueMmaLayout{32, 32, 8, f16, f16, f32};
  }
  case MMAIntrinsic::MFMA_F32_16x16x16_BF16: {
    return OpaqueMmaLayout{16, 16, 16, bf16, bf16, f32};
  }
  case MMAIntrinsic::MFMA_F32_32x32x8_BF16: {
    return OpaqueMmaLayout{32, 32, 8, bf16, bf16, f32};
  }
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ:
  case MMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ: {
    return OpaqueMmaLayout{16, 16, 32, f8E4M3FNUZ, f8E4M3FNUZ, f32};
  }
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2FNUZ: {
    return OpaqueMmaLayout{16, 16, 32, f8E5M2FNUZ, f8E5M2FNUZ, f32};
  }
  case MMAIntrinsic::MFMA_I32_16x16x32_I8: {
    return OpaqueMmaLayout{16, 16, 32, i8, i8, i32};
  }
  case MMAIntrinsic::MFMA_I32_32x32x16_I8: {
    return OpaqueMmaLayout{32, 32, 16, i8, i8, i32};
  }
  case MMAIntrinsic::MFMA_I32_32x32x8_I8: {
    return OpaqueMmaLayout{32, 32, 8, i8, i8, i32};
  }
  case MMAIntrinsic::MFMA_I32_16x16x16_I8: {
    return OpaqueMmaLayout{16, 16, 16, i8, i8, i32};
  }
  case MMAIntrinsic::WMMA_F32_16x16x16_F16: {
    return OpaqueMmaLayout{16, 16, 16, f16, f16, f32};
  }
  case MMAIntrinsic::WMMA_F16_16x16x16_F16: {
    return OpaqueMmaLayout{16, 16, 16, f16, f16, f16};
  }
  case MMAIntrinsic::WMMA_I32_16x16x16_I8: {
    return OpaqueMmaLayout{16, 16, 16, i8, i8, i32};
  }
  // V(Virtual)MFMA instructions which have 2 mfma instructions interleaved
  // along the k dimension.
  case MMAIntrinsic::VMFMA_F32_16x16x32_F16: {
    return OpaqueMmaLayout{16, 16, 32, f16, f16, f32};
  }
  case MMAIntrinsic::VMFMA_F32_32x32x16_F16: {
    return OpaqueMmaLayout{32, 32, 16, f16, f16, f32};
  }
  }
  llvm_unreachable("unhandled mfma layout type");
  return OpaqueMmaLayout{};
}

static std::tuple<PerDimLayoutAttr, PerDimLayoutAttr>
getPerDimLayoutAttrs(MLIRContext *context, TileSwizzle swizzle) {
  // Step 1: obtain the swizzled tile shape, but keeping track of the source
  // dimension indices.
  struct SrcIndexAndSwizzleDim {
    size_t srcIndex;
    TileSwizzle::Dim dim;
  };
  SmallVector<SrcIndexAndSwizzleDim> swizzledShape;
  for (auto [i, e] : llvm::enumerate(swizzle.expandShape)) {
    for (TileSwizzle::Dim d : e) {
      swizzledShape.push_back(SrcIndexAndSwizzleDim{i, d});
    }
  }
  applyPermutationToVector(swizzledShape, swizzle.permutation);

  // Step 2: collect the appropriate labels to use for the swizzled dims.
  LayoutDimension internalLabels[] = {LayoutDimension::VECTORZ,
                                      LayoutDimension::VECTORY,
                                      LayoutDimension::VECTORX};
  LayoutDimension crossThreadLabels[] = {
      LayoutDimension::LANEZ, LayoutDimension::LANEY, LayoutDimension::LANEX};
  auto internalLabelIter = std::end(internalLabels);
  auto crossThreadLabelIter = std::end(crossThreadLabels);
  for (SrcIndexAndSwizzleDim d : swizzledShape) {
    if (d.dim.kind == TileSwizzle::Dim::Kind::Internal) {
      assert(internalLabelIter != std::begin(internalLabels));
      --internalLabelIter;
    } else if (d.dim.kind == TileSwizzle::Dim::Kind::CrossThread) {
      assert(crossThreadLabelIter != std::begin(crossThreadLabels));
      --crossThreadLabelIter;
    } else {
      assert(false && "unexpected dimension kind in intrinsic swizzle");
    }
  }

  // Step 3: put together the result PerDimLayoutAttr'd for the two source dims.
  SmallVector<LayoutDimensionAttr> labels[2];
  SmallVector<int64_t> shape[2];
  for (SrcIndexAndSwizzleDim d : swizzledShape) {
    shape[d.srcIndex].push_back(d.dim.size);
    auto &labelIterRef = (d.dim.kind == TileSwizzle::Dim::Kind::Internal)
                             ? internalLabelIter
                             : crossThreadLabelIter;
    labels[d.srcIndex].push_back(LayoutDimensionAttr::get(
        context, static_cast<LayoutDimension>(*labelIterRef++)));
  }
  return {PerDimLayoutAttr::get(context, labels[0], shape[0]),
          PerDimLayoutAttr::get(context, labels[1], shape[1])};
};

static ConcreteMmaLayout getConcreteMFMALayout(MLIRContext *context,
                                               MMAIntrinsic intrinsic) {
  auto opaque = getOpaqueMFMALayout(context, intrinsic);
  ConcreteMmaLayout concreteLayout;
  concreteLayout.base = opaque;
  auto lhsSwizzle = getIntrinsicSwizzle(intrinsic, MMAFragment::Lhs);
  auto rhsSwizzle = getIntrinsicSwizzle(intrinsic, MMAFragment::Rhs);
  auto accSwizzle = getIntrinsicSwizzle(intrinsic, MMAFragment::Acc);
  std::tie(concreteLayout.aMLayout, concreteLayout.aKLayout) =
      getPerDimLayoutAttrs(context, lhsSwizzle);
  std::tie(concreteLayout.bNLayout, concreteLayout.bKLayout) =
      getPerDimLayoutAttrs(context, rhsSwizzle);
  std::tie(concreteLayout.cMLayout, concreteLayout.cNLayout) =
      getPerDimLayoutAttrs(context, accSwizzle);
  return concreteLayout;
}

//===----------------------------------------------------------------------===//
// MFMA Attributes
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
  auto layout = getOpaqueMFMALayout(context, type);
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

// NOTE: For layout specifications of the WMMA intrinsics
//       below we are assuming subgroupsize of 32.
std::tuple<VectorType, VectorType, VectorType>
MMAAttr::getABCVectorTypes() const {
  // Check https://github.com/ROCm/amd_matrix_instruction_calculator for
  // instruction details. Note here we are returning the number elements, while
  // amd_matrix_instruction_calculator tells us about the number of 32-bit
  // registers. So need to adjust accordingly. All vectors should be 1-D.
  switch (getIntrinsic().getValue()) {
  case MMAIntrinsic::MFMA_F32_16x16x4_F32: {
    auto aType = VectorType::get({1}, getAType());
    auto bType = VectorType::get({1}, getBType());
    auto cType = VectorType::get({4}, getCType());
    return std::make_tuple(aType, bType, cType);
  }
  case MMAIntrinsic::MFMA_I32_16x16x16_I8:
  case MMAIntrinsic::MFMA_F32_16x16x16_F16:
  case MMAIntrinsic::MFMA_F32_16x16x16_BF16: {
    auto aType = VectorType::get({4}, getAType());
    auto bType = VectorType::get({4}, getBType());
    auto cType = VectorType::get({4}, getCType());
    return std::make_tuple(aType, bType, cType);
  }
  case MMAIntrinsic::MFMA_I32_32x32x8_I8:
  case MMAIntrinsic::MFMA_F32_32x32x8_F16:
  case MMAIntrinsic::MFMA_F32_32x32x8_BF16: {
    auto aType = VectorType::get({4}, getAType());
    auto bType = VectorType::get({4}, getBType());
    auto cType = VectorType::get({16}, getCType());
    return std::make_tuple(aType, bType, cType);
  }
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2FNUZ:
  case MMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ:
  case MMAIntrinsic::VMFMA_F32_16x16x32_F16:
  case MMAIntrinsic::MFMA_I32_16x16x32_I8: {
    auto aType = VectorType::get({8}, getAType());
    auto bType = VectorType::get({8}, getBType());
    auto cType = VectorType::get({4}, getCType());
    return std::make_tuple(aType, bType, cType);
  }
  case MMAIntrinsic::VMFMA_F32_32x32x16_F16:
  case MMAIntrinsic::MFMA_I32_32x32x16_I8: {
    auto aType = VectorType::get({8}, getAType());
    auto bType = VectorType::get({8}, getBType());
    auto cType = VectorType::get({16}, getCType());
    return std::make_tuple(aType, bType, cType);
  }
  case MMAIntrinsic::WMMA_F32_16x16x16_F16:
  case MMAIntrinsic::WMMA_I32_16x16x16_I8: {
    auto aType = VectorType::get({16}, getAType());
    auto bType = VectorType::get({16}, getBType());
    auto cType = VectorType::get({8}, getCType());
    return std::make_tuple(aType, bType, cType);
  }
  case MMAIntrinsic::WMMA_F16_16x16x16_F16: {
    auto aType = VectorType::get({16}, getAType());
    auto bType = VectorType::get({16}, getBType());
    auto cType = VectorType::get({16}, getCType());
    return std::make_tuple(aType, bType, cType);
  }
  }
  // This should not happen but just to make GCC happy.
  return std::make_tuple(VectorType{}, VectorType{}, VectorType{});
}

FailureOr<std::tuple<VectorLayoutInterface, VectorLayoutInterface,
                     VectorLayoutInterface>>
MMAAttr::getContractionLayout(vector::ContractionOp contract) const {
  ConcreteMmaLayout layout =
      getConcreteMFMALayout(contract->getContext(), getIntrinsic().getValue());
  return IREE::GPU::getContractionLayout(contract, layout);
}

int64_t MMAAttr::getBlockSize() const {
  switch (getIntrinsic().getValue()) {
  case MMAIntrinsic::MFMA_F32_16x16x4_F32:
  case MMAIntrinsic::MFMA_F32_16x16x16_F16:
  case MMAIntrinsic::MFMA_F32_16x16x16_BF16:
  case MMAIntrinsic::MFMA_I32_16x16x16_I8:
  case MMAIntrinsic::MFMA_F32_32x32x8_F16:
  case MMAIntrinsic::MFMA_F32_32x32x8_BF16:
  case MMAIntrinsic::MFMA_I32_32x32x8_I8:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2FNUZ:
  case MMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ:
  case MMAIntrinsic::VMFMA_F32_16x16x32_F16:
  case MMAIntrinsic::MFMA_I32_16x16x32_I8:
  case MMAIntrinsic::VMFMA_F32_32x32x16_F16:
  case MMAIntrinsic::MFMA_I32_32x32x16_I8:
  case MMAIntrinsic::WMMA_F16_16x16x16_F16:
  case MMAIntrinsic::WMMA_F32_16x16x16_F16:
  case MMAIntrinsic::WMMA_I32_16x16x16_I8: {
    return 1;
  }
  }
  // This should not happen but just to make GCC happy.
  return 0;
}

static int64_t getIntrinsicSubgroupSize(MMAIntrinsic intrinsic) {
  switch (intrinsic) {
  case MMAIntrinsic::MFMA_F32_16x16x4_F32:
  case MMAIntrinsic::MFMA_F32_16x16x16_F16:
  case MMAIntrinsic::MFMA_F32_16x16x16_BF16:
  case MMAIntrinsic::MFMA_I32_16x16x16_I8:
  case MMAIntrinsic::MFMA_F32_32x32x8_F16:
  case MMAIntrinsic::MFMA_F32_32x32x8_BF16:
  case MMAIntrinsic::MFMA_I32_32x32x8_I8:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2FNUZ:
  case MMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ:
  case MMAIntrinsic::VMFMA_F32_16x16x32_F16:
  case MMAIntrinsic::MFMA_I32_16x16x32_I8:
  case MMAIntrinsic::VMFMA_F32_32x32x16_F16:
  case MMAIntrinsic::MFMA_I32_32x32x16_I8: {
    return 64;
  }
  case MMAIntrinsic::WMMA_F32_16x16x16_F16:
  case MMAIntrinsic::WMMA_F16_16x16x16_F16:
  case MMAIntrinsic::WMMA_I32_16x16x16_I8: {
    return 32;
  }
  }
  // This should not happen but just to make GCC happy.
  return 0;
}

int64_t MMAAttr::getSubgroupSize() const {
  return getIntrinsicSubgroupSize(getIntrinsic().getValue());
}

FailureOr<IREE::GPU::MMAScope> MMAAttr::getMmaScope() const {
  return IREE::GPU::MMAScope::Subgroup;
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
  case MMAIntrinsic::VMFMA_F32_16x16x32_F16:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2FNUZ:
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
  case MMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ:
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
  case MMAIntrinsic::VMFMA_F32_32x32x16_F16:
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
  }
  return {};
}

MMASingleSubgroupLayout MMAAttr::getASingleSubgroupLayout() const {
  return getSingleSubgroupLayout(getIntrinsic().getValue(), MMAFragment::Lhs);
}

MMASingleSubgroupLayout MMAAttr::getBSingleSubgroupLayout() const {
  return getSingleSubgroupLayout(getIntrinsic().getValue(), MMAFragment::Rhs);
}

MMASingleSubgroupLayout MMAAttr::getCSingleSubgroupLayout() const {
  return getSingleSubgroupLayout(getIntrinsic().getValue(), MMAFragment::Acc);
}

// Get virtual intrinsics that is composed/based on queried op.
SmallVector<MMAIntrinsic> MMAAttr::getVirtualIntrinsics() const {
  switch (getIntrinsic().getValue()) {
  case MMAIntrinsic::MFMA_F32_16x16x16_F16:
    return {MMAIntrinsic::VMFMA_F32_16x16x32_F16};
  case MMAIntrinsic::MFMA_F32_32x32x8_F16:
    return {MMAIntrinsic::VMFMA_F32_32x32x16_F16};
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ:
    return {MMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ};
  default:
    return {};
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
  switch (getIntrinsic().getValue()) {
  case MMAIntrinsic::MFMA_F32_16x16x4_F32: {
    // Update the lhs and rhs to extract the first element since vector<1xT> is
    // not supoorted by amgpu.mfma op.
    lhs = builder.create<vector::ExtractOp>(loc, lhs, ArrayRef{int64_t{0}});
    rhs = builder.create<vector::ExtractOp>(loc, rhs, ArrayRef{int64_t{0}});
    auto [m, n, k] = getMNKShape();
    return builder
        .create<amdgpu::MFMAOp>(loc, resultType, m, n, k, getBlockSize(), lhs,
                                rhs, acc)
        .getResult();
  }
  case MMAIntrinsic::VMFMA_F32_16x16x32_F16:
  case MMAIntrinsic::VMFMA_F32_32x32x16_F16: {
    // Generate mfma's for K with unrolled kernels.
    const int64_t unrollKFactor = 2;
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
  case MMAIntrinsic::MFMA_I32_16x16x16_I8:
  case MMAIntrinsic::MFMA_F32_16x16x16_F16:
  case MMAIntrinsic::MFMA_F32_16x16x16_BF16:
  case MMAIntrinsic::MFMA_I32_32x32x8_I8:
  case MMAIntrinsic::MFMA_F32_32x32x8_F16:
  case MMAIntrinsic::MFMA_F32_32x32x8_BF16:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_F32_16x16x32_F8E5M2FNUZ:
  case MMAIntrinsic::VMFMA_F32_16x16x32_F8E4M3FNUZ:
  case MMAIntrinsic::MFMA_I32_16x16x32_I8:
  case MMAIntrinsic::MFMA_I32_32x32x16_I8: {
    auto [m, n, k] = getMNKShape();
    return builder
        .create<amdgpu::MFMAOp>(loc, resultType, m, n, k, getBlockSize(), lhs,
                                rhs, acc)
        .getResult();
  }
  case MMAIntrinsic::WMMA_F32_16x16x16_F16:
  case MMAIntrinsic::WMMA_F16_16x16x16_F16:
  case MMAIntrinsic::WMMA_I32_16x16x16_I8: {
    return builder.create<amdgpu::WMMAOp>(loc, resultType, lhs, rhs, acc)
        .getResult();
  }
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

  MMASingleSubgroupLayout subgroupLayout;
  switch (fragment) {
  case IREE::GPU::MMAFragment::Lhs: {
    subgroupLayout = getASingleSubgroupLayout();
    break;
  }
  case IREE::GPU::MMAFragment::Rhs: {
    subgroupLayout = getBSingleSubgroupLayout();
    break;
  }
  case IREE::GPU::MMAFragment::Acc: {
    subgroupLayout = getCSingleSubgroupLayout();
    break;
  }
  }

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

LogicalResult MMAAttr::materializeOperandConcreteShape(
    OpBuilder &builder, IREE::GPU::MMAFragment fragment, Value operand,
    std::optional<ArrayRef<int64_t>> permutation,
    SmallVector<ReassociationIndices> &reassociations,
    RankedTensorType &resultType) const {

  SmallVector<int64_t, 2> outerSizes;
  SmallVector<int64_t, 2> opaqueSizes;
  auto [m, n, k] = getMNKShape();
  switch (fragment) {
  case IREE::GPU::MMAFragment::Lhs: {
    outerSizes = getASingleSubgroupLayout().outer;
    opaqueSizes.append({m, k});
    break;
  }
  case IREE::GPU::MMAFragment::Rhs: {
    outerSizes = getBSingleSubgroupLayout().outer;
    opaqueSizes.append({k, n});
    break;
  }
  case IREE::GPU::MMAFragment::Acc: {
    outerSizes = getCSingleSubgroupLayout().outer;
    opaqueSizes.append({m, n});
    break;
  }
  }
  if (permutation.has_value()) {
    if (permutation.value().size() != outerSizes.size()) {
      return failure();
    }
    applyPermutationToVector(opaqueSizes, permutation.value());
    applyPermutationToVector(outerSizes, permutation.value());
  }

  // Inner tile must have sizes matching the opaque layout.
  auto operandType = llvm::cast<RankedTensorType>(operand.getType());
  ArrayRef<int64_t> operandShape = operandType.getShape();
  SmallVector<int64_t, 2> innerShape(operandShape.end() - opaqueSizes.size(),
                                     operandShape.end());
  if (!llvm::equal(opaqueSizes, innerShape)) {
    return failure();
  }

  // Expand the shape of the inner tile to reflect the MMA thread layout.
  SmallVector<int64_t, 4> resultShape(operandShape.begin(),
                                      operandShape.end() - 2);
  SmallVector<ReassociationIndices> reInds =
      llvm::map_to_vector(llvm::seq<int64_t>(resultShape.size()),
                          [](int64_t idx) -> ReassociationIndices {
                            return ReassociationIndices({idx});
                          });
  int idx = reInds.size();
  for (auto [outer, native] : llvm::zip_equal(outerSizes, opaqueSizes)) {
    // Skip expansion if the outer dim is unit as the SingleSubgroupLayout gives
    // a guarantee that the |element| counts are contiguous within the layout,
    // and a unit outer implies a single offset and size for that dimension.
    if (outer == 1) {
      resultShape.push_back(native);
      reInds.push_back(ReassociationIndices({idx++}));
      continue;
    }

    // Reshape to [outer, native / outer] == [outer, thread * element]. This
    // corresponds to |outer| repetitions of the thread/element sublayout.
    resultShape.push_back(outer);
    assert(native % outer == 0 && "invalid mma layout");
    resultShape.push_back(native / outer);
    reInds.push_back(ReassociationIndices{idx, idx + 1});
    idx += 2;
  }

  reassociations = reInds;
  resultType = operandType.clone(resultShape);
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
  auto opaqueLayout = getOpaqueMFMALayout(ctx, getIntrinsic().getValue());
  return {opaqueLayout.aType, opaqueLayout.bType, opaqueLayout.cType};
}

std::tuple<int64_t, int64_t, int64_t> DataTiledMMAAttr::getMNKShape() const {
  MLIRContext *ctx = getContext();
  auto opaqueLayout = getOpaqueMFMALayout(ctx, getIntrinsic().getValue());
  return {opaqueLayout.mSize * getUnrollM() * getUnrollMToSubgroups(),
          opaqueLayout.nSize * getUnrollN() * getUnrollNToSubgroups(),
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
  // TODO(bjacob): Support WMMA intrinsics.

  // Get the swizzle describing the internal layout of this fragment.
  TileSwizzle swizzle = getSwizzle(*this, fragment);

  LLVM_DEBUG({
    DBGS() << "DataTiledMMAAttr::populateOperandOffsetsSizesStrides\n";
    DBGS() << "    fragment: " << llvm::to_underlying(fragment) << "\n";
    DBGS() << "    swizzle: " << swizzle << "\n";
  });

  // Populate tile sizes.
  MLIRContext *ctx = builder.getContext();
  SmallVector<OpFoldResult> tileSizes = getAsIndexOpFoldResult(
      ctx, sliceSwizzledShape(swizzle, [](TileSwizzle::Dim d) {
        return d.kind != TileSwizzle::Dim::Kind::CrossThread;
      }));

  // Populate tile offsets by delinearizing threadId over the CrossThread dims.
  // Since the AffineDelinearizeIndexOp does not bound the input index, we
  // must bound the threadId by the product of the offset ranges.
  SmallVector<int64_t> tileOffsetsBasis =
      sliceSwizzledShape(swizzle, [](TileSwizzle::Dim d) {
        return d.kind == TileSwizzle::Dim::Kind::CrossThread;
      });

  // Bound for threadId is the product of tileOffsetsBasis.
  OpFoldResult threadIdBound =
      builder.getIndexAttr(ShapedType::getNumElements(tileOffsetsBasis));
  AffineExpr d0 = builder.getAffineDimExpr(0), d1 = builder.getAffineDimExpr(1);
  OpFoldResult boundedThreadId = affine::makeComposedFoldedAffineApply(
      builder, loc, {d0 % d1}, {threadId, threadIdBound});

  SmallVector<OpFoldResult> tileOffsets =
      builder
          .create<affine::AffineDelinearizeIndexOp>(
              loc,
              getValueOrCreateConstantIndexOp(builder, loc, boundedThreadId),
              getAsIndexOpFoldResult(ctx, tileOffsetsBasis))
          ->getResults();

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
  // TODO(bjacob): Support WMMA intrinsics.

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

  // Get a MMAAttr for the intrinsic itself, to reuse MMAAttr::buildMmaOperation
  // to create the target intrinsics.
  auto intrinsicMma = MMAAttr::get(getContext(), getIntrinsic().getValue());
  auto [intrinsicAType, intrinsicBType, intrinsicCType] =
      intrinsicMma.getABCVectorTypes();

  // Loop over the 3 unroll_{m,n,k} dimensions to create the intrinsics.
  for (int mu = 0; mu < getUnrollM(); ++mu) {
    for (int nu = 0; nu < getUnrollN(); ++nu) {
      for (int ku = 0; ku < getUnrollK(); ++ku) {
        // Assume intrinsicMma.buildMmaOperation() success: validation should be
        // completed prior to mutating IR.
        Value lhs = intrinsicsLhs[mu * getUnrollK() + ku];
        Value rhs = intrinsicsRhs[nu * getUnrollK() + ku];
        Value &acc = intrinsicsAcc[mu * getUnrollN() + nu];
        acc = *intrinsicMma.buildMmaOperation(builder, loc, intrinsicCType, lhs,
                                              rhs, acc);
      }
    }
  }

  // Insert the results into the destination accumulator.
  SmallVector<int64_t> accCrossIntrinsicShape =
      sliceSwizzledShape(accSwizzle, [](TileSwizzle::Dim dim) {
        return dim.kind == TileSwizzle::Dim::Kind::CrossIntrinsic;
      });
  LLVM_DEBUG({
    DBGS() << "accCrossIntrinsicShape: ";
    llvm::interleaveComma(accCrossIntrinsicShape, llvm::dbgs());
    llvm::dbgs() << "\n";
  });
  SmallVector<int64_t> strides(intrinsicCType.getRank(), 1);
  SmallVector<int64_t> indices(accCrossIntrinsicShape.size(), 0);
  for (Value intrAcc : intrinsicsAcc) {
    acc = builder.create<vector::InsertStridedSliceOp>(loc, intrAcc, acc,
                                                       indices, strides);
    incrementIndices(indices, accCrossIntrinsicShape);
  }
  return acc;
}

//===----------------------------------------------------------------------===//
// MMA Schedule Attributes
//===----------------------------------------------------------------------===//

/// Gets a unit vector of the given rank, but fills in the given dimensions
/// from the 2 element array |counts|. |dim0| is the position in the returned
/// vector to put the first element of |counts|, and |dim1| is the position to
/// put the second element. For example,
///
/// rank = 3, counts = [5, 7], dim0 = 2, dim1 = 1
/// returns [1, 5, 7]
SmallVector<int64_t> getUnitOfRankWithDims(int64_t rank,
                                           ArrayRef<int64_t> counts,
                                           int64_t dim0, int64_t dim1) {
  assert(counts.size() == 2 &&
         "Unexpected non-rank 2 single subgroup dimension counts");
  SmallVector<int64_t> res(rank, 1);
  res[dim0] = counts[0];
  res[dim1] = counts[1];
  return res;
}

SmallVector<int64_t> getIdentityPerm(int64_t rank) {
  return llvm::to_vector(llvm::seq(static_cast<int64_t>(0), rank));
}

/// Constructs an identity permutation with the given rank, except it applies
/// the given rank-2 |perm| to the two dimensions |dim0| and |dim1|, and then
/// swaps the positions of dim0 and dim1 in the final permutation. For example,
///
/// rank = 3, perm = [1, 0], dim0 = 1, dim1 = 2
/// returns [0, 1, 2]
///
/// This is essentially just applying two rank-2 permutations to two particular
/// dimensions. First it applies |perm|, which corresponds to a permutation
/// needed by the underlying intrinsic, then it does another permutation based
/// on the order of actual dimensions for the MMA fragment. For example, for the
/// B matrix, dim0 = K and dim1 = N, so for the element order of an MFMA
/// 16x16x16, perm would be `[1, 0]`, however if the actual contraction is a
/// matmul_transpose_b, then the element order needs to be [0, 1].
SmallVector<int64_t> getIdentityPermWithSwap(int64_t rank,
                                             ArrayRef<int64_t> perm,
                                             int64_t dim0, int64_t dim1) {
  assert(perm.size() == 2 &&
         "Unexpected non-rank 2 single subgroup dimension order");
  SmallVector<int64_t> res = getIdentityPerm(rank);
  if (perm[0] > perm[1]) {
    std::swap(dim0, dim1);
  }
  if (dim0 > dim1) {
    res[dim0] = dim1;
    res[dim1] = dim0;
  }
  return res;
}

/// Constructs the nested layout given the layout for a single subgroup and the
/// subgroup/batch counts and orders, as well as the dimensions along which to
/// distribute the intrinsic's layout.
///
/// |outerDim| and |innerDim| refer to which dimensions are the outermost and
/// innermost for a canonical MK_KN_MN matrix multiply, for a particular
/// fragment. For example, for the B matrix of an MK_NK_MN matrix multiply,
/// we would have:
///   outerDim = 1 for the K dim
///   innerDim = 0 for the N dim
///
/// For something like MK_NKN_MN with multiple N dims, it would typically be:
///   outerDim = 1 for K
///   innerDim = 2 for the second N dim
///
/// Importantly these two dimensions always refer to the actual dimension
/// positions in the undistributed vector. For each fragment, this means:
///   A: [outerDim, innerDim] = [innerMostMDim, innerMostKDim]
///   B: [outerDim, innerDim] = [innerMostKDim, innerMostNDim]
///   C: [outerDim, innerDim] = [innerMostMDim, innerMostNDim]
///
/// And here inner most is referential to the iteration order, not the order
/// they appear per fragment (because there is no relationship between the
/// dimension order of M in A and in C, for example).
NestedLayoutAttr createNestedLayout(MLIRContext *context, int64_t rank,
                                    int64_t outerDim, int64_t innerDim,
                                    SmallVector<int64_t> subgroupSizes,
                                    SmallVector<int64_t> subgroupStrides,
                                    SmallVector<int64_t> batchCount,
                                    MMASingleSubgroupLayout counts) {

  LLVM_DEBUG({
    llvm::errs() << "Creating Nested Layout for::";
    llvm::errs() << "\n    outerDim = " << outerDim;
    llvm::errs() << "\n    innerDim = " << innerDim;
    llvm::errs() << "\n    subgroupSizes: ";
    llvm::interleaveComma(subgroupSizes, llvm::errs());
    llvm::errs() << "\n    subgroupStrides: ";
    llvm::interleaveComma(subgroupStrides, llvm::errs());
    llvm::errs() << "\n    batchCount: ";
    llvm::interleaveComma(batchCount, llvm::errs());
    llvm::errs() << "\n    counts.outer: ";
    llvm::interleaveComma(counts.outer, llvm::errs());
    llvm::errs() << "\n    counts.thread: ";
    llvm::interleaveComma(counts.thread, llvm::errs());
    llvm::errs() << "\n    counts.element: ";
    llvm::interleaveComma(counts.element, llvm::errs());
    llvm::errs() << "\n    counts.tstrides: ";
    llvm::interleaveComma(counts.tstrides, llvm::errs());
    llvm::errs() << "\n";
  });

  SmallVector<int64_t> outerCount =
      getUnitOfRankWithDims(rank, counts.outer, outerDim, innerDim);
  SmallVector<int64_t> threadCount =
      getUnitOfRankWithDims(rank, counts.thread, outerDim, innerDim);
  SmallVector<int64_t> threadStrides =
      getUnitOfRankWithDims(rank, counts.tstrides, outerDim, innerDim);
  SmallVector<int64_t> elementCount =
      getUnitOfRankWithDims(rank, counts.element, outerDim, innerDim);

  auto layoutAttr = NestedLayoutAttr::get(context, subgroupSizes, batchCount,
                                          outerCount, threadCount, elementCount,
                                          subgroupStrides, threadStrides);
  return layoutAttr;
}

FailureOr<std::tuple<VectorExt::VectorLayoutInterface,
                     VectorExt::VectorLayoutInterface,
                     VectorExt::VectorLayoutInterface>>
MMAScheduleAttr::getContractionLayout(VectorContractOpInfo &opInfo,
                                      linalg::LinalgOp contractOp) const {
  LLVM_DEBUG({
    llvm::errs() << "Getting mma layouts for:\n" << contractOp << "\n";
    llvm::errs() << "For schedule: " << *this << "\n";
  });

  int64_t rank = contractOp.getIteratorTypesArray().size();
  auto mmaAttr = llvm::cast<MMAAttr>(getIntrinsic());
  MLIRContext *context = getContext();

  SmallVector<int64_t> bounds = contractOp.getStaticLoopRanges();
  if (llvm::any_of(bounds,
                   [](int64_t x) { return x == ShapedType::kDynamic; })) {
    return failure();
  }

  if (!llvm::all_of(opInfo.getBatchDims(),
                    [&bounds](int64_t dim) { return bounds[dim] == 1; })) {
    LLVM_DEBUG({ llvm::errs() << "non-unit batch dimension\n"; });
    return failure();
  }

  // Get the concrete nested layout for each matrix. Note that the struct
  // MMASingleSubgroupLayout contains the partial layout for the
  // canonical (M, K) x (K, N) -> (M, N) matmul form; while the specific
  // contract op we are looking at right now may not be exactly in that form.
  // So here we need to permute/transpose the canonical layout to match with
  // the concrete contract op.

  // Note that no matter how we permute/transpose the input contraction
  // problem, the way we view the hardware warps remain the same--that is,
  // from the hardware's perspective, a single warp has the same warp ID no
  // matter what part of the contraction it works on. Similarly here, we are
  // delinearizing the linearized GPU hardware lane ID into a n-D concatenated
  // logical warp+thread using the subgroup/thread basis, so the subgroup
  // basis should remain the same for all A/B/C matrix.

  auto [intrinsicM, intrinsicN, intrinsicK] = mmaAttr.getMNKShape();

  SmallVector<int64_t, 2> subgroupMBasis;
  SmallVector<int64_t, 2> batchMSizes;
  int64_t currMCount = getSubgroupMCount();

  auto divideGreedily = [](int64_t availableSubgroups, int64_t dimSize,
                           int64_t minDimSize) -> std::pair<int64_t, int64_t> {
    int64_t dividableDim = dimSize / minDimSize;
    int64_t subgroupsUsed = std::gcd(availableSubgroups, dividableDim);
    dividableDim /= subgroupsUsed;
    int64_t batchesUsed = dividableDim;
    return {subgroupsUsed, batchesUsed};
  };

  // Greedily break up the M subgroup and batch counts along the "M" iteration
  // bounds. We distribute as many residual subgroups as possible per M dim,
  // and then divide the remaining along batch dims. The inner most M dim is
  // always the one used for the intrinsic, meaning for a valid schedule, the
  // computed batch counts and subgroup basis will satisfy totalMSize /
  // intrinsicM = product(batchMSizes) * product(subgroupMBasis)
  for (auto dim : opInfo.getMDims()) {
    // Get the number of subgroups and batches used for this dimension based
    // on the intrinsic size and the bound size.
    int64_t subgroupsUsed, batchesUsed;
    if (dim == opInfo.getMDims().back()) {
      std::tie(subgroupsUsed, batchesUsed) =
          divideGreedily(currMCount, bounds[dim], intrinsicM);
    } else {
      std::tie(subgroupsUsed, batchesUsed) =
          divideGreedily(currMCount, bounds[dim], 1);
    }
    subgroupMBasis.push_back(subgroupsUsed);
    batchMSizes.push_back(batchesUsed);
    // Update available subgroup count.
    currMCount /= subgroupsUsed;
  }

  SmallVector<int64_t, 2> subgroupNBasis;
  SmallVector<int64_t, 2> batchNSizes;
  int64_t currNCount = getSubgroupNCount();

  // Do the same for N dims.
  for (auto dim : opInfo.getNDims()) {
    // Get the number of subgroups and batches used for this dimension based
    // on the intrinsic size and the bound size.
    int64_t subgroupsUsed, batchesUsed;
    if (dim == opInfo.getNDims().back()) {
      std::tie(subgroupsUsed, batchesUsed) =
          divideGreedily(currNCount, bounds[dim], intrinsicN);
    } else {
      std::tie(subgroupsUsed, batchesUsed) =
          divideGreedily(currNCount, bounds[dim], 1);
    }
    subgroupNBasis.push_back(subgroupsUsed);
    batchNSizes.push_back(batchesUsed);
    // Update available subgroup count.
    currNCount /= subgroupsUsed;
  }

  SmallVector<int64_t> subgroupMStrides(subgroupMBasis.size());
  SmallVector<int64_t> subgroupNStrides(subgroupNBasis.size());

  auto mDimVec = opInfo.getMDims();
  llvm::SmallDenseSet<int64_t> mDims(mDimVec.begin(), mDimVec.end());
  auto nDimVec = opInfo.getNDims();
  llvm::SmallDenseSet<int64_t> nDims(nDimVec.begin(), nDimVec.end());
  // Because we currently require all batch dimensions to be unit, the
  // subgroup basis can be constructed from the M and N bases. To keep things
  // simple, the current heuristic is to distribute the loop dimensions from
  // outer to inner.
  int64_t currStride = 1;
  int64_t currM = subgroupMStrides.size() - 1;
  int64_t currN = subgroupNStrides.size() - 1;
  for (int64_t dim : llvm::reverse(llvm::seq<int64_t>(rank))) {
    if (mDims.contains(dim)) {
      subgroupMStrides[currM] = currStride;
      currStride *= subgroupMBasis[currM];
      currM--;
      continue;
    }

    if (nDims.contains(dim)) {
      subgroupNStrides[currN] = currStride;
      currStride *= subgroupNBasis[currN];
      currN--;
      continue;
    }
  }

  // C matrix layout
  auto [m, n] = opInfo.getResultMNIndex();
  int64_t cRank = opInfo.getCRank();

  // Get the M and N dims w.r.t. the dimensions of the C matrix. cMDims and
  // cNDims are the M and N dimensions of the C matrix in the order they are
  // iterated over in the contraction.
  SmallVector<int64_t> cMDims = opInfo.outMDims;
  SmallVector<int64_t> cNDims = opInfo.outNDims;
  SmallVector<int64_t> cBatchSizes(cRank, 1);
  SmallVector<int64_t> cSubgroupSizes(cRank, 1);
  SmallVector<int64_t> cSubgroupStrides(cRank, 0);
  for (auto [i, dim] : llvm::enumerate(cMDims)) {
    cBatchSizes[dim] = batchMSizes[i];
    cSubgroupSizes[dim] = subgroupMBasis[i];
    cSubgroupStrides[dim] = subgroupMStrides[i];
  }
  for (auto [i, dim] : llvm::enumerate(cNDims)) {
    cBatchSizes[dim] = batchNSizes[i];
    cSubgroupSizes[dim] = subgroupNBasis[i];
    cSubgroupStrides[dim] = subgroupNStrides[i];
  }

  auto cLayout = createNestedLayout(context, cRank, m, n,
                                    /*subgroupCount=*/cSubgroupSizes,
                                    /*subgroupStrides=*/cSubgroupStrides,
                                    /*batchCount=*/cBatchSizes,
                                    mmaAttr.getCSingleSubgroupLayout());
  LLVM_DEBUG({ llvm::errs() << "C layout: " << cLayout << "\n"; });

  // A matrix layout
  auto [afm, bfn] = opInfo.getOperandMNIndex();
  auto [afk, bfk] = opInfo.getOperandKIndex();

  int64_t aRank = opInfo.getARank();

  SmallVector<int64_t> aMDims = opInfo.lhsMDims;
  SmallVector<int64_t> aBatchSizes(aRank, 1);
  SmallVector<int64_t> aSubgroupSizes(aRank, 1);
  SmallVector<int64_t> aSubgroupStrides(aRank, 0);
  for (auto [i, dim] : llvm::enumerate(aMDims)) {
    aBatchSizes[dim] = batchMSizes[i];
    aSubgroupSizes[dim] = subgroupMBasis[i];
    aSubgroupStrides[dim] = subgroupMStrides[i];
  }
  for (auto [kDim, lhsKDim] :
       llvm::zip_equal(opInfo.getKDims(), opInfo.lhsKDim)) {
    aBatchSizes[lhsKDim] = bounds[kDim];
  }
  aBatchSizes[afk] = bounds[opInfo.getKDims().back()] / intrinsicK;

  auto aLayout = createNestedLayout(context, aRank, afm, afk,
                                    /*subgroupCount=*/aSubgroupSizes,
                                    /*subgroupStrides=*/aSubgroupStrides,
                                    /*batchCount=*/aBatchSizes,
                                    mmaAttr.getASingleSubgroupLayout());
  LLVM_DEBUG({ llvm::errs() << "A layout: " << aLayout << "\n"; });

  int64_t bRank = opInfo.getBRank();

  SmallVector<int64_t> bNDims = opInfo.rhsNDims;
  SmallVector<int64_t> bBatchSizes(bRank, 1);
  SmallVector<int64_t> bSubgroupSizes(bRank, 1);
  SmallVector<int64_t> bSubgroupStrides(bRank, 0);
  for (auto [i, dim] : llvm::enumerate(bNDims)) {
    bBatchSizes[dim] = batchNSizes[i];
    bSubgroupSizes[dim] = subgroupNBasis[i];
    bSubgroupStrides[dim] = subgroupNStrides[i];
  }
  for (auto [kDim, rhsKDim] :
       llvm::zip_equal(opInfo.getKDims(), opInfo.rhsKDim)) {
    bBatchSizes[rhsKDim] = bounds[kDim];
  }
  bBatchSizes[bfk] = bounds[opInfo.getKDims().back()] / intrinsicK;

  auto bLayout = createNestedLayout(context, bRank, bfk, bfn,
                                    /*subgroupCount=*/bSubgroupSizes,
                                    /*subgroupStrides=*/bSubgroupStrides,
                                    /*batchCount=*/bBatchSizes,
                                    mmaAttr.getBSingleSubgroupLayout());
  LLVM_DEBUG({ llvm::errs() << "B layout: " << bLayout << "\n"; });

  std::tuple<VectorLayoutInterface, VectorLayoutInterface,
             VectorLayoutInterface>
      result = {aLayout, bLayout, cLayout};
  return result;
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

constexpr StringLiteral kMmaKindName = "mma_kind";

IREE::GPU::MmaInterfaceAttr LoweringConfigAttr::getMmaKind() const {
  return getAttributes().getAs<IREE::GPU::MmaInterfaceAttr>(kMmaKindName);
}

void LoweringConfigAttr::setMmaKind(MLIRContext *context,
                                    SmallVectorImpl<NamedAttribute> &attrs,
                                    IREE::GPU::MmaInterfaceAttr kind) {
  attrs.emplace_back(StringAttr::get(context, kMmaKindName), kind);
}

// TODO: Merge subgroup counts functionality into subgroup tiling level
//       lowering, when we have it implemented.
constexpr StringLiteral kSubgroupMCountName = "subgroup_m_count";
constexpr StringLiteral kSubgroupNCountName = "subgroup_n_count";

std::optional<int64_t> LoweringConfigAttr::getSubgroupMCount() const {
  auto subgroup_m_count_attr =
      getAttributes().getAs<IntegerAttr>(kSubgroupMCountName);
  if (!subgroup_m_count_attr) {
    return std::nullopt;
  }
  return subgroup_m_count_attr.getInt();
}

std::optional<int64_t> LoweringConfigAttr::getSubgroupNCount() const {
  auto subgroup_n_count_attr =
      getAttributes().getAs<IntegerAttr>(kSubgroupNCountName);
  if (!subgroup_n_count_attr) {
    return std::nullopt;
  }
  return subgroup_n_count_attr.getInt();
}

void LoweringConfigAttr::setSubgroupMCount(
    MLIRContext *context, SmallVectorImpl<NamedAttribute> &attrs,
    int64_t subgroup_m_count) {
  attrs.emplace_back(
      StringAttr::get(context, kSubgroupMCountName),
      IntegerAttr::get(IntegerType::get(context, 64), subgroup_m_count));
}

void LoweringConfigAttr::setSubgroupNCount(
    MLIRContext *context, SmallVectorImpl<NamedAttribute> &attrs,
    int64_t subgroup_n_count) {
  attrs.emplace_back(
      StringAttr::get(context, kSubgroupNCountName),
      IntegerAttr::get(IntegerType::get(context, 64), subgroup_n_count));
}

constexpr StringLiteral kPromoteOperandsName = "promote_operands";

std::optional<SmallVector<int64_t>>
LoweringConfigAttr::getPromotedOperandList() const {
  auto array = getAttributes().getAs<ArrayAttr>(kPromoteOperandsName);
  if (!array) {
    return std::nullopt;
  }
  return getIntegerVector(array);
}

void LoweringConfigAttr::setPromotedOperandList(
    MLIRContext *context, SmallVectorImpl<NamedAttribute> &attrs,
    ArrayRef<int64_t> operands) {
  Builder b(context);
  attrs.emplace_back(StringAttr::get(context, kPromoteOperandsName),
                     b.getI64ArrayAttr(operands));
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
