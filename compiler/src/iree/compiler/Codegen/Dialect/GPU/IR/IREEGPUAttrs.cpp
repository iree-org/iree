// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include <numeric>

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "iree-gpu-attrs"

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
  Type f16 = Float16Type::get(context);
  Type f32 = Float32Type::get(context);
  switch (type) {
  case MMAIntrinsic::MFMA_F16_16x16x16_F32: {
    return OpaqueMmaLayout{16, 16, 16, f16, f16, f32};
  }
  case MMAIntrinsic::MFMA_F16_32x32x8_F32: {
    return OpaqueMmaLayout{32, 32, 8, f16, f16, f32};
  }
  case MMAIntrinsic::WMMA_F16_16x16x16_F32: {
    return OpaqueMmaLayout{16, 16, 16, f16, f16, f32};
  }
  case MMAIntrinsic::WMMA_F16_16x16x16_F16: {
    return OpaqueMmaLayout{16, 16, 16, f16, f16, f16};
  }
  }
  llvm_unreachable("unhandled mfma layout type");
  return OpaqueMmaLayout{};
}

static ConcreteMmaLayout getConcreteMFMALayout(MLIRContext *context,
                                               MMAIntrinsic type) {
  auto opaqueLayout = getOpaqueMFMALayout(context, type);

  LayoutDimensionAttr laneX =
      LayoutDimensionAttr::get(context, LayoutDimension::LANEX);
  LayoutDimensionAttr laneY =
      LayoutDimensionAttr::get(context, LayoutDimension::LANEY);
  LayoutDimensionAttr laneZ =
      LayoutDimensionAttr::get(context, LayoutDimension::LANEZ);
  LayoutDimensionAttr vectorX =
      LayoutDimensionAttr::get(context, LayoutDimension::VECTORX);
  LayoutDimensionAttr vectorY =
      LayoutDimensionAttr::get(context, LayoutDimension::VECTORY);
  LayoutDimensionAttr vectorZ =
      LayoutDimensionAttr::get(context, LayoutDimension::VECTORZ);
  (void)laneZ, (void)vectorZ;
  switch (type) {
  case MMAIntrinsic::MFMA_F16_16x16x16_F32: {
    // #outer = #iree_vector_ext.per_dim_layout<[LANEX], [16]>
    // #inner = #iree_vector_ext.per_dim_layout<[LANEY, VECTORX], [4, 4]>
    // #layout_a = #iree_vector_ext.layout<#outer, #inner>
    // #layout_b = #iree_vector_ext.layout<#inner, #outer>
    // #layout_c = #iree_vector_ext.layout<#inner, #outer>

    auto outer = PerDimLayoutAttr::get(context, {laneX}, {16});
    auto inner = PerDimLayoutAttr::get(context, {laneY, vectorX}, {4, 4});
    auto aMLayout = outer;
    auto aKLayout = inner;
    auto bKLayout = inner;
    auto bNLayout = outer;
    auto cMLayout = inner;
    auto cNLayout = outer;
    return ConcreteMmaLayout{opaqueLayout, aMLayout, aKLayout, bKLayout,
                             bNLayout,     cMLayout, cNLayout};
  }
  case MMAIntrinsic::MFMA_F16_32x32x8_F32: {
    // #outer = #iree_vector_ext.per_dim_layout<[LANEX], [32]>
    // #inner1 = #iree_vector_ext.per_dim_layout<[LANEY, VECTORX], [2, 4]>
    // #inner2 = #iree_vector_ext.per_dim_layout<[VECTORY, LANEY, VECTORX],
    //                                           [4, 2, 4]>
    // #layout_a = #iree_vector_ext.layout<#outer, #inner1>
    // #layout_b = #iree_vector_ext.layout<#inner1, #outer>
    // #layout_c = #iree_vector_ext.layout<#inner2, #outer>

    auto outer = PerDimLayoutAttr::get(context, {laneX}, {32});
    auto inner = PerDimLayoutAttr::get(context, {laneY, vectorX}, {2, 4});
    auto aMLayout = outer;
    auto aKLayout = inner;
    auto bKLayout = inner;
    auto bNLayout = outer;
    auto cMLayout =
        PerDimLayoutAttr::get(context, {vectorY, laneY, vectorX}, {4, 2, 4});
    auto cNLayout = outer;
    return ConcreteMmaLayout{opaqueLayout, aMLayout, aKLayout, bKLayout,
                             bNLayout,     cMLayout, cNLayout};
  }
  case MMAIntrinsic::WMMA_F16_16x16x16_F32:
  case MMAIntrinsic::WMMA_F16_16x16x16_F16: {
    // #outer = #iree_vector_ext.per_dim_layout<[LANEX], [16]>
    // #inner = #iree_vector_ext.per_dim_layout<[LANEY, VECTORX], [4, 4]>
    // #layout_a = #iree_vector_ext.layout<#outer, #inner>
    // #layout_b = #iree_vector_ext.layout<#inner, #outer>
    // #layout_c = #iree_vector_ext.layout<#inner, #outer>

    auto outer = PerDimLayoutAttr::get(context, {laneX}, {16});
    auto inner = PerDimLayoutAttr::get(context, {laneY, vectorX}, {1, 16});
    auto aMLayout = outer;
    auto aKLayout = inner;
    auto bKLayout = inner;
    auto bNLayout = outer;
    auto cMLayout =
        PerDimLayoutAttr::get(context, {vectorY, laneY, vectorX}, {8, 2, 1});
    auto cNLayout = PerDimLayoutAttr::get(context, {laneX}, {16});
    return ConcreteMmaLayout{opaqueLayout, aMLayout, aKLayout, bKLayout,
                             bNLayout,     cMLayout, cNLayout};
  }
  default: {
    break;
  }
  }
  llvm_unreachable("unhandled concrete mma type");
  return ConcreteMmaLayout{};
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
  case MMAIntrinsic::MFMA_F16_16x16x16_F32: {
    auto aType = VectorType::get({4}, getAType());
    auto bType = VectorType::get({4}, getBType());
    auto cType = VectorType::get({4}, getCType());
    return std::make_tuple(aType, bType, cType);
  }
  case MMAIntrinsic::MFMA_F16_32x32x8_F32: {
    auto aType = VectorType::get({4}, getAType());
    auto bType = VectorType::get({4}, getBType());
    auto cType = VectorType::get({16}, getCType());
    return std::make_tuple(aType, bType, cType);
  }
  case MMAIntrinsic::WMMA_F16_16x16x16_F32:
  case MMAIntrinsic::WMMA_F16_16x16x16_F16: {
    auto aType = VectorType::get({16}, getAType());
    auto bType = VectorType::get({16}, getBType());
    auto cType = VectorType::get({8}, getCType());
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
  case MMAIntrinsic::MFMA_F16_16x16x16_F32:
  case MMAIntrinsic::MFMA_F16_32x32x8_F32:
  case MMAIntrinsic::WMMA_F16_16x16x16_F16:
  case MMAIntrinsic::WMMA_F16_16x16x16_F32: {
    return 1;
  }
  }
  // This should not happen but just to make GCC happy.
  return 0;
}

int64_t MMAAttr::getSubgroupSize() const {
  switch (getIntrinsic().getValue()) {
  case MMAIntrinsic::MFMA_F16_16x16x16_F32:
  case MMAIntrinsic::MFMA_F16_32x32x8_F32: {
    return 64;
  }
  case MMAIntrinsic::WMMA_F16_16x16x16_F32:
  case MMAIntrinsic::WMMA_F16_16x16x16_F16: {
    return 32;
  }
  }
  // This should not happen but just to make GCC happy.
  return 0;
}

MMAAttr::SingleSubgroupLayout MMAAttr::getASingleSubgroupLayout() const {
  switch (getIntrinsic().getValue()) {
  case MMAIntrinsic::MFMA_F16_16x16x16_F32: {
    return {/*outer=*/{1, 1}, /*thread=*/{16, 4}, /*strides=*/{1, 16},
            /*element=*/{1, 4}};
  }
  case MMAIntrinsic::MFMA_F16_32x32x8_F32: {
    return {/*outer=*/{1, 1}, /*thread=*/{32, 2}, /*strides=*/{1, 32},
            /*element=*/{1, 4}};
  }
  case MMAIntrinsic::WMMA_F16_16x16x16_F32:
  case MMAIntrinsic::WMMA_F16_16x16x16_F16: {
    return {/*outer=*/{1, 1}, /*thread=*/{16, 1}, /*strides=*/{1, 16},
            /*element=*/{1, 16}};
  }
  }
  return {};
}

MMAAttr::SingleSubgroupLayout MMAAttr::getBSingleSubgroupLayout() const {
  switch (getIntrinsic().getValue()) {
  case MMAIntrinsic::MFMA_F16_16x16x16_F32: {
    return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*strides=*/{16, 1},
            /*element=*/{4, 1}};
  }
  case MMAIntrinsic::MFMA_F16_32x32x8_F32: {
    return {/*outer=*/{1, 1}, /*thread=*/{2, 32}, /*strides=*/{32, 1},
            /*element=*/{4, 1}};
  }
  case MMAIntrinsic::WMMA_F16_16x16x16_F32:
  case MMAIntrinsic::WMMA_F16_16x16x16_F16: {
    return {/*outer=*/{1, 1}, /*thread=*/{1, 16}, /*strides=*/{16, 1},
            /*element=*/{16, 1}};
  }
  }
  return {};
}

MMAAttr::SingleSubgroupLayout MMAAttr::getCSingleSubgroupLayout() const {
  switch (getIntrinsic().getValue()) {
  case MMAIntrinsic::MFMA_F16_16x16x16_F32: {
    return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*strides=*/{16, 1},
            /*element=*/{4, 1}};
  }
  case MMAIntrinsic::MFMA_F16_32x32x8_F32: {
    return {/*outer=*/{4, 1}, /*thread=*/{2, 32}, /*strides=*/{32, 1},
            /*element=*/{4, 1}};
  }
  case MMAIntrinsic::WMMA_F16_16x16x16_F32:
  case MMAIntrinsic::WMMA_F16_16x16x16_F16: {
    return {/*outer=*/{8, 1}, /*thread=*/{2, 16}, /*strides=*/{16, 1},
            /*element=*/{1, 1}};
  }
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
  case MMAIntrinsic::MFMA_F16_16x16x16_F32:
  case MMAIntrinsic::MFMA_F16_32x32x8_F32: {
    auto [m, n, k] = getMNKShape();
    return builder
        .create<amdgpu::MFMAOp>(loc, resultType, m, n, k, getBlockSize(), lhs,
                                rhs, acc)
        .getResult();
  }
  case MMAIntrinsic::WMMA_F16_16x16x16_F32:
  case MMAIntrinsic::WMMA_F16_16x16x16_F16: {
    return builder.create<amdgpu::WMMAOp>(loc, resultType, lhs, rhs, acc)
        .getResult();
  }
  }
  return failure();
}

static LogicalResult populateCanonicalOffsetsSizesAndStrides(
    OpBuilder &builder, Location loc, Value laneId,
    ArrayRef<int64_t> permutation, MMAAttr::SingleSubgroupLayout subgroupLayout,
    SmallVector<OpFoldResult> &canonicalOffsets,
    SmallVector<OpFoldResult> &canonicalSizes,
    SmallVector<OpFoldResult> &canonicalStrides) {
  SmallVector<int64_t> rankReducedShape;
  for (auto [outer, thread, element] :
       llvm::zip_equal(subgroupLayout.outer, subgroupLayout.thread,
                       subgroupLayout.element)) {
    if (outer != 1) {
      // TODO: Support this case. Might need a reshape since this makes the
      // slice non-contigious.
      return failure();
    }
    rankReducedShape.push_back(thread * element);
  }

  if (permutation.size() != rankReducedShape.size()) {
    return failure();
  }

  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  canonicalStrides.append(rankReducedShape.size(), one);

  // vtid: virtual thread id
  // tid: lane id
  // vtid = (tid floordiv stride_i) mod size_i.
  SmallVector<OpFoldResult> vtids;
  for (auto [dimSize, dimStride] :
       llvm::zip(subgroupLayout.thread, subgroupLayout.tstrides)) {
    if (dimSize == 1) {
      vtids.push_back(zero);
    }

    // (tid floordiv stride) mod size
    AffineExpr tidExpr = builder.getAffineDimExpr(0);
    AffineMap vtidMap = AffineMap::get(
        /*dims=*/1, /*syms=*/0, tidExpr.floorDiv(dimStride) % dimSize);
    Value vtid = builder.create<affine::AffineApplyOp>(loc, vtidMap, laneId);
    vtids.push_back(vtid);
  }

  int64_t idx = 0;
  for (auto [thread, element] :
       llvm::zip_equal(subgroupLayout.thread, subgroupLayout.element)) {
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
  if (getIntrinsic().getValue() != MMAIntrinsic::MFMA_F16_16x16x16_F32) {
    return failure();
  }

  MMAAttr::SingleSubgroupLayout subgroupLayout;
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
                                    MMAAttr::SingleSubgroupLayout counts) {

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
MMAScheduleAttr::getContractionLayout(vector::ContractionOp contractOp) const {
  VectorContractOpInfo opInfo(contractOp);
  LLVM_DEBUG({
    llvm::errs() << "Getting mma layouts for:\n" << contractOp << "\n";
    llvm::errs() << "For schedule: " << *this << "\n";
  });
  if (opInfo.getKDims().size() != 1) {
    return contractOp->emitError("Unimplemented: > 1 k dims");
  }

  int64_t rank = contractOp.getIteratorTypesArray().size();
  auto mmaAttr = llvm::cast<MMAAttr>(getIntrinsic());
  MLIRContext *context = getContext();

  SmallVector<int64_t> bounds;
  contractOp.getIterationBounds(bounds);
  int64_t batchCount = opInfo.getBatchCount();
  if (batchCount == 1 && bounds[0] != 1) {
    LLVM_DEBUG({ llvm::errs() << "non-unit batch dimension\n"; });
    return failure();
  }

  // Get the concrete nested layout for each matrix. Note that the struct
  // MMAAttr::SingleSubgroupLayout contains the partial layout for the
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

static SmallVector<int64_t> getTileSizes(DictionaryAttr config,
                                         GPU::TilingLevel level) {
  auto sizes = config.getAs<ArrayAttr>(getTilingLevelName(level));
  if (!sizes || !llvm::all_of(sizes.getValue(), llvm::IsaPred<IntegerAttr>)) {
    return {};
  }
  return llvm::map_to_vector(sizes.getValue(), [](Attribute s) -> int64_t {
    return cast<IntegerAttr>(s).getInt();
  });
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

constexpr StringLiteral kMmaKindName = "mma_kind";

IREE::GPU::MmaInterfaceAttr LoweringConfigAttr::getMmaKind() const {
  return getAttributes().getAs<IREE::GPU::MmaInterfaceAttr>(kMmaKindName);
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
// Attribute Registration
//===----------------------------------------------------------------------===//

void IREEGPUDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::GPU
