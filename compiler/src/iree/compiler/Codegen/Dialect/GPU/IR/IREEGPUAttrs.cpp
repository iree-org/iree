// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"

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
                                           MFMAIntrinsic type) {
  Type f16 = Float16Type::get(context);
  Type f32 = Float32Type::get(context);
  switch (type) {
  case MFMAIntrinsic::F16_16x16x16_F32: {
    return OpaqueMmaLayout{16, 16, 16, f16, f16, f32};
  }
  case MFMAIntrinsic::F16_32x32x8_F32: {
    return OpaqueMmaLayout{32, 32, 8, f16, f16, f32};
  }
  }
  llvm_unreachable("unhandled mfma layout type");
  return OpaqueMmaLayout{};
}

static ConcreteMmaLayout getConcreteMFMALayout(MLIRContext *context,
                                               MFMAIntrinsic type) {
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
  case MFMAIntrinsic::F16_16x16x16_F32: {
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
  case MFMAIntrinsic::F16_32x32x8_F32: {
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
  }
  llvm_unreachable("unhandled concrete mfma type");
  return ConcreteMmaLayout{};
}

//===----------------------------------------------------------------------===//
// MFMA Attributes
//===----------------------------------------------------------------------===//

Attribute MFMAAttr::parse(AsmParser &p, Type type) {
  if (failed(p.parseLess()))
    return {};

  FailureOr<MFMAIntrinsicAttr> mfmaIntrinsic =
      FieldParser<MFMAIntrinsicAttr>::parse(p);
  if (failed(mfmaIntrinsic)) {
    p.emitError(p.getCurrentLocation(), "failed to parse mfma type identifier");
    return {};
  }

  if (failed(p.parseGreater()))
    return {};

  return get(p.getContext(), mfmaIntrinsic->getValue());
}

void MFMAAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  os << stringifyMFMAIntrinsic(getIntrinsic().getValue());
  os << ">";
}

MFMAAttr MFMAAttr::get(MLIRContext *context, MFMAIntrinsic type) {
  auto layout = getOpaqueMFMALayout(context, type);
  return Base::get(context, MFMAIntrinsicAttr::get(context, type), layout.mSize,
                   layout.nSize, layout.kSize, layout.aType, layout.bType,
                   layout.cType);
}

std::tuple<VectorType, VectorType, VectorType>
MFMAAttr::getABCVectorTypes() const {
  switch (getIntrinsic().getValue()) {
  case MFMAIntrinsic::F16_16x16x16_F32: {
    auto aType = VectorType::get({4}, getAType());
    auto bType = VectorType::get({4}, getBType());
    auto cType = VectorType::get({4}, getCType());
    return std::make_tuple(aType, bType, cType);
  }
  case MFMAIntrinsic::F16_32x32x8_F32: {
    auto aType = VectorType::get({4}, getAType());
    auto bType = VectorType::get({4}, getBType());
    auto cType = VectorType::get({4, 4}, getCType());
    return std::make_tuple(aType, bType, cType);
  }
  }
  // This should not happen but just to make GCC happy.
  return std::make_tuple(VectorType{}, VectorType{}, VectorType{});
}

FailureOr<std::tuple<VectorLayoutInterface, VectorLayoutInterface,
                     VectorLayoutInterface>>
MFMAAttr::getContractionLayout(vector::ContractionOp contract) const {
  ConcreteMmaLayout layout =
      getConcreteMFMALayout(contract->getContext(), getIntrinsic().getValue());
  return IREE::GPU::getContractionLayout(contract, layout);
}

int64_t MFMAAttr::getBlockSize() const {
  switch (getIntrinsic().getValue()) {
  case MFMAIntrinsic::F16_16x16x16_F32: {
    return 1;
  }
  case MFMAIntrinsic::F16_32x32x8_F32: {
    return 1;
  }
  }
  // This should not happen but just to make GCC happy.
  return 0;
}

MFMAAttr::OuterThreadElement MFMAAttr::getAOuterThreadElementCount() const {
  switch (getIntrinsic().getValue()) {
  case MFMAIntrinsic::F16_16x16x16_F32: {
    return {/*outer=*/{1, 1}, /*thread=*/{16, 4}, /*element=*/{1, 4}};
  }
  case MFMAIntrinsic::F16_32x32x8_F32: {
    return {/*outer=*/{1, 1}, /*thread=*/{32, 2}, /*element=*/{1, 4}};
  }
  }
  return {};
}

MFMAAttr::OuterThreadElement MFMAAttr::getBOuterThreadElementCount() const {
  switch (getIntrinsic().getValue()) {
  case MFMAIntrinsic::F16_16x16x16_F32: {
    return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*element=*/{4, 1}};
  }
  case MFMAIntrinsic::F16_32x32x8_F32: {
    return {/*outer=*/{1, 1}, /*thread=*/{2, 32}, /*element=*/{4, 1}};
  }
  }
  return {};
}

MFMAAttr::OuterThreadElement MFMAAttr::getCOuterThreadElementCount() const {
  switch (getIntrinsic().getValue()) {
  case MFMAIntrinsic::F16_16x16x16_F32: {
    return {/*outer=*/{1, 1}, /*thread=*/{4, 16}, /*element=*/{4, 1}};
  }
  case MFMAIntrinsic::F16_32x32x8_F32: {
    return {/*outer=*/{4, 1}, /*thread=*/{2, 32}, /*element=*/{4, 1}};
  }
  }
  return {};
}

MFMAAttr::OuterThreadElement MFMAAttr::getAOuterThreadElementOrder() const {
  switch (getIntrinsic().getValue()) {
  case MFMAIntrinsic::F16_16x16x16_F32:
  case MFMAIntrinsic::F16_32x32x8_F32: {
    return {/*outer=*/{0, 1}, /*thread=*/{1, 0}, /*element=*/{0, 1}};
  }
  }
  return {};
}

MFMAAttr::OuterThreadElement MFMAAttr::getBOuterThreadElementOrder() const {
  switch (getIntrinsic().getValue()) {
  case MFMAIntrinsic::F16_16x16x16_F32:
  case MFMAIntrinsic::F16_32x32x8_F32: {
    return {/*outer=*/{0, 1}, /*thread=*/{0, 1}, /*element=*/{1, 0}};
  }
  }
  return {};
}

MFMAAttr::OuterThreadElement MFMAAttr::getCOuterThreadElementOrder() const {
  switch (getIntrinsic().getValue()) {
  case MFMAIntrinsic::F16_16x16x16_F32:
  case MFMAIntrinsic::F16_32x32x8_F32: {
    return {/*outer=*/{0, 1}, /*thread=*/{0, 1}, /*element=*/{1, 0}};
  }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// MMA Schedule Attributes
//===----------------------------------------------------------------------===//

std::optional<std::tuple<VectorExt::VectorLayoutInterface,
                         VectorExt::VectorLayoutInterface,
                         VectorExt::VectorLayoutInterface>>
MMAScheduleAttr::getContractionLayout(vector::ContractionOp contractOp) const {
  VectorContractOpInfo opInfo(contractOp);
  if (opInfo.getOpKind() == VectorContractOpInfo::OpKind::UNKNOWN)
    return std::nullopt;

  auto [aM, bN] = *opInfo.getOperandMNIndex();
  auto [aK, bK] = *opInfo.getOperandKIndex();
  auto [cM, cN] = *opInfo.getResultMNIndex();
  SmallVector<int64_t, 2> aPermute = {aM, aK};
  SmallVector<int64_t, 2> bPermute = {bK, bN};
  SmallVector<int64_t, 2> cPermute = {cM, cN};

  // TODO: drop this and permute the following fields
  if (!isIdentityPermutation(aPermute) || !isIdentityPermutation(bPermute) ||
      !isIdentityPermutation(cPermute))
    return std::nullopt;

  auto mfmaAttr = llvm::cast<MFMAAttr>(getIntrinsic());

  // C matrix layout
  MFMAAttr::OuterThreadElement cCounts = mfmaAttr.getCOuterThreadElementCount();
  MFMAAttr::OuterThreadElement cOrders = mfmaAttr.getCOuterThreadElementOrder();

  SmallVector<int64_t, 2> cSubgroupPerWorkgroup = {getSubgroupMCount(),
                                                   getSubgroupNCount()};
  SmallVector<int64_t, 2> cBatchesPerSubgroup = {getSubgroupMTileCount(),
                                                 getSubgroupNTileCount()};
  SmallVector<int64_t, 2> cSubgroupOrder = {0, 1};
  SmallVector<int64_t, 2> cBatchOrder = {0, 1};
  SmallVector<int64_t, 2> cSubgroupBasis = cSubgroupPerWorkgroup;
  SmallVector<int64_t, 2> cThreadBasis = cCounts.thread;

  auto cLayout = NestedLayoutAttr::get(
      getContext(), cSubgroupPerWorkgroup, cSubgroupOrder, cBatchesPerSubgroup,
      cBatchOrder, cCounts.outer, cOrders.outer, cCounts.thread, cOrders.thread,
      cCounts.element, cOrders.element, cSubgroupBasis, cThreadBasis);

  // A matrix layout
  MFMAAttr::OuterThreadElement aCounts = mfmaAttr.getAOuterThreadElementCount();
  MFMAAttr::OuterThreadElement aOrders = mfmaAttr.getAOuterThreadElementOrder();

  SmallVector<int64_t, 2> aSubgroupPerWorkgroup = {getSubgroupMCount(), 1};
  SmallVector<int64_t, 2> aBatchesPerSubgroup = {getSubgroupMTileCount(),
                                                 getSubgroupKTileCount()};
  SmallVector<int64_t, 2> aSubgroupOrder = {0, 1};
  SmallVector<int64_t, 2> aBatchOrder = {0, 1};
  SmallVector<int64_t, 2> aSubgroupBasis = aSubgroupPerWorkgroup;
  SmallVector<int64_t, 2> aThreadBasis = aCounts.thread;

  auto aLayout = NestedLayoutAttr::get(
      getContext(), aSubgroupPerWorkgroup, aSubgroupOrder, aBatchesPerSubgroup,
      aBatchOrder, aCounts.outer, aOrders.outer, aCounts.thread, aOrders.thread,
      aCounts.element, aOrders.element, aSubgroupBasis, aThreadBasis);

  // B matrix layout
  MFMAAttr::OuterThreadElement bCounts = mfmaAttr.getBOuterThreadElementCount();
  MFMAAttr::OuterThreadElement bOrders = mfmaAttr.getBOuterThreadElementOrder();

  SmallVector<int64_t, 2> bSubgroupPerWorkgroup = {1, getSubgroupNCount()};
  SmallVector<int64_t, 2> bBatchesPerSubgroup = {getSubgroupKTileCount(),
                                                 getSubgroupNTileCount()};
  SmallVector<int64_t, 2> bSubgroupOrder = {0, 1};
  SmallVector<int64_t, 2> bBatchOrder = {0, 1};
  SmallVector<int64_t, 2> bSubgroupBasis = bSubgroupPerWorkgroup;
  SmallVector<int64_t, 2> bThreadBasis = bCounts.thread;

  auto bLayout = NestedLayoutAttr::get(
      getContext(), bSubgroupPerWorkgroup, bSubgroupOrder, bBatchesPerSubgroup,
      bBatchOrder, bCounts.outer, bOrders.outer, bCounts.thread, bOrders.thread,
      bCounts.element, bOrders.element, bSubgroupBasis, bThreadBasis);

  return std::make_tuple(aLayout, bLayout, cLayout);
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
