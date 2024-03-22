// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include <numeric>

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
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
  // Check https://github.com/ROCm/amd_matrix_instruction_calculator for
  // instruction details. Note here we are returning the number elements, while
  // amd_matrix_instruction_calculator tells us about the number of 32-bit
  // registers. So need to adjust accordingly. All vectors should be 1-D.
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
    auto cType = VectorType::get({16}, getCType());
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

MFMAAttr::SingleSubgroupLayout MFMAAttr::getASingleSubgroupLayoutCount() const {
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

MFMAAttr::SingleSubgroupLayout MFMAAttr::getBSingleSubgroupLayoutCount() const {
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

MFMAAttr::SingleSubgroupLayout MFMAAttr::getCSingleSubgroupLayoutCount() const {
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

MFMAAttr::SingleSubgroupLayout MFMAAttr::getASingleSubgroupLayoutOrder() const {
  switch (getIntrinsic().getValue()) {
  case MFMAIntrinsic::F16_16x16x16_F32:
  case MFMAIntrinsic::F16_32x32x8_F32: {
    return {/*outer=*/{0, 1}, /*thread=*/{1, 0}, /*element=*/{0, 1}};
  }
  }
  return {};
}

MFMAAttr::SingleSubgroupLayout MFMAAttr::getBSingleSubgroupLayoutOrder() const {
  switch (getIntrinsic().getValue()) {
  case MFMAIntrinsic::F16_16x16x16_F32:
  case MFMAIntrinsic::F16_32x32x8_F32: {
    return {/*outer=*/{0, 1}, /*thread=*/{0, 1}, /*element=*/{1, 0}};
  }
  }
  return {};
}

MFMAAttr::SingleSubgroupLayout MFMAAttr::getCSingleSubgroupLayoutOrder() const {
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

SmallVector<int64_t> getUnitOfRankWithDims(int64_t rank,
                                           ArrayRef<int64_t> counts,
                                           int64_t dim0, int64_t dim1) {
  SmallVector<int64_t> res(rank, 1);
  res[dim0] = counts[0];
  res[dim1] = counts[1];
  return res;
}

SmallVector<int64_t> getIdentityPerm(int64_t rank) {
  return llvm::to_vector(llvm::seq(static_cast<int64_t>(0), rank));
}

SmallVector<int64_t> getIdentityPermWithSwap(int64_t rank,
                                             ArrayRef<int64_t> perm,
                                             int64_t dim0, int64_t dim1) {
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

NestedLayoutAttr permuteAndCreateNestedLayout(
    MLIRContext *context, int64_t rank, int64_t dim0, int64_t dim1,
    ArrayRef<int64_t> permute, SmallVector<int64_t> subgroupCount,
    SmallVector<int64_t> subgroupOrder, SmallVector<int64_t> batchCount,
    SmallVector<int64_t> batchOrder, MFMAAttr::SingleSubgroupLayout counts,
    MFMAAttr::SingleSubgroupLayout orders, ArrayRef<int64_t> subgroupBasis,
    ArrayRef<bool> subgroupActiveIds) {

  LLVM_DEBUG({
    llvm::errs() << "Given:";
    llvm::errs() << "\n    dim0 = " << dim0;
    llvm::errs() << "\n    dim1 = " << dim1;
    llvm::errs() << "\n    subgroupCount: ";
    llvm::interleaveComma(subgroupCount, llvm::errs());
    llvm::errs() << "\n    subgroupOrder: ";
    llvm::interleaveComma(subgroupOrder, llvm::errs());
    llvm::errs() << "\n    batchCount: ";
    llvm::interleaveComma(batchCount, llvm::errs());
    llvm::errs() << "\n    batchOrder: ";
    llvm::interleaveComma(batchOrder, llvm::errs());
    llvm::errs() << "\n    counts.outer: ";
    llvm::interleaveComma(counts.outer, llvm::errs());
    llvm::errs() << "\n    orders.outer: ";
    llvm::interleaveComma(orders.outer, llvm::errs());
    llvm::errs() << "\n    counts.thread: ";
    llvm::interleaveComma(counts.thread, llvm::errs());
    llvm::errs() << "\n    orders.thread: ";
    llvm::interleaveComma(orders.thread, llvm::errs());
    llvm::errs() << "\n    counts.element: ";
    llvm::interleaveComma(counts.element, llvm::errs());
    llvm::errs() << "\n    orders.element: ";
    llvm::interleaveComma(orders.element, llvm::errs());
    llvm::errs() << "\n    subgroupBasis: ";
    llvm::interleaveComma(subgroupBasis, llvm::errs());
    llvm::errs() << "\n    subgroupActiveIds: ";
    llvm::interleaveComma(subgroupActiveIds, llvm::errs());
    llvm::errs() << "\n";
  });

  SmallVector<int64_t> outerOrder =
      getIdentityPermWithSwap(rank, orders.outer, dim0, dim1);
  SmallVector<int64_t> threadOrder =
      getIdentityPermWithSwap(rank, orders.thread, dim0, dim1);
  SmallVector<int64_t> elementOrder =
      getIdentityPermWithSwap(rank, orders.element, dim0, dim1);

  SmallVector<int64_t> threadBasis =
      getUnitOfRankWithDims(rank, counts.thread, dim0, dim1);
  applyPermutationToVector(threadBasis, threadOrder);

  SmallVector<int64_t> outerCount =
      getUnitOfRankWithDims(rank, counts.outer, dim0, dim1);
  SmallVector<int64_t> threadCount =
      getUnitOfRankWithDims(rank, counts.thread, dim0, dim1);
  SmallVector<int64_t> elementCount =
      getUnitOfRankWithDims(rank, counts.element, dim0, dim1);

  // if (permute[0] > permute[1]) {
  //   std::swap(outerCount[dim0], outerCount[dim1]);
  //   std::swap(outerOrder[dim0], outerOrder[dim1]);
  //   std::swap(threadCount[dim0], threadCount[dim1]);
  //   std::swap(threadOrder[dim0], threadOrder[dim1]);
  //   std::swap(elementCount[dim0], elementCount[dim1]);
  //   std::swap(elementOrder[dim0], elementOrder[dim1]);
  // }

  LLVM_DEBUG({
    llvm::errs() << "\nNew layout attr:";
    llvm::errs() << "\n    subgroupCount: ";
    llvm::interleaveComma(subgroupCount, llvm::errs());
    llvm::errs() << "\n    subgroupOrder: ";
    llvm::interleaveComma(subgroupOrder, llvm::errs());
    llvm::errs() << "\n    batchCount: ";
    llvm::interleaveComma(batchCount, llvm::errs());
    llvm::errs() << "\n    batchOrder: ";
    llvm::interleaveComma(batchOrder, llvm::errs());
    llvm::errs() << "\n    outerCount: ";
    llvm::interleaveComma(outerCount, llvm::errs());
    llvm::errs() << "\n    outerOrder: ";
    llvm::interleaveComma(outerOrder, llvm::errs());
    llvm::errs() << "\n    threadCount: ";
    llvm::interleaveComma(threadCount, llvm::errs());
    llvm::errs() << "\n    threadOrder: ";
    llvm::interleaveComma(threadOrder, llvm::errs());
    llvm::errs() << "\n    elementCount: ";
    llvm::interleaveComma(elementCount, llvm::errs());
    llvm::errs() << "\n    elementOrder: ";
    llvm::interleaveComma(elementOrder, llvm::errs());
    llvm::errs() << "\n    subgroupBasis: ";
    llvm::interleaveComma(subgroupBasis, llvm::errs());
    llvm::errs() << "\n    subgroupActiveIds: ";
    llvm::interleaveComma(subgroupActiveIds, llvm::errs());
    llvm::errs() << "\n    threadBasis: ";
    llvm::interleaveComma(threadBasis, llvm::errs());
    llvm::errs() << "\n";
  });

  auto layoutAttr = NestedLayoutAttr::get(
      context, subgroupCount, subgroupOrder, batchCount, batchOrder, outerCount,
      outerOrder, threadCount, threadOrder, elementCount, elementOrder,
      subgroupBasis, subgroupActiveIds, threadBasis,
      SmallVector<bool>(threadBasis.size(), true));
  return layoutAttr;
}

std::optional<std::tuple<VectorExt::VectorLayoutInterface,
                         VectorExt::VectorLayoutInterface,
                         VectorExt::VectorLayoutInterface>>
MMAScheduleAttr::getContractionLayout(vector::ContractionOp contractOp) const {
  VectorContractOpInfo opInfo(contractOp);
  LLVM_DEBUG({
    llvm::errs() << "Getting mma layouts for:\n" << contractOp << "\n";
    llvm::errs() << "For schedule: " << *this << "\n";
  });
  if (opInfo.getOpKind() == VectorContractOpInfo::OpKind::UNKNOWN) {
    LLVM_DEBUG({ llvm::errs() << "Unknown contraction kind\n"; });
    return std::nullopt;
  }

  auto [aM, bN] = *opInfo.getOperandMNIndex();
  auto [aK, bK] = *opInfo.getOperandKIndex();
  auto [cM, cN] = *opInfo.getResultMNIndex();
  SmallVector<int64_t, 2> aPermute = {aM, aK};
  SmallVector<int64_t, 2> bPermute = {bK, bN};
  SmallVector<int64_t, 2> cPermute = {cM, cN};

  auto mfmaAttr = llvm::cast<MFMAAttr>(getIntrinsic());
  MLIRContext *context = getContext();

  SmallVector<int64_t> bounds;
  contractOp.getIterationBounds(bounds);

  int64_t batchCount = opInfo.getBatchCount();
  if (batchCount == 1 && bounds[0] != 1) {
    LLVM_DEBUG({ llvm::errs() << "non-unit batch dimension\n"; });
    return std::nullopt;
  }

  // Get the concrete nested layout for each matrix. Note that the struct
  // MFMAAttr::SingleSubgroupLayout contains the partial layout for the
  // canonical (M, K) x (K, N) -> (M, N) matmul form; while the specific
  // contract op we are looking at right now may not be exactly in that form.
  // So here we need to permute/transpose the canonical layout to match with
  // the concrete contract op.

  // Note that no matter how we permute/transpose the input contraction problem,
  // the way we view the hardware warps remain the same--that is, from the
  // hardware's perspective, a single warp has the same warp ID no matter what
  // part of the contraction it works on. Similarly here, we are delinearizing
  // the linearized GPU hardware lane ID into a n-D concatenated logical
  // warp+thread using the subgroup/thread basis, so the subgroup basis should
  // remain the same for all A/B/C matrix.

  SmallVector<int64_t, 2> subgroupMBasis;
  SmallVector<int64_t, 2> batchMSizes;
  int64_t currMCount = getSubgroupMCount();
  int64_t currMBatch = getSubgroupMTileCount();
  for (auto dim : opInfo.getMDims()) {
    int64_t threads = std::gcd(currMCount, bounds[dim]);
    subgroupMBasis.push_back(threads);
    currMCount /= threads;
    int64_t batchCount = bounds[dim] / threads;
    batchCount = batchCount >= currMBatch ? currMBatch : batchCount;
    batchMSizes.push_back(batchCount);
    currMBatch /= batchCount;
  }

  SmallVector<int64_t, 2> subgroupNBasis;
  SmallVector<int64_t, 2> batchNSizes;
  int64_t currNCount = getSubgroupNCount();
  int64_t currNBatch = getSubgroupNTileCount();
  for (auto dim : opInfo.getNDims()) {
    int64_t threads = std::gcd(currNCount, bounds[dim]);
    subgroupNBasis.push_back(threads);
    currNCount /= threads;
    int64_t batchCount = bounds[dim] / threads;
    batchCount = batchCount >= currNBatch ? currNBatch : batchCount;
    batchNSizes.push_back(batchCount);
    currNBatch /= batchCount;
  }

  SmallVector<int64_t> subgroupBasis;
  if (batchCount == 1) {
    subgroupBasis.push_back(1);
  }
  auto mDimVec = opInfo.getMDims();
  llvm::SmallDenseSet<int64_t> mDims(mDimVec.begin(), mDimVec.end());
  auto nDimVec = opInfo.getNDims();
  llvm::SmallDenseSet<int64_t> nDims(nDimVec.begin(), nDimVec.end());

  int64_t currM = 0;
  int64_t currN = 0;
  for (auto dim : llvm::seq(static_cast<int64_t>(0), opInfo.getCRank())) {
    if (mDims.contains(dim)) {
      subgroupBasis.push_back(subgroupMBasis[currM]);
      mDimVec[currM] = dim;
      currM++;
    }
    if (nDims.contains(dim)) {
      subgroupBasis.push_back(subgroupNBasis[currN]);
      nDimVec[currN] = dim;
      currN++;
    }
  }

  // For threads though, we also need to make sure the basis is consistent
  // across A, B, and C matrix. Though here we need to additionally think it
  // from the matching of how the MMA intrinsics expect the treads organize and
  // how we distribute the large input contraction problem to the threads.
  // The intrinsics expect a certain 2-D (x, y) thread layout, where it's not
  // guaranteed that y is always the fastest moving dimension. But when we
  // distribute the large input contraction problem, we always associate the
  // fastest moving dimension to the innermost thread ID dimension. Therefore,
  // we need to "adjust" the intrinsic thread shape to from the slowest moving
  // dimension to the fastest one. That is, to apply the corresponding order
  // permutation vector. Because how the intrinsics are designed, the end result
  // is actually we are basically guaranteed to see the same thread basis for A,
  // B, and C matrix. But still..

  // C matrix layout
  MFMAAttr::SingleSubgroupLayout cCounts =
      mfmaAttr.getCSingleSubgroupLayoutCount();
  MFMAAttr::SingleSubgroupLayout cOrders =
      mfmaAttr.getCSingleSubgroupLayoutOrder();

  auto [m, n] = opInfo.getResultFullMNIndex();
  int64_t cRank = opInfo.getCRank();

  LLVM_DEBUG({
    llvm::errs() << "Subgroup M Basis: ";
    llvm::interleaveComma(subgroupMBasis, llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "Subgroup N Basis: ";
    llvm::interleaveComma(subgroupNBasis, llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "Batch M Sizes: ";
    llvm::interleaveComma(batchMSizes, llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "Batch N Sizes: ";
    llvm::interleaveComma(batchNSizes, llvm::errs());
    llvm::errs() << "\n";
  });

  // Right now this is assuming at most one batch dim which is outer most and
  // unit.
  SmallVector<int64_t> cMDims = opInfo.outMDims;
  SmallVector<int64_t> cNDims = opInfo.outNDims;
  SmallVector<int64_t> cBatchSizes(cRank, 1);
  SmallVector<int64_t> cSubgroupSizes(cRank, 1);
  SmallVector<int64_t> cOverallOrder = getIdentityPerm(cRank);
  for (auto [i, dim] : llvm::enumerate(cMDims)) {
    cBatchSizes[dim] = batchMSizes[i];
    cSubgroupSizes[dim] = subgroupMBasis[i];
    cOverallOrder[dim] = mDimVec[i];
  }
  for (auto [i, dim] : llvm::enumerate(cNDims)) {
    cBatchSizes[dim] = batchNSizes[i];
    cSubgroupSizes[dim] = subgroupNBasis[i];
    cOverallOrder[dim] = nDimVec[i];
  }

  // Dummy 1 for the k dimension.
  subgroupBasis.push_back(1);

  SmallVector<bool> cActiveSubgroups(cRank + 1, true);
  cActiveSubgroups.back() = false;

  auto cLayout =
      permuteAndCreateNestedLayout(context, cRank, m, n, cPermute,
                                   /*subgroupCount=*/cSubgroupSizes,
                                   /*subgroupOrder=*/cOverallOrder,
                                   /*batchCount=*/cBatchSizes,
                                   /*batchOrder=*/cOverallOrder, cCounts,
                                   cOrders, subgroupBasis, cActiveSubgroups);
  LLVM_DEBUG({ llvm::errs() << "C layout: " << cLayout << "\n"; });

  // A matrix layout
  MFMAAttr::SingleSubgroupLayout aCounts =
      mfmaAttr.getASingleSubgroupLayoutCount();
  MFMAAttr::SingleSubgroupLayout aOrders =
      mfmaAttr.getASingleSubgroupLayoutOrder();

  auto [afm, bfn] = opInfo.getOperandFullMNIndex();
  auto [afk, bfk] = opInfo.getOperandFullKIndex();

  int64_t aRank = opInfo.getARank();

  SmallVector<int64_t> aMDims = opInfo.lhsMDims;
  SmallVector<int64_t> aBatchSizes(aRank, 1);
  SmallVector<int64_t> aSubgroupSizes(aRank, 1);
  SmallVector<int64_t> aSubgroupOrder = getIdentityPerm(aRank);
  SmallVector<int64_t> aBatchOrder = getIdentityPerm(aRank);
  for (auto [i, dim] : llvm::enumerate(aMDims)) {
    aBatchSizes[dim] = batchMSizes[i];
    aSubgroupSizes[dim] = subgroupMBasis[i];
    int64_t j = i + batchCount;
    aSubgroupOrder[dim] = j;
    aBatchOrder[dim] = j >= afk ? j + 1 : j;
  }
  aSubgroupOrder[afk] = aRank - 1;
  aBatchOrder[afk] = afk;
  aBatchSizes[afk] = getSubgroupKTileCount();

  SmallVector<bool> aActiveSubgroups(subgroupBasis.size(), false);
  for (auto mDim : mDims) {
    aActiveSubgroups[mDim] = true;
  }
  if (batchCount == 1) {
    aActiveSubgroups[0] = true;
  }
  aActiveSubgroups.back() = true;

  auto aLayout = permuteAndCreateNestedLayout(
      context, aRank, afm, afk, aPermute,
      /*subgroupCount=*/aSubgroupSizes,
      /*subgroupOrder=*/aSubgroupOrder,
      /*batchCount=*/aBatchSizes,
      /*batchOrder=*/getIdentityPerm(aRank), aCounts, aOrders, subgroupBasis,
      aActiveSubgroups);
  LLVM_DEBUG({ llvm::errs() << "A layout: " << aLayout << "\n"; });

  // B matrix layout
  MFMAAttr::SingleSubgroupLayout bCounts =
      mfmaAttr.getBSingleSubgroupLayoutCount();
  MFMAAttr::SingleSubgroupLayout bOrders =
      mfmaAttr.getBSingleSubgroupLayoutOrder();

  int64_t bRank = opInfo.getBRank();

  SmallVector<int64_t> bNDims = opInfo.rhsNDims;
  SmallVector<int64_t> bBatchSizes(bRank, 1);
  SmallVector<int64_t> bSubgroupSizes(bRank, 1);
  SmallVector<int64_t> bSubgroupOrder = getIdentityPerm(bRank);
  SmallVector<int64_t> bBatchOrder = getIdentityPerm(bRank);
  for (auto [i, dim] : llvm::enumerate(bNDims)) {
    bBatchSizes[dim] = batchNSizes[i];
    bSubgroupSizes[dim] = subgroupNBasis[i];
    int64_t j = i + batchCount;
    bSubgroupOrder[dim] = j;
    bBatchOrder[dim] = j >= bfk ? j + 1 : j;
  }
  bSubgroupOrder[bfk] = bRank - 1;
  bBatchOrder[bfk] = bfk;
  bBatchSizes[bfk] = getSubgroupKTileCount();

  SmallVector<bool> bActiveSubgroups(subgroupBasis.size(), false);
  for (auto nDim : nDims) {
    bActiveSubgroups[nDim] = true;
  }
  if (batchCount == 1) {
    bActiveSubgroups[0] = true;
  }
  bActiveSubgroups.back() = true;

  auto bLayout =
      permuteAndCreateNestedLayout(context, bRank, bfk, bfn, bPermute,
                                   /*subgroupCount=*/bSubgroupSizes,
                                   /*subgroupOrder=*/bSubgroupOrder,
                                   /*batchCount=*/bBatchSizes,
                                   /*batchOrder=*/bBatchOrder, bCounts, bOrders,
                                   subgroupBasis, bActiveSubgroups);
  LLVM_DEBUG({ llvm::errs() << "B layout: " << bLayout << "\n"; });

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
