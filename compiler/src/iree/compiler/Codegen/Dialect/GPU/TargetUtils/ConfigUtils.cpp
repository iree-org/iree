// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Common/TileInferenceUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-gpu-config-utils"

namespace mlir::iree_compiler::IREE::GPU {

constexpr int64_t kCacheLineSizeBits = 128 * 8;
constexpr int64_t kPreferredCopyNumBits = 128;

//===----------------------------------------------------------------------===//
// Lowering Config Selection
//===----------------------------------------------------------------------===//

LogicalResult setDataTiledMultiMmaLoweringConfig(
    IREE::GPU::TargetAttr target, mlir::FunctionOpInterface entryPoint,
    Operation *op, IREE::GPU::UKernelConfigAttr ukernelConfig) {
  auto multiMmaOp = dyn_cast<IREE::Codegen::InnerTiledOp>(op);
  if (!multiMmaOp) {
    return failure();
  }
  auto dataTiledMmaAttr = dyn_cast<DataTiledMMAAttr>(multiMmaOp.getKind());
  if (!dataTiledMmaAttr) {
    return failure();
  }

  LDBG() << "MultiMMA TileAndFuse Config";

  // Compute workgroup size, which is given by the subgroup size times the
  // number of subgroups. The number of subgroups is found by the product of
  // subgroup unrolling factors, since the non-unrolled inner kernel takes a
  // single subgroup.
  const int64_t targetSubgroupSize = dataTiledMmaAttr.getSubgroupSize();
  int64_t flatWorkgroupSize = targetSubgroupSize *
                              dataTiledMmaAttr.getSubgroupsM() *
                              dataTiledMmaAttr.getSubgroupsN();
  std::array<int64_t, 3> workgroupSize{flatWorkgroupSize, 1, 1};

  // Set all workgroup and reduction tile sizes to 1, since the data tiled
  // kernel has the scope of an entire workgroup, and the reduction tiling is
  // already baked into the "opaque" data tiled inner layout of the inner_tiled.
  SmallVector<AffineMap> indexingMaps = multiMmaOp.getIndexingMapsArray();
  mlir::linalg::ContractionDimensions contractionDims =
      mlir::linalg::inferContractionDims(indexingMaps).value();

  int64_t iterationRank = indexingMaps.front().getNumDims();
  SmallVector<int64_t> workgroupTileSizes(iterationRank, 1);
  SmallVector<int64_t> reductionTileSizes(iterationRank, 0);
  for (int64_t kDim : contractionDims.k) {
    workgroupTileSizes[kDim] = 0;
    reductionTileSizes[kDim] = ukernelConfig ? 0 : 1;
  }

  // Set tile sizes.
  MLIRContext *context = multiMmaOp.getContext();
  SmallVector<NamedAttribute> attrs;
  Builder b(context);
  attrs.emplace_back(b.getStringAttr("workgroup"),
                     b.getI64ArrayAttr(workgroupTileSizes));
  attrs.emplace_back(b.getStringAttr("reduction"),
                     b.getI64ArrayAttr(reductionTileSizes));
  if (ukernelConfig) {
    attrs.emplace_back(b.getStringAttr("ukernel"), ukernelConfig);
  } else {
    // Promote operands to use shared memory for LHS and RHS.
    // Don't do that with ukernels: their untiled reduction dimension is too
    // large to fit in shared memory, so they just want global memory and they
    // will take care of moving small chunks at a time into a shared memory
    // operand that will be created together with the ukernel op.
    GPU::appendPromotedOperandsList(context, attrs, {0, 1});
  }
  auto configDict = b.getDictionaryAttr(attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  // Don't add any special padding or prefetching, since the data-tiled layout
  // is already what we want.
  SmallVector<NamedAttribute, 1> pipelineAttrs;
  auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(
      context, /*prefetchSharedMemory=*/false,
      /*no_reduce_shared_memory_bank_conflicts=*/true,
      /*use_igemm_convolution=*/false,
      /*use_direct_convolution=*/false,
      /*reorder_workgroups_strategy=*/std::nullopt);
  pipelineAttrs.emplace_back(
      b.getStringAttr(IREE::GPU::GPUPipelineOptionsAttr::getDictKeyName()),
      pipelineOptions);
  auto pipelineConfig = b.getDictionaryAttr(pipelineAttrs);

  // TODO(qedawkins): Use a shared pipeline identifier here.
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
      workgroupSize, targetSubgroupSize, pipelineConfig);
}

/// Given a target and a matmul problem, try to find an MMA schedule for the
/// problem based on the available mma intrinsics.
static std::optional<GPUMMASchedule> getMmaScheduleFromProblemAndTarget(
    IREE::GPU::TargetAttr target, GPUMatmulShapeType problem,
    bool transposedLhs, bool transposedRhs, bool mustBeAligned = true,
    bool doCPromotion = false, bool scaled = false) {
  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();
  SmallVector<GPUIntrinsicType> intrinsics;
  if (scaled) {
    for (IREE::GPU::ScaledMMAAttr smma : target.getWgp().getScaledMma()) {
      // Intrinsics that do not specify a distribution kind cannot be
      // distributed.
      if (!smma.getDistributionMappingKind())
        continue;
      if (smma.getSubgroupSize() != targetSubgroupSize)
        continue;

      auto [m, n, k, kB] = smma.getScaledMNKShape();
      SmallVector<Type> elementTypes;
      smma.getElementTypes(elementTypes);
      intrinsics.emplace_back(GPUIntrinsicType({m}, {n}, {k, kB}, {},
                                               elementTypes[0], elementTypes[2],
                                               elementTypes[4], smma));
    }
  } else {
    for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
      // Intrinsics that do not specify a distribution kind cannot be
      // distributed.
      if (!mma.getDistributionMappingKind())
        continue;
      if (mma.getSubgroupSize() != targetSubgroupSize)
        continue;

      auto [mSize, nSize, kSize] = mma.getMNKShape();
      auto [aType, bType, cType] = mma.getABCElementTypes();
      intrinsics.emplace_back(mSize, nSize, kSize, aType, bType, cType, mma);
    }
  }
  if (intrinsics.empty()) {
    return std::nullopt;
  }

  GPUMMAHeuristicSeeds seeds;
  assert(problem.aType == problem.bType &&
         "expected the same aType and bType.");
  int64_t inBitWidth = problem.aType.getIntOrFloatBitWidth();

  // Note that the following heuristic seeds are just placeholder values.
  // We need to clean it up and make it adjusting to different targets.
  // See https://github.com/iree-org/iree/issues/16341 for details.
  int64_t mSize = ShapedType::getNumElements(problem.mSizes);
  int64_t nSize = ShapedType::getNumElements(problem.nSizes);
  if (mSize * nSize <= 512 * 512) {
    // For matmuls with small M*N size, we want to distribute M*N onto more
    // workgroups to fill the GPU. Use a smaller bestMNTileCountPerSubgroup
    // and a larger bestKTileCountPerSubgroup.
    seeds = {/*bestSubgroupCountPerWorkgroup=*/4,
             /*bestMNTileCountPerSubgroup=*/4,
             /*bestKTileCountPerSubgroup=*/8,
             /*bestKElementCountPerSubgroup*/ kCacheLineSizeBits / inBitWidth};
  } else {
    seeds = {/*bestSubgroupCountPerWorkgroup=*/4,
             /*bestMNTileCountPerSubgroup=*/16,
             /*bestKTileCountPerSubgroup=*/4,
             /*bestKElementCountPerSubgroup*/ kCacheLineSizeBits / 2 /
                 inBitWidth};
  }
  int64_t maxSharedMemoryBytes = target.getWgp().getMaxWorkgroupMemoryBytes();

  std::optional<int64_t> wgpCount = std::nullopt;
  if (TargetChipAttr chip = target.getChip()) {
    wgpCount = chip.getWgpCount();
  }

  // First try to find a schedule with an exactly matching intrinsic.
  std::optional<GPUMMASchedule> schedule = deduceMMASchedule(
      problem, intrinsics, seeds, maxSharedMemoryBytes, targetSubgroupSize,
      wgpCount, transposedLhs, transposedRhs, /*canUpcastAcc=*/false,
      /*mustBeAligned*/ mustBeAligned, doCPromotion);
  return schedule;
}

/// Helper function to get convolution padding sizes if possible.
static std::optional<ArrayAttr> getPaddingConvSizes(
    Builder &b, int64_t kSize, int64_t kPaddingSize,
    const SmallVector<int64_t> &workgroupTileSizes,
    const SmallVector<int64_t> &mDims, const SmallVector<int64_t> &nDims,
    const SmallVector<int64_t> &batchDims,
    std::optional<mlir::linalg::ConvolutionDimensions> &padConvDims) {
  if (!padConvDims.has_value())
    return std::nullopt;

  SmallVector<unsigned> batchAndImageDims;
  mlir::linalg::ConvolutionDimensions convDims = padConvDims.value();
  bool isBatchLast = !convDims.batch.empty() &&
                     convDims.outputImage.back() < convDims.batch.front();
  if (isBatchLast) {
    batchAndImageDims.append(convDims.outputImage.begin(),
                             convDims.outputImage.end());
    batchAndImageDims.append(convDims.batch.begin(), convDims.batch.end());
  } else {
    batchAndImageDims.append(convDims.batch.begin(), convDims.batch.end());
    batchAndImageDims.append(convDims.outputImage.begin(),
                             convDims.outputImage.end());
  }

  SmallVector<unsigned> concatMDims, concatNDims;
  bool isOutputChannelFirst =
      convDims.outputChannel.back() < convDims.outputImage.front();
  if (isOutputChannelFirst) {
    concatMDims.append(convDims.outputChannel.begin(),
                       convDims.outputChannel.end());
    concatNDims = batchAndImageDims;
  } else {
    concatMDims = batchAndImageDims;
    concatNDims.append(convDims.outputChannel.begin(),
                       convDims.outputChannel.end());
  }

  // Verify that the number of M, N dimensions from IGEMM match the
  // corresponding number of convolution dimensions.
  if (concatMDims.size() != mDims.size() ||
      concatNDims.size() != nDims.size() ||
      convDims.depth.size() != batchDims.size()) {
    return std::nullopt;
  }

  // Padding sizes for parallel dimensions are the same as workgroup tile
  // sizes.
  int64_t totalNumDims = convDims.batch.size() + convDims.outputImage.size() +
                         convDims.outputChannel.size() +
                         convDims.filterLoop.size() +
                         convDims.inputChannel.size() + convDims.depth.size();
  SmallVector<int64_t> paddingConvSizes(totalNumDims, 0);
  if (batchDims.size() != 0) {
    for (auto [dim, bDim] : llvm::zip(convDims.depth, batchDims)) {
      paddingConvSizes[dim] = workgroupTileSizes[bDim];
    }
  }
  for (auto [dim, mDim] : llvm::zip(concatMDims, mDims))
    paddingConvSizes[dim] = workgroupTileSizes[mDim];
  for (auto [dim, nDim] : llvm::zip(concatNDims, nDims))
    paddingConvSizes[dim] = workgroupTileSizes[nDim];

  // To avoid over-padding, no padding for channel dimensions is needed if
  // the product of reduction sizes is already multiples of k padding
  // size. Otherwise, pad the innermost channel dimension.
  // TODO (vivian): Padding the innermost channel dimension to a multiple
  // of vector size may still be needed even if the K-dim is aligned, and
  // this should be validated based on performance.
  if (kSize % kPaddingSize != 0) {
    int64_t innerChannelDim = convDims.inputChannel.back();
    paddingConvSizes[innerChannelDim] = kPaddingSize;
  }
  return b.getI64ArrayAttr(paddingConvSizes);
}

/// Create a lowering config for matmul or IGEMM convolution based on iteration
/// bounds and indexing maps for a given target. This function computes
/// contraction dimensions and deduces an MMA intrinsic schedule to choose tile
/// sizes and the workgroup size. The optional argument `padConvDims` is used to
/// determine the convolution dimensions for padding when creating
/// `padding_conv` config. `padding_conv` attribute is only used when padding
/// convolutions before converting them to IGEMM.
static FailureOr<std::pair<LoweringConfigAttr, int64_t>>
getMatmulOrIGEMMLoweringConfigAndWorkgroupSize(
    SmallVector<int64_t> bounds, ArrayRef<AffineMap> maps,
    ArrayRef<Value> operands, IREE::GPU::TargetAttr target, bool useDirectLoad,
    bool scaled,
    std::optional<mlir::linalg::ConvolutionDimensions> padConvDims = {}) {
  if (target.getWgp().getMma().empty()) {
    return failure();
  }

  SmallVector<unsigned, 2> contractionM, contractionN, contractionK,
      contractionKB, contractionB;
  if (scaled) {
    FailureOr<IREE::LinalgExt::ScaledContractionDimensions>
        scaledContractionDims =
            IREE::LinalgExt::inferScaledContractionDims(maps);
    if (failed(scaledContractionDims)) {
      return failure();
    }
    contractionM = scaledContractionDims->m;
    contractionN = scaledContractionDims->n;
    contractionK = scaledContractionDims->k;
    contractionKB = scaledContractionDims->kB;
    contractionB = scaledContractionDims->batch;
  } else {
    FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
        mlir::linalg::inferContractionDims(maps);
    if (failed(contractionDims)) {
      return failure();
    }
    contractionM = contractionDims->m;
    contractionN = contractionDims->n;
    contractionK = contractionDims->k;
    contractionB = contractionDims->batch;
  }

  if (contractionK.empty() || contractionM.empty() || contractionN.empty()) {
    return failure();
  }

  // TODO(Max191): add dynamic shape support for inner most dims.
  if (ShapedType::isDynamic(bounds[contractionM.back()]) ||
      ShapedType::isDynamic(bounds[contractionN.back()]) ||
      ShapedType::isDynamic(bounds[contractionK.back()])) {
    return failure();
  }

  // We can support unaligned shapes as long as there are no dynamic dimensions
  // as finding padding bounds for dynamic dimensions is not guaranteed.
  // TODO(nirvedhmeshram): Add support so that we can find the bounds
  // information.
  bool canSupportUnaligned = true;

  // Gather all static M, N, and K dimensions to deduce the MMASchedule. Dynamic
  // dimensions will be tiled to 1 in workgroup tiling, so they are ignored when
  // computing an MMA schedule.
  SmallVector<int64_t> mDims, nDims, kDims, batchDims;
  for (int64_t mDim : contractionM) {
    if (ShapedType::isDynamic(bounds[mDim])) {
      canSupportUnaligned = false;
      continue;
    }
    mDims.push_back(mDim);
  }
  for (int64_t nDim : contractionN) {
    if (ShapedType::isDynamic(bounds[nDim])) {
      canSupportUnaligned = false;
      continue;
    }
    nDims.push_back(nDim);
  }
  for (int64_t kDim : contractionK) {
    if (ShapedType::isDynamic(bounds[kDim])) {
      canSupportUnaligned = false;
      continue;
    }
    kDims.push_back(kDim);
  }
  if (scaled) {
    for (int64_t kBDim : contractionKB) {
      if (ShapedType::isDynamic(bounds[kBDim])) {
        canSupportUnaligned = false;
        continue;
      }
      kDims.push_back(kBDim);
    }
  }
  for (int64_t batchDim : contractionB) {
    if (ShapedType::isDynamic(bounds[batchDim])) {
      canSupportUnaligned = false;
      continue;
    }
    batchDims.push_back(batchDim);
  }

  auto getDimBounds = [&](SmallVector<int64_t> dims) -> SmallVector<int64_t> {
    return llvm::map_to_vector(dims, [&](int64_t dim) { return bounds[dim]; });
  };

  assert((operands.size() == 3 || scaled) && "expected 3 operands");
  assert((operands.size() == 5 || !scaled) && "expected 5 operands");

  Value lhs = operands[0];
  Value rhs = operands[1];
  Value init;
  if (scaled) {
    init = operands[4];
  } else {
    init = operands[2];
  }

  Type lhsElemType = getElementTypeOrSelf(lhs);
  Type rhsElemType = getElementTypeOrSelf(rhs);
  Type initElemType = getElementTypeOrSelf(init);

  GPUMatmulShapeType problem{getDimBounds(mDims), getDimBounds(nDims),
                             getDimBounds(kDims), getDimBounds(batchDims),
                             lhsElemType,         rhsElemType,
                             initElemType};

  // Infer if lhs or rhs is transposed to help generate better schedule.
  // TODO: Drop this. This is only a consideration for other pipelines.
  bool transposedLhs =
      kDims.back() !=
      llvm::cast<AffineDimExpr>(maps[0].getResults().back()).getPosition();
  bool transposedRhs =
      nDims.back() !=
      llvm::cast<AffineDimExpr>(maps[1].getResults().back()).getPosition();

  bool mustBeAligned = true;
  bool doCPromotion = false;
  std::optional<GPUMMASchedule> schedule = getMmaScheduleFromProblemAndTarget(
      target, problem, transposedLhs, transposedRhs, /*mustBeAligned*/ true,
      /*doCPromotion*/ false, scaled);

  // TODO (nirvedhmeshram, qedawkins): The performance with this will be bad if
  // the GEMM is accumulating (i.e doesnt have a zero fill dpsInit) as that
  // buffer currently gets materialized as private memory. We need to add
  // missing patterns to fix that.
  if (!schedule && canSupportUnaligned) {
    LDBG() << "Attempting to deduce unaligned TileAndFuse MMA schedulee";
    mustBeAligned = false;
    doCPromotion = true;
    schedule = getMmaScheduleFromProblemAndTarget(
        target, problem, transposedLhs, transposedRhs, mustBeAligned,
        doCPromotion, scaled);
  }

  if (!schedule) {
    LDBG() << "Failed to deduce TileAndFuse MMA schedule";
    return failure();
  }

  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();
  LDBG() << "Target Subgroup size: " << targetSubgroupSize;
  LDBG() << "Schedule: " << schedule;

  SmallVector<int64_t> workgroupTileSizes(bounds.size(), 0);
  SmallVector<int64_t> reductionTileSizes(bounds.size(), 0);
  SmallVector<int64_t> subgroupTileSizes(bounds.size(), 0);
  // Tile all batch dimensions with unit size.
  for (int64_t batch : contractionB) {
    workgroupTileSizes[batch] = 1;
  }

  // Tile all m, n, k and k_b dimensions to 1 except the innermost. Unit dims
  // from this tiling are folded before vectorization.
  for (int64_t m : llvm::drop_end(contractionM)) {
    workgroupTileSizes[m] = 1;
  }
  for (int64_t n : llvm::drop_end(contractionN)) {
    workgroupTileSizes[n] = 1;
  }
  for (int64_t k : llvm::drop_end(contractionK)) {
    reductionTileSizes[k] = 1;
  }
  if (scaled) {
    for (int64_t kB : llvm::drop_end(contractionKB)) {
      reductionTileSizes[kB] = 1;
    }
  }

  // Compute the M/N dimension tile sizes by multiplying subgroup information.
  for (auto [i, mDim] : llvm::enumerate(mDims)) {
    workgroupTileSizes[mDim] =
        schedule->mSubgroupCounts[i] * schedule->mTileSizes[i];
    // Multiply by the intrinsic shape for the inner most dim as we distribute
    // to workgroups before packing to intrinsic.
    if (i == mDims.size() - 1)
      workgroupTileSizes[mDim] *= schedule->mSize;
    subgroupTileSizes[mDim] = schedule->mTileSizes[i];
  }
  for (auto [i, nDim] : llvm::enumerate(nDims)) {
    workgroupTileSizes[nDim] =
        schedule->nSubgroupCounts[i] * schedule->nTileSizes[i];
    // Multiply by the intrinsic shape for the inner most dim as we distribute
    // to workgroups before packing to intrinsic.
    if (i == nDims.size() - 1)
      workgroupTileSizes[nDim] *= schedule->nSize;
    subgroupTileSizes[nDim] = schedule->nTileSizes[i];
  }

  // Similarly the reduction tile size is just the post-packing tile count.
  for (auto [i, kDim] : llvm::enumerate(kDims)) {
    reductionTileSizes[kDim] = schedule->kTileSizes[i];
  }

  IREE::Codegen::InnerTileDescAttrInterface kind = schedule->mmaKind;

  // Attach the MMA schedule as an attribute to the entry point export function
  // for later access in the pipeline.
  MLIRContext *context = lhs.getContext();
  SmallVector<NamedAttribute> attrs;
  Builder b(context);
  attrs.emplace_back(StringAttr::get(context, "workgroup"),
                     b.getI64ArrayAttr(workgroupTileSizes));
  attrs.emplace_back(StringAttr::get(context, "reduction"),
                     b.getI64ArrayAttr(reductionTileSizes));
  attrs.emplace_back(StringAttr::get(context, "subgroup"),
                     b.getI64ArrayAttr(subgroupTileSizes));
  attrs.emplace_back(StringAttr::get(context, "mma_kind"), kind);
  if (mustBeAligned) {
    Attribute useGlobalDma = IREE::GPU::UseGlobalLoadDMAAttr::get(context);
    Attribute promotionArray[] = {useGlobalDma, useGlobalDma};
    ArrayRef<Attribute> promotionTypes =
        useDirectLoad ? ArrayRef<Attribute>(promotionArray)
                      : ArrayRef<Attribute>{};
    GPU::appendPromotedOperandsList(context, attrs, {0, 1}, promotionTypes);
  } else {
    // TODO (nirvedhmeshram, Max191, jerryyin) : Add support so that unaligned
    // shapes do not require c promotion.
    GPU::appendPromotedOperandsList(context, attrs, {0, 1, 2});
    SmallVector<int64_t> paddingTileSizes = workgroupTileSizes;

    // Initialize inner and outer padding sizes from reductionTileSizes.
    for (int64_t kDim : kDims) {
      paddingTileSizes[kDim] = reductionTileSizes[kDim];
    }

    int64_t kPackFactor, innerKDim = contractionK.back();
    if (scaled) {
      auto smmaKind = dyn_cast<IREE::GPU::ScaledMMAAttr>(kind);
      kPackFactor = std::get<2>(smmaKind.getScaledMNKShape());
    } else {
      auto mmaKind = dyn_cast<IREE::GPU::MmaInterfaceAttr>(kind);
      kPackFactor = std::get<2>(mmaKind.getMNKShape());
    }
    paddingTileSizes[innerKDim] *= kPackFactor;

    // Create `padding_conv` attribute when padding convolutions before IGEMM is
    // possible, otherwise fallback to pad IGEMM.
    if (auto attr = getPaddingConvSizes(
            b, bounds[innerKDim], paddingTileSizes[innerKDim],
            workgroupTileSizes, mDims, nDims, batchDims, padConvDims)) {
      attrs.emplace_back(StringAttr::get(context, "padding_conv"), *attr);
    } else {
      attrs.emplace_back(StringAttr::get(context, "padding"),
                         b.getI64ArrayAttr(paddingTileSizes));
    }
  }
  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);
  int64_t flatWorkgroupSize =
      targetSubgroupSize *
      ShapedType::getNumElements(schedule->nSubgroupCounts) *
      ShapedType::getNumElements(schedule->mSubgroupCounts);

  return std::make_pair(loweringConfig, flatWorkgroupSize);
}

LogicalResult setIGEMMConvolutionLoweringConfig(
    IREE::GPU::TargetAttr target, mlir::FunctionOpInterface entryPoint,
    Operation *op, bool useDirectLoad, bool padConv) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || !linalg::isaConvolutionOpInterface(linalgOp)) {
    return failure();
  }

  if (target.getWgp().getMma().empty())
    return failure();

  LDBG() << "IGEMM TileAndFuse Config";
  FailureOr<LinalgExt::IGEMMGenericConvDetails> igemmGenericConvDetails =
      LinalgExt::getIGEMMGenericConvDetails(linalgOp);
  if (failed(igemmGenericConvDetails)) {
    LDBG() << "Unsupported convolution type";
    return failure();
  }
  SmallVector<AffineMap> igemmContractionMaps =
      igemmGenericConvDetails->igemmContractionMaps;
  SmallVector<int64_t> igemmLoopBounds =
      igemmGenericConvDetails->igemmLoopBounds;
  SmallVector<Value> igemmOperands = igemmGenericConvDetails->igemmOperands;

  std::optional<mlir::linalg::ConvolutionDimensions> padConvDims;
  if (padConv)
    padConvDims = igemmGenericConvDetails->convDims;

  SmallVector<int64_t> bounds = igemmLoopBounds;
  FailureOr<std::pair<LoweringConfigAttr, int64_t>> configAndWgSize =
      getMatmulOrIGEMMLoweringConfigAndWorkgroupSize(
          bounds, igemmContractionMaps, igemmOperands, target, useDirectLoad,
          /*scaled*/ false, padConvDims);
  if (failed(configAndWgSize)) {
    return failure();
  }
  std::array<int64_t, 3> workgroupSize = {configAndWgSize->second, 1, 1};
  LoweringConfigAttr loweringConfig = configAndWgSize->first;

  SmallVector<NamedAttribute, 1> pipelineAttrs;
  auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(
      linalgOp->getContext(), /*prefetchSharedMemory=*/true,
      /*no_reduce_shared_memory_bank_conflicts=*/useDirectLoad,
      /*use_igemm_convolution=*/true,
      /*use_direct_convolution=*/false,
      /*reorder_workgroups_strategy=*/std::nullopt);
  pipelineAttrs.emplace_back(
      StringAttr::get(linalgOp->getContext(),
                      IREE::GPU::GPUPipelineOptionsAttr::getDictKeyName()),
      pipelineOptions);
  auto pipelineConfig =
      DictionaryAttr::get(linalgOp->getContext(), pipelineAttrs);
  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();

  // TODO(qedawkins): Use a shared pipeline identifier here.
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
      workgroupSize, targetSubgroupSize, pipelineConfig);
}

LogicalResult setMatmulLoweringConfig(IREE::GPU::TargetAttr target,
                                      mlir::FunctionOpInterface entryPoint,
                                      Operation *op, bool useDirectLoad) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp ||
      (!linalg::isaContractionOpInterface(linalgOp) &&
       !IREE::LinalgExt::isaScaledContractionOpInterface(linalgOp))) {
    return failure();
  }

  SmallVector<int64_t> bounds = linalgOp.getStaticLoopRanges();
  SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
  SmallVector<Value> operands(linalgOp->getOperands());

  LDBG() << "Matmul TileAndFuse Config";

  FailureOr<std::pair<LoweringConfigAttr, int64_t>> configAndWgSize =
      getMatmulOrIGEMMLoweringConfigAndWorkgroupSize(
          bounds, maps, operands, target, useDirectLoad, /*scaled*/ false);

  // TODO (muzasyed) : add generalization for scaled and nonscaled versions of
  // matmul lowering.
  if (failed(configAndWgSize)) {
    // TODO (muzasyed) : Perform padding appropriately for minimizing bank
    // conflicts when dealing with scaled matmuls. For now it is disabled.
    useDirectLoad = true;
    configAndWgSize = getMatmulOrIGEMMLoweringConfigAndWorkgroupSize(
        bounds, maps, operands, target, useDirectLoad, /*scaled*/ true);
  }

  if (failed(configAndWgSize)) {
    return failure();
  }
  std::array<int64_t, 3> workgroupSize = {configAndWgSize->second, 1, 1};
  LoweringConfigAttr loweringConfig = configAndWgSize->first;

  SmallVector<NamedAttribute, 1> pipelineAttrs;
  auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(
      linalgOp->getContext(), /*prefetchSharedMemory=*/true,
      /*no_reduce_shared_memory_bank_conflicts=*/useDirectLoad,
      /*use_igemm_convolution=*/false,
      /*use_direct_convolution=*/false,
      /*reorder_workgroups_strategy=*/std::nullopt);
  pipelineAttrs.emplace_back(
      StringAttr::get(linalgOp->getContext(),
                      IREE::GPU::GPUPipelineOptionsAttr::getDictKeyName()),
      pipelineOptions);
  auto pipelineConfig =
      DictionaryAttr::get(linalgOp->getContext(), pipelineAttrs);
  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();

  // TODO(qedawkins): Use a shared pipeline identifier here.
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
      workgroupSize, targetSubgroupSize, pipelineConfig);
}

/// Helper to identify contraction like operations for operand promotiong.
static bool isNonMatvecContraction(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return false;
  }
  SmallVector<int64_t> bounds = linalgOp.getStaticLoopRanges();
  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    return false;
  }

  if (contractionDims->k.size() < 1 || contractionDims->m.size() < 1 ||
      contractionDims->n.size() < 1) {
    return false;
  }

  auto getElementCount = [&](ArrayRef<unsigned> dims) {
    int64_t acc = 1;
    for (auto mDim : dims) {
      int64_t size = bounds[mDim];
      if (ShapedType::isDynamic(size)) {
        return size;
      }
      acc *= size;
    }
    return acc;
  };
  return getElementCount(contractionDims->m) != 1 &&
         getElementCount(contractionDims->n) != 1;
}

// To find the number of vector elements per work-item, find a
// bit width that is representative of the computation.
static unsigned getRepresentativeBitWidth(linalg::LinalgOp linalgOp) {
  // Check all the inputs with permutation indexing maps. Use
  // the maximum of those to get the bit width.
  std::optional<unsigned> maxBitWidth;
  auto updateElementTypeBitWidth = [&](Value v) {
    auto elementType = getElementTypeOrSelf(v);
    if (!elementType.isIntOrFloat()) {
      return;
    }
    unsigned bitWidth = elementType.getIntOrFloatBitWidth();
    if (maxBitWidth) {
      maxBitWidth = std::max(maxBitWidth.value(), bitWidth);
      return;
    }
    maxBitWidth = bitWidth;
  };
  for (OpOperand *input : linalgOp.getDpsInputOperands()) {
    AffineMap inputOperandMap = linalgOp.getMatchingIndexingMap(input);
    if (!inputOperandMap.isPermutation()) {
      continue;
    }
    updateElementTypeBitWidth(input->get());
  }
  if (maxBitWidth) {
    return maxBitWidth.value();
  }

  // If none of the operands have permutation inputs, use the result.
  // Dont bother about the indexing map.
  for (OpOperand &output : linalgOp.getDpsInitsMutable()) {
    updateElementTypeBitWidth(output.get());
  }
  if (maxBitWidth) {
    return maxBitWidth.value();
  }

  // Fall back, just be a word.
  return 32;
}

static bool elementHasPowerOfTwoBitwidth(Value operand) {
  Type elementType = getElementTypeOrSelf(operand.getType());
  return elementType.isIntOrFloat() &&
         llvm::isPowerOf2_64(elementType.getIntOrFloatBitWidth());
}

struct DistributionInfo {
  SmallVector<unsigned int> partitionableLoops;
  SmallVector<int64_t> loopBounds;
  unsigned minBitwidth = 0;
  unsigned representativeBitWidth = 0;
  bool vectorizable = false;
};

static FailureOr<DistributionInfo> collectOpDistributionInfo(Operation *op) {
  DistributionInfo distInfo;
  // MapScatterOp doesn't fit the LinalgOp interface, so use special case logic
  // to get the distribution info.
  if (auto mapScatterOp = dyn_cast<IREE::LinalgExt::MapScatterOp>(op)) {
    distInfo.partitionableLoops =
        llvm::to_vector(llvm::seq<unsigned int>(mapScatterOp.getInputRank()));
    distInfo.vectorizable = false;
    distInfo.minBitwidth = mapScatterOp.getInputType().getElementTypeBitWidth();
    distInfo.representativeBitWidth = distInfo.minBitwidth;
    distInfo.loopBounds =
        SmallVector<int64_t>(mapScatterOp.getInputType().getShape());
    return distInfo;
  }

  // PackOp doesn't fit the LinalgOp interface, since it is a RelayoutOp, so
  // we have to use special case logic to get the distribution info.
  if (auto packOp = dyn_cast<linalg::PackOp>(op)) {
    distInfo.partitionableLoops =
        llvm::to_vector(llvm::seq<unsigned int>(packOp.getDestRank()));
    distInfo.vectorizable =
        llvm::all_of(op->getResults(), elementHasPowerOfTwoBitwidth);
    distInfo.minBitwidth = packOp.getDestType().getElementTypeBitWidth();
    distInfo.representativeBitWidth = distInfo.minBitwidth;
    distInfo.loopBounds = SmallVector<int64_t>(packOp.getDestType().getShape());
    return distInfo;
  }

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  // Bail out on multi result cases as consumer fusion currently does not
  // support multi result ops.
  if (!linalgOp || linalgOp.getNumDpsInits() != 1) {
    return failure();
  }

  // This pipeline requires tensor semantics. Also fail for gather semantics
  // for now to simplify tile + fuse.
  if (!linalgOp.hasPureTensorSemantics() || linalgOp.hasIndexSemantics()) {
    return failure();
  }

  linalgOp.getParallelDims(distInfo.partitionableLoops);

  // Bail out if op is not tilable.
  if (distInfo.partitionableLoops.empty()) {
    return failure();
  }

  // Whether we can try to use the vectorization pipeline.
  distInfo.loopBounds = linalgOp.getStaticLoopRanges();
  bool isProjPerm =
      llvm::all_of(linalgOp.getIndexingMapsArray(),
                   [](AffineMap map) { return map.isProjectedPermutation(); });
  bool isPowTwo = llvm::all_of(op->getOperands(), elementHasPowerOfTwoBitwidth);

  // Require all affine maps to be projected permutation so that we can
  // generate vector transfer ops.
  distInfo.vectorizable = isProjPerm && isPowTwo;

  distInfo.minBitwidth = getMinElementBitwidth(linalgOp);
  distInfo.representativeBitWidth = getRepresentativeBitWidth(linalgOp);
  return distInfo;
};

LogicalResult setTileAndFuseLoweringConfig(IREE::GPU::TargetAttr target,
                                           mlir::FunctionOpInterface entryPoint,
                                           Operation *op) {
  FailureOr<DistributionInfo> maybeDistInfo = collectOpDistributionInfo(op);
  if (failed(maybeDistInfo)) {
    return failure();
  }

  // TODO(Max191): Drop this check for reshapes in the dispatch once we can
  // codegen larger tile sizes with reshapes in the dispatch.
  bool hasReshapes = false;
  entryPoint->walk([&](Operation *opInEntryPoint) {
    if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(opInEntryPoint)) {
      hasReshapes = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  DistributionInfo distInfo = maybeDistInfo.value();

  const int subgroupSize = target.getPreferredSubgroupSize();
  const unsigned loopDepth = distInfo.loopBounds.size();

  // Configurations we need to decide.
  int64_t flatWorkgroupSize = 1;
  SmallVector<int64_t> workgroupTileSizes(loopDepth, 0);
  SmallVector<int64_t> threadTileSizes(loopDepth, 0);

  // Find constraints on workgroup tile sizes due to pack or unpack ops in the
  // dispatch. If there are no pack or unpack ops present, then these multiples
  // will be 1, which means there is no constraint on workgroup tile sizes.
  //
  // TODO(Max191): Getting the workgroup size multiples is needed for current
  // pack and unpack GPU codegen. Ideally, we won't rely on propagating pack
  // and unpack tile size information during lowering strategy selection, and
  // this logic should be dropped once we have a better solution.
  SmallVector<int64_t> workgroupTileSizeMultiples =
      getWorkgroupSizeMultiples(cast<TilingInterface>(op));

  // Common case for all linalg ops.

  // The core idea is to distribute the partitioned loops to the workgroup
  // dimensions. The goal is to fill up the GPU as much as possible, which means
  // 1) distributing to as many threads as possible, and 2) avoid assigning too
  // many threads to handle out-of-bound elements (thus idle).

  // Make sure we use a tile size that results in some integral number of bytes.
  const unsigned scaleToByte =
      std::max(8 / distInfo.minBitwidth, static_cast<unsigned>(1));

  // Distribute workload to the given `numThreads` by allowing a potental loss.
  auto distributeToThreads = [&](int64_t numThreads,
                                 std::optional<int64_t> lossFactor =
                                     std::nullopt) -> int64_t {
    LDBG() << "Loss factor: " << lossFactor;
    // Initialize the configuration.
    flatWorkgroupSize = 1;
    // Initialize thread tiling along all partitioned loops with size 1, and
    // workgroup tiling with the required tile size multiples. This may lead
    // to larger workgroup tiles than the number of threads in the workgroup,
    // but it is unavoidable.
    for (int64_t loopIndex : distInfo.partitionableLoops) {
      workgroupTileSizes[loopIndex] = workgroupTileSizeMultiples[loopIndex];
      threadTileSizes[loopIndex] = 1;
    }

    // Special case: if all the parallel dimensions are dynamic (and, just to be
    // defensive, there aren't workgroup tile size constraints requested), we'll
    // let the later tiling logic mask off excess lanes in the last workgroup
    // since that's cheaper than using insufficient parallelism.
    if (!distInfo.partitionableLoops.empty() && !lossFactor &&
        llvm::all_of(distInfo.partitionableLoops, [&](int64_t shapeDim) {
          return ShapedType::isDynamic(distInfo.loopBounds[shapeDim]) &&
                 workgroupTileSizeMultiples[shapeDim] == 1;
        })) {
      LDBG() << "Fully dynamic parallel dimension";
      int64_t innerShapeDim = distInfo.partitionableLoops.back();
      workgroupTileSizes[innerShapeDim] *= numThreads;
      flatWorkgroupSize *= numThreads;
      return 1;
    }
    // Scan from the innermost shape dimension and try to deduce the
    // configuration for the corresponding GPU workgroup dimension.
    int64_t wgDim = 0;
    for (auto shapeDim : llvm::reverse(distInfo.partitionableLoops)) {
      int64_t loopBound = distInfo.loopBounds[shapeDim];
      // Skip dynamic dimensions.
      if (ShapedType::isDynamic(loopBound))
        continue;

      // Try to find some power of two that can divide the current shape dim
      // size. This vector keeps the candidate tile sizes.
      SmallVector<int64_t, 8> candidates;

      // Ensure vectorization works with the `workgroupTileMultiple`.
      int64_t workgroupTileMultiple = workgroupTileSizeMultiples[shapeDim];
      unsigned numVectorElements =
          std::max(4u, 128 / distInfo.representativeBitWidth);
      int64_t vectorizableCandidate = numVectorElements * numThreads;
      // For smaller shapes, we reduce `numVectorElements` as we may not find
      // work for all threads otherwise and we dont have vectorization enabled
      // with loss.
      while (distInfo.vectorizable && (vectorizableCandidate > loopBound) &&
             numVectorElements > 4) {
        numVectorElements /= 2;
        vectorizableCandidate = numVectorElements * numThreads;
      }
      distInfo.vectorizable =
          distInfo.vectorizable &&
          vectorizableCandidate % workgroupTileMultiple == 0;

      if (distInfo.vectorizable && wgDim == 0 && !lossFactor) {
        candidates.push_back(vectorizableCandidate);
      }

      // Try all power of two multiples of `workgroupTileMultiple` up to the
      // subgroup size.
      uint64_t maxCandidate =
          std::max<uint64_t>(1, llvm::PowerOf2Ceil(llvm::divideCeil(
                                    numThreads, workgroupTileMultiple)));
      for (unsigned i = maxCandidate; i >= 1; i >>= 1) {
        candidates.push_back(i * workgroupTileMultiple);
      }
      LDBG() << "Base candidate tile sizes: "
             << llvm::interleaved_array(candidates);

      int64_t candidateWorkgroupSize = 1;
      for (int64_t candidate : candidates) {
        int64_t scaledTileSize = candidate * scaleToByte;
        if (loopBound % scaledTileSize != 0) {
          if (!lossFactor)
            continue;
          // Skip this candidate if it causes many threads to be idle.
          int64_t idleThreads = candidate - (loopBound % scaledTileSize);
          if (idleThreads > candidate / *lossFactor)
            continue;
        }
        // If the workload is too small and we cannot distribute to more than 2
        // workgroups, try a smaller tile size to increase parallelism.
        if (distInfo.partitionableLoops.size() == 1 &&
            candidate > subgroupSize &&
            llvm::divideCeil(loopBound, scaledTileSize) <= 2) {
          continue;
        }
        // Try to let each thread handle 4 elements if this is the workgroup x
        // dimension.
        // TODO: Try to take into account element type bit width to get
        // 4xdword reads instead of 4x{elements}.
        if (distInfo.vectorizable && wgDim == 0 && !lossFactor &&
            candidate % numVectorElements == 0 && !hasReshapes) {
          // Use size-1 vectors to increase parallelism if larger ones causes
          // idle threads in the subgroup.
          bool hasIdleThreads = distInfo.partitionableLoops.size() == 1 &&
                                candidate <= subgroupSize;
          int vectorSize = hasIdleThreads ? 1 : numVectorElements;
          LDBG() << "Use vector size: " << vectorSize;
          threadTileSizes[shapeDim] = vectorSize * scaleToByte;
          candidateWorkgroupSize = candidate / vectorSize;
          assert(numThreads % (candidate / vectorSize) == 0);
          numThreads /= candidate / vectorSize;
        } else {
          // When the workgroupTileMultiple is not a Po2, then the candidate
          // may not evenly divide the numThreads. In this case, we get some
          // idle threads in the last iteration of the workgroup tile. Verify
          // that the idle threads are within the lossFactor.
          int64_t maybeCandidateWorkgroupSize = candidate;
          if (numThreads % candidate != 0) {
            maybeCandidateWorkgroupSize =
                std::min<int64_t>(1ll << llvm::Log2_64(candidate), numThreads);
            int64_t idleThreads = candidate % maybeCandidateWorkgroupSize;
            if (idleThreads != 0 &&
                (!lossFactor || idleThreads > candidate / *lossFactor)) {
              continue;
            }
          }
          if (wgDim == 0)
            distInfo.vectorizable = false;
          threadTileSizes[shapeDim] = scaleToByte;
          candidateWorkgroupSize = maybeCandidateWorkgroupSize;
          numThreads /= candidateWorkgroupSize;
        }
        workgroupTileSizes[shapeDim] = scaledTileSize;
        LDBG() << "Chosen workgroup tile size: " << scaledTileSize;
        assert(numThreads >= 1);
        break;
      }

      flatWorkgroupSize *= candidateWorkgroupSize;

      // Stop if we have distributed all threads.
      if (numThreads == 1)
        break;
      wgDim++;
    }
    return numThreads;
  };

  // First try to see if we can use up all threads without any loss.
  int64_t newNumThreads = subgroupSize;
  if (distributeToThreads(newNumThreads) != 1) {
    // Otherwise, allow larger and larger loss factor.

    // Threads for distribution. Use `minPreferredNumThreads` at least, but no
    // more than 4 subgroups.
    int64_t minPreferredNumThreads = std::reduce(
        workgroupTileSizeMultiples.begin(), workgroupTileSizeMultiples.end(), 1,
        std::multiplies<int64_t>());
    int64_t numThreads =
        std::min<int64_t>(4 * subgroupSize, minPreferredNumThreads);
    // If minPreferredNumThreads is small, use at least 32 or subgroupSize
    // threads, whichever is larger.
    numThreads =
        std::max<int64_t>(std::max<int64_t>(subgroupSize, 32), numThreads);
    // We can tolerate (1 / lossFactor) of threads in the workgroup to be idle.
    int64_t lossFactor = 32;

    for (; lossFactor >= 1; lossFactor >>= 1) {
      if (distributeToThreads(numThreads, lossFactor) == 1)
        break;
    }
  }

  // Heuristic value chosen to limit maximum vector sizes when tiling below.
  const unsigned maxVectorSize = 32;

  // Try to tile all reductions by some small factor, preferrably 4, when
  // possible. This gives us a chance to perform vector4 load if an input has
  // its innnermost dimension being reduction. It also avoids generating too
  // many instructions when unrolling vector later. We limit the expected
  // vector size by estimating it from the size of the iteration space tile and
  // limit it to a reasonable value. We process the loops from inner most to
  // outer most to try to align loads along inner dimensions.
  int64_t vectorSize = 1;
  int64_t numLoops = distInfo.loopBounds.size();
  SmallVector<int64_t> loopTileSizes(numLoops, 0);
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    SmallVector<utils::IteratorType> iterTypes =
        linalgOp.getIteratorTypesArray();
    for (auto [reverseIdx, iter] : llvm::enumerate(llvm::reverse(iterTypes))) {
      unsigned i = numLoops - reverseIdx - 1;
      if (linalg::isReductionIterator(iter) || i >= workgroupTileSizes.size() ||
          workgroupTileSizes[i] == 0) {
        int64_t tileSize = getReductionTilingFactor(distInfo.loopBounds[i]);
        if (vectorSize * tileSize > maxVectorSize) {
          tileSize = 1;
        }
        vectorSize *= tileSize;
        loopTileSizes[i] = tileSize;
      }
    }
  }

  // Attach the MMA schedule as an attribute to the entry point export function
  // for later access in the pipeline.
  MLIRContext *context = op->getContext();
  SmallVector<NamedAttribute> attrs;
  Builder b(context);
  attrs.emplace_back(StringAttr::get(context, "workgroup"),
                     b.getI64ArrayAttr(workgroupTileSizes));

  attrs.emplace_back(StringAttr::get(context, "thread"),
                     b.getI64ArrayAttr(threadTileSizes));

  if (isNonMatvecContraction(op)) {
    GPU::appendPromotedOperandsList(context, attrs, {0, 1});
  }

  if (llvm::any_of(loopTileSizes, [](int64_t s) { return s != 0; })) {
    attrs.emplace_back(StringAttr::get(context, "reduction"),
                       b.getI64ArrayAttr(loopTileSizes));
  }
  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  LDBG() << "Selected tile and fuse lowering config: " << loweringConfig
         << "\n";

  // TODO(qedawkins): Use a shared pipeline identifier here.
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
      {flatWorkgroupSize, 1, 1}, subgroupSize, DictionaryAttr());
}

LogicalResult setScatterLoweringConfig(IREE::GPU::TargetAttr target,
                                       mlir::FunctionOpInterface entryPoint,
                                       Operation *op) {
  auto scatter = dyn_cast<IREE::LinalgExt::ScatterOp>(op);
  if (!scatter) {
    return failure();
  }

  // TODO: Support non-unique indices.
  if (!scatter.getUniqueIndices()) {
    return failure();
  }

  // Various problem parameters.
  int64_t loopDepth = scatter.getLoopIteratorTypes().size();
  int64_t elemBits = scatter.getOriginalType().getElementTypeBitWidth();
  SmallVector<int64_t> loopBounds = scatter.getStaticLoopRanges();

  // Configurations we need to decide.
  int64_t flatWorkgroupSize = target.getPreferredSubgroupSize();
  SmallVector<int64_t> workgroupTileSizes(loopDepth, 1);
  SmallVector<int64_t> threadTileSizes(loopDepth, 1);
  int64_t vectorSize = kPreferredCopyNumBits / elemBits;

  bool innerDynamic = ShapedType::isDynamic(loopBounds.back());

  // Do not bother trying to vectorize if there are no vectorizable dims.
  if (loopDepth == 1) {
    vectorSize = 1;
  } else if (!innerDynamic) {
    // Use the largest power of 2 that divides the inner most non-scattered dim.
    vectorSize = std::gcd(vectorSize, loopBounds.back());
  }

  threadTileSizes.back() = vectorSize;
  int64_t residualInnerSize =
      innerDynamic ? loopBounds.back() : loopBounds.back() / vectorSize;

  // If the inner most dim is dynamic or exceeds the expected number of threads,
  // Only distribute threads along the inner most dimension.
  if (ShapedType::isDynamic(residualInnerSize) ||
      residualInnerSize >= flatWorkgroupSize) {
    workgroupTileSizes.back() = vectorSize * flatWorkgroupSize;
  } else { // residualInnerSize < flatWorkgroupSize
    // Floordiv to overestimate the required number of threads.
    int64_t residualThreads = flatWorkgroupSize / residualInnerSize;
    workgroupTileSizes.back() = residualInnerSize * vectorSize;
    for (int64_t i = loopDepth - 2, e = 0; i >= e; --i) {
      if (residualThreads <= 1) {
        break;
      }

      bool dynamicDim = ShapedType::isDynamic(loopBounds[i]);
      workgroupTileSizes[i] = dynamicDim
                                  ? residualThreads
                                  : std::min(residualThreads, loopBounds[i]);
      residualThreads = dynamicDim ? 1 : residualThreads / loopBounds[i];
    }
  }

  int64_t numBatch = scatter.getBatchRank();
  // Currently bufferization will fail if the only dimension distributed to
  // workgroups is the batch dims because the workgroup level slice will fold
  // away and cause a mismatch. To work around this we ensure that at least one
  // inner dim is always at least partially distributed to workgroups.
  if (llvm::all_of_zip(llvm::drop_begin(workgroupTileSizes, numBatch),
                       llvm::drop_begin(loopBounds, numBatch),
                       [](int64_t tileSize, int64_t bound) {
                         return tileSize == bound || tileSize == 0;
                       })) {
    bool hasNonUnitInnerSlice = false;
    for (int i = numBatch, e = loopDepth; i < e; ++i) {
      if (workgroupTileSizes[i] > 1) {
        workgroupTileSizes[i] /= 2;
        hasNonUnitInnerSlice = true;
        break;
      }
    }
    // If the inner most slice is a single element then we have to bail out.
    // TODO: Support this case.
    if (!hasNonUnitInnerSlice) {
      return failure();
    }
  }

  // Attach the MMA schedule as an attribute to the entry point export function
  // for later access in the pipeline.
  MLIRContext *context = scatter.getContext();
  SmallVector<NamedAttribute> attrs;
  Builder b(context);
  attrs.emplace_back(StringAttr::get(context, "workgroup"),
                     b.getI64ArrayAttr(workgroupTileSizes));

  attrs.emplace_back(StringAttr::get(context, "thread"),
                     b.getI64ArrayAttr(threadTileSizes));
  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  LDBG() << "Selected tile and fuse lowering config: " << loweringConfig
         << "\n";

  // TODO(qedawkins): Use a shared pipeline identifier here.
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, scatter, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
      {flatWorkgroupSize, 1, 1}, flatWorkgroupSize, DictionaryAttr());
}

LogicalResult
setDirectConvolutionLoweringConfig(IREE::GPU::TargetAttr target,
                                   mlir::FunctionOpInterface entryPoint,
                                   Operation *computeOp) {
  auto op = dyn_cast<linalg::LinalgOp>(computeOp);
  if (target.getWgp().getMma().empty())
    return failure();

  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();

  SmallVector<int64_t> bounds = op.getStaticLoopRanges();
  FailureOr<mlir::linalg::ConvolutionDimensions> convolutionDims =
      mlir::linalg::inferConvolutionDims(op);
  if (failed(convolutionDims)) {
    return failure();
  }

  // This strategy turns non-strided/dilated convolution problems into matmul
  // problems by tiling certain dimensions to 1:
  //  - Batch dimensions (parallel shared by the image and output)
  //  - Filter dimensions (reduction on the filter, and convolved on the image)
  //  - All output image dimensions except the outermost one
  //
  // After this, the remaining non-unit dimensions are:
  //  - One output image dimension corresponding to the M dimension of a matmul.
  //  - The output channel dimension, corresponding to the N dimension.
  //  - The input channel dimension, corresponding to the K dimension.

  // TODO: Relax this condition to strictly alignment requirements.
  if (convolutionDims->outputChannel.size() < 1 ||
      convolutionDims->inputChannel.size() < 1 ||
      convolutionDims->filterLoop.size() < 1 ||
      convolutionDims->outputImage.size() < 1 ||
      convolutionDims->depth.size() != 0) {
    return failure();
  }

  auto isAllOnesList = [](ArrayRef<int64_t> list) {
    return llvm::all_of(list, [](int64_t i) { return i == 1; });
  };

  // TODO: Support non-unit strides/dilations.
  if (!isAllOnesList(convolutionDims->strides) ||
      !isAllOnesList(convolutionDims->dilations)) {
    return failure();
  }

  int64_t mDim = convolutionDims->outputImage.back();
  int64_t nDim = convolutionDims->outputChannel.back();
  // TODO: Support NCHW convolutions. This is just a matmul_transpose_a, however
  // the distribution patterns currently do not support that variant.
  if (mDim > nDim) {
    return failure();
  }
  int64_t kDim = convolutionDims->inputChannel.back();

  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value init = op.getDpsInitOperand(0)->get();

  Type lhsElemType = getElementTypeOrSelf(lhs);
  Type rhsElemType = getElementTypeOrSelf(rhs);
  Type initElemType = getElementTypeOrSelf(init);

  // TODO(Max191): Support multiple M/N/K dimension problems for MMASchedules
  // once the pipeline is able to support it. After adding multiple dimensions,
  // all instances of schedule->m/nSubgroupCounts[0] and
  // schedule->m/n/kTileSizes[0] need to use the full list of sizes instead of
  // just the first element.
  GPUMatmulShapeType problem{bounds[mDim], bounds[nDim], bounds[kDim],
                             lhsElemType,  rhsElemType,  initElemType};

  // Helper fn to store mma information.
  auto storeMmaInfo = [](IREE::GPU::MmaInterfaceAttr mma,
                         SmallVector<GPUIntrinsicType> &intrinsics) {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    intrinsics.emplace_back(mSize, nSize, kSize, aType, bType, cType, mma);
  };

  SmallVector<GPUIntrinsicType> intrinsics;
  intrinsics.reserve(target.getWgp().getMma().size());
  MLIRContext *context = op.getContext();
  for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
    if (mma.getSubgroupSize() != targetSubgroupSize)
      continue;
    storeMmaInfo(mma, intrinsics);
    // Store info on virtual intrinsics based on current mma if any
    for (IREE::GPU::VirtualMMAIntrinsic virtualIntrinsic :
         mma.getVirtualIntrinsics()) {
      auto virtualMma =
          IREE::GPU::VirtualMMAAttr::get(context, virtualIntrinsic);
      storeMmaInfo(virtualMma, intrinsics);
    }
  }

  if (intrinsics.empty())
    return failure();

  // Note that the following heuristic seeds are just placeholder values.
  // We need to clean it up and make it adjusting to different targets.
  // See https://github.com/iree-org/iree/issues/16341 for details.
  GPUMMAHeuristicSeeds seeds{/*bestSubgroupCountPerWorkgroup=*/4,
                             /*bestMNTileCountPerSubgroup=*/8,
                             /*bestKTileCountPerSubgroup=*/2};

  int64_t maxSharedMemoryBytes = target.getWgp().getMaxWorkgroupMemoryBytes();

  std::optional<int64_t> wgpCount = std::nullopt;
  if (IREE::GPU::TargetChipAttr chip = target.getChip()) {
    wgpCount = chip.getWgpCount();
  }

  // First try to find a schedule with an exactly matching intrinsic.
  FailureOr<GPUMMASchedule> schedule =
      deduceMMASchedule(problem, intrinsics, seeds, maxSharedMemoryBytes,
                        targetSubgroupSize, wgpCount);
  if (failed(schedule)) {
    // Then try again by allowing upcasting accumulator.
    schedule =
        deduceMMASchedule(problem, intrinsics, seeds, maxSharedMemoryBytes,
                          targetSubgroupSize, wgpCount,
                          /*transposedLhs*/ false, /*transposedRhs*/ false,
                          /*canUpcastAcc=*/true);
  }
  if (failed(schedule)) {
    return failure();
  }

  int64_t flatWorkgroupSize =
      targetSubgroupSize *
      ShapedType::getNumElements(schedule->nSubgroupCounts) *
      ShapedType::getNumElements(schedule->mSubgroupCounts);
  std::array<int64_t, 3> workgroupSize{flatWorkgroupSize, 1, 1};

  SmallVector<int64_t> workgroupTileSizes(op.getNumLoops(), 0);
  SmallVector<int64_t> reductionTileSizes(op.getNumLoops(), 0);
  SmallVector<int64_t> subgroupTileSizes(op.getNumLoops(), 0);
  SmallVector<int64_t> convTileSizes(op.getNumLoops(), 0);
  // Tile all batch dimensions with unit size.
  for (int64_t batch : convolutionDims->batch) {
    workgroupTileSizes[batch] = 1;
  }
  // Tile all output image dimensions with unit size except the last one.
  for (int64_t oi : llvm::drop_end(convolutionDims->outputImage)) {
    workgroupTileSizes[oi] = 1;
  }
  for (int64_t oc : llvm::drop_end(convolutionDims->outputChannel)) {
    workgroupTileSizes[oc] = 1;
  }
  for (int64_t ic : llvm::drop_end(convolutionDims->inputChannel)) {
    reductionTileSizes[ic] = 1;
  }
  // Compute the M/N dimension tile size by multiply subgroup information.
  workgroupTileSizes[mDim] =
      schedule->mSubgroupCounts[0] * schedule->mTileSizes[0] * schedule->mSize;
  subgroupTileSizes[mDim] = schedule->mTileSizes[0];
  workgroupTileSizes[nDim] =
      schedule->nSubgroupCounts[0] * schedule->nTileSizes[0] * schedule->nSize;
  subgroupTileSizes[nDim] = schedule->nTileSizes[0];

  reductionTileSizes[kDim] = schedule->kTileSizes[0] * schedule->kSize;

  Builder b(context);
  SmallVector<NamedAttribute, 4> attrs = {
      NamedAttribute("workgroup", b.getI64ArrayAttr(workgroupTileSizes)),
      NamedAttribute("reduction", b.getI64ArrayAttr(reductionTileSizes)),
      NamedAttribute("conv_reduction", b.getI64ArrayAttr(convTileSizes)),
      NamedAttribute("subgroup", b.getI64ArrayAttr(subgroupTileSizes))};
  IREE::GPU::appendPromotedOperandsList(context, attrs, {0, 1});
  IREE::GPU::setMmaKind(context, attrs, schedule->mmaKind);
  // IREE::GPU::setSubgroupMCount(context, attrs, schedule->mSubgroupCounts[0]);
  // IREE::GPU::setSubgroupNCount(context, attrs, schedule->nSubgroupCounts[0]);

  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  SmallVector<NamedAttribute, 1> pipelineAttrs;

  // Prefetch shared memory is kept off.
  auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(
      context, /*prefetchSharedMemory=*/false,
      /*no_reduce_shared_memory_bank_conflicts=*/false,
      /*use_igemm_convolution=*/false,
      /*use_direct_convolution=*/true,
      /*reorder_workgroups_strategy=*/std::nullopt);
  pipelineAttrs.emplace_back(
      IREE::GPU::GPUPipelineOptionsAttr::getDictKeyName(), pipelineOptions);

  auto pipelineConfig = DictionaryAttr::get(context, pipelineAttrs);

  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig,
      Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse, workgroupSize,
      targetSubgroupSize, pipelineConfig);
}

//====---------------------------------------------------------------------===//
// Sort Pipeline Configuration
//====---------------------------------------------------------------------===//

LogicalResult setSortConfig(IREE::GPU::TargetAttr target,
                            mlir::FunctionOpInterface entryPoint,
                            Operation *op) {
  assert(isa<IREE::LinalgExt::SortOp>(op) && "expected linalg_ext.sort op");
  MLIRContext *context = op->getContext();
  Builder b(context);

  const int64_t subgroupSize = target.getPreferredSubgroupSize();
  auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
  SmallVector<unsigned> partitionedLoops =
      interfaceOp.getPartitionableLoops(std::nullopt);

  auto createLoweringConfig = [&](ArrayRef<int64_t> workgroupSizes,
                                  ArrayRef<int64_t> threadSizes) {
    NamedAttribute attrs[2] = {
        NamedAttribute("workgroup", b.getI64ArrayAttr(workgroupSizes)),
        NamedAttribute("thread", b.getI64ArrayAttr(threadSizes))};
    auto configDict = b.getDictionaryAttr(attrs);
    return IREE::GPU::LoweringConfigAttr::get(context, configDict);
  };

  if (partitionedLoops.empty()) {
    IREE::GPU::LoweringConfigAttr loweringConfig =
        createLoweringConfig(int64_t{0}, int64_t{0});
    return setOpConfigAndEntryPointFnTranslation(
        entryPoint, op, loweringConfig,
        IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
        {1, 1, 1}, subgroupSize, DictionaryAttr());
  }

  unsigned numLoops = cast<ShapedType>(op->getResult(0).getType()).getRank();

  // To get peak occupancy we need a workgroup size of at least two warps.
  std::array<int64_t, 3> workgroupSize = {2 * subgroupSize, 1, 1};
  SmallVector<int64_t> workgroupTileSizes(numLoops, 1);
  SmallVector<int64_t> threadTileSizes(numLoops, 1);

  // Set all non-parallel loops to zero tile size.
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto depth : llvm::seq<int64_t>(0, numLoops)) {
    if (!partitionedLoopsSet.count(depth)) {
      workgroupTileSizes[depth] = 0;
      threadTileSizes[depth] = 0;
    }
  }

  // Tile to have one element per thread.
  ArrayRef loopBounds = cast<IREE::LinalgExt::SortOp>(op).getOperandShape();
  int64_t residualWorkgroupSize = workgroupSize[0];
  for (int64_t depth = numLoops - 1; depth >= 0; --depth) {
    if (!partitionedLoopsSet.contains(depth)) {
      continue;
    }
    if (ShapedType::isDynamic(loopBounds[depth])) {
      continue;
    }
    if (residualWorkgroupSize % loopBounds[depth] == 0) {
      workgroupTileSizes[depth] = loopBounds[depth];
      residualWorkgroupSize /= loopBounds[depth];
      continue;
    }
    if (loopBounds[depth] % residualWorkgroupSize == 0) {
      workgroupTileSizes[depth] = residualWorkgroupSize;
      break;
    }
  }

  IREE::GPU::LoweringConfigAttr loweringConfig =
      createLoweringConfig(workgroupTileSizes, threadTileSizes);
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
      workgroupSize, subgroupSize, DictionaryAttr());
}

//===----------------------------------------------------------------------===//
// Lowering Config Attributes
//===----------------------------------------------------------------------===//

GPUPipelineOptions
getPipelineOptions(FunctionOpInterface funcOp,
                   IREE::Codegen::TranslationInfoAttr translationInfo) {
  GPUPipelineOptions pipelineOptions = {};
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);

  if (DictionaryAttr config = translationInfo.getConfiguration()) {
    std::optional<NamedAttribute> maybePipelineOptionsAttr =
        config.getNamed(GPUPipelineOptionsAttr::getDictKeyName());
    if (!maybePipelineOptionsAttr.has_value()) {
      return pipelineOptions;
    }
    auto pipelineOptionsAttr =
        cast<GPUPipelineOptionsAttr>(maybePipelineOptionsAttr->getValue());
    BoolAttr prefetchSharedMemory =
        pipelineOptionsAttr.getPrefetchSharedMemory();
    if (prefetchSharedMemory) {
      pipelineOptions.prefetchSharedMemory = prefetchSharedMemory.getValue();
    }
    BoolAttr noReduceBankConflicts =
        pipelineOptionsAttr.getNoReduceSharedMemoryBankConflicts();
    if (noReduceBankConflicts) {
      pipelineOptions.enableReduceSharedMemoryBankConflicts =
          !noReduceBankConflicts.getValue();
    }
    BoolAttr useIgemmConvolution = pipelineOptionsAttr.getUseIgemmConvolution();
    if (useIgemmConvolution) {
      pipelineOptions.useIgemmConvolution = useIgemmConvolution.getValue();
    }
    BoolAttr useDirectConvolution =
        pipelineOptionsAttr.getUseDirectConvolution();
    if (useDirectConvolution) {
      pipelineOptions.useDirectConvolution = useDirectConvolution.getValue();
    }
    ReorderWorkgroupsStrategyAttr reorderWorkgroupsStrategy =
        pipelineOptionsAttr.getReorderWorkgroupsStrategy();
    if (reorderWorkgroupsStrategy) {
      pipelineOptions.reorderStrategy = reorderWorkgroupsStrategy.getValue();
    }
  }

  pipelineOptions.enableUkernels = targetAttr && hasUkernel(targetAttr);

  LLVM_DEBUG(llvm::dbgs() << "GPU Pipeline Options: " << pipelineOptions
                          << "\n");
  return pipelineOptions;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const GPUPipelineOptions &options) {
  StringRef reorderStr = "<not set>";
  if (options.reorderStrategy) {
    switch (options.reorderStrategy.value()) {
    case ReorderWorkgroupsStrategy::Transpose:
      reorderStr = "transpose";
      break;
    case ReorderWorkgroupsStrategy::None:
      reorderStr = "none";
      break;
    default:
      assert(false && "Unhandled reorder option");
    }
  }

  return os << "{" << "enableReduceSharedMemoryBankConflicts = "
            << options.enableReduceSharedMemoryBankConflicts << ", "
            << ", prefetchSharedMemory = " << options.prefetchSharedMemory
            << ", useIgemmConvolution = " << options.useIgemmConvolution
            << ", useDirectConvolution = " << options.useDirectConvolution
            << ", reorderWorkgroupsStrategy = " << reorderStr
            << ", enableUkernels = " << options.enableUkernels << "}";
}

} // namespace mlir::iree_compiler::IREE::GPU
