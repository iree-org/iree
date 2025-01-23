// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-gpu-config-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler::IREE::GPU {

constexpr int64_t kCacheLineSizeBits = 128 * 8;
constexpr int64_t kPreferredCopyNumBits = 128;

//===----------------------------------------------------------------------===//
// Workgroup Size Multiples
//===----------------------------------------------------------------------===//

static SmallVector<int64_t> getDefaultValueMultiples(Value v) {
  auto shapedType = dyn_cast<ShapedType>(v.getType());
  return shapedType ? SmallVector<int64_t>(shapedType.getRank(), 1)
                    : SmallVector<int64_t>();
}

/// Compute the workgroup tile size multiples for pack or unpack based on the
/// inner tile sizes. Returns a pair of vectors `{srcMultiples, destMultiples}`
/// for the multiples required for the source and destination tensors. Allows
/// an optional set of initial packed and unpacked operand multiples, which, if
/// present will be composed with the computed multiples for the src and dest
/// by taking the LCM of each corresponding multiple.
template <typename PackOrUnPackOpTy>
static std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
inferWorkgroupTileMultiplesFromPackUnPack(
    PackOrUnPackOpTy op,
    std::optional<SmallVector<int64_t>> initialPackedMultiples = std::nullopt,
    std::optional<SmallVector<int64_t>> initialUnPackedMultiples =
        std::nullopt) {
  static_assert(llvm::is_one_of<PackOrUnPackOpTy, tensor::PackOp,
                                tensor::UnPackOp>::value);
  LDBG("Inferring workgroup tile size multiples from " << op->getName() << ":\n"
                                                       << op);
  // Initialize the list of multiples for the packed and unpack inputs.
  int64_t unPackedRank = (std::is_same<PackOrUnPackOpTy, tensor::PackOp>::value)
                             ? op.getSourceRank()
                             : op.getDestRank();
  SmallVector<int64_t> innerTiles = op.getStaticTiles();
  // If the tiles are dynamic, then no inference can be made.
  if (ShapedType::isDynamicShape(innerTiles)) {
    LLVM_DEBUG(DBGS() << "Cannot infer multiple for dynamic inner tiles."
                         "Defaulting to all 1.");
    return {getDefaultValueMultiples(op.getSource()),
            getDefaultValueMultiples(op.getDest())};
  }

  // Otherwise, adjust the multiples of the packed and unpacked inputs for the
  // inner tiles.
  int64_t packedRank = unPackedRank + innerTiles.size();
  SmallVector<int64_t> packedMultiples(packedRank, 1);
  SmallVector<int64_t> unPackedMultiples(unPackedRank, 1);
  for (auto [i, tile, pos] :
       llvm::enumerate(innerTiles, op.getInnerDimsPos())) {
    // We need to invert the outerDimsPerm vector to compute the outerTileIdx.
    // This is slightly counterintuitive, but the reason this works is because:
    //  - The outerDimsPerm is a mapping from the permuted index to the
    //    unpermuted index, since outerDimsPerm[i] will give you the element
    //    position from the unpermuted tensor that belongs at position `i` in
    //    the permuted tensor.
    //  - We have the innerDimPos, which is an index in the unpermuted tensor,
    //    so we need a mapping from unpermuted tensor indices to the permuted
    //    tensor indices, which is the inverse of outerDimsPerm.
    int64_t outerTileIdx = invertPermutationVector(op.getOuterDimsPerm())[pos];
    int64_t innerTileIdx = i + innerTiles.size();
    // Compute the LCM with the initial multiples for both the inner tile and
    // the corresponding outer tile. The multiples for the packedMultiples will
    // then be these LCMs, and the multiple for the unPackedMultipes will be the
    // product of these LCMs.
    int64_t lcmInnerTileMultiple = tile;
    int64_t lcmOuterTileMultiple = 1;
    if (initialPackedMultiples) {
      lcmInnerTileMultiple = std::lcm(
          initialPackedMultiples.value()[innerTileIdx], lcmInnerTileMultiple);
      if (lcmInnerTileMultiple != tile) {
        LLVM_DEBUG(DBGS() << "Cannot find a compatible tile size multiple. "
                          << "Defaulting to all 1.");
        return {getDefaultValueMultiples(op.getSource()),
                getDefaultValueMultiples(op.getDest())};
      }
      lcmOuterTileMultiple = std::lcm(
          initialPackedMultiples.value()[outerTileIdx], lcmOuterTileMultiple);
    }
    if (initialUnPackedMultiples) {
      int64_t unPackedMultiple = lcmOuterTileMultiple * tile;
      int64_t lcmUnPackedMultiple =
          std::lcm(initialUnPackedMultiples.value()[pos], unPackedMultiple);
      lcmOuterTileMultiple = lcmUnPackedMultiple / tile;
    }
    packedMultiples[innerTileIdx] = lcmInnerTileMultiple;
    packedMultiples[outerTileIdx] = lcmOuterTileMultiple;
    unPackedMultiples[pos] = lcmOuterTileMultiple * lcmInnerTileMultiple;
  }

  SmallVector<int64_t> srcMultiples =
      (std::is_same<PackOrUnPackOpTy, tensor::PackOp>::value)
          ? unPackedMultiples
          : packedMultiples;
  SmallVector<int64_t> destMultiples =
      (std::is_same<PackOrUnPackOpTy, tensor::PackOp>::value)
          ? packedMultiples
          : unPackedMultiples;
  LLVM_DEBUG({
    DBGS() << "\nInferred " << op->getName() << " multiples:\nsrc: [";
    llvm::interleaveComma(srcMultiples, llvm::dbgs());
    llvm::dbgs() << "]\nresult: [";
    llvm::interleaveComma(destMultiples, llvm::dbgs());
    llvm::dbgs() << "]\n\n";
  });
  return {srcMultiples, destMultiples};
}

/// Given some initial operand, result, and iteration space multiples, compute
/// the least common multiples for each dimension of the iteration space, and
/// adjust the given multiples so all operands, results, and iteration
/// dimensions agree.
static void inferWorkgroupTileMultiplesFromLinalgOp(
    linalg::LinalgOp linalgOp, SmallVector<int64_t> &iterationMultiples,
    SmallVector<SmallVector<int64_t>> &operandMultiples,
    SmallVector<SmallVector<int64_t>> &resultMultiples) {
  LDBG("Inferring workgroup tile size multiples for linalgOp:\n" << linalgOp);
  auto dbgsPrintMultiples = [](SmallVector<SmallVector<int64_t>> multiples) {
    LLVM_DEBUG({
      for (auto [i, m] : llvm::enumerate(multiples)) {
        DBGS() << "operand " << i << ": [";
        llvm::interleaveComma(m, llvm::dbgs());
        llvm::dbgs() << "]\n";
      }
    });
  };
  LDBG("\noperandMultiples:\n");
  dbgsPrintMultiples(operandMultiples);
  LDBG("\nresultMultiples:\n");
  dbgsPrintMultiples(resultMultiples);

  // Actual logic starts here.
  SmallVector<int64_t> linalgOpMultiples(
      linalgOp.getIteratorTypesArray().size(), 1);
  for (auto [operandIdx, map, multiples] :
       llvm::enumerate(linalgOp.getIndexingMapsArray(), operandMultiples)) {
    for (auto [idx, dim] : llvm::enumerate(map.getResults())) {
      auto dimExpr = dyn_cast<AffineDimExpr>(dim);
      if (!dimExpr) {
        continue;
      }
      int64_t dimPos = dimExpr.getPosition();
      int64_t lcm = std::lcm(iterationMultiples[dimPos], multiples[idx]);
      // If the operand is a DPS init, then include the result dims in the LCM.
      int64_t dpsInitIdx = operandIdx - linalgOp.getNumDpsInputs();
      if (dpsInitIdx >= 0) {
        lcm = std::lcm(resultMultiples[dpsInitIdx][idx], lcm);
        resultMultiples[dpsInitIdx][idx] = lcm;
      }
      iterationMultiples[dimPos] = lcm;
      operandMultiples[operandIdx][idx] = lcm;
    }
  }

  LLVM_DEBUG({
    DBGS() << "\niterationMultiples: [";
    llvm::interleaveComma(iterationMultiples, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });
}

/// Given a set of multiples, and reassociations for expansion, return the
/// corresponding expanded set of multiples, where the product of the expanded
/// multiples is equal to the original collapsed multiple within each
/// reassociation group. If there are dynamic shapes in the expanded dims, then
/// returns all ones as the expanded multiples, since inferring the appropriate
/// multiples is not possible in the dynamic case.
///
/// Example (parenthesis added to expanded groups for readability):
/// Input:
///   collapsedMultiples = [4, 32, 64]
///   expandedShape = [(7, 2), (8, 2, 4), (3, 32)]
///   reassociations = [[0, 1], [2, 3, 4], [5, 6]]
/// Output:
///   expandedMultiples = [(2, 2), (4, 2, 4), (2, 32)]
static SmallVector<int64_t>
expandMultiples(ArrayRef<int64_t> collapsedMultiples,
                ArrayRef<int64_t> expandedShape,
                SmallVector<ReassociationIndices> reassociations) {
  SmallVector<int64_t> expandedMultiples(expandedShape.size(), 1);
  for (auto [multiple, group] :
       llvm::zip_equal(collapsedMultiples, reassociations)) {
    if (group.size() == 1) {
      expandedMultiples[group[0]] = multiple;
      continue;
    }
    int64_t residualMultiple = multiple;
    for (int i = group.size() - 1; i >= 0; --i) {
      int64_t expandedSize = expandedShape[group[i]];
      if (ShapedType::isDynamic(expandedSize)) {
        LLVM_DEBUG(DBGS() << "Cannot infer multiple with dynamic size. "
                             "defaulting to all 1");
        expandedMultiples = SmallVector<int64_t>(expandedShape.size(), 1);
        return expandedMultiples;
      }
      if (residualMultiple % expandedSize != 0) {
        LLVM_DEBUG(DBGS() << "Expanded size does not divide producer "
                             "multiple. Defaulting to all 1");
        expandedMultiples = SmallVector<int64_t>(expandedShape.size(), 1);
        return expandedMultiples;
      }
      if (residualMultiple >= expandedSize) {
        expandedMultiples[group[i]] = expandedSize;
        residualMultiple /= expandedSize;
        continue;
      }
      expandedMultiples[group[i]] = residualMultiple;
      residualMultiple = 1;
      break;
    }
  }
  return expandedMultiples;
}

/// Find a set of required workgroup tile size mulitples for the given OpResult
/// by walking the producer chain of the OpResult's owner, and finding ops that
/// require specific tile size multiples. For now, the only ops that need
/// special workgroup tile size multiples are pack and unpack ops. The returned
/// list of multiples represent the required multiples for the workgroup tile
/// slice of the `result` tensor after tiling and distributing to workgroups.
static SmallVector<int64_t> inferResultWorkgroupTileMultiples(OpResult result) {
  LDBG("Inferring workgroup tile size multiples for result:\n" << result);
  // Gather multiples for all operands from producers.
  Operation *op = result.getOwner();
  auto getOperandMultiples = [&]() -> SmallVector<SmallVector<int64_t>> {
    SmallVector<SmallVector<int64_t>> operandMultiples;
    for (Value operand : op->getOperands()) {
      auto producerResult = dyn_cast<OpResult>(operand);
      if (!producerResult) {
        operandMultiples.push_back(getDefaultValueMultiples(operand));
        continue;
      }
      operandMultiples.push_back(
          inferResultWorkgroupTileMultiples(producerResult));
    }
    return operandMultiples;
  };
  // Propagate the operand multiples through the given operation to compute
  // the multiples for the desired result.
  return llvm::TypeSwitch<Operation *, SmallVector<int64_t>>(op)
      .Case<tensor::ExpandShapeOp>([&](tensor::ExpandShapeOp expandOp) {
        SmallVector<int64_t> srcMultiples = getOperandMultiples()[0];
        LDBG("Inferring workgroup tile size multiples for "
             << expandOp->getName() << " result.\n");
        SmallVector<int64_t> resultMultiples = expandMultiples(
            /*collapsedMultiples=*/srcMultiples,
            /*expandedShape=*/expandOp.getResultType().getShape(),
            /*reassociations=*/expandOp.getReassociationIndices());
        LLVM_DEBUG({
          DBGS() << "\nInferred expand_shape result multiples: [";
          llvm::interleaveComma(resultMultiples, llvm::dbgs());
          llvm::dbgs() << "]\n";
        });
        return resultMultiples;
      })
      .Case<tensor::PackOp>([&](tensor::PackOp packOp) {
        SmallVector<int64_t> srcMultiples = getOperandMultiples()[0];
        return inferWorkgroupTileMultiplesFromPackUnPack(
                   packOp, /*initialPackedMultiples=*/std::nullopt,
                   /*initialUnPackedMultiples=*/srcMultiples)
            .second;
      })
      .Case<tensor::UnPackOp>([&](tensor::UnPackOp unPackOp) {
        SmallVector<int64_t> srcMultiples = getOperandMultiples()[0];
        return inferWorkgroupTileMultiplesFromPackUnPack(
                   unPackOp, /*initialPackedMultiples=*/srcMultiples,
                   /*initialUnPackedMultiples=*/std::nullopt)
            .second;
      })
      .Case<linalg::LinalgOp>([&](linalg::LinalgOp linalgOp) {
        SmallVector<SmallVector<int64_t>> operandMultiples =
            getOperandMultiples();
        LDBG("Inferring workgroup tile size multiples for linalg op result #"
             << result.getResultNumber() << ":\n"
             << result);
        SmallVector<SmallVector<int64_t>> resultMultiples = llvm::map_to_vector(
            linalgOp->getResults(), getDefaultValueMultiples);
        SmallVector<int64_t> iterationMultiples(
            linalgOp.getIteratorTypesArray().size(), 1);
        inferWorkgroupTileMultiplesFromLinalgOp(
            linalgOp, iterationMultiples, operandMultiples, resultMultiples);
        return resultMultiples[result.getResultNumber()];
      })
      .Default([&](Operation *) {
        LDBG("Unsupported operation. Defualting to all 1: " << result);
        return getDefaultValueMultiples(result);
      });
}

/// Find a set of required workgroup tile size mulitples for the given OpOperand
/// by walking the use chain of the OpOperand's owner, and finding ops that
/// require specific tile size multiples. For now, the only ops that need
/// special workgroup tile size multiples are pack and unpack ops. The returned
/// list of multiples represent the required multiples for the workgroup tile
/// slice of the `use` tensor after tiling and distributing to workgroups.
static SmallVector<int64_t> inferUseWorkgroupTileMultiples(OpOperand *use) {
  LDBG("Inferring workgroup tile size multiples for operand "
       << use->getOperandNumber() << " of user:\n"
       << *use->getOwner());
  // Gather multiples for all operands from producers.
  Operation *op = use->getOwner();
  auto getResultMultiples = [&]() -> SmallVector<SmallVector<int64_t>> {
    SmallVector<SmallVector<int64_t>> resultMultiples;
    for (Value result : op->getResults()) {
      for (OpOperand &opUse : result.getUses()) {
        resultMultiples.push_back(inferUseWorkgroupTileMultiples(&opUse));
      }
    }
    return resultMultiples;
  };
  // Propagate the operand multiples through the given operation to compute
  // the multiples for the desired result.
  return llvm::TypeSwitch<Operation *, SmallVector<int64_t>>(op)
      .Case<tensor::CollapseShapeOp>([&](tensor::CollapseShapeOp collapseOp) {
        SmallVector<int64_t> destMultiples = getResultMultiples()[0];
        LDBG("Inferring workgroup tile size multiples for "
             << collapseOp->getName() << "source.\n");
        SmallVector<int64_t> srcMultiples = expandMultiples(
            /*collapsedMultiples=*/destMultiples,
            /*expandedShape=*/collapseOp.getSrcType().getShape(),
            /*reassociations=*/collapseOp.getReassociationIndices());
        LLVM_DEBUG({
          DBGS() << "\nInferred collapse_shape source multiples: ";
          llvm::interleaveComma(srcMultiples, llvm::dbgs());
          llvm::dbgs() << "]\n";
        });
        return srcMultiples;
      })
      .Case<tensor::PackOp>([&](tensor::PackOp packOp) {
        SmallVector<int64_t> destMultiples = getResultMultiples()[0];
        return inferWorkgroupTileMultiplesFromPackUnPack(
                   packOp, /*initialPackedMultiples=*/destMultiples,
                   /*initialUnPackedMultiples=*/std::nullopt)
            .first;
      })
      .Case<tensor::UnPackOp>([&](tensor::UnPackOp unpackOp) {
        SmallVector<int64_t> destMultiples = getResultMultiples()[0];
        return inferWorkgroupTileMultiplesFromPackUnPack(
                   unpackOp, /*initialPackedMultiples=*/std::nullopt,
                   /*initialUnPackedMultiples=*/destMultiples)
            .first;
      })
      .Default([&](Operation *) {
        LDBG("Unsupported operation. Defualting to all 1: " << use->get());
        return getDefaultValueMultiples(use->get());
      });
}

/// Walks the producer and consumer chains of the `tilingOp`, and looks for ops
/// that require specific workgroup tile size multiples. Right now, the only ops
/// that require a specific multiple are pack and unpack, since the workgroup
/// tile sizes need to be multiples of the inner_tiles. After walking the IR and
/// finding multiples for the slices of the `tilingOp` operands and results, the
/// function computes and returns the multiples of the `tilingOp` iteration
/// space. The function may fail to find a valid set of workgroup size
/// multiples, in which case the function will fallback to returning a list of
/// all 1, meaning no constraints on the workgroup tile sizes.
static SmallVector<int64_t>
getWorkgroupSizeMultiples(TilingInterface tilingOp) {
  LDBG("Computing workgroup tile size multiples for: "
       << *tilingOp.getOperation());
  auto linalgOp = dyn_cast<linalg::LinalgOp>(tilingOp.getOperation());
  if (!linalgOp) {
    LDBG("Only LinalgOp is implemented. Defaulting to all 1 multiples.");
    return SmallVector<int64_t>(tilingOp.getLoopIteratorTypes().size(), 1);
  }

  // Get operand and result multiples for the op.
  SmallVector<SmallVector<int64_t>> operandMultiples;
  for (Value operand : tilingOp->getOperands()) {
    auto result = dyn_cast<OpResult>(operand);
    operandMultiples.push_back(result
                                   ? inferResultWorkgroupTileMultiples(result)
                                   : getDefaultValueMultiples(operand));
  }
  SmallVector<SmallVector<int64_t>> resultMultiples;
  for (Value result : tilingOp->getResults()) {
    for (OpOperand &use : result.getUses()) {
      resultMultiples.push_back(inferUseWorkgroupTileMultiples(&use));
    }
  }

  // Infer the workgroup tile size multiples for the iteration space of
  // `tilingOp` based on the multiples of the operand and result workgroup
  // tile multiples.
  SmallVector<int64_t> tileSizeMultiples(tilingOp.getLoopIteratorTypes().size(),
                                         1);
  inferWorkgroupTileMultiplesFromLinalgOp(linalgOp, tileSizeMultiples,
                                          operandMultiples, resultMultiples);
  return tileSizeMultiples;
}

//===----------------------------------------------------------------------===//
// Lowering Config Selection
//===----------------------------------------------------------------------===//

LogicalResult setDataTiledMultiMmaLoweringConfig(
    IREE::GPU::TargetAttr target, mlir::FunctionOpInterface entryPoint,
    Operation *op, IREE::GPU::UKernelConfigAttr ukernelConfig) {
  auto multiMmaOp = dyn_cast<IREE::GPU::MultiMmaOp>(op);
  if (!multiMmaOp) {
    return failure();
  }
  auto dataTiledMmaAttr = dyn_cast<DataTiledMMAAttr>(multiMmaOp.getKind());
  if (!dataTiledMmaAttr) {
    return failure();
  }

  LDBG("MultiMMA TileAndFuse Config");

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
  // already baked into the "opaque" data tiled inner layout of the multi_mma.
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
  SmallVector<NamedAttribute, 1> attrs;
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
    GPU::setPromotedOperandList(context, attrs, {0, 1});
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
    bool doCPromotion = false) {
  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();
  SmallVector<GPUMatmulShapeType> intrinsics;
  for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
    // Intrinsics that do not specify a scope cannot be distributed.
    if (failed(mma.getMmaScope()))
      continue;
    if (mma.getSubgroupSize() != targetSubgroupSize)
      continue;

    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();
    intrinsics.emplace_back(mSize, nSize, kSize, aType, bType, cType);
  }
  if (intrinsics.empty())
    return std::nullopt;

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

  // First try to find a schedule with an exactly matching intrinsic.
  std::optional<GPUMMASchedule> schedule = deduceMMASchedule(
      problem, intrinsics, seeds, maxSharedMemoryBytes, targetSubgroupSize,
      transposedLhs, transposedRhs, /*canUpcastAcc=*/false,
      /*mustBeAligned*/ mustBeAligned, doCPromotion);
  return schedule;
}

/// Create a matmul lowering config based on iteration bounds and indexing
/// maps for a given target. This function computes contraction dimensions
/// and deduces an MMA intrinsic schedule to choose tile sizes and the
/// workgroup size.
static FailureOr<std::pair<LoweringConfigAttr, int64_t>>
getMatmulLoweringConfigAndWorkgroupSize(SmallVector<int64_t> bounds,
                                        ArrayRef<AffineMap> maps,
                                        ArrayRef<Value> operands,
                                        IREE::GPU::TargetAttr target) {
  if (target.getWgp().getMma().empty())
    return failure();

  mlir::linalg::ContractionDimensions contractionDims =
      mlir::linalg::inferContractionDims(maps).value();

  if (contractionDims.k.empty() || contractionDims.m.empty() ||
      contractionDims.n.empty()) {
    return failure();
  }

  // TODO(Max191): add dynamic shape support for inner most dims.
  if (ShapedType::isDynamic(bounds[contractionDims.m.back()]) ||
      ShapedType::isDynamic(bounds[contractionDims.n.back()]) ||
      ShapedType::isDynamic(bounds[contractionDims.k.back()])) {
    return failure();
  }

  // Gather all static M, N, and K dimensions to deduce the MMASchedule. Dynamic
  // dimensions will be tiled to 1 in workgroup tiling, so they are ignored when
  // computing an MMA schedule.
  SmallVector<int64_t> mDims, nDims, kDims, batchDims;
  for (int64_t mDim : contractionDims.m) {
    if (!ShapedType::isDynamic(bounds[mDim])) {
      mDims.push_back(mDim);
    }
  }
  for (int64_t nDim : contractionDims.n) {
    if (!ShapedType::isDynamic(bounds[nDim])) {
      nDims.push_back(nDim);
    }
  }
  for (int64_t kDim : contractionDims.k) {
    if (!ShapedType::isDynamic(bounds[kDim])) {
      kDims.push_back(kDim);
    }
  }

  for (int64_t batchDim : contractionDims.batch) {
    if (!ShapedType::isDynamic(bounds[batchDim])) {
      batchDims.push_back(batchDim);
    }
  }

  auto getDimBounds = [&](SmallVector<int64_t> dims) -> SmallVector<int64_t> {
    return llvm::map_to_vector(dims, [&](int64_t dim) { return bounds[dim]; });
  };

  assert(operands.size() == 3 && "expected 3 operands");
  Value lhs = operands[0];
  Value rhs = operands[1];
  Value init = operands[2];

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
      target, problem, transposedLhs, transposedRhs);

  // TODO (nirvedhmeshram, qedawkins): The performance with this will be bad if
  // the GEMM is accumulating (i.e doesnt have a zero fill dpsInit) as that
  // buffer currently gets materialized as private memory. We need to add
  // missing patterns to fix that.
  if (!schedule) {
    LDBG("Attempting to deduce unaligned TileAndFuse MMA schedulee");
    mustBeAligned = false;
    doCPromotion = true;
    schedule = getMmaScheduleFromProblemAndTarget(target, problem,
                                                  transposedLhs, transposedRhs,
                                                  mustBeAligned, doCPromotion);
  }

  if (!schedule) {
    LDBG("Failed to deduce TileAndFuse MMA schedule");
    return failure();
  }

  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();
  LDBG("Target Subgroup size: " << targetSubgroupSize);
  LDBG("Schedule: " << schedule);

  SmallVector<int64_t> workgroupTileSizes(bounds.size(), 0);
  SmallVector<int64_t> reductionTileSizes(bounds.size(), 0);
  SmallVector<int64_t> subgroupTileSizes(bounds.size(), 0);
  // Tile all batch dimensions with unit size.
  for (int64_t batch : contractionDims.batch) {
    workgroupTileSizes[batch] = 1;
  }

  // Tile all m, n, and k dimensions to 1 except the innermost. Unit dims
  // from this tiling are folded before vectorization.
  for (int64_t m : llvm::drop_end(contractionDims.m)) {
    workgroupTileSizes[m] = 1;
  }
  for (int64_t n : llvm::drop_end(contractionDims.n)) {
    workgroupTileSizes[n] = 1;
  }
  for (int64_t k : llvm::drop_end(contractionDims.k)) {
    reductionTileSizes[k] = 1;
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

  IREE::GPU::MmaInterfaceAttr mmaKind =
      target.getWgp().getMma()[schedule->index];

  // Attach the MMA schedule as an attribute to the entry point export function
  // for later access in the pipeline.
  MLIRContext *context = lhs.getContext();
  SmallVector<NamedAttribute, 1> attrs;
  Builder b(context);
  attrs.emplace_back(StringAttr::get(context, "workgroup"),
                     b.getI64ArrayAttr(workgroupTileSizes));
  attrs.emplace_back(StringAttr::get(context, "reduction"),
                     b.getI64ArrayAttr(reductionTileSizes));
  attrs.emplace_back(StringAttr::get(context, "subgroup"),
                     b.getI64ArrayAttr(subgroupTileSizes));
  attrs.emplace_back(StringAttr::get(context, "mma_kind"), mmaKind);
  if (mustBeAligned) {
    GPU::setPromotedOperandList(context, attrs, {0, 1});
  } else {
    // TODO (nirvedhmeshram, Max191, jerryyin) : Add support so that unaligned
    // shapes do not require c promotion.
    GPU::setPromotedOperandList(context, attrs, {0, 1, 2});
    SmallVector<int64_t> paddingTileSizes = workgroupTileSizes;

    // Initialize inner and outer padding sizes from reductionTileSizes.
    for (int64_t kDim : kDims) {
      paddingTileSizes[kDim] = reductionTileSizes[kDim];
    }

    int64_t innerKDim = contractionDims.k.back();
    int64_t kPackFactor = std::get<2>(mmaKind.getMNKShape());
    paddingTileSizes[innerKDim] *= kPackFactor;

    attrs.emplace_back(StringAttr::get(context, "padding"),
                       b.getI64ArrayAttr(paddingTileSizes));
  }
  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);
  int64_t flatWorkgroupSize =
      targetSubgroupSize *
      ShapedType::getNumElements(schedule->nSubgroupCounts) *
      ShapedType::getNumElements(schedule->mSubgroupCounts);

  return std::make_pair(loweringConfig, flatWorkgroupSize);
}

LogicalResult
setIGEMMConvolutionLoweringConfig(IREE::GPU::TargetAttr target,
                                  mlir::FunctionOpInterface entryPoint,
                                  Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || !linalg::isaConvolutionOpInterface(linalgOp)) {
    return failure();
  }

  if (target.getWgp().getMma().empty())
    return failure();

  LDBG("IGEMM TileAndFuse Config");
  FailureOr<LinalgExt::IGEMMGenericConvDetails> igemmGenericConvDetails =
      LinalgExt::getIGEMMGenericConvDetails(linalgOp);
  if (failed(igemmGenericConvDetails)) {
    LDBG("Unsupported convolution type");
    return failure();
  }
  SmallVector<AffineMap> igemmContractionMaps =
      igemmGenericConvDetails->igemmContractionMaps;
  SmallVector<int64_t> igemmLoopBounds =
      igemmGenericConvDetails->igemmLoopBounds;
  SmallVector<Value> igemmOperands = igemmGenericConvDetails->igemmOperands;

  SmallVector<int64_t> bounds = igemmLoopBounds;
  FailureOr<std::pair<LoweringConfigAttr, int64_t>> configAndWgSize =
      getMatmulLoweringConfigAndWorkgroupSize(bounds, igemmContractionMaps,
                                              igemmOperands, target);
  if (failed(configAndWgSize)) {
    return failure();
  }
  std::array<int64_t, 3> workgroupSize = {configAndWgSize->second, 1, 1};
  LoweringConfigAttr loweringConfig = configAndWgSize->first;

  SmallVector<NamedAttribute, 1> pipelineAttrs;
  auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(
      linalgOp->getContext(), /*prefetchSharedMemory=*/true,
      /*no_reduce_shared_memory_bank_conflicts=*/false,
      /*use_igemm_convolution=*/true,
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
                                      Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp)) {
    return failure();
  }

  SmallVector<int64_t> bounds = linalgOp.getStaticLoopRanges();
  SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
  SmallVector<Value> operands(linalgOp->getOperands());

  LDBG("Matmul TileAndFuse Config");

  FailureOr<std::pair<LoweringConfigAttr, int64_t>> configAndWgSize =
      getMatmulLoweringConfigAndWorkgroupSize(bounds, maps, operands, target);
  if (failed(configAndWgSize)) {
    return failure();
  }
  std::array<int64_t, 3> workgroupSize = {configAndWgSize->second, 1, 1};
  LoweringConfigAttr loweringConfig = configAndWgSize->first;

  SmallVector<NamedAttribute, 1> pipelineAttrs;
  auto pipelineOptions = IREE::GPU::GPUPipelineOptionsAttr::get(
      linalgOp->getContext(), /*prefetchSharedMemory=*/true,
      /*no_reduce_shared_memory_bank_conflicts=*/false,
      /*use_igemm_convolution=*/false,
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
static bool isNonMatvecContraction(linalg::LinalgOp linalgOp) {
  SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
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

LogicalResult setTileAndFuseLoweringConfig(IREE::GPU::TargetAttr target,
                                           mlir::FunctionOpInterface entryPoint,
                                           Operation *op) {
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

  SmallVector<unsigned int> partitionableLoops;
  linalgOp.getParallelDims(partitionableLoops);

  // Bail out if op is not tilable.
  if (partitionableLoops.empty()) {
    return failure();
  }

  const int subgroupSize = target.getPreferredSubgroupSize();
  const unsigned loopDepth = linalgOp.getNumLoops();

  // Configurations we need to decide.
  int64_t flatWorkgroupSize = 1;
  SmallVector<int64_t> workgroupTileSizes(loopDepth, 0);
  SmallVector<int64_t> threadTileSizes(loopDepth, 0);

  // Find constraints on workgroup tile sizes due to pack or unpack ops in the
  // dispatch. If there are no pack or unpack ops present, then these multiples
  // will be 1, which means there is no constraint on workgroup tile sizes.
  SmallVector<int64_t> workgroupTileSizeMultiples =
      getWorkgroupSizeMultiples(cast<TilingInterface>(op));

  // Common case for all linalg ops.

  // The core idea is to distribute the partitioned loops to the workgroup
  // dimensions. The goal is to fill up the GPU as much as possible, which means
  // 1) distributing to as many threads as possible, and 2) avoid assigning too
  // many threads to handle out-of-bound elements (thus idle).

  auto elementHasPowerOfTwoBitwidth = [](Value operand) {
    Type elementType = getElementTypeOrSelf(operand.getType());
    return isa<IntegerType, FloatType>(elementType) &&
           llvm::isPowerOf2_64(IREE::Util::getTypeBitWidth(elementType));
  };

  // Whether we can try to use the vectorization pipeline.
  SmallVector<int64_t> loopBounds = linalgOp.getStaticLoopRanges();
  bool projPerm =
      llvm::all_of(linalgOp.getIndexingMapsArray(),
                   [](AffineMap map) { return map.isProjectedPermutation(); });
  bool powTwo =
      llvm::all_of(linalgOp->getOperands(), elementHasPowerOfTwoBitwidth);

  // Require all affine maps to be projected permutation so that we can
  // generate vector transfer ops.
  bool vectorizable = projPerm && powTwo;

  const unsigned minBitwidth = getMinElementBitwidth(linalgOp);
  // Make sure we use a tile size that results in some integral number of bytes.
  const unsigned scaleToByte =
      std::max(8 / minBitwidth, static_cast<unsigned>(1));

  // Distribute workload to the given `numThreads` by allowing a potental loss.
  auto distributeToThreads = [&](int64_t numThreads,
                                 std::optional<int64_t> lossFactor =
                                     std::nullopt) {
    LDBG("Loss factor: " << lossFactor << "\n");
    // Initialize the configuration.
    flatWorkgroupSize = 1;
    // Initialize thread tiling along all partitioned loops with size 1, and
    // workgroup tiling with the required tile size multiples. This may lead
    // to larger workgroup tiles than the number of threads in the workgroup,
    // but it is unavoidable.
    for (int64_t loopIndex : partitionableLoops) {
      workgroupTileSizes[loopIndex] = workgroupTileSizeMultiples[loopIndex];
      threadTileSizes[loopIndex] = 1;
    }

    // Scan from the innermost shape dimension and try to deduce the
    // configuration for the corresponding GPU workgroup dimension.
    int64_t wgDim = 0;
    for (auto shapeDim : llvm::reverse(partitionableLoops)) {
      int64_t loopBound = loopBounds[shapeDim];
      // Skip dynamic dimensions.
      if (ShapedType::isDynamic(loopBound))
        continue;

      // Try to find some power of two that can divide the current shape dim
      // size. This vector keeps the candidate tile sizes.
      SmallVector<int64_t, 8> candidates;

      // Ensure vectorization works with the `workgroupTileMultiple`.
      int64_t workgroupTileMultiple = workgroupTileSizeMultiples[shapeDim];
      vectorizable =
          vectorizable && 4 * numThreads % workgroupTileMultiple == 0;
      // For the inner most workgroup dim, try to see if we can have 4
      // elements per thread. This enables vectorization.
      if (vectorizable && wgDim == 0 && !lossFactor) {
        candidates.push_back(4 * numThreads);
      }
      // Try all power of two multiples of `workgroupTileMultiple` up to the
      // subgroup size.
      uint64_t maxCandidate =
          std::max<uint64_t>(1, llvm::PowerOf2Ceil(llvm::divideCeil(
                                    numThreads, workgroupTileMultiple)));
      for (unsigned i = maxCandidate; i >= 1; i >>= 1) {
        candidates.push_back(i * workgroupTileMultiple);
      }
      LLVM_DEBUG({
        llvm::dbgs() << "Base candidate tile sizes: [";
        llvm::interleaveComma(candidates, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });

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
        if (partitionableLoops.size() == 1 && candidate > subgroupSize &&
            llvm::divideCeil(loopBound, scaledTileSize) <= 2) {
          continue;
        }

        // Try to let each thread handle 4 elements if this is the workgroup x
        // dimension.
        // TODO: Try to take into account element type bit width to get
        // 4xdword reads instead of 4x{elements}.
        if (vectorizable && wgDim == 0 && !lossFactor && candidate % 4 == 0) {
          // Use size-1 vectors to increase parallelism if larger ones causes
          // idle threads in the subgroup.
          bool hasIdleThreads =
              partitionableLoops.size() == 1 && candidate <= subgroupSize;
          int vectorSize = hasIdleThreads ? 1 : 4;
          LLVM_DEBUG(llvm::dbgs() << "Use vector size: " << vectorSize << "\n");
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
            vectorizable = false;
          threadTileSizes[shapeDim] = scaleToByte;
          candidateWorkgroupSize = maybeCandidateWorkgroupSize;
          numThreads /= candidateWorkgroupSize;
        }
        workgroupTileSizes[shapeDim] = scaledTileSize;
        LLVM_DEBUG(llvm::dbgs()
                   << "Chosen workgroup tile size: " << scaledTileSize << "\n");
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

    // Threads for distribution. Use `minPreferrefNumThreads` at least, but no
    // more than 4 subgroups.
    int64_t minPreferrefNumThreads = std::reduce(
        workgroupTileSizeMultiples.begin(), workgroupTileSizeMultiples.end(), 1,
        std::multiplies<int64_t>());
    int64_t numThreads =
        std::min<int64_t>(4 * subgroupSize, minPreferrefNumThreads);
    // If minPreferrefNumThreads is small, use at least 32 or subgroupSize
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
  int64_t numLoops = linalgOp.getNumLoops();
  SmallVector<utils::IteratorType> iterTypes = linalgOp.getIteratorTypesArray();
  SmallVector<int64_t> loopTileSizes(numLoops, 0);
  for (auto [reverseIdx, iter] : llvm::enumerate(llvm::reverse(iterTypes))) {
    unsigned i = numLoops - reverseIdx - 1;
    if (linalg::isReductionIterator(iter) || i >= workgroupTileSizes.size() ||
        workgroupTileSizes[i] == 0) {
      int64_t tileSize = getReductionTilingFactor(loopBounds[i]);
      if (vectorSize * tileSize > maxVectorSize) {
        tileSize = 1;
      }
      vectorSize *= tileSize;
      loopTileSizes[i] = tileSize;
    }
  }

  for (auto dim : partitionableLoops) {
    if (workgroupTileSizes[dim] == loopBounds[dim]) {
      workgroupTileSizes[dim] = 0;
    }
  }

  // Attach the MMA schedule as an attribute to the entry point export function
  // for later access in the pipeline.
  MLIRContext *context = linalgOp.getContext();
  SmallVector<NamedAttribute, 1> attrs;
  Builder b(context);
  attrs.emplace_back(StringAttr::get(context, "workgroup"),
                     b.getI64ArrayAttr(workgroupTileSizes));

  attrs.emplace_back(StringAttr::get(context, "thread"),
                     b.getI64ArrayAttr(threadTileSizes));

  if (isNonMatvecContraction(linalgOp)) {
    GPU::setPromotedOperandList(context, attrs, {0, 1});
  }

  if (llvm::any_of(loopTileSizes, [](int64_t s) { return s != 0; })) {
    attrs.emplace_back(StringAttr::get(context, "reduction"),
                       b.getI64ArrayAttr(loopTileSizes));
  }

  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  LDBG("Selected tile and fuse lowering config: " << loweringConfig << "\n");

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
  SmallVector<int64_t> loopBounds = scatter.getStaticLoopRanges().value_or(
      SmallVector<int64_t>(loopDepth, ShapedType::kDynamic));

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
  SmallVector<NamedAttribute, 1> attrs;
  Builder b(context);
  attrs.emplace_back(StringAttr::get(context, "workgroup"),
                     b.getI64ArrayAttr(workgroupTileSizes));

  attrs.emplace_back(StringAttr::get(context, "thread"),
                     b.getI64ArrayAttr(threadTileSizes));

  auto configDict = DictionaryAttr::get(context, attrs);
  auto loweringConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);

  LDBG("Selected tile and fuse lowering config: " << loweringConfig << "\n");

  // TODO(qedawkins): Use a shared pipeline identifier here.
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, scatter, loweringConfig,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTileAndFuse,
      {flatWorkgroupSize, 1, 1}, flatWorkgroupSize, DictionaryAttr());
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
            << ", reorderWorkgroupsStrategy = " << reorderStr
            << ", enableUkernels = " << options.enableUkernels << "}";
}

} // namespace mlir::iree_compiler::IREE::GPU
