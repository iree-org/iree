// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "iree-codegen-tile-inference-utils"

namespace mlir::iree_compiler {

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
  static_assert(llvm::is_one_of<PackOrUnPackOpTy, linalg::PackOp,
                                linalg::UnPackOp>::value);
  LDBG() << "Inferring workgroup tile size multiples from " << op->getName()
         << ":\n"
         << op;
  // Initialize the list of multiples for the packed and unpack inputs.
  int64_t unPackedRank = (std::is_same<PackOrUnPackOpTy, linalg::PackOp>::value)
                             ? op.getSourceRank()
                             : op.getDestRank();
  SmallVector<int64_t> innerTiles = op.getStaticTiles();
  // If the tiles are dynamic, then no inference can be made.
  if (ShapedType::isDynamicShape(innerTiles)) {
    LDBG() << "Cannot infer multiple for dynamic inner tiles. Defaulting to "
              "all 1.";
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
    ArrayRef<int64_t> outerDimsPerm = op.getOuterDimsPerm();
    int64_t outerTileIdx = outerDimsPerm.empty()
                               ? pos
                               : invertPermutationVector(outerDimsPerm)[pos];
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
        LDBG() << "Cannot find a compatible tile size multiple. Defaulting to "
                  "all 1.";
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
      std::is_same_v<PackOrUnPackOpTy, linalg::PackOp> ? unPackedMultiples
                                                       : packedMultiples;
  SmallVector<int64_t> destMultiples =
      std::is_same_v<PackOrUnPackOpTy, linalg::PackOp> ? packedMultiples
                                                       : unPackedMultiples;
  LDBG() << "Inferred " << op->getName() << " multiples";
  LDBG() << "src: " << llvm::interleaved_array(srcMultiples);
  LDBG() << "result: " << llvm::interleaved_array(destMultiples);
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
  LDBG() << "Inferring workgroup tile size multiples for linalgOp:\n"
         << linalgOp;
  auto dbgsPrintMultiples = [](SmallVector<SmallVector<int64_t>> multiples) {
    LLVM_DEBUG({
      for (auto [i, m] : llvm::enumerate(multiples)) {
        LDBG() << "operand " << i << ": " << llvm::interleaved_array(m);
      }
    });
  };
  LDBG() << "\noperandMultiples:\n";
  dbgsPrintMultiples(operandMultiples);
  LDBG() << "\nresultMultiples:\n";
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

  LDBG() << "\niterationMultiples: "
         << llvm::interleaved_array(iterationMultiples);
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
        LDBG()
            << "Cannot infer multiple with dynamic size. defaulting to all 1";
        expandedMultiples = SmallVector<int64_t>(expandedShape.size(), 1);
        return expandedMultiples;
      }
      if (residualMultiple % expandedSize != 0) {
        LDBG() << "Expanded size does not divide producer multiple. Defaulting "
                  "to all 1";
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
  LDBG() << "Inferring workgroup tile size multiples for result:\n" << result;
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
        LDBG() << "Inferring workgroup tile size multiples for "
               << expandOp->getName() << " result.\n";
        SmallVector<int64_t> resultMultiples = expandMultiples(
            /*collapsedMultiples=*/srcMultiples,
            /*expandedShape=*/expandOp.getResultType().getShape(),
            /*reassociations=*/expandOp.getReassociationIndices());
        LDBG() << "\nInferred expand_shape result multiples: "
               << llvm::interleaved_array(resultMultiples);
        return resultMultiples;
      })
      .Case<linalg::PackOp>([&](linalg::PackOp packOp) {
        SmallVector<int64_t> srcMultiples = getOperandMultiples()[0];
        return inferWorkgroupTileMultiplesFromPackUnPack(
                   packOp, /*initialPackedMultiples=*/std::nullopt,
                   /*initialUnPackedMultiples=*/srcMultiples)
            .second;
      })
      .Case<linalg::UnPackOp>([&](linalg::UnPackOp unPackOp) {
        SmallVector<int64_t> srcMultiples = getOperandMultiples()[0];
        return inferWorkgroupTileMultiplesFromPackUnPack(
                   unPackOp, /*initialPackedMultiples=*/srcMultiples,
                   /*initialUnPackedMultiples=*/std::nullopt)
            .second;
      })
      .Case<linalg::LinalgOp>([&](linalg::LinalgOp linalgOp) {
        SmallVector<SmallVector<int64_t>> operandMultiples =
            getOperandMultiples();
        LDBG()
            << "Inferring workgroup tile size multiples for linalg op result #"
            << result.getResultNumber() << ":\n"
            << result;
        SmallVector<SmallVector<int64_t>> resultMultiples = llvm::map_to_vector(
            linalgOp->getResults(), getDefaultValueMultiples);
        SmallVector<int64_t> iterationMultiples(
            linalgOp.getIteratorTypesArray().size(), 1);
        inferWorkgroupTileMultiplesFromLinalgOp(
            linalgOp, iterationMultiples, operandMultiples, resultMultiples);
        return resultMultiples[result.getResultNumber()];
      })
      .Default([&](Operation *) {
        LDBG() << "Unsupported operation. Defualting to all 1: " << result;
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
  LDBG() << "Inferring workgroup tile size multiples for operand "
         << use->getOperandNumber() << " of user:\n"
         << *use->getOwner();
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
        LDBG() << "Inferring workgroup tile size multiples for "
               << collapseOp->getName() << "source.\n";
        SmallVector<int64_t> srcMultiples = expandMultiples(
            /*collapsedMultiples=*/destMultiples,
            /*expandedShape=*/collapseOp.getSrcType().getShape(),
            /*reassociations=*/collapseOp.getReassociationIndices());
        LDBG() << "\nInferred collapse_shape source multiples: "
               << llvm::interleaved_array(srcMultiples);
        return srcMultiples;
      })
      .Case<linalg::PackOp>([&](linalg::PackOp packOp) {
        SmallVector<int64_t> destMultiples = getResultMultiples()[0];
        return inferWorkgroupTileMultiplesFromPackUnPack(
                   packOp, /*initialPackedMultiples=*/destMultiples,
                   /*initialUnPackedMultiples=*/std::nullopt)
            .first;
      })
      .Case<linalg::UnPackOp>([&](linalg::UnPackOp unpackOp) {
        SmallVector<int64_t> destMultiples = getResultMultiples()[0];
        return inferWorkgroupTileMultiplesFromPackUnPack(
                   unpackOp, /*initialPackedMultiples=*/std::nullopt,
                   /*initialUnPackedMultiples=*/destMultiples)
            .first;
      })
      .Default([&](Operation *) {
        LDBG() << "Unsupported operation. Defaulting to all 1: " << use->get();
        return getDefaultValueMultiples(use->get());
      });
}

static SmallVector<int64_t> lcmMultiples(ArrayRef<int64_t> a,
                                         ArrayRef<int64_t> b) {
  SmallVector<int64_t> lcm;
  for (auto [aMultiple, bMultiple] : llvm::zip_equal(a, b)) {
    lcm.push_back(std::lcm(aMultiple, bMultiple));
  }
  return lcm;
}

SmallVector<int64_t> getWorkgroupSizeMultiples(TilingInterface tilingOp) {
  LDBG() << "Computing workgroup tile size multiples for: "
         << *tilingOp.getOperation();

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
    SmallVector<int64_t> multiples = getDefaultValueMultiples(result);
    for (OpOperand &use : result.getUses()) {
      multiples = lcmMultiples(multiples, inferUseWorkgroupTileMultiples(&use));
    }
    resultMultiples.push_back(multiples);
  }

  if (auto packOp = dyn_cast<linalg::PackOp>(tilingOp.getOperation())) {
    SmallVector<int64_t> initialUnPackedMultiples = operandMultiples.front();
    SmallVector<int64_t> initialPackedMultiples = resultMultiples.front();
    return inferWorkgroupTileMultiplesFromPackUnPack(
               packOp, initialPackedMultiples, initialUnPackedMultiples)
        .second;
  }
  auto linalgOp = dyn_cast<linalg::LinalgOp>(tilingOp.getOperation());
  if (!linalgOp) {
    LDBG() << "Only LinalgOp and PackOp are implemented. Defaulting to all 1 "
              "multiples.";
    return SmallVector<int64_t>(tilingOp.getLoopIteratorTypes().size(), 1);
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

} // namespace mlir::iree_compiler
