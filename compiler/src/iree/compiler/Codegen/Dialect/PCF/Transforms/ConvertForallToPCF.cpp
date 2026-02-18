// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Utils/RewriteUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/IRMapping.h"

#define DEBUG_TYPE "iree-pcf-convert-forall-to-pcf"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_TESTCONVERTFORALLTOLOOPSPASS
#define GEN_PASS_DEF_TESTCONVERTFORALLTOGENERICNESTPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Shared Utilities
//===----------------------------------------------------------------------===//

/// Returns true if the forall op has LocalMappingAttr mapping attributes,
/// or the mapping is empty/not present.
static bool hasEmptyOrLocalMapping(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mapping = forallOp.getMapping();
  if (!mapping || mapping->empty()) {
    return true;
  }
  return llvm::all_of(mapping.value(),
                      llvm::IsaPred<IREE::Codegen::LocalMappingAttr>);
}

/// Returns the permutation from mapping attributes based on their relative
/// processor IDs. Lower IDs indicate faster varying dimensions. The returned
/// permutation maps from the fastest-to-slowest order back to the original
/// forall dimension order.
///
/// Example: mapping = [local_mapping<1>, local_mapping<0>]
/// - local_mapping<0> has lower ID than local_mapping<1>
/// - So dimension 1 (id 0) is faster, dimension 0 (id 1) is slower
/// - Returns [1, 0]: position 0 in linearized order maps to dim 1, etc.
static SmallVector<int64_t> getMappingPermutation(ArrayAttr mapping) {
  auto mappingRange = mapping.getAsRange<DeviceMappingAttrInterface>();
  int64_t mappingBase =
      cast<DeviceMappingAttrInterface>(
          *std::min_element(mappingRange.begin(), mappingRange.end(),
                            [](auto a, auto b) {
                              return a.getMappingId() < b.getMappingId();
                            }))
          .getMappingId();
  return llvm::map_to_vector(
      mappingRange, [&](auto a) { return a.getMappingId() - mappingBase; });
}

/// Returns the permutation for processor ID ordering from the forall's mapping.
/// For unspecified or empty mappings, returns a reversed sequence (assumes
/// natural fastest-to-slowest is reverse of dimension order).
static FailureOr<SmallVector<int64_t>>
getProcessorIdPermutation(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mappingAttr = forallOp.getMapping();
  // Unspecified mappings indicate sequential foralls which we can choose the
  // iteration order for.
  if (!mappingAttr) {
    return llvm::to_vector(llvm::reverse(llvm::seq(forallOp.getRank())));
  }
  // Empty mappings are unsupported at the moment. It's unclear when a forall
  // with an empty mapping would be useful or important.
  if (mappingAttr.value().empty()) {
    return SmallVector<int64_t>{};
  }

  SmallVector<int64_t> perm = getMappingPermutation(mappingAttr.value());
  if (!isPermutationVector(perm)) {
    return failure();
  }
  return perm;
}

/// Validates that the forall op can be converted.
static LogicalResult matchForallConversion(scf::ForallOp forallOp) {
  scf::InParallelOp terminator = forallOp.getTerminator();
  for (Operation &op : terminator.getBody()->getOperations()) {
    // Bail on terminator ops other than parallel insert slice since we don't
    // know how to convert it.
    auto insertSliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(&op);
    if (!insertSliceOp) {
      return failure();
    }

    // Bail on non-shared outs destinations.
    auto bbArgDest = dyn_cast<BlockArgument>(insertSliceOp.getDest());
    if (!bbArgDest || bbArgDest.getOwner()->getParentOp() != forallOp) {
      return failure();
    }
  }

  for (BlockArgument bbArg : forallOp.getRegionIterArgs()) {
    for (OpOperand &use : bbArg.getUses()) {
      // Skip users outside of the terminator. These are replaced with the init.
      if (use.getOwner()->getParentOp() != terminator) {
        continue;
      }

      // Bail if the use is not on the dest of the insert slice.
      auto insertSliceUser =
          cast<tensor::ParallelInsertSliceOp>(use.getOwner());
      if (use != insertSliceUser.getDestMutable()) {
        return failure();
      }
    }
  }
  // Validate that the mapping permutation is valid.
  if (failed(getProcessorIdPermutation(forallOp))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// scf.forall -> pcf.loop Implementation
//===----------------------------------------------------------------------===//

static PCF::LoopOp convertForallToPCFLoopImpl(RewriterBase &rewriter,
                                              scf::ForallOp forallOp,
                                              PCF::ScopeAttrInterface scope,
                                              int64_t numIds) {
  assert(succeeded(matchForallConversion(forallOp)) &&
         "converting unsupported forall op");

  // Maps from fastest -> slowest to current order.
  SmallVector<int64_t> perm = *getProcessorIdPermutation(forallOp);

  // Maps from current order to fastest -> slowest.
  SmallVector<int64_t> invPerm = invertPermutationVector(perm);

  // Get the permuted ubs/lbs/steps and save them for later since we need them
  // to reconstruct the correct ids.
  SmallVector<OpFoldResult> mixedUbs = forallOp.getMixedUpperBound();
  applyPermutationToVector(mixedUbs, invPerm);
  SmallVector<OpFoldResult> mixedLbs = forallOp.getMixedLowerBound();
  applyPermutationToVector(mixedLbs, invPerm);
  SmallVector<OpFoldResult> mixedStep = forallOp.getMixedStep();
  applyPermutationToVector(mixedStep, invPerm);
  // Permute the ivs of the body to match the original mapping order by
  // permuting the uses before we move it over to the new op.
  permuteValues(rewriter, forallOp.getLoc(), forallOp.getInductionVars(), perm);

  scf::InParallelOp terminator = forallOp.getTerminator();
  MutableArrayRef<BlockArgument> bodySharedOuts = forallOp.getRegionIterArgs();

  // Replace non-insert slice users outside of the `scf.forall.in_parallel` with
  // the init values.
  ValueRange inits = forallOp.getDpsInits();
  for (auto [init, bbArg] : llvm::zip_equal(inits, bodySharedOuts)) {
    rewriter.replaceUsesWithIf(bbArg, init, [&](OpOperand &use) {
      return use.getOwner()->getParentOp() != terminator;
    });
  }

  Location loc = forallOp.getLoc();

  AffineExpr s0, s1, s2;
  bindSymbols(rewriter.getContext(), s0, s1, s2);
  AffineExpr numIters = (s0 - s1).ceilDiv(s2);
  SmallVector<Value> iterationCounts;
  for (auto [ub, lb, step] : llvm::zip_equal(mixedUbs, mixedLbs, mixedStep)) {
    OpFoldResult iterCount = affine::makeComposedFoldedAffineApply(
        rewriter, loc, numIters, ArrayRef<OpFoldResult>{ub, lb, step});
    iterationCounts.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, iterCount));
  }

  int64_t numDelinIds = numIds > 0 && iterationCounts.size() > numIds
                            ? iterationCounts.size() - numIds + 1
                            : 0;
  // Reverse the delinearization basis because affine.delinearize_index is from
  // slowest to fastest varying.
  SmallVector<Value> delinearizationBasis(
      llvm::reverse(ArrayRef<Value>(iterationCounts).take_back(numDelinIds)));
  if (!delinearizationBasis.empty()) {
    AffineExpr mul = s0 * s1;
    Value total = delinearizationBasis.front();
    total = std::accumulate(
        delinearizationBasis.begin() + 1, delinearizationBasis.end(), total,
        [&](Value l, Value r) {
          OpFoldResult acc =
              affine::makeComposedFoldedAffineApply(rewriter, loc, mul, {l, r});
          return getValueOrCreateConstantIndexOp(rewriter, loc, acc);
        });
    // Replace the first |numDelinIds| entries with their product.
    iterationCounts.erase(iterationCounts.end() - numDelinIds,
                          iterationCounts.end());
    iterationCounts.push_back(total);
  }

  auto loopOp = PCF::LoopOp::create(rewriter, loc, scope, iterationCounts,
                                    forallOp.getDpsInits());
  SmallVector<Value> argReplacements;
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(loopOp.getBody());
    // If/else here to avoid popping off the front of a vector.
    if (!delinearizationBasis.empty()) {
      auto delin = affine::AffineDelinearizeIndexOp::create(
          rewriter, loc, loopOp.getIdArgs().back(), delinearizationBasis,
          /*hasOuterBound=*/true);
      argReplacements.append(loopOp.getIdArgs().begin(),
                             loopOp.getIdArgs().end() - 1);
      // Add replacements in reverse to get fastest -> slowest.
      auto delinResultsReverse = llvm::reverse(delin.getResults());
      argReplacements.append(delinResultsReverse.begin(),
                             delinResultsReverse.end());
    } else {
      argReplacements.append(loopOp.getIdArgs().begin(),
                             loopOp.getIdArgs().end());
    }

    // id * step + lb.
    AffineExpr applyLbAndStep = s0 * s1 + s2;
    for (auto [id, step, lb] :
         llvm::zip_equal(argReplacements, mixedStep, mixedLbs)) {
      OpFoldResult newId = affine::makeComposedFoldedAffineApply(
          rewriter, loc, applyLbAndStep, {id, step, lb});
      id = getValueOrCreateConstantIndexOp(rewriter, loc, newId);
    }
  }

  // Add parent only sync scope to the body arg types.
  Attribute syncScope = PCF::SyncOnReturnAttr::get(rewriter.getContext());
  for (BlockArgument regionRefArg : loopOp.getRegionRefArgs()) {
    auto srefType = cast<PCF::ShapedRefType>(regionRefArg.getType());
    auto newSrefType = PCF::ShapedRefType::get(
        rewriter.getContext(), srefType.getShape(), srefType.getElementType(),
        srefType.getScope(), syncScope);
    regionRefArg.setType(newSrefType);
  }

  rewriter.setInsertionPoint(terminator);
  llvm::SmallDenseMap<Value, Value> argToReplacementMap;
  for (auto [bbArg, refArg] :
       llvm::zip_equal(bodySharedOuts, loopOp.getRegionRefArgs())) {
    argToReplacementMap[bbArg] = refArg;
  }

  // Iterate the insert_slice ops in the order to retain the order of writes.
  SmallVector<tensor::ParallelInsertSliceOp> insertOps(
      terminator.getBody()->getOps<tensor::ParallelInsertSliceOp>());
  for (tensor::ParallelInsertSliceOp insertSliceOp : insertOps) {
    PCF::WriteSliceOp::create(
        rewriter, insertSliceOp.getLoc(), insertSliceOp.getSource(),
        argToReplacementMap[insertSliceOp.getDest()],
        insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
        insertSliceOp.getMixedStrides());
    rewriter.eraseOp(insertSliceOp);
  }

  // Replace the terminator with the new terminator kind.
  rewriter.replaceOpWithNewOp<PCF::ReturnOp>(terminator);

  // Use the inits as the replacements for the shared outs bbargs to appease
  // `inlineBlockBefore`. By this point all of their users have been replaced
  // or erased so it doesn't matter what goes here.
  argReplacements.append(inits.begin(), inits.end());
  rewriter.inlineBlockBefore(forallOp.getBody(), loopOp.getBody(),
                             loopOp.getBody()->end(), argReplacements);

  rewriter.replaceOp(forallOp, loopOp);
  return loopOp;
}

//===----------------------------------------------------------------------===//
// scf.forall -> pcf.generic nest Implementation
//===----------------------------------------------------------------------===//

static PCF::GenericOp
convertForallToGenericNestImpl(RewriterBase &rewriter, scf::ForallOp forallOp,
                               ArrayRef<PCF::ScopeAttrInterface> scopes) {
  assert(succeeded(matchForallConversion(forallOp)) &&
         "converting unsupported forall op");
  assert(!scopes.empty() && "at least one scope required");

  Location loc = forallOp.getLoc();
  MLIRContext *ctx = rewriter.getContext();

  // Get forall outputs to base the tied results on.
  ValueRange outputs = forallOp.getOutputs();
  TypeRange resultTypes = forallOp.getResultTypes();
  scf::InParallelOp terminator = forallOp.getTerminator();
  MutableArrayRef<BlockArgument> bodySharedOuts = forallOp.getRegionIterArgs();

  // Replace non-insert slice users outside of the `scf.forall.in_parallel` with
  // the init values.
  ValueRange inits = forallOp.getDpsInits();
  for (auto [init, bbArg] : llvm::zip_equal(inits, bodySharedOuts)) {
    rewriter.replaceUsesWithIf(bbArg, init, [&](OpOperand &use) {
      return use.getOwner()->getParentOp() != terminator;
    });
  }

  // Create nested pcf.generic ops for each scope.
  SmallVector<bool> isTied(outputs.size(), true);
  Attribute syncScope = PCF::SyncOnReturnAttr::get(ctx);

  SmallVector<Value> allIds, allCounts;

  // Create outermost generic with inits and results. Use the scope's native
  // number of processor IDs to determine iteration dimensions.
  int64_t outermostNumIds = scopes[0].getNativeNumProcessorIds();
  PCF::GenericOp outermostGeneric = PCF::GenericOp::create(
      rewriter, loc, resultTypes, scopes[0], outputs, /*dynamicSizes=*/{},
      isTied, outermostNumIds, /*syncOnReturn=*/false);

  // Add sync scope to sref types for outermost generic.
  for (BlockArgument regionRefArg : outermostGeneric.getRegionRefArgs()) {
    auto srefType = cast<PCF::ShapedRefType>(regionRefArg.getType());
    auto newSrefType = PCF::ShapedRefType::get(ctx, srefType.getShape(),
                                               srefType.getElementType(),
                                               srefType.getScope(), syncScope);
    regionRefArg.setType(newSrefType);
  }

  // Collect id/count args from outermost generic.
  allIds.append(outermostGeneric.getIdArgs().begin(),
                outermostGeneric.getIdArgs().end());
  allCounts.append(outermostGeneric.getCountArgs().begin(),
                   outermostGeneric.getCountArgs().end());

  // Set insertion point to end of execute block for next generic or body.
  Block &outermostBlock = outermostGeneric.getRegion().front();
  rewriter.setInsertionPointToEnd(&outermostBlock);

  // Create inner generics (no inits, no results).
  for (size_t i = 1; i < scopes.size(); ++i) {
    PCF::ScopeAttrInterface scope = scopes[i];
    int64_t numIds = scope.getNativeNumProcessorIds();
    PCF::GenericOp generic = PCF::GenericOp::create(
        rewriter, loc, TypeRange{}, scope, ValueRange{}, /*dynamicSizes=*/{},
        SmallVector<bool>{}, numIds, /*syncOnReturn=*/false);

    // Same here, collect id/count args from this generic and update the
    // insertion point for the next one.
    allIds.append(generic.getIdArgs().begin(), generic.getIdArgs().end());
    allCounts.append(generic.getCountArgs().begin(),
                     generic.getCountArgs().end());

    Block &executeBlock = generic.getRegion().front();
    rewriter.setInsertionPointToEnd(&executeBlock);
  }

  // In innermost block, linearize all ids if there are multiple.
  // linearId = id[0] * count[1] * ... * count[n-1] + id[1] * count[2] * ... +
  // ... + id[n-1] totalWorkers = count[0] * count[1] * ... * count[n-1]
  Value linearId;
  if (allIds.size() == 1) {
    // Shortcut single id to avoid creating IR.
    linearId = allIds[0];
  } else {
    // Use affine.linearize_index with the |counts| as the linearization basis.
    SmallVector<OpFoldResult> countOfrs = llvm::map_to_vector(
        allCounts, [](Value v) -> OpFoldResult { return v; });
    linearId =
        affine::AffineLinearizeIndexOp::create(rewriter, loc, allIds, countOfrs,
                                               /*disjoint=*/false);
  }

  // Compute total workers as product of all counts.
  Value totalWorkers = allCounts[0];
  for (size_t i = 1; i < allCounts.size(); ++i) {
    totalWorkers =
        arith::MulIOp::create(rewriter, loc, totalWorkers, allCounts[i]);
  }

  // Compute total iteration count from forall bounds.
  SmallVector<OpFoldResult> mixedUbs = forallOp.getMixedUpperBound();
  Value totalIters = arith::ConstantIndexOp::create(rewriter, loc, 1);
  for (OpFoldResult ub : mixedUbs) {
    Value dim = getValueOrCreateConstantIndexOp(rewriter, loc, ub);
    totalIters = arith::MulIOp::create(rewriter, loc, totalIters, dim);
  }

  // Compute chunk bounds: chunkSize = ceildiv(total, totalWorkers). We'll
  // create a spillover scf.forall that iterates from the start of the current
  // chunk (call it start) to min(start + chunkSize, total). This greedily
  // allocates an equal number of iterations to all workers except the last.
  Value chunkSize =
      arith::CeilDivUIOp::create(rewriter, loc, totalIters, totalWorkers);
  Value linearLb = arith::MulIOp::create(rewriter, loc, linearId, chunkSize);
  Value linearUbRaw = arith::AddIOp::create(rewriter, loc, linearLb, chunkSize);
  Value linearUb =
      arith::MinUIOp::create(rewriter, loc, linearUbRaw, totalIters);

  // Create the inner scf.forall with linearized bounds (no body builder).
  SmallVector<OpFoldResult> lbs = {linearLb};
  SmallVector<OpFoldResult> ubs = {linearUb};
  SmallVector<OpFoldResult> steps = {rewriter.getIndexAttr(1)};

  auto innerForall = scf::ForallOp::create(rewriter, loc, lbs, ubs, steps,
                                           /*outputs=*/ValueRange{},
                                           /*mapping=*/std::nullopt);

  // The scf.forall without body builder creates an empty block with a
  // terminator. Remove the terminator so we can populate the body.
  rewriter.eraseOp(innerForall.getBody()->getTerminator());

  // Get the permutation from mapping. The permutation value at position i
  // indicates the "speed rank" of dimension i (lower = faster varying).
  // For example, [#iree_codegen.local_mapping<1>,
  // #iree_codegen.local_mapping<0>] gives perm = [1, 0]:
  //   - dim 0 has rank 1 (slower)
  //   - dim 1 has rank 0 (faster)
  SmallVector<int64_t> perm = *getProcessorIdPermutation(forallOp);

  // Compute bounds ordered from slowest to fastest for delinearization.
  // affine.delinearize_index expects bounds from slowest to fastest varying.
  // Sort dimensions by decreasing speed rank (slowest first).
  SmallVector<std::pair<int64_t, int64_t>> rankAndDim;
  for (size_t i = 0; i < perm.size(); ++i) {
    rankAndDim.emplace_back(perm[i], i);
  }
  // Sort by rank descending (highest rank = slowest first).
  llvm::sort(rankAndDim, [](auto &a, auto &b) { return a.first > b.first; });

  // Build permuted upper bounds (slowest to fastest).
  SmallVector<OpFoldResult> permutedUbs;
  SmallVector<int64_t> delinToOrigDim; // Maps delinearized result index to
                                       // original forall dim.
  for (auto [rank, dim] : rankAndDim) {
    permutedUbs.push_back(mixedUbs[dim]);
    delinToOrigDim.push_back(dim);
  }

  // Map old induction variables to new linearized index.
  // For 1D: directly use the induction variable.
  // For multi-D: need delinearization with permutation handling.
  IRMapping mapping;
  if (forallOp.getRank() == 1) {
    // Shortcut rank-1 foralls to avoid creating the delinearize_index.
    mapping.map(forallOp.getInductionVars()[0],
                innerForall.getInductionVars()[0]);
  } else {
    // For multi-dimensional forall, we need to delinearize.
    // Delinearize the iteration index back to multi-D indices.
    rewriter.setInsertionPointToStart(innerForall.getBody());
    Value linearIdx = innerForall.getInductionVars()[0];
    auto delinearized = affine::AffineDelinearizeIndexOp::create(
        rewriter, loc, linearIdx, permutedUbs, /*hasOuterBound=*/true);
    // The delinearized results are in permuted order (slowest to fastest).
    // Map them back to the corresponding forall induction variables.
    // delinToOrigDim[i] tells us which original forall dim corresponds to
    // delinearized result i.
    for (size_t i = 0; i < delinToOrigDim.size(); ++i) {
      int64_t origDim = delinToOrigDim[i];
      mapping.map(forallOp.getInductionVars()[origDim],
                  delinearized.getResult(i));
    }
  }

  // Clone body operations (except terminator) into inner forall.
  rewriter.setInsertionPointToEnd(innerForall.getBody());

  for (Operation &op : forallOp.getBody()->without_terminator()) {
    rewriter.clone(op, mapping);
  }

  // Convert parallel_insert_slice to pcf.write_slice.
  llvm::SmallDenseMap<Value, Value> bbArgToSref;
  for (auto [bbArg, refArg] :
       llvm::zip_equal(bodySharedOuts, outermostGeneric.getRegionRefArgs())) {
    bbArgToSref[bbArg] = refArg;
  }

  for (Operation &op : terminator.getBody()->getOperations()) {
    if (auto insertOp = dyn_cast<tensor::ParallelInsertSliceOp>(&op)) {
      Value src = mapping.lookupOrDefault(insertOp.getSource());
      Value destRef = bbArgToSref[insertOp.getDest()];

      // Map offsets, sizes, and strides through the mapping.
      SmallVector<OpFoldResult> offsets, sizes, strides;
      for (OpFoldResult offset : insertOp.getMixedOffsets()) {
        if (auto val = dyn_cast<Value>(offset)) {
          offsets.push_back(mapping.lookupOrDefault(val));
        } else {
          offsets.push_back(offset);
        }
      }
      for (OpFoldResult size : insertOp.getMixedSizes()) {
        if (auto val = dyn_cast<Value>(size)) {
          sizes.push_back(mapping.lookupOrDefault(val));
        } else {
          sizes.push_back(size);
        }
      }
      for (OpFoldResult stride : insertOp.getMixedStrides()) {
        if (auto val = dyn_cast<Value>(stride)) {
          strides.push_back(mapping.lookupOrDefault(val));
        } else {
          strides.push_back(stride);
        }
      }

      PCF::WriteSliceOp::create(rewriter, loc, src, destRef, offsets, sizes,
                                strides);
    }
  }

  // Add an empty in_parallel terminator to the inner forall.
  scf::InParallelOp::create(rewriter, loc);

  // Add pcf.return terminators to all generic blocks, from innermost to
  // outermost. We need to walk from innermost out, adding returns.
  // The innermost is where we currently are.
  Operation *current = innerForall.getOperation();
  while (current) {
    Operation *parent = current->getParentOp();
    if (auto generic = dyn_cast_or_null<PCF::GenericOp>(parent)) {
      Block &block = generic.getRegion().front();
      rewriter.setInsertionPointToEnd(&block);
      PCF::ReturnOp::create(rewriter, loc);
      current = generic.getOperation();
    } else {
      // Break on the first non-generic (should be the parent of all the IR we
      // just created).
      break;
    }
  }

  return outermostGeneric;
}

//===----------------------------------------------------------------------===//
// scf.forall -> pcf.loop Pass
//===----------------------------------------------------------------------===//

struct TestConvertForallToLoopsPass final
    : impl::TestConvertForallToLoopsPassBase<TestConvertForallToLoopsPass> {
  void runOnOperation() override {
    SmallVector<scf::ForallOp> opsToConvert;
    getOperation()->walk([&](scf::ForallOp forallOp) {
      // Empty mapping, no mapping, and local mapping all map to
      // `pcf.sequential`. If it is a local mapping, then the lowering pattern
      // will automatically handle any mapping permutation based on the mapping
      // attribute's relative id.
      if (hasEmptyOrLocalMapping(forallOp)) {
        opsToConvert.push_back(forallOp);
      }
    });

    IRRewriter rewriter(getOperation());
    PCF::ScopeAttrInterface sequentialScope =
        PCF::SequentialAttr::get(&getContext());
    for (scf::ForallOp forallOp : opsToConvert) {
      rewriter.setInsertionPoint(forallOp);
      if (failed(convertForallToPCFLoop(rewriter, forallOp, sequentialScope))) {
        forallOp->emitOpError("failed to convert forall");
        return signalPassFailure();
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// scf.forall -> pcf.generic nest Pass
//===----------------------------------------------------------------------===//

struct TestConvertForallToGenericNestPass final
    : impl::TestConvertForallToGenericNestPassBase<
          TestConvertForallToGenericNestPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // Build scope list based on numSequentialScopes.
    SmallVector<PCF::ScopeAttrInterface> scopeAttrs;
    for (int64_t i = 0; i < numSequentialScopes; ++i) {
      scopeAttrs.push_back(PCF::SequentialAttr::get(ctx));
    }

    if (scopeAttrs.empty()) {
      emitError(getOperation()->getLoc()) << "no scopes specified";
      return signalPassFailure();
    }

    IRRewriter rewriter(ctx);
    SmallVector<scf::ForallOp> forallOps;
    getOperation()->walk([&](scf::ForallOp forallOp) {
      // Only convert foralls with empty mapping or local_mapping attributes.
      if (hasEmptyOrLocalMapping(forallOp)) {
        forallOps.push_back(forallOp);
      }
    });

    for (scf::ForallOp forallOp : forallOps) {
      rewriter.setInsertionPoint(forallOp);
      FailureOr<PCF::GenericOp> result =
          convertForallToGenericNest(rewriter, forallOp, scopeAttrs);
      if (failed(result)) {
        forallOp.emitError("failed to convert forall to generic nest");
        return signalPassFailure();
      }
      // Replace forall results with generic results.
      rewriter.replaceOp(forallOp, result->getResults());
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API: scf.forall -> pcf.loop
//===----------------------------------------------------------------------===//

FailureOr<PCF::LoopOp> convertForallToPCFLoop(RewriterBase &rewriter,
                                              scf::ForallOp forallOp,
                                              PCF::ScopeAttrInterface scope,
                                              int64_t numIds) {
  if (failed(matchForallConversion(forallOp))) {
    return failure();
  }
  return convertForallToPCFLoopImpl(rewriter, forallOp, scope, numIds);
}

//===----------------------------------------------------------------------===//
// Public API: scf.forall -> pcf.generic nest
//===----------------------------------------------------------------------===//

FailureOr<PCF::GenericOp>
convertForallToGenericNest(RewriterBase &rewriter, scf::ForallOp forallOp,
                           ArrayRef<PCF::ScopeAttrInterface> scopes) {
  if (scopes.empty()) {
    return forallOp.emitError("at least one scope required");
  }

  if (failed(matchForallConversion(forallOp))) {
    return failure();
  }

  return convertForallToGenericNestImpl(rewriter, forallOp, scopes);
}

} // namespace mlir::iree_compiler::IREE::PCF
