// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PreprocessingExtensions.h"

#include "iree-dialects/Dialect/LinalgTransform/SimplePatternRewriter.h"
#include "iree/compiler/Utils/EquivalenceUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

namespace mlir::iree_compiler {

IREE::transform_dialect::PreprocessingExtensions::PreprocessingExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.cpp.inc"
      >();
}

void registerTransformDialectPreprocessingExtension(DialectRegistry &registry) {
  registry.addExtensions<IREE::transform_dialect::PreprocessingExtensions>();
}

//===----------------------------------------------------------------------===//
// GetNearestSymbolTableOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
IREE::transform_dialect::GetNearestSymbolTableOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  auto tableOp = SymbolTable::getNearestSymbolTable(target);
  if (!tableOp) {
    return emitDefaultDefiniteFailure(target);
  }
  results.push_back(tableOp);
  return DiagnosedSilenceableFailure::success();
}

void IREE::transform_dialect::GetNearestSymbolTableOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::producesHandle(getResult(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ImportSymbolOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure IREE::transform_dialect::ImportSymbolOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  auto symbolOp = SymbolTable::lookupNearestSymbolFrom(*this, getSymbol());
  if (!symbolOp) {
    return emitDefiniteFailure() << "could not find corresponding symbol op";
  }
  // Require isolated from above as the clone does not make sense with escaping
  // values.
  if (!symbolOp->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return emitDefiniteFailure()
           << "target symbol op is not isolated from above";
  }
  StringRef symbol = getSymbol().getLeafReference();
  SmallVector<Operation *> results;
  for (Operation *payloadOp : state.getPayloadOps(getSymbolTable())) {
    if (!payloadOp->hasTrait<OpTrait::SymbolTable>()) {
      return emitDefiniteFailure()
             << "target symbol table " << payloadOp << " is not a symbol table";
    }
    SymbolTable symbolTable(payloadOp);

    if (Operation *preExistingSymbolOp = symbolTable.lookup(symbol)) {
      if (getForceImport()) {
        // If we want to overwrite pre-existing symbols, just erase it here.
        symbolTable.erase(preExistingSymbolOp);
      } else if (getIfUndefined()) {
        // Skip if we want to use the symbol that is already there.
        results.push_back(preExistingSymbolOp);
        continue;
      } else {
        return emitDefiniteFailure()
               << "target symbol " << symbol << " is already defined";
      }
    }

    // Symbol table ops must have exactly one region with exactly one block.
    // Simply clone the target symbol op into the single block.
    rewriter.setInsertionPointToStart(&payloadOp->getRegion(0).front());
    results.push_back(rewriter.clone(*symbolOp));
  }
  transformResults.set(cast<OpResult>(getClonedSymbol()), results);
  return DiagnosedSilenceableFailure::success();
}

void IREE::transform_dialect::ImportSymbolOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getSymbolTable(), effects);
  transform::producesHandle(getClonedSymbol(), effects);
  transform::modifiesPayload(effects);
}

LogicalResult IREE::transform_dialect::ImportSymbolOp::verify() {
  if (getForceImport() && getIfUndefined()) {
    return emitOpError()
           << "force_import and if_undefined are mutually exclusive";
  }
  if (!SymbolTable::lookupNearestSymbolFrom(*this, getSymbol())) {
    return emitOpError() << "invalid import of undefined symbol";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MatchCastCompatibleDagFromRootOp
//===----------------------------------------------------------------------===//

static bool isCastableToTensorType(Type from, RankedTensorType to) {
  auto tensorType = dyn_cast<RankedTensorType>(from);
  if (!tensorType) {
    return false;
  }
  if (tensorType.getRank() != to.getRank()) {
    return false;
  }
  if (tensorType.getElementType() != to.getElementType()) {
    return false;
  }
  auto isAlignment = to.getEncoding()
                         ? dyn_cast<DenseBoolArrayAttr>(to.getEncoding())
                         : nullptr;
  // Bail if the alignment attribute is present and not the same size as the
  // tensor rank.
  if (isAlignment && isAlignment.getSize() != to.getRank()) {
    return false;
  }
  SmallVector<bool> alignmentStatuses(tensorType.getRank(), false);
  for (auto [fromSize, toSize, align] : llvm::zip_equal(
           tensorType.getShape(), to.getShape(),
           isAlignment ? isAlignment.asArrayRef() : alignmentStatuses)) {
    bool fromIsDynamic = ShapedType::isDynamic(fromSize);
    bool toIsDynamic = ShapedType::isDynamic(toSize);
    // Alignment check is invalid for dynamic dimensions. If the target
    // dimension is dynamic we can always cast to it.
    if (toIsDynamic) {
      continue;
    }
    // Casting a dynamic dimension to a static one is never valid.
    if (fromIsDynamic) {
      return false;
    }
    // Check for either the static alignment or size equality.
    if (align) {
      if (fromSize % toSize != 0) {
        return false;
      }
    } else if (toSize != fromSize) {
      return false;
    }
  }
  return true;
}

// Compares the regions between two operations in lockstep for equality.
static DiagnosedSilenceableFailure
compareOperationRegions(transform::TransformOpInterface transformOp,
                        Operation *target, Operation *payload) {
  if (target->getNumRegions() != payload->getNumRegions()) {
    return transformOp.emitSilenceableError() << "region count mismatch";
  }
  for (auto [r0, r1] :
       llvm::zip_equal(target->getRegions(), payload->getRegions())) {
    if (!isStructurallyEquivalentTo(r0, r1)) {
      return transformOp.emitSilenceableError()
             << "target op does not match specified body";
    }
  }
  return DiagnosedSilenceableFailure::success();
}

// Helper to check whether two operations are equivalent up to cast
// compatibility of their arguments (i.e. the arguments of the payload
// can be casted to the arguments of the target).
static DiagnosedSilenceableFailure
compareCastCompatibleOperations(transform::TransformOpInterface transformOp,
                                Operation *target, Operation *payload) {
  if (target->getName() != payload->getName()) {
    return transformOp.emitSilenceableError()
           << "target operation name " << target->getName()
           << " does not match payload " << payload->getName();
  }

  if (target->getAttrDictionary() != payload->getAttrDictionary()) {
    return transformOp.emitSilenceableError()
           << "target attribute dictionary " << target->getAttrDictionary()
           << " does not match payload attribute dictionary "
           << payload->getAttrDictionary();
  }

  if (target->getNumResults() != payload->getNumResults()) {
    return transformOp.emitSilenceableError() << "result count mismatch";
  }

  if (target->getNumOperands() != payload->getNumOperands()) {
    return transformOp.emitSilenceableError() << "operand count mismatch";
  }
  for (auto [targetType, payloadType] :
       llvm::zip_equal(target->getOperandTypes(), payload->getOperandTypes())) {
    if (auto targetTensorType = dyn_cast<RankedTensorType>(targetType)) {
      if (!isCastableToTensorType(payloadType, targetTensorType)) {
        return transformOp.emitSilenceableError()
               << "operand tensor type mismatch";
      }
    } else if (targetType != payloadType) {
      return transformOp.emitSilenceableError() << "operand type mismatch";
    }
  }

  return compareOperationRegions(transformOp, target, payload);
}

DiagnosedSilenceableFailure
IREE::transform_dialect::MatchCastCompatibleDagFromRootOp::matchOperation(
    Operation *current, transform::TransformResults &results,
    transform::TransformState &state) {
  Operation *targetDagRoot = getRegion().front().back().getPrevNode();

  // Reserve the list of inputs based on the number of block arguments in
  // the operation region.
  int64_t numInputs = getRegion().getNumArguments();
  SmallVector<Value> inputs(numInputs, nullptr);

  // Maps from target/payload op to the order in which they were first
  // processed. This is used to verify that two uses actually point to the
  // same node in the dag.
  llvm::MapVector<Operation *, int64_t> targetDagOrder;
  llvm::MapVector<Operation *, int64_t> payloadDagOrder;
  int64_t index = 0;

  auto compareDagOrder = [&](Operation *target, Operation *payload) {
    return targetDagOrder.find(target) == payloadDagOrder.find(payload);
  };

  // Populate the paired worklist with the current target and payload root ops.
  SmallVector<Operation *> targetWorklist = {targetDagRoot};
  SmallVector<Operation *> payloadWorklist = {current};
  while (!targetWorklist.empty()) {
    Operation *targetOp = targetWorklist.pop_back_val();
    Operation *payloadOp = payloadWorklist.pop_back_val();

    if (targetOp->hasAttr("match.operation_name_only")) {
      if (targetOp->getName() != payloadOp->getName()) {
        return emitSilenceableError() << "only operation name op mismatch";
      }
      // Do not recurse and do not require any specific structure beyond the
      // operation name.
      continue;
    }

    // Verify that if already processed, both operations are at the same
    // position.
    if (targetDagOrder.contains(targetOp) ||
        payloadDagOrder.contains(payloadOp)) {
      if (!compareDagOrder(targetOp, payloadOp)) {
        return emitSilenceableError() << "dag mismatch";
      }
      continue;
    }

    // Verify general operation equality (name, attributes, regions).
    DiagnosedSilenceableFailure diag =
        compareCastCompatibleOperations(*this, targetOp, payloadOp);
    if (!diag.succeeded()) {
      diag.attachNote() << "While processing operation " << *payloadOp;
      return diag;
    }

    for (auto [payloadOperand, targetOperand] :
         llvm::zip_equal(payloadOp->getOperands(), targetOp->getOperands())) {
      // If the target value is a block argument, map the payload value to the
      // associated input and don't process its producer.
      if (auto targetBlockArg = dyn_cast<BlockArgument>(targetOperand)) {
        if (targetBlockArg.getOwner() != &getRegion().front()) {
          return emitDefiniteFailure() << "Invalid block argument in target";
        }
        int64_t argIdx = targetBlockArg.getArgNumber();
        if (inputs[argIdx] && inputs[argIdx] != targetOperand) {
          return emitSilenceableError()
                 << "input operand with conflicting uses";
        }
        inputs[argIdx] = payloadOperand;
        continue;
      }

      Operation *payloadDefiningOp = payloadOperand.getDefiningOp();
      if (!payloadDefiningOp) {
        return emitSilenceableError()
               << "early termination of the operation dag";
      }

      // Check whether the producer was already processed, and if so make sure
      // the target and payload match.
      Operation *targetDefiningOp = targetOperand.getDefiningOp();
      if (targetDagOrder.contains(targetDefiningOp) ||
          payloadDagOrder.contains(payloadDefiningOp)) {
        if (!compareDagOrder(targetDefiningOp, payloadDefiningOp)) {
          return emitSilenceableError() << "dag mismatch";
        }
        continue;
      }
      // Pop the producer of this value onto the worklist.
      targetWorklist.push_back(targetDefiningOp);
      payloadWorklist.push_back(payloadDefiningOp);
    }

    // Mark the current target + payload as processed.
    targetDagOrder[targetOp] = index;
    payloadDagOrder[payloadOp] = index++;
  }

  // Verify that all input arguments were successfully matched.
  if (llvm::any_of(inputs, [](Value in) { return !in; })) {
    return emitSilenceableError() << "failed to match all input nodes";
  }

  results.setValues(cast<OpResult>(getInputs()), inputs);
  results.setValues(cast<OpResult>(getOutputs()), current->getResults());
  return DiagnosedSilenceableFailure::success();
}

LogicalResult
IREE::transform_dialect::MatchCastCompatibleDagFromRootOp::verify() {
  auto &body = getRegion().front();
  if (llvm::range_size(body.getOperations()) < 2) {
    return emitOpError() << "match region must contain at least one operation";
  }
  // TODO: Region verification that it includes a single DAG.
  return success();
}

//===----------------------------------------------------------------------===//
// MatchCastCompatibleTypesOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
IREE::transform_dialect::MatchCastCompatibleTypesOp::matchValue(
    Value current, transform::TransformResults &results,
    transform::TransformState &state) {
  if (auto targetTensorType = dyn_cast<RankedTensorType>(getTargetType())) {
    if (!isCastableToTensorType(current.getType(), targetTensorType)) {
      return emitSilenceableError()
             << "Type " << current.getType() << " is not castable to "
             << targetTensorType;
    }
    return DiagnosedSilenceableFailure::success();
  }
  return emitSilenceableError() << "unhandled target type: " << getTargetType();
}

//===----------------------------------------------------------------------===//
// MatchRegionsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
IREE::transform_dialect::MatchRegionsOp::matchOperation(
    Operation *current, transform::TransformResults &results,
    transform::TransformState &state) {
  Operation *comparisonTarget = &getRegion().front().front();
  return compareOperationRegions(*this, comparisonTarget, current);
}

LogicalResult IREE::transform_dialect::MatchRegionsOp::verify() {
  auto &body = getRegion().front();
  if (llvm::range_size(body.getOperations()) != 2) {
    return emitOpError() << "match region must contain exactly one operation";
  }
  Operation *target = &body.front();
  if (target->getNumRegions() == 0) {
    return emitOpError() << "contained operation for comparison must have at "
                            "least one region";
  }
  return success();
}

} // namespace mlir::iree_compiler

#define GET_OP_CLASSES
#include "iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.cpp.inc"
