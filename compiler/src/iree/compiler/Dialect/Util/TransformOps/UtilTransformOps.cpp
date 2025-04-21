// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/TransformOps/UtilTransformOps.h"

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::Util {

LogicalResult
IREE::Util::transform_dialect::CreateSerializedModuleOp::verify() {
  if (!getBody().hasOneBlock()) {
    return emitOpError() << "expected single block body";
  }
  Block *body = &getBody().front();
  if (body->getNumArguments() != 1) {
    return emitOpError() << "expected body with single block argument.";
  }
  if (!isa<transform::TransformHandleTypeInterface>(
          body->getArgument(0).getType())) {
    return emitOpError()
           << "expected body argument to be a transform op handle type";
  }
  if (!body->empty()) {
    for (Operation &op : *body) {
      if (!isa<transform::TransformOpInterface>(&op)) {
        InFlightDiagnostic diag = emitOpError()
                                  << "expected children ops to implement "
                                     "TransformOpInterface";
        diag.attachNote(op.getLoc()) << "op without interface";
        return diag;
      }
    }
  }
  return success();
}

void IREE::Util::transform_dialect::CreateSerializedModuleOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::producesHandle(getOperation()->getOpResults(), effects);
}

DiagnosedSilenceableFailure
IREE::Util::transform_dialect::CreateSerializedModuleOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &transformResults,
    transform::TransformState &state) {

  // Create a temporary module that will be erased once out of scope.
  OwningOpRef<ModuleOp> tempModule(ModuleOp::create(getLoc()));

  // Map the temporary module to the block argument of the body.
  // This should never fail per the verifier.
  transform::TransformState::RegionScope scope =
      state.make_region_scope(getBody());
  if (failed(state.mapBlockArguments(getBody().front().getArgument(0),
                                     tempModule->getOperation()))) {
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  // Apply the contained ops one by one.
  for (Operation &transform : getBody().front()) {
    DiagnosedSilenceableFailure result =
        state.applyTransform(cast<transform::TransformOpInterface>(transform));
    // TODO: Support better error propagation.
    if (result.isSilenceableFailure())
      return DiagnosedSilenceableFailure::definiteFailure();
    // Pass through the error message from definite failures.
    if (result.isDefiniteFailure())
      return result;
  }

  // Serialize the module as bytecode to a string.
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  if (failed(writeBytecodeToFile(tempModule->getOperation(), os))) {
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  auto bufferSize = static_cast<int64_t>(buffer.size());
  auto bufferShape =
      VectorType::get(bufferSize, IntegerType::get(rewriter.getContext(), 8));
  auto serializedModule = DenseElementsAttr::getFromRawBuffer(
      bufferShape, ArrayRef(buffer.data(), buffer.data() + bufferSize));

  // Return the serialized module as a DenseElementsAttr.
  transformResults.setParams(cast<OpResult>(getResult()), {serializedModule});
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// DeserializeModuleOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
IREE::Util::transform_dialect::DeserializeModuleOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  ArrayRef<Attribute> params = state.getParams(getModule());
  if (params.size() != 1) {
    return emitDefiniteFailure() << "requires exactly one parameter associated";
  }

  auto serializedModule =
      dyn_cast<IREE::Util::SerializableAttrInterface>(params[0]);
  if (!serializedModule) {
    return emitDefiniteFailure() << "input must be serializable to deserialize";
  }

  auto containerOps = state.getPayloadOps(getContainer());
  if (!llvm::hasSingleElement(containerOps)) {
    return emitDefiniteFailure()
           << "Only one op can be specified as a container";
  }

  auto containerOp = dyn_cast<ModuleOp>(*containerOps.begin());
  if (!containerOp) {
    return emitDefiniteFailure() << "Expected module op as a container";
  }

  SmallVector<char, 0> bytecode;
  if (failed(serializedModule.serializeToVector(
          getLoc(), llvm::endianness::native, bytecode))) {
    return emitDefiniteFailure() << "Failed to deserialize module";
  }

  // Invoke parseSourceString directly to construct the deserialized module
  // at the end of the container block.
  ParserConfig config(rewriter.getContext());
  LocationAttr sourceFileLoc;
  if (failed(parseSourceString(StringRef(bytecode.data(), bytecode.size()),
                               containerOp.getBody(0), config,
                               /*sourceName=*/"", &sourceFileLoc))) {
    return emitDefiniteFailure() << "Failed to deserialize module";
  }
  return DiagnosedSilenceableFailure::success();
}

void IREE::Util::transform_dialect::DeserializeModuleOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getContainerMutable(), effects);
  transform::onlyReadsHandle(getModuleMutable(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// GetNearestSymbolTableOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
IREE::Util::transform_dialect::GetNearestSymbolTableOp::applyToOne(
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

void IREE::Util::transform_dialect::GetNearestSymbolTableOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// LookupNearestSymbolFromSelfOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
IREE::Util::transform_dialect::LookupNearestSymbolFromSelfOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  auto symbolOp = SymbolTable::lookupNearestSymbolFrom(*this, getSymbol());
  if (!symbolOp) {
    return emitDefiniteFailure() << "could not find symbol " << getSymbol();
  }
  transformResults.set(cast<OpResult>(getTargetSymbol()), {symbolOp});
  return DiagnosedSilenceableFailure::success();
}

void IREE::Util::transform_dialect::LookupNearestSymbolFromSelfOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ImportSymbolOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
IREE::Util::transform_dialect::ImportSymbolOp::apply(
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

void IREE::Util::transform_dialect::ImportSymbolOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getSymbolTableMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

LogicalResult IREE::Util::transform_dialect::ImportSymbolOp::verify() {
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
// CastAndCallOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure IREE::Util::transform_dialect::CastAndCallOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  SmallVector<Value> inputs;
  if (getInputs())
    llvm::append_range(inputs, state.getPayloadValues(getInputs()));

  SetVector<Value> outputs;
  if (getOutputs()) {
    for (auto output : state.getPayloadValues(getOutputs()))
      outputs.insert(output);

    // Verify that the set of output values to be replaced is unique.
    if (outputs.size() !=
        llvm::range_size(state.getPayloadValues(getOutputs()))) {
      return emitSilenceableFailure(getLoc())
             << "cast and call output values must be unique";
    }
  }

  // Get the insertion point for the call.
  auto insertionOps = state.getPayloadOps(getInsertionPoint());
  if (!llvm::hasSingleElement(insertionOps)) {
    return emitSilenceableFailure(getLoc())
           << "Only one op can be specified as an insertion point";
  }
  bool insertAfter = getInsertAfter();
  Operation *insertionPoint = *insertionOps.begin();

  // Check that all inputs dominate the insertion point, and the insertion
  // point dominates all users of the outputs.
  DominanceInfo dom(insertionPoint);
  for (Value output : outputs) {
    for (Operation *user : output.getUsers()) {
      // If we are inserting after the insertion point operation, the
      // insertion point operation must properly dominate the user. Otherwise
      // basic dominance is enough.
      bool doesDominate = insertAfter
                              ? dom.properlyDominates(insertionPoint, user)
                              : dom.dominates(insertionPoint, user);
      if (!doesDominate) {
        return emitDefiniteFailure()
               << "User " << user << " is not dominated by insertion point "
               << insertionPoint;
      }
    }
  }

  for (Value input : inputs) {
    // If we are inserting before the insertion point operation, the
    // input must properly dominate the insertion point operation. Otherwise
    // basic dominance is enough.
    bool doesDominate = insertAfter
                            ? dom.dominates(input, insertionPoint)
                            : dom.properlyDominates(input, insertionPoint);
    if (!doesDominate) {
      return emitDefiniteFailure()
             << "input " << input << " does not dominate insertion point "
             << insertionPoint;
    }
  }

  // Get the function to call. This can either be specified by symbol or as a
  // transform handle.
  IREE::Util::FuncOp targetFunction = nullptr;
  if (getFunctionName()) {
    targetFunction = SymbolTable::lookupNearestSymbolFrom<IREE::Util::FuncOp>(
        insertionPoint, *getFunctionName());
    if (!targetFunction) {
      return emitDefiniteFailure()
             << "unresolved symbol " << *getFunctionName();
    }
  } else if (getFunction()) {
    auto payloadOps = state.getPayloadOps(getFunction());
    if (!llvm::hasSingleElement(payloadOps)) {
      return emitDefiniteFailure() << "requires a single function to call";
    }
    targetFunction = dyn_cast<IREE::Util::FuncOp>(*payloadOps.begin());
    if (!targetFunction) {
      return emitDefiniteFailure() << "invalid non-function callee";
    }
  } else {
    llvm_unreachable("invalid CastAndCall op without a function to call");
    return emitDefiniteFailure();
  }

  // Verify that the function argument and result lengths match the inputs and
  // outputs given to this op.
  if (targetFunction.getNumArguments() != inputs.size()) {
    return emitSilenceableFailure(targetFunction.getLoc())
           << "mismatch between number of function arguments "
           << targetFunction.getNumArguments() << " and number of inputs "
           << inputs.size();
  }
  if (targetFunction.getNumResults() != outputs.size()) {
    return emitSilenceableFailure(targetFunction.getLoc())
           << "mismatch between number of function results "
           << targetFunction->getNumResults() << " and number of outputs "
           << outputs.size();
  }

  // Gather all specified converters.
  mlir::TypeConverter converter;
  if (!getRegion().empty()) {
    for (Operation &op : getRegion().front()) {
      cast<transform::TypeConverterBuilderOpInterface>(&op)
          .populateTypeMaterializations(converter);
    }
  }

  if (insertAfter)
    rewriter.setInsertionPointAfter(insertionPoint);
  else
    rewriter.setInsertionPoint(insertionPoint);

  for (auto [input, type] :
       llvm::zip_equal(inputs, targetFunction.getArgumentTypes())) {
    if (input.getType() != type) {
      Value newInput = converter.materializeSourceConversion(
          rewriter, input.getLoc(), type, input);
      if (!newInput) {
        return emitDefiniteFailure() << "Failed to materialize conversion of "
                                     << input << " to type " << type;
      }
      input = newInput;
    }
  }

  SmallVector<Value> replacements;
  Operation *exceptedUser = nullptr;
  if (getInlineCall()) {
    // TODO: Support inlining multi-block functions using `scf.execute_region`
    // (needed in case the callsite parent requires a single block region).
    if (!targetFunction.getBody().hasOneBlock()) {
      return emitDefiniteFailure()
             << "Expected single block function, got "
             << targetFunction.getBody().getBlocks().size() << " blocks.";
    }

    Region *targetRegion = rewriter.getInsertionBlock()->getParent();

    // Temporarily clone the function body into the region containing the
    // insertion point. This will never cross pass nesting scopes unless the
    // insertion point itself does, which would be a misuse of this transform.
    //
    // We do this instead of cloning the function to avoid the possibility of
    // a nested pass manager picking up the temporary function and processing
    // it. (this likely isn't actually possible per the implementation of the
    // pass manager, but skipping cloning the symbol is 100% safe).
    //
    // Note that this means the function being called is assumed to also not
    // be undergoing modification (again no way to verify this, only specify
    // it as misuse).
    mlir::IRMapping mapper;
    targetFunction.getFunctionBody().cloneInto(targetRegion, mapper);

    Block *body = &targetRegion->getBlocks().back();
    Operation *terminator = body->getTerminator();

    // Inlining the block removes it from the parent region.
    rewriter.inlineBlockBefore(body, &*rewriter.getInsertionPoint(), inputs);
    replacements = terminator->getOperands();
    rewriter.eraseOp(terminator);
  } else {
    auto callOp = rewriter.create<IREE::Util::CallOp>(
        insertionPoint->getLoc(), targetFunction.getResultTypes(),
        targetFunction.getName(), inputs, /*tied_operands=*/ArrayAttr{},
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
    exceptedUser = callOp;
    replacements = callOp->getOpResults();
    if (getResults().size() != 0) {
      results.set(cast<OpResult>(getResults()[0]), {callOp});
    }
  }

  // Cast the call results back to the expected types. If any conversions fail
  // this is a definite failure as the call has been constructed at this point.
  for (auto [output, newOutput] : llvm::zip_equal(outputs, replacements)) {
    Value convertedOutput = newOutput;
    if (output.getType() != newOutput.getType()) {
      convertedOutput = converter.materializeTargetConversion(
          rewriter, output.getLoc(), output.getType(), newOutput);
      if (!convertedOutput) {
        return emitDefiniteFailure()
               << "Failed to materialize conversion of " << newOutput
               << " to type " << output.getType();
      }
    }
    if (exceptedUser) {
      rewriter.replaceAllUsesExcept(output, convertedOutput, exceptedUser);
    } else {
      rewriter.replaceAllUsesWith(output, convertedOutput);
    }
  }
  return DiagnosedSilenceableFailure::success();
}

LogicalResult IREE::Util::transform_dialect::CastAndCallOp::verify() {
  if (!getRegion().empty()) {
    for (Operation &op : getRegion().front()) {
      if (!isa<transform::TypeConverterBuilderOpInterface>(&op)) {
        InFlightDiagnostic diag = emitOpError()
                                  << "expected children ops to implement "
                                     "TypeConverterBuilderOpInterface";
        diag.attachNote(op.getLoc()) << "op without interface";
        return diag;
      }
    }
  }
  if (!getFunction() && !getFunctionName()) {
    return emitOpError() << "expected a function handle or name to call";
  }
  if (getFunction() && getFunctionName()) {
    return emitOpError() << "function handle and name are mutually exclusive";
  }
  if (getNumResults() > 1) {
    return emitOpError()
           << "produces at most one result as a handle to the call";
  }
  if (getInlineCall() && getNumResults() != 0) {
    return emitOpError() << "inlining mode does not produce a result";
  }
  return success();
}

void IREE::Util::transform_dialect::CastAndCallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getInsertionPointMutable(), effects);
  if (getInputs())
    transform::onlyReadsHandle(getInputsMutable(), effects);
  if (getOutputs())
    transform::onlyReadsHandle(getOutputsMutable(), effects);
  if (getFunction())
    transform::onlyReadsHandle(getFunctionMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

class UtilTransformDialectExtension
    : public transform::TransformDialectExtension<
          UtilTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareGeneratedDialect<IREE::Util::UtilDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Dialect/Util/TransformOps/UtilTransformOps.cpp.inc"
        >();
  }
};

void registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<UtilTransformDialectExtension>();
}

} // namespace mlir::iree_compiler::IREE::Util

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Util/TransformOps/UtilTransformOps.cpp.inc"
