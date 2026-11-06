// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements IREE-specific preprocessing for XLA inputs.

#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Rewriters.h"
#include "compiler/plugins/input/StableHLO/Conversion/StableHLOSignatureMetadata.h"
#include "iree/compiler/Bindings/Native/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_FLATTENPUBLICTUPLESIGNATURES
#define GEN_PASS_DEF_FLATTENTUPLESINCFG
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h.inc"

namespace {
// Given a set of types, unpack to a list of a types, removing all tuples.
void untupleTypes(TypeRange types, llvm::SmallVectorImpl<Type> &newTypes) {
  for (Type type : types) {
    if (auto tupleTy = dyn_cast<TupleType>(type)) {
      untupleTypes(tupleTy.getTypes(), newTypes);
    } else {
      newTypes.push_back(type);
    }
  }
}

static bool hasTupleSignature(FunctionType type) {
  return llvm::any_of(
      llvm::concat<const Type>(type.getInputs(), type.getResults()),
      llvm::IsaPred<TupleType>);
}

// Applies the metadata contract shared with final 1:N type conversion, using
// depth-first tuple leaves as this pass's signature expansion.
static FailureOr<SignatureMetadata>
remapTupleSignatureMetadata(Operation *op, FunctionType type,
                            SignatureMetadata metadata) {
  if (!hasTupleSignature(type)) {
    return metadata;
  }
  TypeConverter converter;
  converter.addConversion([](Type type, SmallVectorImpl<Type> &converted) {
    untupleTypes(TypeRange{type}, converted);
    return success();
  });
  return remapSignatureMetadata(op, type, metadata, converter);
}

template <typename T>
bool hasTuples(T values) {
  bool isTuple = false;
  for (auto val : values) {
    isTuple |= isa<TupleType>(val.getType());
  }

  return isTuple;
}

Value processTuple(Type type, Location loc, Block &block, OpBuilder &builder) {
  auto tupleType = dyn_cast<TupleType>(type);
  if (!tupleType) {
    return block.addArgument(type, loc);
  }

  llvm::SmallVector<Value> values;
  values.reserve(tupleType.size());
  for (Type subtype : tupleType.getTypes()) {
    values.push_back(processTuple(subtype, loc, block, builder));
  }

  return mlir::stablehlo::TupleOp::create(builder, loc, tupleType, values);
}

void copyOperationAttrs(Operation *oldOp, Operation *newOp) {
  for (NamedAttribute oldAttr : oldOp->getAttrs()) {
    // Don't copy segment attributes as these correspond to the number operands,
    // which may be different.
    if (oldAttr.getName() == "operandSegmentSizes" ||
        oldAttr.getName() == "resultSegmentSizes") {
      continue;
    }

    newOp->setAttr(oldAttr.getName(), oldAttr.getValue());
  }
}

void recursiveUntuple(Value value, Location loc, OpBuilder &builder,
                      llvm::SmallVectorImpl<Value> &newValues) {
  auto tupleType = dyn_cast<TupleType>(value.getType());
  if (!tupleType) {
    // We can return the value as is.
    newValues.push_back(value);
    return;
  }

  for (auto [idx, subType] : llvm::enumerate(tupleType.getTypes())) {
    auto elementOp = mlir::stablehlo::GetTupleElementOp::create(
        builder, loc, subType, value, builder.getI32IntegerAttr(idx));
    recursiveUntuple(elementOp.getResult(), loc, builder, newValues);
  }
}

Value recursiveRetuple(Type oldType, Operation::result_range *values,
                       OpBuilder &builder, Location loc) {
  auto tupleType = dyn_cast<TupleType>(oldType);
  if (!tupleType) {
    Value returnValue = *values->begin();
    *values = {values->begin() + 1, values->end()};
    return returnValue;
  }

  llvm::SmallVector<Value> subValues;
  for (Type subType : tupleType.getTypes()) {
    subValues.push_back(recursiveRetuple(subType, values, builder, loc));
  }

  return mlir::stablehlo::TupleOp::create(builder, loc, tupleType, subValues)
      .getResult();
}

template <typename T>
void untupleValues(T values, llvm::SmallVectorImpl<Value> &newValues,
                   OpBuilder &builder, Location loc) {
  for (auto operand : values) {
    recursiveUntuple(operand, loc, builder, newValues);
  }
}

class DetupleReturnOp final : public OpRewritePattern<func::ReturnOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &builder) const override {
    if (!hasTuples(op.getOperands())) {
      return builder.notifyMatchFailure(op, "No detupling required");
    }

    llvm::SmallVector<Value> newOperands;
    untupleValues(op.getOperands(), newOperands, builder, op.getLoc());

    mlir::func::ReturnOp::create(builder, op->getLoc(), newOperands);
    builder.eraseOp(op);
    return success();
  }
};

class DetupleCallOp final : public OpRewritePattern<func::CallOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(func::CallOp oldOp,
                                PatternRewriter &builder) const override {
    if (!hasTuples(oldOp.getOperands()) && !hasTuples(oldOp.getResults())) {
      return builder.notifyMatchFailure(oldOp, "No detupling required");
    }

    auto metadata = remapTupleSignatureMetadata(
        oldOp, oldOp.getCalleeType(),
        {oldOp.getArgAttrsAttr(), oldOp.getResAttrsAttr(),
         oldOp->getAttrOfType<ArrayAttr>("iree.abi.tied_operands")});
    if (failed(metadata)) {
      return failure();
    }
    llvm::SmallVector<Value> newArgs;
    untupleValues(oldOp.getOperands(), newArgs, builder, oldOp.getLoc());

    SmallVector<Type> resultTypes;
    untupleTypes(oldOp.getResultTypes(), resultTypes);
    auto newOp = func::CallOp::create(builder, oldOp->getLoc(),
                                      oldOp.getCallee(), resultTypes, newArgs);
    newOp->setDiscardableAttrs(oldOp->getDiscardableAttrDictionary());
    newOp.setArgAttrsAttr(metadata->argAttrs);
    newOp.setResAttrsAttr(metadata->resultAttrs);
    newOp.setNoInline(oldOp.getNoInline());
    if (metadata->tiedOperands) {
      newOp->setAttr("iree.abi.tied_operands", metadata->tiedOperands);
    }

    auto newResults = newOp.getResults();
    llvm::SmallVector<Value> retupledResults;
    for (auto oldResult : oldOp.getResults()) {
      auto newResult = recursiveRetuple(oldResult.getType(), &newResults,
                                        builder, oldOp->getLoc());
      retupledResults.push_back(newResult);
    }

    builder.replaceOp(oldOp, retupledResults);
    return success();
  }
};

class DetupleBranchOp final : public OpRewritePattern<cf::BranchOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(cf::BranchOp oldOp,
                                PatternRewriter &builder) const override {
    if (!hasTuples(oldOp.getOperands())) {
      return builder.notifyMatchFailure(oldOp, "No detupling required");
    }

    llvm::SmallVector<Value> newArgs;
    untupleValues(oldOp.getOperands(), newArgs, builder, oldOp.getLoc());

    auto newOp =
        cf::BranchOp::create(builder, oldOp.getLoc(), oldOp.getDest(), newArgs);

    copyOperationAttrs(oldOp, newOp);
    builder.eraseOp(oldOp);
    return success();
  }
};

class DetupleConditionOp final : public OpRewritePattern<cf::CondBranchOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(cf::CondBranchOp oldOp,
                                PatternRewriter &builder) const override {
    if (!hasTuples(oldOp.getOperands())) {
      return builder.notifyMatchFailure(oldOp, "No detupling required");
    }

    llvm::SmallVector<Value> trueArgs;
    untupleValues(oldOp.getTrueOperands(), trueArgs, builder, oldOp.getLoc());

    llvm::SmallVector<Value> falseArgs;
    untupleValues(oldOp.getFalseOperands(), falseArgs, builder, oldOp.getLoc());

    auto newOp = cf::CondBranchOp::create(
        builder, oldOp.getLoc(), oldOp.getCondition(), oldOp.getTrueDest(),
        trueArgs, oldOp.getFalseDest(), falseArgs);

    copyOperationAttrs(oldOp, newOp);

    builder.eraseOp(oldOp);
    return success();
  }
};

static void convertFunction(func::FuncOp oldFunction, func::FuncOp newFunction,
                            const SignatureMetadata &metadata) {
  OpBuilder builder(newFunction.getBody());
  IRMapping mapping;

  auto xlaAbiParam = StringAttr::get(newFunction.getContext(),
                                     "xla_entry_computation_parameter_layouts");
  auto xlaAbiLayout = StringAttr::get(newFunction.getContext(),
                                      "xla_entry_computation_result_layout");

  for (NamedAttribute attr : oldFunction->getAttrs()) {
    // Slot attributes are remapped separately. XLA layout attributes have no
    // flattened-signature contract.
    if (llvm::is_contained(
            {oldFunction.getFunctionTypeAttrName(), xlaAbiParam, xlaAbiLayout},
            attr.getName())) {
      continue;
    }

    if (attr.getName() == oldFunction.getArgAttrsAttrName() ||
        attr.getName() == oldFunction.getResAttrsAttrName()) {
      continue;
    }
    newFunction->setAttr(attr.getName(), attr.getValue());
  }

  if (metadata.argAttrs) {
    newFunction.setAllArgAttrs(metadata.argAttrs);
  }
  if (metadata.resultAttrs) {
    newFunction.setAllResultAttrs(metadata.resultAttrs);
  }
  newFunction.getBlocks().clear();
  for (Block &oldBlock : oldFunction.getBlocks()) {
    Block *newBlock = builder.createBlock(&newFunction.getBody());
    for (BlockArgument oldArg : oldBlock.getArguments()) {
      Value newTuple = processTuple(oldArg.getType(), oldFunction.getLoc(),
                                    *newBlock, builder);
      mapping.map(oldArg, newTuple);
    }
    mapping.map(&oldBlock, newBlock);
  }

  // Convert all ops in the blocks.
  for (Block &oldBlock : oldFunction.getBlocks()) {
    builder.setInsertionPointToEnd(mapping.lookupOrNull(&oldBlock));
    for (Operation &oldOp : oldBlock.getOperations()) {
      builder.clone(oldOp, mapping);
    }
  }
}

// Returns the source declaration of `funcOp` in the format of the native ABI,
// with token leaves shown as the fences they lower to.
StringAttr formatPublicDeclaration(func::FuncOp funcOp) {
  auto model = funcOp->getAttrOfType<StringAttr>("iree.abi.model");
  AttrTypeReplacer replacer;
  replacer.addReplacement([](mlir::stablehlo::TokenType type) -> Type {
    return IREE::HAL::FenceType::get(type.getContext());
  });
  return IREE::ABI::formatSourceDeclaration(
      model && model.getValue() == "coarse-fences"
          ? IREE::ABI::InvocationModel::CoarseFences
          : IREE::ABI::InvocationModel::Sync,
      funcOp, funcOp.getName(), funcOp.getArgAttrsAttr(),
      funcOp.getResAttrsAttr(),
      [&](Type type) { return replacer.replace(type); });
}

struct FlattenPublicTupleSignatures final
    : impl::FlattenPublicTupleSignaturesBase<FlattenPublicTupleSignatures> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    for (auto funcOp : llvm::to_vector(moduleOp.getOps<func::FuncOp>())) {
      FunctionType type = funcOp.getFunctionType();
      if (funcOp.isPrivate() || (!hasTupleType(type.getInputs()) &&
                                 !hasTupleType(type.getResults()))) {
        continue;
      }
      auto metadata = remapTupleSignatureMetadata(
          funcOp, type,
          {funcOp.getAllArgAttrs(), funcOp.getAllResultAttrs(), {}});
      if (failed(metadata)) {
        return signalPassFailure();
      }
      FailureOr<SmallVector<FlatCall>> calls = prepareCallers(funcOp);
      if (failed(calls)) {
        return signalPassFailure();
      }
      if (!funcOp->hasAttr("iree.abi.declaration")) {
        funcOp->setAttr("iree.abi.declaration",
                        formatPublicDeclaration(funcOp));
      }
      SmallVector<Type> inputs, results;
      untupleTypes(type.getInputs(), inputs);
      untupleTypes(type.getResults(), results);
      flattenCallers(funcOp, *calls, results);
      if (!funcOp.isExternal()) {
        flattenBody(funcOp);
      }
      funcOp.setFunctionType(FunctionType::get(&getContext(), inputs, results));
      funcOp.setArgAttrsAttr(metadata->argAttrs);
      funcOp.setResAttrsAttr(metadata->resultAttrs);
    }
  }

private:
  static bool hasTupleType(TypeRange types) {
    return llvm::any_of(types, llvm::IsaPred<TupleType>);
  }

  // Rebuilds tuple arguments from leaf arguments at entry and returns leaves.
  static void flattenBody(func::FuncOp funcOp) {
    Block &entry = funcOp.front();
    unsigned oldArgCount = entry.getNumArguments();
    OpBuilder builder = OpBuilder::atBlockBegin(&entry);
    for (unsigned index = 0; index < oldArgCount; ++index) {
      BlockArgument oldArg = entry.getArgument(index);
      Value replacement =
          processTuple(oldArg.getType(), oldArg.getLoc(), entry, builder);
      oldArg.replaceAllUsesWith(replacement);
    }
    entry.eraseArguments(0, oldArgCount);
    funcOp.walk([&](func::ReturnOp returnOp) {
      OpBuilder returnBuilder(returnOp);
      SmallVector<Value> leaves;
      for (Value operand : returnOp.getOperands()) {
        recursiveUntuple(operand, returnOp.getLoc(), returnBuilder, leaves);
      }
      returnOp.getOperandsMutable().assign(leaves);
    });
  }

  // A caller and its metadata over the flattened slots.
  struct FlatCall {
    func::CallOp callOp;
    ArrayAttr argAttrs;
    ArrayAttr resultAttrs;
    ArrayAttr ties;
  };

  // Flattens the metadata of every caller of `funcOp`, failing before any
  // caller changes.
  static FailureOr<SmallVector<FlatCall>> prepareCallers(func::FuncOp funcOp) {
    auto module = funcOp->getParentOfType<ModuleOp>();
    std::optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(funcOp, module);
    if (!uses) {
      return funcOp.emitError("cannot enumerate uses of a public function");
    }
    FunctionType type = funcOp.getFunctionType();
    SmallVector<FlatCall> calls;
    llvm::SmallPtrSet<Operation *, 8> seenCalls;
    for (const SymbolTable::SymbolUse &use : *uses) {
      auto callOp = dyn_cast<func::CallOp>(use.getUser());
      if (!callOp || callOp.getCalleeAttr() != use.getSymbolRef()) {
        return use.getUser()->emitError(
            "public tuple signatures support only direct calls");
      }
      if (!seenCalls.insert(callOp).second) {
        continue;
      }
      auto metadata = remapTupleSignatureMetadata(
          callOp, type,
          {callOp.getArgAttrsAttr(), callOp.getResAttrsAttr(),
           callOp->getAttrOfType<ArrayAttr>("iree.abi.tied_operands")});
      if (failed(metadata)) {
        return failure();
      }
      FlatCall call{callOp, metadata->argAttrs, metadata->resultAttrs,
                    metadata->tiedOperands};
      calls.push_back(call);
    }
    return calls;
  }

  static void flattenCallers(func::FuncOp funcOp, ArrayRef<FlatCall> calls,
                             ArrayRef<Type> results) {
    for (const FlatCall &call : calls) {
      func::CallOp callOp = call.callOp;
      OpBuilder builder(callOp);
      SmallVector<Value> operands;
      for (Value operand : callOp.getOperands()) {
        recursiveUntuple(operand, callOp.getLoc(), builder, operands);
      }
      auto newCall = func::CallOp::create(builder, callOp.getLoc(),
                                          funcOp.getName(), results, operands);
      newCall->setDiscardableAttrs(callOp->getDiscardableAttrDictionary());
      if (call.ties) {
        newCall->setAttr("iree.abi.tied_operands", call.ties);
      }
      newCall.setArgAttrsAttr(call.argAttrs);
      newCall.setResAttrsAttr(call.resultAttrs);
      newCall.setNoInline(callOp.getNoInline());
      Operation::result_range leaves = newCall.getResults();
      SmallVector<Value> replacements;
      for (Type type : callOp.getResultTypes()) {
        replacements.push_back(
            recursiveRetuple(type, &leaves, builder, callOp.getLoc()));
      }
      callOp.replaceAllUsesWith(replacements);
      callOp.erase();
    }
  }
};

struct TupleFunctionPlan {
  func::FuncOp function;
  SignatureMetadata metadata;
};

// Inspect symbol uses once while all original functions and calls are live.
// Only direct calls and Native result-dimension references have a contract for
// a changed tuple signature.
static LogicalResult
validateTupleSymbolUses(ModuleOp module, ArrayRef<TupleFunctionPlan> plans) {
  llvm::SmallDenseSet<Operation *> functions;
  for (const TupleFunctionPlan &plan : plans) {
    func::FuncOp function = plan.function;
    if (hasTupleSignature(function.getFunctionType())) {
      functions.insert(function);
    }
  }
  if (functions.empty()) {
    return success();
  }
  auto uses = SymbolTable::getSymbolUses(&module.getBodyRegion());
  if (!uses) {
    return module.emitError("cannot enumerate uses of tuple functions");
  }
  SymbolTableCollection symbols;
  for (const SymbolTable::SymbolUse &use : *uses) {
    Operation *callee =
        symbols.lookupNearestSymbolFrom(use.getUser(), use.getSymbolRef());
    if (!functions.contains(callee)) {
      continue;
    }
    auto call = dyn_cast<func::CallOp>(use.getUser());
    if (call && call.getCalleeAttr() == use.getSymbolRef()) {
      continue;
    }
    auto owner = dyn_cast<func::FuncOp>(use.getUser());
    if (owner && owner->getAttrOfType<SymbolRefAttr>("iree.abi.result_dims") ==
                     use.getSymbolRef()) {
      continue;
    }
    return use.getUser()->emitError(
        "tuple signatures support only direct calls and result-dimension "
        "helper references");
  }
  return success();
}

static LogicalResult validateTupleCallMetadata(ModuleOp module) {
  WalkResult invalid = module.walk([](Operation *op) {
    if (auto call = dyn_cast<func::CallOp>(op)) {
      if (failed(remapTupleSignatureMetadata(
              call, call.getCalleeType(),
              {call.getArgAttrsAttr(), call.getResAttrsAttr(),
               call->getAttrOfType<ArrayAttr>("iree.abi.tied_operands")}))) {
        return WalkResult::interrupt();
      }
    } else if (auto call = dyn_cast<func::CallIndirectOp>(op)) {
      if (hasTuples(call.getOperands()) || hasTuples(call.getResults())) {
        call.emitError("tuple signatures support only direct calls");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(invalid.wasInterrupted());
}

static FailureOr<SmallVector<TupleFunctionPlan>>
prepareTupleFunctions(ModuleOp module) {
  SmallVector<TupleFunctionPlan> plans;
  for (auto funcOp : module.getOps<func::FuncOp>()) {
    auto metadata = remapTupleSignatureMetadata(
        funcOp, funcOp.getFunctionType(),
        {funcOp.getAllArgAttrs(), funcOp.getAllResultAttrs(), {}});
    if (failed(metadata)) {
      return failure();
    }
    plans.push_back({funcOp, *metadata});
  }
  if (failed(validateTupleSymbolUses(module, plans)) ||
      failed(validateTupleCallMetadata(module))) {
    return failure();
  }
  return plans;
}

struct FlattenTuplesInCFG final
    : impl::FlattenTuplesInCFGBase<FlattenTuplesInCFG> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = moduleOp.getContext();
    Builder builder(ctx);

    auto signatures = prepareTupleFunctions(moduleOp);
    if (failed(signatures)) {
      return signalPassFailure();
    }

    // Build a list of (oldFunction, newFunction) for all functions we need to
    // replace. This will ensure that when we go to convert function bodies we
    // have only new functions defined.
    SmallVector<std::pair<func::FuncOp, func::FuncOp>> convertedFunctions;
    for (const TupleFunctionPlan &plan : *signatures) {
      func::FuncOp oldFunction = plan.function;
      FunctionType oldFunctionType = oldFunction.getFunctionType();

      llvm::SmallVector<Type> newInputTypes;
      untupleTypes(oldFunctionType.getInputs(), newInputTypes);

      llvm::SmallVector<Type> newResultTypes;
      untupleTypes(oldFunctionType.getResults(), newResultTypes);

      FunctionType newFunctionType =
          builder.getFunctionType(newInputTypes, newResultTypes);
      func::FuncOp newFunction =
          func::FuncOp::create(oldFunction.getLoc(), oldFunction.getName(),
                               newFunctionType, oldFunction->getDialectAttrs());
      convertedFunctions.push_back({oldFunction, newFunction});

      // Perform the actual body conversion now that we have proper signatures.
      convertFunction(oldFunction, newFunction, plan.metadata);
    }

    // Replace functions in the module.
    for (auto [oldFunction, newFunction] : convertedFunctions) {
      oldFunction.erase();
      moduleOp.push_back(newFunction);
    }

    // Run canonicalization patterns to cancel out remaining tuple ops. We need
    // to run these manually here because StableHLO does not define
    // folds/canonicalization patterns for its ops.
    RewritePatternSet patterns(ctx);
    patterns.insert<DetupleCallOp, DetupleConditionOp, DetupleReturnOp,
                    DetupleBranchOp>(ctx);
    populateCanonicalizationPatterns(ctx, &patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::stablehlo
