// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/ConvertVMToEmitC.h"

#include "iree/compiler/Dialect/Util/Conversion/PreserveCompilerHints.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Utils/CallingConvention.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

// TODO(simon-camp/marbre): Use this function throughout the conversions.
Optional<std::string> getCType(Type type) {
  if (auto iType = type.dyn_cast<IntegerType>()) {
    switch (iType.getWidth()) {
      case 32:
      case 64:
        return std::string("int") + std::to_string(iType.getWidth()) +
               std::string("_t");
    }
  }

  if (auto fType = type.dyn_cast<FloatType>()) {
    switch (fType.getWidth()) {
      case 32:
        return std::string("float");
      case 64:
        return std::string("double");
    }
  }

  if (auto oType = type.dyn_cast<emitc::OpaqueType>()) {
    return std::string(oType.getValue());
  }

  return None;
}

LogicalResult convertFuncOp(IREE::VM::FuncOp funcOp,
                            VMAnalysisCache &vmAnalysisCache) {
  auto ctx = funcOp.getContext();
  auto loc = funcOp.getLoc();

  OpBuilder builder(funcOp);

  auto moduleOp = funcOp.getOperation()->getParentOfType<IREE::VM::ModuleOp>();

  FunctionType funcType = funcOp.getType();
  std::string name =
      std::string(moduleOp.getName()) + "_" + std::string(funcOp.getName());
  std::string moduleTypeName = (moduleOp.getName() + "_t*").str();
  std::string moduleStateTypeName = (moduleOp.getName() + "_state_t*").str();

  Type stackType = mlir::emitc::OpaqueType::get(ctx, "iree_vm_stack_t*");
  Type moduleType = mlir::emitc::OpaqueType::get(ctx, moduleTypeName);
  Type moduleStateType = mlir::emitc::OpaqueType::get(ctx, moduleStateTypeName);

  SmallVector<Type, 3> inputTypes = {stackType, moduleType, moduleStateType};
  SmallVector<Type, 1> outputTypes;

  for (auto &inputType : funcType.getInputs()) {
    inputTypes.push_back(inputType);
  }

  for (auto &resultType : funcType.getResults()) {
    Optional<std::string> cType = getCType(resultType);
    if (!cType.hasValue()) {
      return funcOp.emitError() << "unable to emit C type";
    }
    std::string cPtrType = cType.getValue() + std::string("*");
    Type type = mlir::emitc::OpaqueType::get(ctx, cPtrType);
    inputTypes.push_back(type);
    outputTypes.push_back(type);
  }

  auto newFuncType = mlir::FunctionType::get(
      ctx, {inputTypes}, {mlir::emitc::OpaqueType::get(ctx, "iree_status_t")});

  auto newFuncOp = builder.create<mlir::FuncOp>(loc, name, newFuncType);
  newFuncOp.getOperation()->setAttr("emitc.static", UnitAttr::get(ctx));

  Optional<std::string> callingConvention = makeCallingConventionString(funcOp);

  // Annotate new function with calling convention string which gets used in
  // the CModuleTarget.
  newFuncOp.getOperation()->setAttr(
      "calling_convention", StringAttr::get(ctx, callingConvention.getValue()));

  // This call shold be equivalent to rewriter.inlineRegionBefore()
  newFuncOp.getBody().getBlocks().splice(newFuncOp.end(),
                                         funcOp.getBody().getBlocks());

  Block &entryBlock = newFuncOp.getBlocks().front();

  entryBlock.insertArgument(static_cast<unsigned>(0), stackType);
  entryBlock.insertArgument(static_cast<unsigned>(1), moduleType);
  entryBlock.insertArgument(static_cast<unsigned>(2), moduleStateType);

  entryBlock.addArguments(outputTypes);

  auto ptr = vmAnalysisCache.find(funcOp.getOperation());
  if (ptr == vmAnalysisCache.end()) {
    return funcOp.emitError() << "parent func op not found in cache.";
  }

  // Add constant ops for refs
  const int numRefs =
      ptr->second.registerAllocation.getMaxRefRegisterOrdinal() + 1;

  builder.setInsertionPointToStart(&entryBlock);

  for (int i = 0; i < numRefs; i++) {
    auto refOp = builder.create<emitc::ConstantOp>(
        /*location=*/loc,
        /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t"),
        /*value=*/emitc::OpaqueAttr::get(ctx, "{0}"));

    // Mark local refs so that we can release them before a return operation
    refOp.getOperation()->setAttr("ref_ordinal", builder.getIndexAttr(i));
  }

  vmAnalysisCache.insert(
      std::make_pair(newFuncOp.getOperation(), std::move(ptr->second)));

  if (failed(funcOp.replaceAllSymbolUses(name, moduleOp)))
    return funcOp.emitError() << "unable to update symbol name in module";

  return success();
}

Optional<std::string> buildFunctionName(IREE::VM::ModuleOp &moduleOp,
                                        IREE::VM::ImportOp &importOp) {
  auto callingConvention = makeImportCallingConventionString(importOp);
  if (!callingConvention.hasValue()) {
    return None;
  }
  return std::string("call_") + callingConvention.getValue() + "_import";
}

template <typename SrcOpTy>
Optional<emitc::ApplyOp> createVmTypeDefPtr(ConversionPatternRewriter &rewriter,
                                            SrcOpTy srcOp, Type elementType) {
  auto ctx = srcOp.getContext();
  auto loc = srcOp.getLoc();

  // TODO(simon-camp): Cleanup this up
  StringRef elementTypeConstructor;
  std::string elementTypeConstructorArg;
  if (auto intType = elementType.template dyn_cast<IntegerType>()) {
    unsigned int bitWidth = intType.getIntOrFloatBitWidth();
    elementTypeConstructor = "iree_vm_type_def_make_value_type";
    elementTypeConstructorArg =
        std::string("IREE_VM_VALUE_TYPE_I") + std::to_string(bitWidth);
  } else if (auto refType =
                 elementType.template dyn_cast<IREE::VM::RefType>()) {
    auto objType = refType.getObjectType();

    elementTypeConstructor = "iree_vm_type_def_make_ref_type";
    if (objType.template isa<IREE::VM::BufferType>()) {
      elementTypeConstructorArg = "iree_vm_buffer_type_id()";
    } else if (objType.template isa<IREE::VM::ListType>()) {
      elementTypeConstructorArg = "iree_vm_list_type_id()";
    } else {
      srcOp.emitError() << "Unhandled ref object type " << objType;
      return None;
    }
  } else if (auto opaqueType =
                 elementType.template dyn_cast<IREE::VM::OpaqueType>()) {
    elementTypeConstructor = "iree_vm_type_def_make_variant_type";
    elementTypeConstructorArg = "";
  } else {
    srcOp.emitError() << "Unhandled element type " << elementType;
    return None;
  }

  auto elementTypeOp = rewriter.template create<emitc::CallOp>(
      /*location=*/loc,
      /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_type_def_t"),
      /*callee=*/StringAttr::get(ctx, elementTypeConstructor),
      /*args=*/
      ArrayAttr::get(ctx,
                     {emitc::OpaqueAttr::get(ctx, elementTypeConstructorArg)}),
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/ArrayRef<Value>{});

  auto elementTypePtrOp = rewriter.template create<emitc::ApplyOp>(
      /*location=*/loc,
      /*result=*/emitc::OpaqueType::get(ctx, "iree_vm_type_def_t*"),
      /*applicableOperator=*/StringAttr::get(ctx, "&"),
      /*operand=*/elementTypeOp.getResult(0));

  return elementTypePtrOp;
}

void releaseLocalRefs(ConversionPatternRewriter &rewriter, Location location,
                      mlir::FuncOp funcOp) {
  auto ctx = funcOp.getContext();

  // Release local refs
  for (auto constantOp : funcOp.getOps<emitc::ConstantOp>()) {
    Operation *op = constantOp.getOperation();
    if (!op->hasAttr("ref_ordinal")) continue;

    auto refPtrOp = rewriter.create<emitc::ApplyOp>(
        /*location=*/location,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t*"),
        /*applicableOperator=*/StringAttr::get(ctx, "&"),
        /*operand=*/constantOp.getResult());

    rewriter.create<emitc::CallOp>(
        /*location=*/location,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "iree_vm_ref_release"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refPtrOp.getResult()});
  }
}

/// Generate a call op with one result and split the current block into a
/// continuation and failure block based on the truthiness of the result
/// value, i.e. a truthy value branches to the continuation block.
emitc::CallOp failableCall(
    ConversionPatternRewriter &rewriter, Location location, Type type,
    StringAttr callee, ArrayAttr args, ArrayAttr templateArgs,
    ArrayRef<Value> operands,
    const std::function<void(emitc::CallOp &)> &failureBlockBuilder,
    bool negateResult = false) {
  auto ctx = rewriter.getContext();

  auto callOp = rewriter.create<emitc::CallOp>(
      /*location=*/location,
      /*type=*/type,
      /*callee=*/callee,
      /*args=*/args,
      /*templateArgs=*/templateArgs,
      /*operands=*/operands);

  Type boolType = rewriter.getIntegerType(1);

  auto conditionI1 = rewriter.create<emitc::CallOp>(
      /*location=*/location,
      /*type=*/boolType,
      /*callee=*/StringAttr::get(ctx, "EMITC_CAST"),
      /*args=*/
      ArrayAttr::get(ctx, {rewriter.getIndexAttr(0), TypeAttr::get(boolType)}),
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/ArrayRef<Value>{callOp.getResult(0)});

  if (negateResult)
    conditionI1 = rewriter.create<emitc::CallOp>(
        /*location=*/location,
        /*type=*/boolType,
        /*callee=*/StringAttr::get(ctx, "EMITC_NOT"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{conditionI1.getResult(0)});

  // Start by splitting the block into two. The part before will contain the
  // condition, and the part after will contain the continuation point.
  Block *condBlock = rewriter.getInsertionBlock();
  Block::iterator opPosition = rewriter.getInsertionPoint();
  Block *continuationBlock = rewriter.splitBlock(condBlock, opPosition);

  // Create a new block for the target of the failure.
  Block *failureBlock;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Region *parentRegion = condBlock->getParent();
    failureBlock = rewriter.createBlock(parentRegion, parentRegion->end());

    failureBlockBuilder(callOp);
  }

  rewriter.setInsertionPointToEnd(condBlock);
  auto branchOp = rewriter.create<CondBranchOp>(
      location, conditionI1.getResult(0), continuationBlock, failureBlock);

  rewriter.setInsertionPoint(continuationBlock, opPosition);

  return callOp;
}

emitc::CallOp returnIfError(ConversionPatternRewriter &rewriter,
                            Location location, StringAttr callee,
                            ArrayAttr args, ArrayAttr templateArgs,
                            ArrayRef<Value> operands) {
  auto blockBuilder = [&rewriter, &location](emitc::CallOp &callOp) {
    auto ctx = rewriter.getContext();

    Block *block = rewriter.getBlock();
    mlir::FuncOp funcOp = cast<mlir::FuncOp>(block->getParentOp());

    releaseLocalRefs(rewriter, location, funcOp);

    rewriter.create<mlir::ReturnOp>(location, callOp.getResult(0));
  };

  auto ctx = rewriter.getContext();
  Type type = emitc::OpaqueType::get(ctx, "iree_status_t");
  return failableCall(rewriter, location, type, callee, args, templateArgs,
                      operands, blockBuilder, /*negateResult=*/true);
}

emitc::CallOp failListNull(ConversionPatternRewriter &rewriter,
                           Location location, Type type, StringAttr callee,
                           ArrayAttr args, ArrayAttr templateArgs,
                           ArrayRef<Value> operands) {
  auto blockBuilder = [&rewriter, &location](emitc::CallOp &callOp) {
    auto ctx = rewriter.getContext();

    Block *block = rewriter.getBlock();
    mlir::FuncOp funcOp = cast<mlir::FuncOp>(block->getParentOp());

    releaseLocalRefs(rewriter, location, funcOp);

    auto statusOp = rewriter.create<emitc::CallOp>(
        /*location=*/location,
        /*type=*/mlir::emitc::OpaqueType::get(ctx, "iree_status_t"),
        /*callee=*/StringAttr::get(ctx, "iree_make_status"),
        /*args=*/
        ArrayAttr::get(ctx, {mlir::emitc::OpaqueAttr::get(
                                ctx, "IREE_STATUS_INVALID_ARGUMENT")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    rewriter.create<mlir::ReturnOp>(location, statusOp.getResult(0));
  };

  return failableCall(rewriter, location, type, callee, args, templateArgs,
                      operands, blockBuilder);
}

mlir::CallOp failableCall(
    ConversionPatternRewriter &rewriter, Location location,
    mlir::FuncOp &callee, ArrayRef<Value> operands,
    const std::function<void(mlir::CallOp &)> &failureBlockBuilder,
    bool negateResult = false) {
  auto ctx = rewriter.getContext();

  auto callOp = rewriter.create<mlir::CallOp>(
      /*location=*/location,
      /*callee=*/callee,
      /*operands=*/operands);

  Type boolType = rewriter.getIntegerType(1);

  auto conditionI1 = rewriter.create<emitc::CallOp>(
      /*location=*/location,
      /*type=*/boolType,
      /*callee=*/StringAttr::get(ctx, "EMITC_CAST"),
      /*args=*/
      ArrayAttr::get(ctx, {rewriter.getIndexAttr(0), TypeAttr::get(boolType)}),
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/ArrayRef<Value>{callOp.getResult(0)});

  if (negateResult)
    conditionI1 = rewriter.create<emitc::CallOp>(
        /*location=*/location,
        /*type=*/boolType,
        /*callee=*/StringAttr::get(ctx, "EMITC_NOT"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{conditionI1.getResult(0)});

  // Start by splitting the block into two. The part before will contain the
  // condition, and the part after will contain the continuation point.
  Block *condBlock = rewriter.getInsertionBlock();
  Block::iterator opPosition = rewriter.getInsertionPoint();
  Block *continuationBlock = rewriter.splitBlock(condBlock, opPosition);

  // Create a new block for the target of the failure.
  Block *failureBlock;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Region *parentRegion = condBlock->getParent();
    failureBlock = rewriter.createBlock(parentRegion, parentRegion->end());

    failureBlockBuilder(callOp);
  }

  rewriter.setInsertionPointToEnd(condBlock);
  auto branchOp = rewriter.create<CondBranchOp>(
      location, conditionI1.getResult(0), continuationBlock, failureBlock);

  rewriter.setInsertionPoint(continuationBlock, opPosition);

  return callOp;
}

mlir::CallOp returnIfError(ConversionPatternRewriter &rewriter,
                           Location location, mlir::FuncOp &callee,
                           ArrayRef<Value> operands) {
  auto blockBuilder = [&rewriter, &location](mlir::CallOp &callOp) {
    auto ctx = rewriter.getContext();

    Block *block = rewriter.getBlock();
    mlir::FuncOp funcOp = cast<mlir::FuncOp>(block->getParentOp());

    releaseLocalRefs(rewriter, location, funcOp);

    rewriter.create<mlir::ReturnOp>(location, callOp.getResult(0));
  };

  return failableCall(rewriter, location, callee, operands, blockBuilder,
                      /*negateResult=*/true);
}

SmallVector<Attribute, 4> indexSequence(int64_t n, MLIRContext *ctx) {
  return llvm::to_vector<4>(
      llvm::map_range(llvm::seq<int64_t>(0, n), [&ctx](int64_t i) -> Attribute {
        return IntegerAttr::get(IndexType::get(ctx), i);
      }));
}

template <typename AccessOpTy, typename ResultOpTy>
ResultOpTy lookupSymbolRef(AccessOpTy accessOp, StringRef attrName) {
  FlatSymbolRefAttr globalAttr =
      accessOp.getOperation()->template getAttrOfType<FlatSymbolRefAttr>(
          attrName);
  ResultOpTy globalOp =
      accessOp.getOperation()
          ->template getParentOfType<IREE::VM::ModuleOp>()
          .template lookupSymbol<ResultOpTy>(globalAttr.getValue());
  return globalOp;
}

// Convert vm operations to emitc calls. The resultiong call has the ops
// operands as arguments followed by an argument for every attribute.
template <typename SrcOpTy>
class GenericOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

 public:
  GenericOpConversion(MLIRContext *context, StringRef funcName)
      : OpConversionPattern<SrcOpTy>(context), funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      SrcOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();

    auto type = op.getOperation()->getResultTypes();
    StringAttr callee = StringAttr::get(ctx, funcName);

    // Default to an empty args attribute, which results in the operands being
    // printed as the arguments to the function call.
    ArrayAttr args;
    ArrayAttr templateArgs;

    // If the operation has attributes, we need to explicitely build the args
    // attribute of the emitc call op. This consists of index attributes for
    // the operands, followed by the source op attributes themselves.
    if (op->getAttrs().size() > 0) {
      SmallVector<Attribute, 4> args_ =
          indexSequence(operands.size(), op.getContext());

      for (NamedAttribute attr : op->getAttrs()) {
        args_.push_back(attr.second);
      }

      args = rewriter.getArrayAttr(args_);
    }

    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, type, callee, args,
                                               templateArgs, operands);

    return success();
  }

  StringRef funcName;
};

class CallOpConversion : public OpConversionPattern<IREE::VM::CallOp> {
  using OpConversionPattern<IREE::VM::CallOp>::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::CallOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    mlir::FuncOp funcOp =
        lookupSymbolRef<IREE::VM::CallOp, mlir::FuncOp>(op, "callee");
    IREE::VM::ImportOp importOp =
        lookupSymbolRef<IREE::VM::CallOp, IREE::VM::ImportOp>(op, "callee");

    if (!funcOp && !importOp)
      return op.emitError() << "lookup of callee failed";

    if (funcOp && importOp)
      return op.emitError() << "lookup of callee ambiguous";

    const bool isImported = importOp != nullptr;

    return isImported ? rewriteImportedCall(op, operands, rewriter, importOp)
                      : rewriteInternalCall(op, operands, rewriter, funcOp);
  }

  LogicalResult rewriteInternalCall(IREE::VM::CallOp op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter,
                                    mlir::FuncOp funcOp) const {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    SmallVector<Value, 4> updatedOperands;
    SmallVector<Value, 4> resultOperands;

    auto parentFuncOp = op.getOperation()->getParentOfType<mlir::FuncOp>();

    BlockArgument stackArg = parentFuncOp.getArgument(0);
    BlockArgument moduleArg = parentFuncOp.getArgument(1);
    BlockArgument moduleStateArg = parentFuncOp.getArgument(2);

    updatedOperands = {stackArg, moduleArg, moduleStateArg};

    for (const Value &operand : operands) updatedOperands.push_back(operand);

    // Create a variable for every result and a pointer to it as output
    // parameter to the call.
    for (OpResult result : op.getResults()) {
      auto resultOp = rewriter.create<emitc::ConstantOp>(
          /*location=*/loc,
          /*resultType=*/result.getType(),
          /*value=*/emitc::OpaqueAttr::get(ctx, ""));

      Optional<std::string> cType = getCType(result.getType());
      if (!cType.hasValue()) return op.emitError() << "unable to emit C type";

      std::string cPtrType = cType.getValue() + std::string("*");
      auto resultPtrOp = rewriter.create<emitc::ApplyOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, cPtrType),
          /*applicableOperator=*/StringAttr::get(ctx, "&"),
          /*operand=*/resultOp);

      resultOperands.push_back(resultOp.getResult());
      updatedOperands.push_back(resultPtrOp.getResult());
    }

    auto callOp = returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/funcOp,
        /*operands=*/updatedOperands);

    for (auto &pair : llvm::enumerate(op.getResults())) {
      size_t index = pair.index();
      OpResult result = pair.value();
      result.replaceAllUsesWith(resultOperands[index]);
    }

    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult rewriteImportedCall(IREE::VM::CallOp op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter,
                                    IREE::VM::ImportOp importOp) const {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    SmallVector<Value, 4> updatedOperands;
    SmallVector<Value, 4> resultOperands;

    auto moduleOp =
        importOp.getOperation()->getParentOfType<IREE::VM::ModuleOp>();

    Optional<std::string> funcName = buildFunctionName(moduleOp, importOp);

    if (!funcName.hasValue())
      return op.emitError() << "Couldn't build name to imported function";

    int importOrdinal = importOp.ordinal().getValue().getZExtValue();

    auto funcOp = op.getOperation()->template getParentOfType<mlir::FuncOp>();
    BlockArgument stackArg = funcOp.getArgument(0);
    BlockArgument stateArg = funcOp.getArgument(2);

    auto imports = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_function_t*"),
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "imports")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{stateArg});

    auto import = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_function_t*"),
        /*callee=*/StringAttr::get(ctx, "EMITC_ARRAY_ELEMENT_ADDRESS"),
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                             rewriter.getUI32IntegerAttr(importOrdinal)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{imports.getResult(0)});

    updatedOperands = {stackArg, import.getResult(0)};

    for (const Value &operand : operands) updatedOperands.push_back(operand);

    // Create a variable for every result and a pointer to it as output
    // parameter to the call.
    for (OpResult result : op.getResults()) {
      auto resultOp = rewriter.create<emitc::ConstantOp>(
          /*location=*/loc,
          /*resultType=*/result.getType(),
          /*value=*/emitc::OpaqueAttr::get(ctx, ""));

      Optional<std::string> cType = getCType(result.getType());
      if (!cType.hasValue()) return op.emitError() << "unable to emit C type";

      std::string cPtrType = cType.getValue() + std::string("*");
      auto resultPtrOp = rewriter.create<emitc::ApplyOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, cPtrType),
          /*applicableOperator=*/StringAttr::get(ctx, "&"),
          /*operand=*/resultOp);

      resultOperands.push_back(resultOp.getResult());
      updatedOperands.push_back(resultPtrOp.getResult());
    }

    auto callOp = returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, funcName.getValue()),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/updatedOperands);

    for (auto &pair : llvm::enumerate(op.getResults())) {
      size_t index = pair.index();
      OpResult result = pair.value();
      result.replaceAllUsesWith(resultOperands[index]);
    }

    rewriter.eraseOp(op);

    return success();

    return op.emitError() << "calls to imported function not supported yet";
  }
};

template <typename CmpOpTy>
class CompareRefOpConversion : public OpConversionPattern<CmpOpTy> {
 public:
  using OpConversionPattern<CmpOpTy>::OpConversionPattern;

  CompareRefOpConversion(MLIRContext *context, StringRef funcName,
                         VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<CmpOpTy>(context),
        funcName(funcName),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      CmpOpTy cmpOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = cmpOp.getContext();
    auto loc = cmpOp.getLoc();

    auto funcOp =
        cmpOp.getOperation()->template getParentOfType<mlir::FuncOp>();
    auto ptr = vmAnalysisCache.find(funcOp.getOperation());
    if (ptr == vmAnalysisCache.end()) {
      return cmpOp.emitError() << "parent func op not found in cache.";
    }
    ValueLiveness &valueLiveness = ptr->second.valueLiveness;

    bool moveLhs =
        valueLiveness.isLastValueUse(cmpOp.lhs(), cmpOp.getOperation());
    bool moveRhs =
        valueLiveness.isLastValueUse(cmpOp.rhs(), cmpOp.getOperation());

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        /*op=*/cmpOp,
        /*type=*/cmpOp.getType(),
        /*callee=*/StringAttr::get(ctx, funcName),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    if (moveLhs) {
      rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "iree_vm_ref_release"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{operands[0]});
    }

    // NOTE: If lhs and rhs alias we call release twice on the same argument.
    if (moveRhs) {
      rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "iree_vm_ref_release"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{operands[1]});
    }

    return success();
  }

  StringRef funcName;
  VMAnalysisCache &vmAnalysisCache;
};

class CompareRefNotZeroOpConversion
    : public OpConversionPattern<IREE::VM::CmpNZRefOp> {
  using OpConversionPattern<IREE::VM::CmpNZRefOp>::OpConversionPattern;

 public:
  CompareRefNotZeroOpConversion(MLIRContext *context,
                                VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<IREE::VM::CmpNZRefOp>(context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::CmpNZRefOp cmpOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = cmpOp.getContext();
    auto loc = cmpOp.getLoc();

    auto funcOp = cmpOp.getOperation()->getParentOfType<mlir::FuncOp>();
    auto ptr = vmAnalysisCache.find(funcOp.getOperation());
    if (ptr == vmAnalysisCache.end()) {
      return cmpOp.emitError() << "parent func op not found in cache.";
    }
    ValueLiveness &valueLiveness = ptr->second.valueLiveness;

    bool move =
        valueLiveness.isLastValueUse(cmpOp.operand(), cmpOp.getOperation());

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        /*op=*/cmpOp,
        /*type=*/cmpOp.getType(),
        /*callee=*/StringAttr::get(ctx, "vm_cmp_nz_ref"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    if (move) {
      rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "iree_vm_ref_release"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{operands[0]});
    }

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

template <typename ConstOpTy>
class ConstOpConversion : public OpRewritePattern<ConstOpTy> {
 public:
  using OpRewritePattern<ConstOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstOpTy constOp,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(constOp, constOp.getType(),
                                                   constOp.value());
    return success();
  }
};

template <typename ConstZeroOpTy>
class ConstZeroOpConversion : public OpRewritePattern<ConstZeroOpTy> {
 public:
  using OpRewritePattern<ConstZeroOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstZeroOpTy constZeroOp,
                                PatternRewriter &rewriter) const final {
    auto type = constZeroOp.getType();
    Attribute value;

    if (type.template isa<IntegerType>()) {
      value = rewriter.getIntegerAttr(type, 0);
    } else if (type.template isa<FloatType>()) {
      value = rewriter.getFloatAttr(type, 0.0);
    } else {
      return failure();
    }

    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(constZeroOp, type, value);
    return success();
  }
};

class ConstRefZeroOpConversion
    : public OpRewritePattern<IREE::VM::ConstRefZeroOp> {
 public:
  using OpRewritePattern<IREE::VM::ConstRefZeroOp>::OpRewritePattern;

  ConstRefZeroOpConversion(MLIRContext *context,
                           VMAnalysisCache &vmAnalysisCache)
      : OpRewritePattern<IREE::VM::ConstRefZeroOp>(context),
        vmAnalysisCache(vmAnalysisCache) {}

  LogicalResult matchAndRewrite(IREE::VM::ConstRefZeroOp constRefZeroOp,
                                PatternRewriter &rewriter) const final {
    auto ctx = constRefZeroOp.getContext();
    auto loc = constRefZeroOp.getLoc();

    auto funcOp =
        constRefZeroOp.getOperation()->getParentOfType<mlir::FuncOp>();

    auto ptr = vmAnalysisCache.find(funcOp.getOperation());
    if (ptr == vmAnalysisCache.end()) {
      return constRefZeroOp.emitError() << "parent func op not found in cache.";
    }
    RegisterAllocation &registerAllocation = ptr->second.registerAllocation;

    int32_t ordinal =
        registerAllocation.mapToRegister(constRefZeroOp.getResult()).ordinal();

    emitc::ConstantOp refOp = nullptr;
    for (auto constantOp : funcOp.getOps<emitc::ConstantOp>()) {
      Operation *op = constantOp.getOperation();
      if (!op->hasAttr("ref_ordinal")) continue;
      if (op->getAttr("ref_ordinal")
              .cast<IntegerAttr>()
              .getValue()
              .getZExtValue() == ordinal) {
        refOp = constantOp;
        break;
      }
    }

    if (!refOp)
      return constRefZeroOp.emitError()
             << "Corresponding ref for ordinal '" << ordinal << "' not found";

    auto refPtrOp = rewriter.replaceOpWithNewOp<emitc::ApplyOp>(
        /*op=*/constRefZeroOp,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t*"),
        /*applicableOperator=*/StringAttr::get(ctx, "&"),
        /*operand=*/refOp);

    rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "iree_vm_ref_release"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{constRefZeroOp.result()});
    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

class ConstRefRodataOpConversion
    : public OpConversionPattern<IREE::VM::ConstRefRodataOp> {
 public:
  using OpConversionPattern<IREE::VM::ConstRefRodataOp>::OpConversionPattern;

  ConstRefRodataOpConversion(MLIRContext *context,
                             VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<IREE::VM::ConstRefRodataOp>(context),
        vmAnalysisCache(vmAnalysisCache) {}

  LogicalResult matchAndRewrite(
      IREE::VM::ConstRefRodataOp constRefRodataOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto ctx = constRefRodataOp.getContext();
    auto loc = constRefRodataOp.getLoc();

    auto rodataOp =
        lookupSymbolRef<IREE::VM::ConstRefRodataOp, IREE::VM::RodataOp>(
            constRefRodataOp, "rodata");
    if (!rodataOp) {
      return constRefRodataOp.emitError() << "Unable to find RodataOp";
    }

    auto funcOp = constRefRodataOp.getOperation()
                      ->template getParentOfType<mlir::FuncOp>();

    BlockArgument stateArg = funcOp.getArgument(2);
    auto rodataBuffersPtr = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_buffer_t*"),
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "rodata_buffers")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{stateArg});

    auto byteBufferPtrOp = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_buffer_t*"),
        /*callee=*/StringAttr::get(ctx, "EMITC_ARRAY_ELEMENT_ADDRESS"),
        /*args=*/
        ArrayAttr::get(ctx,
                       {rewriter.getIndexAttr(0),
                        rewriter.getUI32IntegerAttr(static_cast<uint32_t>(
                            rodataOp.ordinal().getValue().getZExtValue()))}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{rodataBuffersPtr.getResult(0)});

    auto typeIdOp = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_buffer_type_id"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    auto ptr = vmAnalysisCache.find(funcOp.getOperation());
    if (ptr == vmAnalysisCache.end()) {
      return constRefRodataOp.emitError()
             << "parent func op not found in cache.";
    }
    RegisterAllocation &registerAllocation = ptr->second.registerAllocation;

    int32_t ordinal =
        registerAllocation.mapToRegister(constRefRodataOp.getResult())
            .ordinal();

    emitc::ConstantOp refOp = nullptr;
    for (auto constantOp : funcOp.getOps<emitc::ConstantOp>()) {
      Operation *op = constantOp.getOperation();
      if (!op->hasAttr("ref_ordinal")) continue;
      if (op->getAttr("ref_ordinal")
              .cast<IntegerAttr>()
              .getValue()
              .getZExtValue() == ordinal) {
        refOp = constantOp;
        break;
      }
    }

    if (!refOp)
      return constRefRodataOp.emitError()
             << "Corresponding ref for ordinal '" << ordinal << "' not found";

    auto refPtrOp = rewriter.replaceOpWithNewOp<emitc::ApplyOp>(
        /*op=*/constRefRodataOp,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t*"),
        /*applicableOperator=*/StringAttr::get(ctx, "&"),
        /*operand=*/refOp);

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_ref_wrap_retain"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{byteBufferPtrOp.getResult(0), typeIdOp.getResult(0),
                        refPtrOp.getResult()});

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

class BranchOpConversion : public OpConversionPattern<IREE::VM::BranchOp> {
  using OpConversionPattern<IREE::VM::BranchOp>::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::BranchOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    Type boolType = rewriter.getIntegerType(1);

    rewriter.replaceOpWithNewOp<mlir::BranchOp>(op, op.dest(),
                                                op.destOperands());

    return success();
  }
};

class CondBranchOpConversion
    : public OpConversionPattern<IREE::VM::CondBranchOp> {
  using OpConversionPattern<IREE::VM::CondBranchOp>::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::CondBranchOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    Type boolType = rewriter.getIntegerType(1);

    auto condition = rewriter.create<IREE::VM::CmpNZI32Op>(
        loc, rewriter.getI32Type(), op.condition());
    auto conditionI1 = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/boolType,
        /*callee=*/StringAttr::get(ctx, "EMITC_CAST"),
        /*args=*/
        ArrayAttr::get(ctx,
                       {rewriter.getIndexAttr(0), TypeAttr::get(boolType)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{condition.getResult()});

    rewriter.replaceOpWithNewOp<mlir::CondBranchOp>(
        op, conditionI1.getResult(0), op.trueDest(), op.trueDestOperands(),
        op.falseDest(), op.falseDestOperands());

    return success();
  }
};

class ReturnOpConversion : public OpConversionPattern<IREE::VM::ReturnOp> {
  using OpConversionPattern<IREE::VM::ReturnOp>::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::ReturnOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    auto funcOp = op.getOperation()->getParentOfType<mlir::FuncOp>();

    releaseLocalRefs(rewriter, loc, funcOp);

    // The result variables are the last N arguments of the function.
    unsigned int firstOutputArgumentIndex =
        funcOp.getNumArguments() - operands.size();

    for (auto &operand : llvm::enumerate(operands)) {
      unsigned int argumentIndex = firstOutputArgumentIndex + operand.index();
      BlockArgument resultArgument = funcOp.getArgument(argumentIndex);

      if (operand.value().getType().isa<IREE::VM::RefType>()) {
        return op.emitError("ref types are not supported as function results.");
      }

      rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "EMITC_DEREF_ASSIGN"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{resultArgument, operand.value()});
    }

    auto status = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
        /*callee=*/StringAttr::get(ctx, "iree_ok_status"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, status.getResult(0));

    return success();
  }
};

class FailOpConversion : public OpConversionPattern<IREE::VM::FailOp> {
  using OpConversionPattern<IREE::VM::FailOp>::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::FailOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    Block *block = rewriter.getInsertionBlock();
    Region *parentRegion = block->getParent();
    Block *passthroughBlock;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      passthroughBlock =
          rewriter.createBlock(parentRegion, parentRegion->end());

      auto funcOp = op.getOperation()->getParentOfType<mlir::FuncOp>();

      releaseLocalRefs(rewriter, loc, funcOp);

      auto status = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
          /*callee=*/StringAttr::get(ctx, "iree_ok_status"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{});

      rewriter.create<mlir::ReturnOp>(loc, status.getResult(0));
    }
    Block *failureBlock;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      failureBlock = rewriter.createBlock(parentRegion, parentRegion->end());

      auto funcOp = op.getOperation()->getParentOfType<mlir::FuncOp>();

      releaseLocalRefs(rewriter, loc, funcOp);

      std::string message = std::string("\"") +
                            op.message().getValueOr("").str() +
                            std::string("\"");

      auto messageOp = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_string_view_t"),
          /*callee=*/StringAttr::get(ctx, "iree_make_cstring_view"),
          /*args=*/
          ArrayAttr::get(ctx, {mlir::emitc::OpaqueAttr::get(ctx, message)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{});

      auto messageSizeOp = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_host_size_t"),
          /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_MEMBER"),
          /*args=*/
          ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                               emitc::OpaqueAttr::get(ctx, "size")}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{messageOp.getResult(0)});

      auto messageSizeIntOp = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "int"),
          /*callee=*/StringAttr::get(ctx, "EMITC_CAST"),
          /*args=*/
          ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                               emitc::OpaqueAttr::get(ctx, "int")}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{messageSizeOp.getResult(0)});

      auto messageDataOp = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "const char*"),
          /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_MEMBER"),
          /*args=*/
          ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                               emitc::OpaqueAttr::get(ctx, "data")}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{messageOp.getResult(0)});

      auto status = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
          /*callee=*/StringAttr::get(ctx, "iree_status_allocate_f"),
          /*args=*/
          ArrayAttr::get(ctx,
                         {mlir::emitc::OpaqueAttr::get(
                              ctx, "IREE_STATUS_FAILED_PRECONDITION"),
                          mlir::emitc::OpaqueAttr::get(ctx, "\"<vm>\""),
                          rewriter.getI32IntegerAttr(0),
                          mlir::emitc::OpaqueAttr::get(ctx, "\"%.*s\""),
                          rewriter.getIndexAttr(0), rewriter.getIndexAttr(1)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{messageSizeIntOp.getResult(0),
                          messageDataOp.getResult(0)});

      rewriter.create<mlir::ReturnOp>(loc, status.getResult(0));
    }

    rewriter.replaceOpWithNewOp<IREE::VM::CondBranchOp>(
        op, op.status(), failureBlock, passthroughBlock);

    return success();
  }
};

template <typename LoadOpTy, typename GlobalOpTy>
class GlobalLoadOpConversion : public OpConversionPattern<LoadOpTy> {
  using OpConversionPattern<LoadOpTy>::OpConversionPattern;

 public:
  GlobalLoadOpConversion(MLIRContext *context, StringRef funcName)
      : OpConversionPattern<LoadOpTy>(context), funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      LoadOpTy loadOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = loadOp.getContext();
    auto loc = loadOp.getLoc();

    GlobalOpTy globalOp =
        lookupSymbolRef<LoadOpTy, GlobalOpTy>(loadOp, "global");
    if (!globalOp) {
      return loadOp.emitError() << "Unable to find GlobalOp";
    }

    auto funcOp =
        loadOp.getOperation()->template getParentOfType<mlir::FuncOp>();

    BlockArgument stateArg = funcOp.getArgument(2);
    auto rwDataPtr = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "uint8_t*"),
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "rwdata")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{stateArg});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        /*op=*/loadOp,
        /*type=*/loadOp.getOperation()->getResultTypes(),
        /*callee=*/StringAttr::get(ctx, funcName),
        /*args=*/
        rewriter.getArrayAttr(
            {rewriter.getIndexAttr(0),
             rewriter.getUI32IntegerAttr(static_cast<uint32_t>(
                 globalOp.ordinal().getValue().getZExtValue()))}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{rwDataPtr.getResult(0)});

    return success();
  }

  StringRef funcName;
};

template <typename StoreOpTy, typename GlobalOpTy>
class GlobalStoreOpConversion : public OpConversionPattern<StoreOpTy> {
  using OpConversionPattern<StoreOpTy>::OpConversionPattern;

 public:
  GlobalStoreOpConversion(MLIRContext *context, StringRef funcName)
      : OpConversionPattern<StoreOpTy>(context), funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      StoreOpTy storeOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = storeOp.getContext();
    auto loc = storeOp.getLoc();

    GlobalOpTy globalOp =
        lookupSymbolRef<StoreOpTy, GlobalOpTy>(storeOp, "global");
    if (!globalOp) {
      return storeOp.emitError() << "Unable to find GlobalOp";
    }

    auto funcOp =
        storeOp.getOperation()->template getParentOfType<mlir::FuncOp>();

    BlockArgument stateArg = funcOp.getArgument(2);
    auto rwDataPtr = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "uint8_t*"),
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "rwdata")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{stateArg});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        /*op=*/storeOp,
        /*type=*/storeOp.getOperation()->getResultTypes(),
        /*callee=*/StringAttr::get(ctx, funcName),
        /*args=*/
        rewriter.getArrayAttr(
            {rewriter.getIndexAttr(0),
             rewriter.getUI32IntegerAttr(static_cast<uint32_t>(
                 globalOp.ordinal().getValue().getZExtValue())),
             rewriter.getIndexAttr(1)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{rwDataPtr.getResult(0), operands[0]});

    return success();
  }

  StringRef funcName;
};

// Convert vm list operations to two emitc calls. The wrapping ref pointer is
// first dereferenced and the result is used as the argument of the specified
// function name.
template <typename SrcOpTy>
class ListOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

 public:
  ListOpConversion(MLIRContext *context, StringRef funcName,
                   size_t listArgumentIndex, bool failable)
      : OpConversionPattern<SrcOpTy>(context),
        funcName(funcName),
        listArgumentIndex(listArgumentIndex),
        failable(failable) {}

 private:
  LogicalResult matchAndRewrite(
      SrcOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    if (listArgumentIndex >= operands.size()) {
      return op.emitError() << " index for list argument out of range";
    }

    Value listOperand = operands[listArgumentIndex];

    // deref
    auto refOp = rewriter.create<emitc::ApplyOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t"),
        /*applicableOperator=*/StringAttr::get(ctx, "*"),
        /*operand=*/listOperand);

    auto listDerefOp = failListNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_list_t*"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_deref"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refOp.getResult()});

    // Replace the one list argument (which is wrapped in a ref) with the
    // unwrapped list.
    SmallVector<Value, 4> updatedOperands;
    for (auto &operand : llvm::enumerate(operands)) {
      if (operand.index() == listArgumentIndex) {
        updatedOperands.push_back(listDerefOp.getResult(0));
      } else {
        updatedOperands.push_back(operand.value());
      }
    }

    if (failable) {
      auto callOp = returnIfError(
          /*rewriter=*/rewriter,
          /*location=*/loc,
          /*callee=*/StringAttr::get(ctx, funcName),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>(updatedOperands));

      rewriter.replaceOp(op, ArrayRef<Value>{});
    } else {
      rewriter.replaceOpWithNewOp<emitc::CallOp>(
          /*op=*/op,
          /*type=*/op.getOperation()->getResultTypes(),
          /*callee=*/StringAttr::get(ctx, funcName),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>(updatedOperands));
    }

    return success();
  }

  StringRef funcName;

  // The index of the list argument. This gets replaced in the conversion.
  size_t listArgumentIndex;

  // Whether the function call can fail, i.e. it returns an iree_status_t.
  bool failable;
};

class ListAllocOpConversion
    : public OpConversionPattern<IREE::VM::ListAllocOp> {
 public:
  using OpConversionPattern<IREE::VM::ListAllocOp>::OpConversionPattern;

  ListAllocOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                        VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<IREE::VM::ListAllocOp>(typeConverter, context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::ListAllocOp allocOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = allocOp.getContext();
    auto loc = allocOp.getLoc();

    Type convertedType = typeConverter->convertType(allocOp.getType());

    if (!convertedType) {
      return allocOp.emitOpError() << "type conversion failed";
    }

    auto elementType = allocOp.getType()
                           .cast<IREE::VM::RefType>()
                           .getObjectType()
                           .cast<IREE::VM::ListType>()
                           .getElementType();

    Optional<emitc::ApplyOp> elementTypePtrOp =
        createVmTypeDefPtr(rewriter, allocOp, elementType);

    if (!elementTypePtrOp.hasValue()) {
      return failure();
    }

    auto listOp = rewriter.create<emitc::ConstantOp>(
        /*location=*/loc,
        /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_list_t*"),
        /*value=*/emitc::OpaqueAttr::get(ctx, "NULL"));

    auto listPtrOp = rewriter.create<emitc::ApplyOp>(
        /*location=*/loc,
        /*result=*/emitc::OpaqueType::get(ctx, "iree_vm_list_t**"),
        /*applicableOperator=*/StringAttr::get(ctx, "&"),
        /*operand=*/listOp.getResult());

    auto funcOp =
        allocOp.getOperation()->template getParentOfType<mlir::FuncOp>();

    BlockArgument stateArg = funcOp.getArgument(2);
    auto allocatorOp = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_allocator_t"),
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "allocator")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{stateArg});

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_create"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{elementTypePtrOp.getValue().getResult(), operands[0],
                        allocatorOp.getResult(0), listPtrOp.getResult()});

    auto ptr = vmAnalysisCache.find(funcOp.getOperation());
    if (ptr == vmAnalysisCache.end()) {
      return allocOp.emitError() << "parent func op not found in cache.";
    }
    RegisterAllocation &registerAllocation = ptr->second.registerAllocation;

    int32_t ordinal =
        registerAllocation.mapToRegister(allocOp.getResult()).ordinal();

    emitc::ConstantOp refOp = nullptr;
    for (auto constantOp : funcOp.getOps<emitc::ConstantOp>()) {
      Operation *op = constantOp.getOperation();
      if (!op->hasAttr("ref_ordinal")) continue;
      if (op->getAttr("ref_ordinal")
              .cast<IntegerAttr>()
              .getValue()
              .getZExtValue() == ordinal) {
        refOp = constantOp;
        break;
      }
    }

    if (!refOp)
      return allocOp.emitError()
             << "Corresponding ref for ordinal '" << ordinal << "' not found";

    auto refPtrOp = rewriter.replaceOpWithNewOp<emitc::ApplyOp>(
        /*op=*/allocOp,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t*"),
        /*applicableOperator=*/StringAttr::get(ctx, "&"),
        /*operand=*/refOp);

    auto refTypeOp = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_type_id"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_ref_wrap_assign"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{listOp.getResult(), refTypeOp.getResult(0),
                        refPtrOp.getResult()});

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

template <typename GetOpTy>
class ListGetOpConversion : public OpConversionPattern<GetOpTy> {
  using OpConversionPattern<GetOpTy>::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      GetOpTy getOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getOp.getContext();
    auto loc = getOp.getLoc();

    Optional<StringRef> valueTypeEnum;
    Optional<StringRef> valueExtractor;

    std::tie(valueTypeEnum, valueExtractor) =
        TypeSwitch<Operation *,
                   std::pair<Optional<StringRef>, Optional<StringRef>>>(
            getOp.getOperation())
            .Case<IREE::VM::ListGetI32Op>([&](auto op) {
              return std::make_pair(StringRef("IREE_VM_VALUE_TYPE_I32"),
                                    StringRef("iree_vm_value_get_i32"));
            })
            .template Case<IREE::VM::ListGetI64Op>([&](auto op) {
              return std::make_pair(StringRef("IREE_VM_VALUE_TYPE_I64"),
                                    StringRef("iree_vm_value_get_i64"));
            })
            .Default([](Operation *) { return std::make_pair(None, None); });

    if (!valueTypeEnum.hasValue() || !valueExtractor.hasValue()) {
      return getOp.emitOpError() << "element type not handled";
    }

    auto valueOp = rewriter.create<emitc::ConstantOp>(
        /*location=*/loc,
        /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_value_t"),
        /*value=*/emitc::OpaqueAttr::get(ctx, ""));

    auto valuePtrOp = rewriter.create<emitc::ApplyOp>(
        /*location=*/loc,
        /*result=*/emitc::OpaqueType::get(ctx, "iree_vm_value_t*"),
        /*applicableOperator=*/StringAttr::get(ctx, "&"),
        /*operand=*/valueOp.getResult());

    auto refOp = rewriter.create<emitc::ApplyOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t"),
        /*applicableOperator=*/StringAttr::get(ctx, "*"),
        /*operand=*/operands[0]);

    auto listDerefOp = failListNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_list_t*"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_deref"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refOp.getResult()});

    auto getValueOp = returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_get_value_as"),
        /*args=*/
        ArrayAttr::get(ctx,
                       {rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
                        emitc::OpaqueAttr::get(ctx, valueTypeEnum.getValue()),
                        rewriter.getIndexAttr(2)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{listDerefOp.getResult(0), getOp.index(),
                        valuePtrOp.getResult()});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        /*op=*/getOp,
        /*type=*/getOp.getType(),
        /*callee=*/StringAttr::get(ctx, valueExtractor.getValue()),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{valuePtrOp.getResult()});

    return success();
  }
};

class ListGetRefOpConversion
    : public OpConversionPattern<IREE::VM::ListGetRefOp> {
 public:
  using OpConversionPattern<IREE::VM::ListGetRefOp>::OpConversionPattern;

  ListGetRefOpConversion(MLIRContext *context, VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<IREE::VM::ListGetRefOp>(context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::ListGetRefOp getOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getOp.getContext();
    auto loc = getOp.getLoc();

    auto listRefOp = rewriter.create<emitc::ApplyOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t"),
        /*applicableOperator=*/StringAttr::get(ctx, "*"),
        /*operand=*/operands[0]);

    auto listDerefOp = failListNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_list_t*"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_deref"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{listRefOp.getResult()});

    auto funcOp = getOp.getOperation()->getParentOfType<mlir::FuncOp>();

    auto ptr = vmAnalysisCache.find(funcOp.getOperation());
    if (ptr == vmAnalysisCache.end()) {
      return getOp.emitError() << "parent func op not found in cache.";
    }
    RegisterAllocation &registerAllocation = ptr->second.registerAllocation;

    int32_t ordinal =
        registerAllocation.mapToRegister(getOp.getResult()).ordinal();

    emitc::ConstantOp refOp = nullptr;
    for (auto constantOp : funcOp.getOps<emitc::ConstantOp>()) {
      Operation *op = constantOp.getOperation();
      if (!op->hasAttr("ref_ordinal")) continue;
      if (op->getAttr("ref_ordinal")
              .cast<IntegerAttr>()
              .getValue()
              .getZExtValue() == ordinal) {
        refOp = constantOp;
        break;
      }
    }

    if (!refOp)
      return getOp.emitError()
             << "Corresponding ref for ordinal '" << ordinal << "' not found";

    auto refPtrOp = rewriter.replaceOpWithNewOp<emitc::ApplyOp>(
        /*op=*/getOp,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t*"),
        /*applicableOperator=*/StringAttr::get(ctx, "&"),
        /*operand=*/refOp);

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_get_ref_retain"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{listDerefOp.getResult(0), getOp.index(),
                        refPtrOp.getResult()});

    Type elementType = getOp.getResult().getType();

    auto elementTypePtrOp = createVmTypeDefPtr(rewriter, getOp, elementType);

    if (!elementTypePtrOp.hasValue()) {
      return failure();
    }

    // Build the following expression:
    // (ref->type != IREE_VM_REF_TYPE_NULL &&
    // (iree_vm_type_def_is_value(type_def) || ref->type != type_def->ref_type))
    emitc::CallOp invalidType;
    {
      auto refType = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/
          emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
          /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
          /*args=*/
          ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                               emitc::OpaqueAttr::get(ctx, "type")}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{refPtrOp.getResult()});

      auto refTypeNull = rewriter.create<emitc::ConstantOp>(
          /*location=*/loc,
          /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
          /*value=*/emitc::OpaqueAttr::get(ctx, "IREE_VM_REF_TYPE_NULL"));

      auto typedefIsValue = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/rewriter.getIntegerType(1),
          /*callee=*/StringAttr::get(ctx, "iree_vm_type_def_is_value"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{elementTypePtrOp.getValue().getResult()});

      auto typedefRefType = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/
          emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
          /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
          /*args=*/
          ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                               emitc::OpaqueAttr::get(ctx, "ref_type")}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{elementTypePtrOp.getValue().getResult()});

      auto refTypeIsNotNull = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/rewriter.getIntegerType(1),
          /*callee=*/StringAttr::get(ctx, "EMITC_NE"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{refType.getResult(0), refTypeNull.getResult()});

      auto refTypesDontMatch = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/rewriter.getIntegerType(1),
          /*callee=*/StringAttr::get(ctx, "EMITC_NE"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{refType.getResult(0), typedefRefType.getResult(0)});

      auto invalidRefType = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/rewriter.getIntegerType(1),
          /*callee=*/StringAttr::get(ctx, "EMITC_OR"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{typedefIsValue.getResult(0),
                          refTypesDontMatch.getResult(0)});

      invalidType = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/rewriter.getIntegerType(1),
          /*callee=*/StringAttr::get(ctx, "EMITC_AND"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{refTypeIsNotNull.getResult(0),
                          invalidRefType.getResult(0)});
    }

    // Start by splitting the block into two. The part before will contain the
    // condition, and the part after will contain the continuation point.
    Block *condBlock = rewriter.getInsertionBlock();
    Block::iterator opPosition = rewriter.getInsertionPoint();
    Block *continuationBlock = rewriter.splitBlock(condBlock, opPosition);

    // Create a new block for the target of the failure.
    Block *failureBlock;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Region *parentRegion = condBlock->getParent();
      failureBlock = rewriter.createBlock(parentRegion, parentRegion->end());

      rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "iree_vm_ref_release"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{refPtrOp.getResult()});

      rewriter.create<mlir::BranchOp>(loc, continuationBlock);
    }

    rewriter.setInsertionPointToEnd(condBlock);
    auto branchOp = rewriter.create<CondBranchOp>(
        loc, invalidType.getResult(0), failureBlock, continuationBlock);

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

template <typename SetOpTy>
class ListSetOpConversion : public OpConversionPattern<SetOpTy> {
  using OpConversionPattern<SetOpTy>::OpConversionPattern;

 private:
  LogicalResult matchAndRewrite(
      SetOpTy setOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = setOp.getContext();
    auto loc = setOp.getLoc();

    Optional<StringRef> valueConstructor =
        TypeSwitch<Operation *, Optional<StringRef>>(setOp.getOperation())
            .Case<IREE::VM::ListSetI32Op>(
                [&](auto op) { return StringRef("iree_vm_value_make_i32"); })
            .template Case<IREE::VM::ListSetI64Op>(
                [&](auto op) { return StringRef("iree_vm_value_make_i64"); })
            .Default([](Operation *) { return None; });

    if (!valueConstructor.hasValue()) {
      return setOp.emitOpError() << " not handled";
    }

    auto valueOp = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_value_t"),
        /*callee=*/StringAttr::get(ctx, valueConstructor.getValue()),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{setOp.value()});

    auto valuePtrOp = rewriter.create<emitc::ApplyOp>(
        /*location=*/loc,
        /*result=*/emitc::OpaqueType::get(ctx, "iree_vm_value_t*"),
        /*applicableOperator=*/StringAttr::get(ctx, "&"),
        /*operand=*/valueOp.getResult(0));

    auto refOp = rewriter.create<emitc::ApplyOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t"),
        /*applicableOperator=*/StringAttr::get(ctx, "*"),
        /*operand=*/operands[0]);

    auto listDerefOp = failListNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_list_t*"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_deref"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refOp.getResult()});

    auto callOp = returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_set_value"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{listDerefOp.getResult(0), setOp.index(),
                        valuePtrOp.getResult()});

    rewriter.eraseOp(setOp);

    return success();
  }
};

class ListSetRefOpConversion
    : public OpConversionPattern<IREE::VM::ListSetRefOp> {
 public:
  using OpConversionPattern<IREE::VM::ListSetRefOp>::OpConversionPattern;

  ListSetRefOpConversion(MLIRContext *context, VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<IREE::VM::ListSetRefOp>(context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::ListSetRefOp setOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = setOp.getContext();
    auto loc = setOp.getLoc();

    auto refOp = rewriter.create<emitc::ApplyOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t"),
        /*applicableOperator=*/StringAttr::get(ctx, "*"),
        /*operand=*/operands[0]);

    auto listDerefOp = failListNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_list_t*"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_deref"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refOp.getResult()});

    auto funcOp = setOp.getOperation()->getParentOfType<mlir::FuncOp>();
    auto ptr = vmAnalysisCache.find(funcOp.getOperation());
    if (ptr == vmAnalysisCache.end()) {
      return setOp.emitError() << "parent func op not found in cache.";
    }
    ValueLiveness &valueLiveness = ptr->second.valueLiveness;

    bool move =
        valueLiveness.isLastValueUse(setOp.value(), setOp.getOperation());

    auto callOp = returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_set_ref_retain"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{listDerefOp.getResult(0), setOp.index(), operands[2]});

    rewriter.eraseOp(setOp);

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};
}  // namespace

void populateVMToEmitCPatterns(MLIRContext *context,
                               IREE::VM::EmitCTypeConverter &typeConverter,
                               OwningRewritePatternList &patterns,
                               VMAnalysisCache &vmAnalysisCache) {
  populatePreserveCompilerHintsPatterns(context, patterns);

  // CFG
  patterns.insert<BranchOpConversion>(context);
  patterns.insert<CallOpConversion>(context);
  patterns.insert<CondBranchOpConversion>(context);
  patterns.insert<FailOpConversion>(context);
  patterns.insert<ReturnOpConversion>(context);

  // Globals
  patterns.insert<
      GlobalLoadOpConversion<IREE::VM::GlobalLoadI32Op, IREE::VM::GlobalI32Op>>(
      context, "vm_global_load_i32");
  patterns.insert<GlobalStoreOpConversion<IREE::VM::GlobalStoreI32Op,
                                          IREE::VM::GlobalI32Op>>(
      context, "vm_global_store_i32");

  // Constants
  patterns.insert<ConstOpConversion<IREE::VM::ConstI32Op>>(context);
  patterns.insert<ConstZeroOpConversion<IREE::VM::ConstI32ZeroOp>>(context);
  patterns.insert<ConstRefZeroOpConversion>(context, vmAnalysisCache);
  patterns.insert<ConstRefRodataOpConversion>(context, vmAnalysisCache);

  // List ops
  patterns.insert<ListAllocOpConversion>(typeConverter, context,
                                         vmAnalysisCache);
  patterns.insert<ListOpConversion<IREE::VM::ListReserveOp>>(
      context, "iree_vm_list_reserve", 0, true);
  patterns.insert<ListOpConversion<IREE::VM::ListResizeOp>>(
      context, "iree_vm_list_resize", 0, true);
  patterns.insert<ListOpConversion<IREE::VM::ListSizeOp>>(
      context, "iree_vm_list_size", 0, false);
  patterns.insert<ListGetOpConversion<IREE::VM::ListGetI32Op>>(context);
  patterns.insert<ListGetRefOpConversion>(context, vmAnalysisCache);
  patterns.insert<ListSetOpConversion<IREE::VM::ListSetI32Op>>(context);
  patterns.insert<ListSetRefOpConversion>(context, vmAnalysisCache);

  // Conditional assignment ops
  patterns.insert<GenericOpConversion<IREE::VM::SelectI32Op>>(context,
                                                              "vm_select_i32");

  // Native integer arithmetic ops
  patterns.insert<GenericOpConversion<IREE::VM::AddI32Op>>(context,
                                                           "vm_add_i32");
  patterns.insert<GenericOpConversion<IREE::VM::SubI32Op>>(context,
                                                           "vm_sub_i32");
  patterns.insert<GenericOpConversion<IREE::VM::MulI32Op>>(context,
                                                           "vm_mul_i32");
  patterns.insert<GenericOpConversion<IREE::VM::DivI32SOp>>(context,
                                                            "vm_div_i32s");
  patterns.insert<GenericOpConversion<IREE::VM::DivI32UOp>>(context,
                                                            "vm_div_i32u");
  patterns.insert<GenericOpConversion<IREE::VM::RemI32SOp>>(context,
                                                            "vm_rem_i32s");
  patterns.insert<GenericOpConversion<IREE::VM::RemI32UOp>>(context,
                                                            "vm_rem_i32u");
  patterns.insert<GenericOpConversion<IREE::VM::FMAI32Op>>(context,
                                                           "vm_fma_i32");
  patterns.insert<GenericOpConversion<IREE::VM::NotI32Op>>(context,
                                                           "vm_not_i32");
  patterns.insert<GenericOpConversion<IREE::VM::AndI32Op>>(context,
                                                           "vm_and_i32");
  patterns.insert<GenericOpConversion<IREE::VM::OrI32Op>>(context, "vm_or_i32");
  patterns.insert<GenericOpConversion<IREE::VM::XorI32Op>>(context,
                                                           "vm_xor_i32");

  // Casting and type conversion/emulation ops
  patterns.insert<GenericOpConversion<IREE::VM::TruncI32I8Op>>(
      context, "vm_trunc_i32i8");
  patterns.insert<GenericOpConversion<IREE::VM::TruncI32I16Op>>(
      context, "vm_trunc_i32i16");
  patterns.insert<GenericOpConversion<IREE::VM::ExtI8I32SOp>>(context,
                                                              "vm_ext_i8i32s");
  patterns.insert<GenericOpConversion<IREE::VM::ExtI8I32UOp>>(context,
                                                              "vm_ext_i8i32u");
  patterns.insert<GenericOpConversion<IREE::VM::ExtI16I32SOp>>(
      context, "vm_ext_i16i32s");
  patterns.insert<GenericOpConversion<IREE::VM::ExtI16I32UOp>>(
      context, "vm_ext_i16i32u");

  // Native bitwise shift and rotate ops
  patterns.insert<GenericOpConversion<IREE::VM::ShlI32Op>>(context,
                                                           "vm_shl_i32");
  patterns.insert<GenericOpConversion<IREE::VM::ShrI32SOp>>(context,
                                                            "vm_shr_i32s");
  patterns.insert<GenericOpConversion<IREE::VM::ShrI32UOp>>(context,
                                                            "vm_shr_i32u");

  // Comparison ops
  patterns.insert<GenericOpConversion<IREE::VM::CmpEQI32Op>>(context,
                                                             "vm_cmp_eq_i32");
  patterns.insert<GenericOpConversion<IREE::VM::CmpNEI32Op>>(context,
                                                             "vm_cmp_ne_i32");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTI32SOp>>(context,
                                                              "vm_cmp_lt_i32s");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTI32UOp>>(context,
                                                              "vm_cmp_lt_i32u");
  patterns.insert<GenericOpConversion<IREE::VM::CmpNZI32Op>>(context,
                                                             "vm_cmp_nz_i32");
  patterns.insert<CompareRefOpConversion<IREE::VM::CmpEQRefOp>>(
      context, "vm_cmp_eq_ref", vmAnalysisCache);
  patterns.insert<CompareRefOpConversion<IREE::VM::CmpNERefOp>>(
      context, "vm_cmp_ne_ref", vmAnalysisCache);
  patterns.insert<CompareRefNotZeroOpConversion>(context, vmAnalysisCache);

  // ExtF32: Globals
  patterns.insert<
      GlobalLoadOpConversion<IREE::VM::GlobalLoadF32Op, IREE::VM::GlobalF32Op>>(
      context, "vm_global_load_f32");
  patterns.insert<GlobalStoreOpConversion<IREE::VM::GlobalStoreF32Op,
                                          IREE::VM::GlobalF32Op>>(
      context, "vm_global_store_f32");

  // ExtF32: Native floating-point constants
  patterns.insert<ConstOpConversion<IREE::VM::ConstF32Op>>(context);
  patterns.insert<ConstZeroOpConversion<IREE::VM::ConstF32ZeroOp>>(context);

  // ExtF32: Conditional assignment
  patterns.insert<GenericOpConversion<IREE::VM::SelectF32Op>>(context,
                                                              "vm_select_f32");

  // ExtF32: Native floating-point arithmetic
  patterns.insert<GenericOpConversion<IREE::VM::AddF32Op>>(context,
                                                           "vm_add_f32");
  patterns.insert<GenericOpConversion<IREE::VM::SubF32Op>>(context,
                                                           "vm_sub_f32");
  patterns.insert<GenericOpConversion<IREE::VM::MulF32Op>>(context,
                                                           "vm_mul_f32");
  patterns.insert<GenericOpConversion<IREE::VM::DivF32Op>>(context,
                                                           "vm_div_f32");
  patterns.insert<GenericOpConversion<IREE::VM::RemF32Op>>(context,
                                                           "vm_rem_f32");
  patterns.insert<GenericOpConversion<IREE::VM::FMAF32Op>>(context,
                                                           "vm_fma_f32");
  patterns.insert<GenericOpConversion<IREE::VM::AbsF32Op>>(context,
                                                           "vm_abs_f32");
  patterns.insert<GenericOpConversion<IREE::VM::NegF32Op>>(context,
                                                           "vm_neg_f32");
  patterns.insert<GenericOpConversion<IREE::VM::CeilF32Op>>(context,
                                                            "vm_ceil_f32");
  patterns.insert<GenericOpConversion<IREE::VM::FloorF32Op>>(context,
                                                             "vm_floor_f32");

  patterns.insert<GenericOpConversion<IREE::VM::AtanF32Op>>(context,
                                                            "vm_atan_f32");
  patterns.insert<GenericOpConversion<IREE::VM::Atan2F32Op>>(context,
                                                             "vm_atan2_f32");
  patterns.insert<GenericOpConversion<IREE::VM::CosF32Op>>(context,
                                                           "vm_cos_f32");
  patterns.insert<GenericOpConversion<IREE::VM::SinF32Op>>(context,
                                                           "vm_sin_f32");
  patterns.insert<GenericOpConversion<IREE::VM::ExpF32Op>>(context,
                                                           "vm_exp_f32");
  patterns.insert<GenericOpConversion<IREE::VM::Exp2F32Op>>(context,
                                                            "vm_exp2_f32");
  patterns.insert<GenericOpConversion<IREE::VM::ExpM1F32Op>>(context,
                                                             "vm_expm1_f32");
  patterns.insert<GenericOpConversion<IREE::VM::LogF32Op>>(context,
                                                           "vm_log_f32");
  patterns.insert<GenericOpConversion<IREE::VM::Log10F32Op>>(context,
                                                             "vm_log10_f32");
  patterns.insert<GenericOpConversion<IREE::VM::Log1pF32Op>>(context,
                                                             "vm_log1p_f32");
  patterns.insert<GenericOpConversion<IREE::VM::Log2F32Op>>(context,
                                                            "vm_log2_f32");
  patterns.insert<GenericOpConversion<IREE::VM::PowF32Op>>(context,
                                                           "vm_pow_f32");
  patterns.insert<GenericOpConversion<IREE::VM::RsqrtF32Op>>(context,
                                                             "vm_rsqrt_f32");
  patterns.insert<GenericOpConversion<IREE::VM::SqrtF32Op>>(context,
                                                            "vm_sqrt_f32");
  patterns.insert<GenericOpConversion<IREE::VM::TanhF32Op>>(context,
                                                            "vm_tanh_f32");

  // ExtF32: Casting and type conversion/emulation
  patterns.insert<GenericOpConversion<IREE::VM::CastSI32F32Op>>(
      context, "vm_cast_si32f32");
  patterns.insert<GenericOpConversion<IREE::VM::CastUI32F32Op>>(
      context, "vm_cast_ui32f32");
  patterns.insert<GenericOpConversion<IREE::VM::CastF32SI32Op>>(
      context, "vm_cast_f32si32");
  patterns.insert<GenericOpConversion<IREE::VM::CastF32UI32Op>>(
      context, "vm_cast_f32ui32");

  // ExtF32: Comparison ops
  patterns.insert<GenericOpConversion<IREE::VM::CmpEQF32OOp>>(context,
                                                              "vm_cmp_eq_f32o");
  patterns.insert<GenericOpConversion<IREE::VM::CmpEQF32UOp>>(context,
                                                              "vm_cmp_eq_f32u");
  patterns.insert<GenericOpConversion<IREE::VM::CmpNEF32OOp>>(context,
                                                              "vm_cmp_ne_f32o");
  patterns.insert<GenericOpConversion<IREE::VM::CmpNEF32UOp>>(context,
                                                              "vm_cmp_ne_f32u");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTF32OOp>>(context,
                                                              "vm_cmp_lt_f32o");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTF32UOp>>(context,
                                                              "vm_cmp_lt_f32u");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTEF32OOp>>(
      context, "vm_cmp_lte_f32o");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTEF32UOp>>(
      context, "vm_cmp_lte_f32u");
  patterns.insert<GenericOpConversion<IREE::VM::CmpNaNF32Op>>(context,
                                                              "vm_cmp_nan_f32");

  // ExtI64: Globals
  patterns.insert<
      GlobalLoadOpConversion<IREE::VM::GlobalLoadI64Op, IREE::VM::GlobalI64Op>>(
      context, "vm_global_load_i64");
  patterns.insert<GlobalStoreOpConversion<IREE::VM::GlobalStoreI64Op,
                                          IREE::VM::GlobalI64Op>>(
      context, "vm_global_store_i64");

  // ExtI64: Constants
  patterns.insert<ConstOpConversion<IREE::VM::ConstI64Op>>(context);
  patterns.insert<ConstZeroOpConversion<IREE::VM::ConstI64ZeroOp>>(context);

  // ExtI64: List ops
  patterns.insert<ListGetOpConversion<IREE::VM::ListGetI64Op>>(context);
  patterns.insert<ListSetOpConversion<IREE::VM::ListSetI64Op>>(context);

  // ExtI64: Conditional assignment ops
  patterns.insert<GenericOpConversion<IREE::VM::SelectI64Op>>(context,
                                                              "vm_select_i64");
  // ExtI64: Native integer arithmetic ops
  patterns.insert<GenericOpConversion<IREE::VM::AddI64Op>>(context,
                                                           "vm_add_i64");
  patterns.insert<GenericOpConversion<IREE::VM::SubI64Op>>(context,
                                                           "vm_sub_i64");
  patterns.insert<GenericOpConversion<IREE::VM::MulI64Op>>(context,
                                                           "vm_mul_i64");
  patterns.insert<GenericOpConversion<IREE::VM::DivI64SOp>>(context,
                                                            "vm_div_i64s");
  patterns.insert<GenericOpConversion<IREE::VM::DivI64UOp>>(context,
                                                            "vm_div_i64u");
  patterns.insert<GenericOpConversion<IREE::VM::RemI64SOp>>(context,
                                                            "vm_rem_i64s");
  patterns.insert<GenericOpConversion<IREE::VM::RemI64UOp>>(context,
                                                            "vm_rem_i64u");
  patterns.insert<GenericOpConversion<IREE::VM::FMAI64Op>>(context,
                                                           "vm_fma_i64");
  patterns.insert<GenericOpConversion<IREE::VM::NotI64Op>>(context,
                                                           "vm_not_i64");
  patterns.insert<GenericOpConversion<IREE::VM::AndI64Op>>(context,
                                                           "vm_and_i64");
  patterns.insert<GenericOpConversion<IREE::VM::OrI64Op>>(context, "vm_or_i64");
  patterns.insert<GenericOpConversion<IREE::VM::XorI64Op>>(context,
                                                           "vm_xor_i64");

  // ExtI64: Casting and type conversion/emulation ops
  patterns.insert<GenericOpConversion<IREE::VM::TruncI64I32Op>>(
      context, "vm_trunc_i64i32");
  patterns.insert<GenericOpConversion<IREE::VM::ExtI32I64SOp>>(
      context, "vm_ext_i32i64s");
  patterns.insert<GenericOpConversion<IREE::VM::ExtI32I64UOp>>(
      context, "vm_ext_i32i64u");

  // ExtI64: Native bitwise shift and rotate ops
  patterns.insert<GenericOpConversion<IREE::VM::ShlI64Op>>(context,
                                                           "vm_shl_i64");
  patterns.insert<GenericOpConversion<IREE::VM::ShrI64SOp>>(context,
                                                            "vm_shr_i64s");
  patterns.insert<GenericOpConversion<IREE::VM::ShrI64UOp>>(context,
                                                            "vm_shr_i64u");

  // ExtI64: Comparison ops
  patterns.insert<GenericOpConversion<IREE::VM::CmpEQI64Op>>(context,
                                                             "vm_cmp_eq_i64");
  patterns.insert<GenericOpConversion<IREE::VM::CmpNEI64Op>>(context,
                                                             "vm_cmp_ne_i64");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTI64SOp>>(context,
                                                              "vm_cmp_lt_i64s");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTI64UOp>>(context,
                                                              "vm_cmp_lt_i64u");
  patterns.insert<GenericOpConversion<IREE::VM::CmpNZI64Op>>(context,
                                                             "vm_cmp_nz_i64");
}

namespace IREE {
namespace VM {

namespace {

// A pass converting IREE VM operations into the EmitC dialect.
class ConvertVMToEmitCPass
    : public PassWrapper<ConvertVMToEmitCPass,
                         OperationPass<IREE::VM::ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::emitc::EmitCDialect, mlir::BuiltinDialect,
                    mlir::StandardOpsDialect, IREE::Util::UtilDialect>();
  }

  StringRef getArgument() const override { return "iree-convert-vm-to-emitc"; }

  StringRef getDescription() const override {
    return "Convert VM Ops to the EmitC dialect";
  }

  void runOnOperation() override {
    IREE::VM::ModuleOp module = getOperation();

    ConversionTarget target(getContext());
    EmitCTypeConverter typeConverter;

    // Run analysis passes
    VMAnalysisCache vmAnalysisCache;

    // Convert vm.func ops to std.func with the calling convntion used by EmitC.
    // We convert these upfront to make sure vm.call ops always reference
    // std.func ops with the correct calling convention during the conversion.
    SmallVector<IREE::VM::FuncOp, 4> funcsToRemove;
    for (auto funcOp : module.getOps<IREE::VM::FuncOp>()) {
      Operation *op = funcOp.getOperation();
      vmAnalysisCache.insert(std::make_pair(
          op, VMAnalysis{RegisterAllocation(op), ValueLiveness(op)}));

      if (failed(convertFuncOp(funcOp, vmAnalysisCache)))
        return signalPassFailure();
      funcsToRemove.push_back(funcOp);
    }

    for (auto &funcOp : funcsToRemove) funcOp.erase();

    OwningRewritePatternList patterns(&getContext());
    populateVMToEmitCPatterns(&getContext(), typeConverter, patterns,
                              vmAnalysisCache);

    target.addLegalDialect<mlir::emitc::EmitCDialect, mlir::BuiltinDialect,
                           mlir::StandardOpsDialect>();

    target.addDynamicallyLegalOp<IREE::Util::DoNotOptimizeOp>(
        [&](IREE::Util::DoNotOptimizeOp op) {
          return typeConverter.isLegal(op.getResultTypes());
        });

    // Structural ops
    target.addLegalOp<IREE::VM::ModuleOp>();
    target.addLegalOp<IREE::VM::ModuleTerminatorOp>();
    target.addLegalOp<IREE::VM::ExportOp>();
    target.addLegalOp<IREE::VM::ImportOp>();

    // Global ops
    target.addLegalOp<IREE::VM::GlobalI32Op>();
    target.addLegalOp<IREE::VM::GlobalI64Op>();
    target.addLegalOp<IREE::VM::GlobalF32Op>();
    target.addLegalOp<IREE::VM::RodataOp>();

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createConvertVMToEmitCPass() {
  return std::make_unique<ConvertVMToEmitCPass>();
}

}  // namespace VM
}  // namespace IREE

static PassRegistration<IREE::VM::ConvertVMToEmitCPass> pass;

}  // namespace iree_compiler
}  // namespace mlir
