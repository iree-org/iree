// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/ConvertVMToEmitC.h"

#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/VMAnalysis.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Utils/CallingConvention.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Math/IR/Math.h"
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

LogicalResult sanitizeSymbolNames(IREE::VM::ModuleOp &moduleOp) {
  auto replaceIllegalChars = [](std::string &symbolName) {
    // replace illegal characters in symbol name with "_"
    std::replace(symbolName.begin(), symbolName.end(), '>', '_');
    std::replace(symbolName.begin(), symbolName.end(), '<', '_');
  };

  std::string symbolName(moduleOp.getName());

  // Sanitize name
  replaceIllegalChars(symbolName);

  if (failed(moduleOp.replaceAllSymbolUses(
          StringAttr::get(moduleOp.getContext(), symbolName), moduleOp)))
    return failure();

  SymbolTable::setSymbolName(moduleOp, symbolName);

  for (auto rodataOp : moduleOp.getOps<IREE::VM::RodataOp>()) {
    std::string symbolName(rodataOp.getName());
    replaceIllegalChars(symbolName);

    // Sanitize name
    if (failed(rodataOp.replaceAllSymbolUses(
            StringAttr::get(moduleOp.getContext(), symbolName), moduleOp)))
      return failure();
    SymbolTable::setSymbolName(rodataOp, symbolName);
  }
  return success();
}

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

  if (type.isa<IREE::VM::RefType>()) {
    return std::string("iree_vm_ref_t*");
  }

  return None;
}

/// Create a call to memset to clear a struct
LogicalResult clearStruct(OpBuilder builder, Value structValue,
                          bool isPointer) {
  auto ctx = structValue.getContext();
  auto loc = structValue.getLoc();

  Type type = structValue.getType();

  if (!type.isa<emitc::OpaqueType>()) {
    return failure();
  }

  Optional<std::string> cType = getCType(type);

  if (!cType.hasValue()) {
    return failure();
  }

  Value structPointerValue;
  Value sizeValue;

  if (isPointer) {
    std::string pointerType = cType.getValue();
    if (pointerType.back() != '*') {
      return failure();
    }

    std::string nonPointerType = pointerType.substr(0, pointerType.size() - 1);

    auto size = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/builder.getI32Type(),
        /*callee=*/StringAttr::get(ctx, "sizeof"),
        /*args=*/
        ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(ctx, nonPointerType)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    structPointerValue = structValue;
    sizeValue = size.getResult(0);
  } else {
    std::string cPtrType = cType.getValue() + "*";

    auto structPointer = builder.create<emitc::ApplyOp>(
        /*location=*/loc,
        /*result=*/emitc::OpaqueType::get(ctx, cPtrType),
        /*applicableOperator=*/StringAttr::get(ctx, "&"),
        /*operand=*/structValue);

    auto size = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/builder.getI32Type(),
        /*callee=*/StringAttr::get(ctx, "sizeof"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{structValue});

    structPointerValue = structPointer.getResult();
    sizeValue = size.getResult(0);
  }

  builder.create<emitc::CallOp>(
      /*location=*/loc,
      /*type=*/TypeRange{},
      /*callee=*/StringAttr::get(ctx, "memset"),
      /*args=*/
      ArrayAttr::get(ctx,
                     {builder.getIndexAttr(0), builder.getUI32IntegerAttr(0),
                      builder.getIndexAttr(1)}),
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/
      ArrayRef<Value>{structPointerValue, sizeValue});

  return success();
}

Optional<Value> findRefByOrdinal(mlir::FuncOp &parentFuncOp,
                                 VMAnalysisCache &vmAnalysisCache,
                                 size_t ordinal) {
  auto ctx = parentFuncOp.getContext();

  // Search block arguments
  int refArgCounter = 0;
  for (BlockArgument arg : parentFuncOp.getArguments()) {
    assert(!arg.getType().isa<IREE::VM::RefType>());

    if (arg.getType() == emitc::OpaqueType::get(ctx, "iree_vm_ref_t*")) {
      if (ordinal == refArgCounter++) {
        return arg;
      }
    }
  }

  auto ptr = vmAnalysisCache.find(parentFuncOp.getOperation());
  if (ptr == vmAnalysisCache.end()) {
    return None;
  }

  emitc::ApplyOp applyOp = ptr->second.lookupLocalRef(ordinal);
  return applyOp.getResult();
}

/// Find a local ref of type `iree_vm_ref_t*` matching the ordinal of the given
/// `IREE::VM::Ref` value.
Optional<Value> findRef(mlir::FuncOp &parentFuncOp,
                        VMAnalysisCache &vmAnalysisCache, Value refResult) {
  assert(refResult.getType().isa<IREE::VM::RefType>());

  auto ptr = vmAnalysisCache.find(parentFuncOp.getOperation());
  if (ptr == vmAnalysisCache.end()) {
    parentFuncOp.emitError() << "parent func op not found in cache.";
    return None;
  }

  int32_t ordinal = ptr->second.getRefRegisterOrdinal(refResult);

  Optional<Value> result =
      findRefByOrdinal(parentFuncOp, vmAnalysisCache, ordinal);

  return result;
}

LogicalResult convertFuncOp(IREE::VM::FuncOp funcOp,
                            IREE::VM::EmitCTypeConverter &typeConverter,
                            VMAnalysisCache &vmAnalysisCache,
                            SmallVector<BlockArgument, 4> &blockArgsToRemove) {
  auto ctx = funcOp.getContext();
  auto loc = funcOp.getLoc();

  OpBuilder builder(funcOp);

  auto moduleOp = funcOp.getOperation()->getParentOfType<IREE::VM::ModuleOp>();

  FunctionType funcType = funcOp.getType();
  std::string name =
      std::string(moduleOp.getName()) + "_" + std::string(funcOp.getName());
  std::string moduleTypeName = (moduleOp.getName() + "_t*").str();
  std::string moduleStateTypeName = (moduleOp.getName() + "_state_t*").str();

  Type stackType = emitc::OpaqueType::get(ctx, "iree_vm_stack_t*");
  Type moduleType = emitc::OpaqueType::get(ctx, moduleTypeName);
  Type moduleStateType = emitc::OpaqueType::get(ctx, moduleStateTypeName);

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
    std::string cPtrType;
    // We pass refs as iree_vm_ref_t* regardless of whether it is an in or out
    // parameter
    if (resultType.isa<IREE::VM::RefType>()) {
      cPtrType = cType.getValue();
    } else {
      cPtrType = cType.getValue() + std::string("*");
    }
    Type type = emitc::OpaqueType::get(ctx, cPtrType);
    inputTypes.push_back(type);
    outputTypes.push_back(type);
  }

  auto newFuncType = mlir::FunctionType::get(
      ctx, {inputTypes}, {emitc::OpaqueType::get(ctx, "iree_status_t")});

  auto newFuncOp = builder.create<mlir::FuncOp>(loc, name, newFuncType);

  newFuncOp.getOperation()->setAttr("emitc.static", UnitAttr::get(ctx));

  Optional<std::string> callingConvention = makeCallingConventionString(funcOp);

  // Annotate new function with calling convention string which gets used in
  // the CModuleTarget.
  newFuncOp.getOperation()->setAttr(
      "vm.calling_convention",
      StringAttr::get(ctx, callingConvention.getValue()));

  // This call shold be equivalent to rewriter.inlineRegionBefore()
  newFuncOp.getBody().getBlocks().splice(newFuncOp.end(),
                                         funcOp.getBody().getBlocks());

  Block &entryBlock = newFuncOp.getBlocks().front();

  if (!entryBlock.hasNoPredecessors()) {
    return funcOp.emitError()
           << "branches to the entry block are not supported for now.";
  }

  entryBlock.insertArgument(static_cast<unsigned>(0), stackType);
  entryBlock.insertArgument(static_cast<unsigned>(1), moduleType);
  entryBlock.insertArgument(static_cast<unsigned>(2), moduleStateType);

  entryBlock.addArguments(outputTypes);

  auto ptr = vmAnalysisCache.find(funcOp.getOperation());
  if (ptr == vmAnalysisCache.end()) {
    return funcOp.emitError() << "parent func op not found in cache.";
  }

  vmAnalysisCache.insert(
      std::make_pair(newFuncOp.getOperation(), std::move(ptr->second)));

  // ptr gets invalidated, reset it
  ptr = vmAnalysisCache.find(newFuncOp.getOperation());
  if (ptr == vmAnalysisCache.end()) {
    return funcOp.emitError()
           << "newly created mlir::FuncOp not found in cache.";
  }

  // Add constant ops for local refs
  const int numRefArgs = llvm::count_if(inputTypes, [](Type inputType) {
    return inputType.isa<IREE::VM::RefType>();
  });
  const int numLocalRefs = ptr->second.getNumRefRegisters() - numRefArgs;

  builder.setInsertionPointToStart(&entryBlock);

  ptr->second.numRefArguments = numRefArgs;

  for (int i = 0; i < numLocalRefs; i++) {
    auto refOp = builder.create<emitc::ConstantOp>(
        /*location=*/loc,
        /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t"),
        /*value=*/emitc::OpaqueAttr::get(ctx, ""));

    auto refPtrOp = builder.create<emitc::ApplyOp>(
        /*location=*/loc,
        /*result=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t*"),
        /*applicableOperator=*/StringAttr::get(ctx, "&"),
        /*operand=*/refOp.getResult());

    // Cache local refs so that we can release them before a return operation.
    // Here we rely on the fact that the register allocation maps arguments in
    // the first slots.
    ptr->second.cacheLocalRef(i + numRefArgs, refPtrOp);

    if (failed(
            clearStruct(builder, refPtrOp.getResult(), /*isPointer=*/true))) {
      return failure();
    }
  }

  for (Block &block : llvm::drop_begin(newFuncOp.getBlocks(), 1)) {
    for (BlockArgument blockArg : block.getArguments()) {
      if (!blockArg.getType().isa<IREE::VM::RefType>()) {
        continue;
      }
      blockArgsToRemove.push_back(blockArg);
    }
  }

  if (failed(
          funcOp.replaceAllSymbolUses(builder.getStringAttr(name), moduleOp)))
    return funcOp.emitError() << "unable to update symbol name in module";

  return success();
}

/// Remove block arguments
LogicalResult removeBlockArguments(
    IREE::VM::ModuleOp moduleOp,
    SmallVector<BlockArgument, 4> &blockArgsToRemove) {
  for (auto &blockArg : blockArgsToRemove) {
    assert(blockArg.getType().isa<IREE::VM::RefType>());
    assert(blockArg.use_empty());
    Block *block = blockArg.getOwner();

    for (auto pred : block->getPredecessors()) {
      auto terminator = pred->getTerminator();
      if (auto branchOp = dyn_cast<mlir::BranchOp>(terminator)) {
        branchOp.eraseOperand(blockArg.getArgNumber());
      } else if (auto condBranchOp = dyn_cast<mlir::CondBranchOp>(terminator)) {
        if (condBranchOp.getTrueDest() == block) {
          condBranchOp.eraseTrueOperand(blockArg.getArgNumber());
        } else {
          condBranchOp.eraseFalseOperand(blockArg.getArgNumber());
        }
      }
    }
    block->eraseArgument(blockArg.getArgNumber());
  }

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

Optional<emitc::ApplyOp> createVmTypeDefPtr(ConversionPatternRewriter &rewriter,
                                            Operation *srcOp,
                                            Type elementType) {
  auto ctx = srcOp->getContext();
  auto loc = srcOp->getLoc();

  // Map from type to enum values of type iree_vm_value_type_t and
  // iree_vm_ref_type_t
  mlir::DenseMap<Type, std::pair<std::string, std::string>> valueTypeMap = {
      {IntegerType::get(ctx, 8),
       {"IREE_VM_VALUE_TYPE_I8", "IREE_VM_REF_TYPE_NULL"}},
      {IntegerType::get(ctx, 16),
       {"IREE_VM_VALUE_TYPE_I16", "IREE_VM_REF_TYPE_NULL"}},
      {IntegerType::get(ctx, 32),
       {"IREE_VM_VALUE_TYPE_I32", "IREE_VM_REF_TYPE_NULL"}},
      {IntegerType::get(ctx, 64),
       {"IREE_VM_VALUE_TYPE_I64", "IREE_VM_REF_TYPE_NULL"}},
      {Float32Type::get(ctx),
       {"IREE_VM_VALUE_TYPE_F32", "IREE_VM_REF_TYPE_NULL"}},
      {Float64Type::get(ctx),
       {"IREE_VM_VALUE_TYPE_F64", "IREE_VM_REF_TYPE_NULL"}},
      {IREE::VM::OpaqueType::get(ctx),
       {"IREE_VM_VALUE_TYPE_NONE", "IREE_VM_REF_TYPE_NULL"}},
  };

  auto elementTypeOp = rewriter.create<emitc::ConstantOp>(
      /*location=*/loc,
      /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_type_def_t"),
      /*value=*/emitc::OpaqueAttr::get(ctx, ""));

  if (failed(clearStruct(rewriter, elementTypeOp.getResult(),
                         /*isPointer=*/false))) {
    return None;
  }

  auto ptr = valueTypeMap.find((elementType));
  if (ptr != valueTypeMap.end()) {
    rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_MEMBER_ASSIGN"),
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "value_type"),
                             emitc::OpaqueAttr::get(ctx, ptr->second.first)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{elementTypeOp.getResult()});

    rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_MEMBER_ASSIGN"),
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "ref_type"),
                             emitc::OpaqueAttr::get(ctx, ptr->second.second)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{elementTypeOp.getResult()});
  } else {
    if (!elementType.isa<IREE::VM::RefType>()) {
      return None;
    }
    Type objType = elementType.cast<IREE::VM::RefType>().getObjectType();

    std::string typeName;

    if (objType.isa<IREE::VM::ListType>()) {
      typeName = "!vm.list";
    } else {
      llvm::raw_string_ostream sstream(typeName);
      objType.print(sstream);
      sstream.flush();
    }

    // Remove leading '!' and wrap in quotes
    typeName = std::string("\"") + typeName.substr(1) + std::string("\"");

    auto typeNameCStringView = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_string_view_t"),
        /*callee=*/StringAttr::get(ctx, "iree_make_cstring_view"),
        /*args=*/ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(ctx, typeName)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    auto typeDescriptor = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/
        emitc::OpaqueType::get(ctx, "const iree_vm_ref_type_descriptor_t*"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_ref_lookup_registered_type"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{typeNameCStringView.getResult(0)});

    // TODDO(simon-camp) typeDescriptor might be NULL
    auto typeDescriptorType = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "type")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{typeDescriptor.getResult(0)});

    rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_MEMBER_ASSIGN"),
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "ref_type"),
                             rewriter.getIndexAttr(1)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{elementTypeOp.getResult(),
                        typeDescriptorType.getResult(0)});
  }

  auto elementTypePtrOp = rewriter.create<emitc::ApplyOp>(
      /*location=*/loc,
      /*result=*/emitc::OpaqueType::get(ctx, "iree_vm_type_def_t*"),
      /*applicableOperator=*/StringAttr::get(ctx, "&"),
      /*operand=*/elementTypeOp.getResult());

  return elementTypePtrOp;
}

/// Releases refs which are local to the function as well as ref arguments.
void releaseRefs(OpBuilder &builder, Location location, mlir::FuncOp funcOp,
                 VMAnalysisCache &vmAnalysisCache) {
  auto ctx = builder.getContext();

  auto ptr = vmAnalysisCache.find(funcOp.getOperation());
  assert(ptr != vmAnalysisCache.end());

  auto &localRefs = ptr->second.localRefs();
  for (auto pair : localRefs) {
    Operation *op = pair.second;

    assert(isa<emitc::ApplyOp>(op));

    Value localRef = cast<emitc::ApplyOp>(op).getResult();

    builder.create<emitc::CallOp>(
        /*location=*/location,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "iree_vm_ref_release"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{localRef});
  }

  // We only release the original arguments not the results which were appended
  // as further operands.
  size_t refArgumentsReleased = 0;
  for (auto arg : funcOp.getArguments()) {
    if (arg.getType() == emitc::OpaqueType::get(ctx, "iree_vm_ref_t*")) {
      if (ptr->second.numRefArguments <= refArgumentsReleased++) {
        break;
      }
      builder.create<emitc::CallOp>(
          /*location=*/location,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "iree_vm_ref_release"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{arg});
    }
  }
}

/// Generate an emitc.call op with one result and split the current block into a
/// continuation and failure block based on the truthiness of the result
/// value, i.e. a truthy value branches to the continuation block when
/// `negateCondition` is false.
emitc::CallOp failableCall(
    OpBuilder &builder, Location location, Type type, StringAttr callee,
    ArrayAttr args, ArrayAttr templateArgs, ArrayRef<Value> operands,
    const std::function<void(emitc::CallOp &)> &failureBlockBuilder,
    bool negateCondition = false) {
  auto ctx = builder.getContext();

  auto callOp = builder.create<emitc::CallOp>(
      /*location=*/location,
      /*type=*/type,
      /*callee=*/callee,
      /*args=*/args,
      /*templateArgs=*/templateArgs,
      /*operands=*/operands);

  Type boolType = builder.getIntegerType(1);

  auto conditionI1 = builder.create<emitc::CallOp>(
      /*location=*/location,
      /*type=*/boolType,
      /*callee=*/StringAttr::get(ctx, "EMITC_CAST"),
      /*args=*/
      ArrayAttr::get(ctx, {builder.getIndexAttr(0), TypeAttr::get(boolType)}),
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/ArrayRef<Value>{callOp.getResult(0)});

  // Start by splitting the block into two. The part before will contain the
  // condition, and the part after will contain the continuation point.
  Block *condBlock = builder.getInsertionBlock();
  Block::iterator opPosition = builder.getInsertionPoint();
  Block *continuationBlock = condBlock->splitBlock(opPosition);

  // Create a new block for the target of the failure.
  Block *failureBlock;
  {
    OpBuilder::InsertionGuard guard(builder);
    Region *parentRegion = condBlock->getParent();
    failureBlock = builder.createBlock(parentRegion, parentRegion->end());

    failureBlockBuilder(callOp);
  }

  builder.setInsertionPointToEnd(condBlock);
  builder.create<IREE::VM::CondBranchOp>(
      location, conditionI1.getResult(0),
      negateCondition ? failureBlock : continuationBlock,
      negateCondition ? continuationBlock : failureBlock);

  builder.setInsertionPointToStart(continuationBlock);

  return callOp;
}

emitc::CallOp returnIfError(OpBuilder &builder, Location location,
                            StringAttr callee, ArrayAttr args,
                            ArrayAttr templateArgs, ArrayRef<Value> operands,
                            VMAnalysisCache &vmAnalysisCache) {
  auto blockBuilder = [&builder, &location,
                       &vmAnalysisCache](emitc::CallOp &callOp) {
    Block *block = builder.getBlock();
    mlir::FuncOp funcOp = cast<mlir::FuncOp>(block->getParentOp());

    releaseRefs(builder, location, funcOp, vmAnalysisCache);

    builder.create<mlir::ReturnOp>(location, callOp.getResult(0));
  };

  auto ctx = builder.getContext();
  Type type = emitc::OpaqueType::get(ctx, "iree_status_t");
  return failableCall(builder, location, type, callee, args, templateArgs,
                      operands, blockBuilder, /*negateResult=*/true);
}

emitc::CallOp failListNull(OpBuilder &builder, Location location, Type type,
                           StringAttr callee, ArrayAttr args,
                           ArrayAttr templateArgs, ArrayRef<Value> operands,
                           VMAnalysisCache &vmAnalysisCache) {
  auto blockBuilder = [&builder, &location,
                       &vmAnalysisCache](emitc::CallOp &callOp) {
    auto ctx = builder.getContext();

    Block *block = builder.getBlock();
    mlir::FuncOp funcOp = cast<mlir::FuncOp>(block->getParentOp());

    releaseRefs(builder, location, funcOp, vmAnalysisCache);

    auto statusOp = builder.create<emitc::CallOp>(
        /*location=*/location,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
        /*callee=*/StringAttr::get(ctx, "iree_make_status"),
        /*args=*/
        ArrayAttr::get(
            ctx, {emitc::OpaqueAttr::get(ctx, "IREE_STATUS_INVALID_ARGUMENT")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    builder.create<mlir::ReturnOp>(location, statusOp.getResult(0));
  };

  return failableCall(builder, location, type, callee, args, templateArgs,
                      operands, blockBuilder);
}

/// Generate a mlir.call op with one result and split the current block into a
/// continuation and failure block based on the truthiness of the result
/// value, i.e. a truthy value branches to the continuation block when
/// `negateCondition` is false.
mlir::CallOp failableCall(
    OpBuilder &builder, Location location, mlir::FuncOp &callee,
    ArrayRef<Value> operands,
    const std::function<void(mlir::CallOp &)> &failureBlockBuilder,
    bool negateCondition = false) {
  auto ctx = builder.getContext();

  auto callOp = builder.create<mlir::CallOp>(
      /*location=*/location,
      /*callee=*/callee,
      /*operands=*/operands);

  Type boolType = builder.getIntegerType(1);

  auto conditionI1 = builder.create<emitc::CallOp>(
      /*location=*/location,
      /*type=*/boolType,
      /*callee=*/StringAttr::get(ctx, "EMITC_CAST"),
      /*args=*/
      ArrayAttr::get(ctx, {builder.getIndexAttr(0), TypeAttr::get(boolType)}),
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/ArrayRef<Value>{callOp.getResult(0)});

  // Start by splitting the block into two. The part before will contain the
  // condition, and the part after will contain the continuation point.
  Block *condBlock = builder.getInsertionBlock();
  Block::iterator opPosition = builder.getInsertionPoint();
  Block *continuationBlock = condBlock->splitBlock(opPosition);

  // Create a new block for the target of the failure.
  Block *failureBlock;
  {
    OpBuilder::InsertionGuard guard(builder);
    Region *parentRegion = condBlock->getParent();
    failureBlock = builder.createBlock(parentRegion, parentRegion->end());

    failureBlockBuilder(callOp);
  }

  builder.setInsertionPointToEnd(condBlock);
  builder.create<IREE::VM::CondBranchOp>(
      location, conditionI1.getResult(0),
      negateCondition ? failureBlock : continuationBlock,
      negateCondition ? continuationBlock : failureBlock);

  builder.setInsertionPoint(continuationBlock, opPosition);

  return callOp;
}

mlir::CallOp returnIfError(OpBuilder &builder, Location location,
                           mlir::FuncOp &callee, ArrayRef<Value> operands,
                           VMAnalysisCache &vmAnalysisCache) {
  auto blockBuilder = [&builder, &location,
                       &vmAnalysisCache](mlir::CallOp &callOp) {
    Block *block = builder.getBlock();
    mlir::FuncOp funcOp = cast<mlir::FuncOp>(block->getParentOp());

    releaseRefs(builder, location, funcOp, vmAnalysisCache);

    builder.create<mlir::ReturnOp>(location, callOp.getResult(0));
  };

  return failableCall(builder, location, callee, operands, blockBuilder,
                      /*negateResult=*/true);
}

LogicalResult createAPIFunctions(IREE::VM::ModuleOp moduleOp,
                                 VMAnalysisCache &vmAnalysisCache) {
  auto ctx = moduleOp.getContext();
  auto loc = moduleOp.getLoc();

  OpBuilder builder(moduleOp);
  builder.setInsertionPoint(moduleOp.getBody()->getTerminator());

  std::string moduleName{moduleOp.getName()};

  // destroy
  {
    OpBuilder::InsertionGuard guard(builder);

    auto funcType = mlir::FunctionType::get(
        ctx, {emitc::OpaqueType::get(ctx, "void*")}, {});

    auto funcOp =
        builder.create<mlir::FuncOp>(loc, moduleName + "_destroy", funcType);

    vmAnalysisCache.insert(
        std::make_pair(funcOp.getOperation(), VMAnalysis{{}, {}}));

    funcOp.getOperation()->setAttr("emitc.static", UnitAttr::get(ctx));

    Block *entryBlock = funcOp.addEntryBlock();

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleTypeName = moduleName + "_t*";

    auto castedModuleOp = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, moduleTypeName),
        /*callee=*/StringAttr::get(ctx, "EMITC_CAST"),
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, moduleTypeName)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{funcOp.getArgument(0)});

    auto allocatorOp = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_allocator_t"),
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "allocator")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{castedModuleOp.getResult(0)});

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "iree_allocator_free"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{allocatorOp.getResult(0), castedModuleOp.getResult(0)});

    builder.create<mlir::ReturnOp>(loc);
  }

  // alloc_state
  {
    OpBuilder::InsertionGuard guard(builder);

    auto funcType = mlir::FunctionType::get(
        ctx,
        {emitc::OpaqueType::get(ctx, "void*"),
         emitc::OpaqueType::get(ctx, "iree_allocator_t"),
         emitc::OpaqueType::get(ctx, "iree_vm_module_state_t**")},
        {emitc::OpaqueType::get(ctx, "iree_status_t")});

    auto funcOp = builder.create<mlir::FuncOp>(loc, moduleName + "_alloc_state",
                                               funcType);

    vmAnalysisCache.insert(
        std::make_pair(funcOp.getOperation(), VMAnalysis{{}, {}}));

    funcOp.getOperation()->setAttr("emitc.static", UnitAttr::get(ctx));

    Block *entryBlock = funcOp.addEntryBlock();

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleStateTypeName = moduleName + "_state_t*";

    auto stateOp = builder.create<emitc::ConstantOp>(
        /*location=*/loc,
        /*resultType=*/emitc::OpaqueType::get(ctx, moduleStateTypeName),
        /*value=*/emitc::OpaqueAttr::get(ctx, "NULL"));

    auto stateSize = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_host_size_t"),
        /*callee=*/StringAttr::get(ctx, "sizeof"),
        /*args=*/
        ArrayAttr::get(ctx,
                       {emitc::OpaqueAttr::get(ctx, moduleName + "_state_t")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    auto statePtr = builder.create<emitc::ApplyOp>(
        /*location=*/loc,
        /*result=*/emitc::OpaqueType::get(ctx, moduleStateTypeName + "*"),
        /*applicableOperator=*/StringAttr::get(ctx, "&"),
        /*operand=*/stateOp.getResult());

    auto voidPtr = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "void**"),
        /*callee=*/StringAttr::get(ctx, "EMITC_CAST"),
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "void**")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{statePtr.getResult()});

    returnIfError(
        builder, loc, StringAttr::get(ctx, "iree_allocator_malloc"), {}, {},
        {funcOp.getArgument(1), stateSize.getResult(0), voidPtr.getResult(0)},
        /*vmAnalysisCache=*/vmAnalysisCache);

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "memset"),
        /*args=*/
        ArrayAttr::get(ctx,
                       {builder.getIndexAttr(0), builder.getUI32IntegerAttr(0),
                        builder.getIndexAttr(1)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{stateOp.getResult(), stateSize.getResult(0)});

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER_ASSIGN"),
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "allocator"),
                             builder.getIndexAttr(1)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{stateOp.getResult(), funcOp.getArgument(1)});

    // Initialize buffers
    for (auto rodataOp : moduleOp.getOps<IREE::VM::RodataOp>()) {
      auto ordinal = rodataOp.ordinal().getValue().getZExtValue();

      std::string bufferName = moduleName + "_" + rodataOp.getName().str();

      auto bufferVoid = builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "void*"),
          /*callee=*/StringAttr::get(ctx, "EMITC_CAST"),
          /*args=*/
          ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(ctx, bufferName),
                               emitc::OpaqueAttr::get(ctx, "void*")}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{});

      auto bufferSize = builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_host_size_t"),
          /*callee=*/StringAttr::get(ctx, "sizeof"),
          /*args=*/
          ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(ctx, bufferName)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{});

      auto byteSpan = builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_byte_span_t"),
          /*callee=*/StringAttr::get(ctx, "iree_make_byte_span"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{bufferVoid.getResult(0), bufferSize.getResult(0)});

      auto allocator = builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_allocator_t"),
          /*callee=*/StringAttr::get(ctx, "iree_allocator_null"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{});

      auto buffers = builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_buffer_t*"),
          /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
          /*args=*/
          ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                               emitc::OpaqueAttr::get(ctx, "rodata_buffers")}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{stateOp.getResult()});

      auto buffer = builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_buffer_t*"),
          /*callee=*/StringAttr::get(ctx, "EMITC_ARRAY_ELEMENT_ADDRESS"),
          /*args=*/
          ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                               builder.getUI32IntegerAttr(ordinal)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{buffers.getResult(0)});

      builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "iree_vm_buffer_initialize"),
          /*args=*/
          ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(
                                   ctx, "IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE"),
                               builder.getIndexAttr(0), builder.getIndexAttr(1),
                               builder.getIndexAttr(2)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{byteSpan.getResult(0), allocator.getResult(0),
                          buffer.getResult(0)});
    }

    // Zero out refs from state struct.
    auto ordinal_counts = moduleOp.ordinal_counts();

    if (!ordinal_counts.hasValue()) {
      return moduleOp.emitError()
             << "ordinal_counts attribute not found. The OrdinalAllocationPass "
                "must be run before.";
    }

    const int numGlobalRefs = ordinal_counts.getValue().global_refs();

    auto refs = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t*"),
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "refs")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{stateOp.getResult()});

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/builder.getI32Type(),
        /*callee=*/StringAttr::get(ctx, "sizeof"),
        /*args=*/
        ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(ctx, "iree_vm_ref_t")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    for (int i = 0; i < numGlobalRefs; i++) {
      auto refPtrOp = builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t*"),
          /*callee=*/StringAttr::get(ctx, "EMITC_ARRAY_ELEMENT_ADDRESS"),
          /*args=*/
          ArrayAttr::get(
              ctx, {builder.getIndexAttr(0), builder.getUI32IntegerAttr(i)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{refs.getResult(0)});

      if (failed(clearStruct(builder, refPtrOp.getResult(0),
                             /*isPointer=*/true))) {
        return failure();
      }
    }

    auto baseStateOp = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_module_state_t*"),
        /*callee=*/StringAttr::get(ctx, "EMITC_CAST"),
        /*args=*/
        ArrayAttr::get(
            ctx, {builder.getIndexAttr(0),
                  emitc::OpaqueAttr::get(ctx, "iree_vm_module_state_t*")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{stateOp.getResult()});

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "EMITC_DEREF_ASSIGN_VALUE"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{funcOp.getArgument(2), baseStateOp.getResult(0)});

    auto status = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
        /*callee=*/StringAttr::get(ctx, "iree_ok_status"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    builder.create<mlir::ReturnOp>(loc, status.getResult(0));
  }

  // free_state
  {
    OpBuilder::InsertionGuard guard(builder);

    auto funcType = mlir::FunctionType::get(
        ctx,
        {emitc::OpaqueType::get(ctx, "void*"),
         emitc::OpaqueType::get(ctx, "iree_vm_module_state_t*")},
        {});

    auto funcOp =
        builder.create<mlir::FuncOp>(loc, moduleName + "_free_state", funcType);

    vmAnalysisCache.insert(
        std::make_pair(funcOp.getOperation(), VMAnalysis{{}, {}}));

    funcOp.getOperation()->setAttr("emitc.static", UnitAttr::get(ctx));

    Block *entryBlock = funcOp.addEntryBlock();

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleStateTypeName = moduleName + "_state_t*";

    auto stateOp = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, moduleStateTypeName),
        /*callee=*/StringAttr::get(ctx, "EMITC_CAST"),
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, moduleStateTypeName)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{funcOp.getArgument(1)});

    // Release refs from state struct.
    auto ordinal_counts = moduleOp.ordinal_counts();

    if (!ordinal_counts.hasValue()) {
      return moduleOp.emitError()
             << "ordinal_counts attribute not found. The OrdinalAllocationPass "
                "must be run before.";
    }
    const int numGlobalRefs = ordinal_counts.getValue().global_refs();

    auto refs = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t*"),
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "refs")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{stateOp.getResult(0)});

    for (int i = 0; i < numGlobalRefs; i++) {
      auto refPtrOp = builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t*"),
          /*callee=*/StringAttr::get(ctx, "EMITC_ARRAY_ELEMENT_ADDRESS"),
          /*args=*/
          ArrayAttr::get(
              ctx, {builder.getIndexAttr(0), builder.getUI32IntegerAttr(i)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{refs.getResult(0)});

      builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "iree_vm_ref_release"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{refPtrOp.getResult(0)});
    }

    auto allocatorOp = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_allocator_t"),
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "allocator")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{stateOp.getResult(0)});

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "iree_allocator_free"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{allocatorOp.getResult(0), stateOp.getResult(0)});

    builder.create<mlir::ReturnOp>(loc);
  }

  // resolve_import
  {
    OpBuilder::InsertionGuard guard(builder);

    auto funcType = mlir::FunctionType::get(
        ctx,
        {
            emitc::OpaqueType::get(ctx, "void*"),
            emitc::OpaqueType::get(ctx, "iree_vm_module_state_t*"),
            emitc::OpaqueType::get(ctx, "iree_host_size_t"),
            emitc::OpaqueType::get(ctx, "const iree_vm_function_t*"),
            emitc::OpaqueType::get(ctx, "const iree_vm_function_signature_t*"),
        },
        {emitc::OpaqueType::get(ctx, "iree_status_t")});

    auto funcOp = builder.create<mlir::FuncOp>(
        loc, moduleName + "_resolve_import", funcType);

    vmAnalysisCache.insert(
        std::make_pair(funcOp.getOperation(), VMAnalysis{{}, {}}));

    funcOp.getOperation()->setAttr("emitc.static", UnitAttr::get(ctx));

    Block *entryBlock = funcOp.addEntryBlock();

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleStateTypeName = moduleName + "_state_t*";

    auto stateOp = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, moduleStateTypeName),
        /*callee=*/StringAttr::get(ctx, "EMITC_CAST"),
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, moduleStateTypeName)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{funcOp.getArgument(1)});

    auto imports = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_function_t*"),
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "imports")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{stateOp.getResult(0)});

    auto import = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_function_t*"),
        /*callee=*/StringAttr::get(ctx, "EMITC_ARRAY_ELEMENT_ADDRESS"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{imports.getResult(0), funcOp.getArgument(2)});

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "EMITC_DEREF_ASSIGN_PTR"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{import.getResult(0), funcOp.getArgument(3)});

    auto status = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
        /*callee=*/StringAttr::get(ctx, "iree_ok_status"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    builder.create<mlir::ReturnOp>(loc, status.getResult(0));
  }

  // create
  {
    OpBuilder::InsertionGuard guard(builder);

    auto funcType = mlir::FunctionType::get(
        ctx,
        {emitc::OpaqueType::get(ctx, "iree_allocator_t"),
         emitc::OpaqueType::get(ctx, "iree_vm_module_t**")},
        {emitc::OpaqueType::get(ctx, "iree_status_t")});

    auto funcOp =
        builder.create<mlir::FuncOp>(loc, moduleName + "_create", funcType);

    vmAnalysisCache.insert(
        std::make_pair(funcOp.getOperation(), VMAnalysis{{}, {}}));

    funcOp.getOperation()->setAttr("emitc.static", UnitAttr::get(ctx));

    // This function needs an iree_vm_native_module_descriptor_t that is emitted
    // by the CModuleTarget at the moment. So we add a marker to this function
    // and delay the printing of it.
    funcOp.getOperation()->setAttr("vm.emit_at_end", UnitAttr::get(ctx));

    Block *entryBlock = funcOp.addEntryBlock();

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleTypeName = moduleName + "_t*";

    auto module = builder.create<emitc::ConstantOp>(
        /*location=*/loc,
        /*resultType=*/emitc::OpaqueType::get(ctx, moduleTypeName),
        /*value=*/emitc::OpaqueAttr::get(ctx, "NULL"));

    auto moduleSize = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_host_size_t"),
        /*callee=*/StringAttr::get(ctx, "sizeof"),
        /*args=*/
        ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(ctx, moduleName + "_t")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    auto modulePtr = builder.create<emitc::ApplyOp>(
        /*location=*/loc,
        /*result=*/emitc::OpaqueType::get(ctx, moduleTypeName + "*"),
        /*applicableOperator=*/StringAttr::get(ctx, "&"),
        /*operand=*/module.getResult());

    auto voidPtr = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "void**"),
        /*callee=*/StringAttr::get(ctx, "EMITC_CAST"),
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "void**")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{modulePtr.getResult()});

    returnIfError(
        builder, loc, StringAttr::get(ctx, "iree_allocator_malloc"), {}, {},
        {funcOp.getArgument(0), moduleSize.getResult(0), voidPtr.getResult(0)},
        /*vmAnalysisCache=*/vmAnalysisCache);

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "memset"),
        /*args=*/
        ArrayAttr::get(ctx,
                       {builder.getIndexAttr(0), builder.getUI32IntegerAttr(0),
                        builder.getIndexAttr(1)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{module.getResult(), moduleSize.getResult(0)});

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER_ASSIGN"),
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "allocator"),
                             builder.getIndexAttr(1)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{module.getResult(), funcOp.getArgument(0)});

    auto vmModule = builder.create<emitc::ConstantOp>(
        /*location=*/loc,
        /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_module_t"),
        /*value=*/emitc::OpaqueAttr::get(ctx, ""));

    auto vmModulePtr = builder.create<emitc::ApplyOp>(
        /*location=*/loc,
        /*result=*/emitc::OpaqueType::get(ctx, "iree_vm_module_t*"),
        /*applicableOperator=*/StringAttr::get(ctx, "&"),
        /*operand=*/vmModule.getResult());

    auto vmInitializeStatus = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_module_initialize"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{vmModulePtr.getResult(), module.getResult()});

    Type boolType = builder.getIntegerType(1);

    auto vmInitializeIsOk = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/boolType,
        /*callee=*/StringAttr::get(ctx, "iree_status_is_ok"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{vmInitializeStatus.getResult(0)});

    // Start by splitting the block into two. The part before will contain the
    // condition, and the part after will contain the continuation point.
    Block *condBlock = builder.getInsertionBlock();
    Block::iterator opPosition = builder.getInsertionPoint();
    Block *continuationBlock = condBlock->splitBlock(opPosition);

    // Create a new block for the target of the failure.
    Block *failureBlock;
    {
      OpBuilder::InsertionGuard guard(builder);
      Region *parentRegion = condBlock->getParent();
      failureBlock = builder.createBlock(parentRegion, parentRegion->end());

      builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "iree_allocator_free"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{funcOp.getArgument(0), module.getResult()});

      builder.create<mlir::ReturnOp>(loc, vmInitializeStatus.getResult(0));
    }

    builder.setInsertionPointToEnd(condBlock);

    builder.create<IREE::VM::CondBranchOp>(loc, vmInitializeIsOk.getResult(0),
                                           continuationBlock, failureBlock);

    builder.setInsertionPointToStart(continuationBlock);

    // Set function pointers
    for (std::string funcName :
         {"destroy", "alloc_state", "free_state", "resolve_import"}) {
      builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_MEMBER_ASSIGN"),
          /*args=*/
          ArrayAttr::get(
              ctx,
              {builder.getIndexAttr(0), emitc::OpaqueAttr::get(ctx, funcName),
               emitc::OpaqueAttr::get(ctx, moduleName + "_" + funcName)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{vmModule.getResult()});
    }

    std::string descriptoPtr = "&" + moduleName + "_descriptor_";

    auto status = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_native_module_create"),
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, descriptoPtr),
                             builder.getIndexAttr(1), builder.getIndexAttr(2)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{vmModulePtr.getResult(), funcOp.getArgument(0),
                        funcOp.getArgument(1)});

    builder.create<mlir::ReturnOp>(loc, status.getResult(0));
  }

  return success();
}

SmallVector<Attribute, 4> indexSequence(int64_t n, MLIRContext *ctx) {
  return llvm::to_vector<4>(
      llvm::map_range(llvm::seq<int64_t>(0, n), [&ctx](int64_t i) -> Attribute {
        return IntegerAttr::get(IndexType::get(ctx), i);
      }));
}

template <typename ResultOpTy>
ResultOpTy lookupSymbolRef(Operation *accessOp, StringRef attrName) {
  FlatSymbolRefAttr globalAttr =
      accessOp->getAttrOfType<FlatSymbolRefAttr>(attrName);
  ResultOpTy globalOp =
      accessOp->getParentOfType<IREE::VM::ModuleOp>().lookupSymbol<ResultOpTy>(
          globalAttr.getValue());
  return globalOp;
}

// Convert vm operations to emitc calls. The resultiong call has the ops
// operands as arguments followed by an argument for every attribute.
template <typename SrcOpTy, typename Adaptor = typename SrcOpTy::Adaptor>
class GenericOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

 public:
  GenericOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                      StringRef funcName)
      : OpConversionPattern<SrcOpTy>(typeConverter, context),
        funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      SrcOpTy op, Adaptor adaptor,
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
          indexSequence(adaptor.getOperands().size(), op.getContext());

      for (NamedAttribute attr : op->getAttrs()) {
        args_.push_back(attr.getValue());
      }

      args = rewriter.getArrayAttr(args_);
    }

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        op, type, callee, args, templateArgs, adaptor.getOperands());

    return success();
  }

  StringRef funcName;
};

class FuncOpConversion : public OpConversionPattern<mlir::FuncOp> {
 public:
  using OpConversionPattern<mlir::FuncOp>::OpConversionPattern;

  FuncOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<mlir::FuncOp>(typeConverter, context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      mlir::FuncOp funcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TypeConverter::SignatureConversion signatureConverter(
        funcOp.getType().getNumInputs());
    TypeConverter typeConverter;
    for (const auto &arg : llvm::enumerate(funcOp.getArguments())) {
      Type convertedType =
          getTypeConverter()->convertType(arg.value().getType());
      signatureConverter.addInputs(arg.index(), convertedType);
    }

    rewriter.applySignatureConversion(&funcOp.getBody(), signatureConverter);

    // Creates a new function with the updated signature.
    rewriter.updateRootInPlace(funcOp, [&] {
      funcOp.setType(
          rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                                   funcOp.getType().getResults()));
    });
    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

template <typename CallOpTy>
class CallOpConversion : public OpConversionPattern<CallOpTy> {
 public:
  using Adaptor = typename CallOpTy::Adaptor;
  using OpConversionPattern<CallOpTy>::OpConversionPattern;

  CallOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<CallOpTy>(typeConverter, context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      CallOpTy op, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    mlir::FuncOp funcOp =
        lookupSymbolRef<mlir::FuncOp>(op.getOperation(), "callee");
    IREE::VM::ImportOp importOp =
        lookupSymbolRef<IREE::VM::ImportOp>(op.getOperation(), "callee");

    if (!funcOp && !importOp)
      return op.emitError() << "lookup of callee failed";

    if (funcOp && importOp)
      return op.emitError() << "lookup of callee ambiguous";

    const bool isImported = importOp != nullptr;

    return isImported ? rewriteImportedCall(op.getOperation(), adaptor,
                                            rewriter, importOp)
                      : rewriteInternalCall(op.getOperation(), adaptor,
                                            rewriter, funcOp);
  }

  LogicalResult rewriteInternalCall(Operation *op, Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    mlir::FuncOp funcOp) const {
    auto loc = op->getLoc();

    SmallVector<Value, 4> updatedOperands;
    SmallVector<Value, 4> resultOperands;

    auto parentFuncOp = op->getParentOfType<mlir::FuncOp>();

    BlockArgument stackArg = parentFuncOp.getArgument(0);
    BlockArgument moduleArg = parentFuncOp.getArgument(1);
    BlockArgument moduleStateArg = parentFuncOp.getArgument(2);

    updatedOperands = {stackArg, moduleArg, moduleStateArg};

    if (failed(updateOperands(op, op->getOperands(), rewriter, updatedOperands,
                              resultOperands))) {
      return failure();
    };

    returnIfError(/*rewriter=*/rewriter, /*location=*/loc, /*callee=*/funcOp,
                  /*operands=*/updatedOperands,
                  /*vmAnalysisCache=*/vmAnalysisCache);

    if (failed(updateResults(op, rewriter, resultOperands))) {
      return failure();
    }

    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult rewriteImportedCall(Operation *op, Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    IREE::VM::ImportOp importOp) const {
    auto ctx = op->getContext();
    auto loc = op->getLoc();

    SmallVector<Value, 4> updatedOperands;
    SmallVector<Value, 4> resultOperands;

    auto moduleOp =
        importOp.getOperation()->getParentOfType<IREE::VM::ModuleOp>();

    Optional<std::string> funcName = buildFunctionName(moduleOp, importOp);

    if (!funcName.hasValue())
      return op->emitError() << "Couldn't build name to imported function";

    int importOrdinal = importOp.ordinal().getValue().getZExtValue();

    auto funcOp = op->getParentOfType<mlir::FuncOp>();
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

    auto isVariadic = [](APInt segmentSize) {
      return segmentSize.getSExtValue() != -1;
    };

    if (auto variadicOp = dyn_cast<IREE::VM::CallVariadicOp>(op)) {
      DenseIntElementsAttr segmentSizes = variadicOp.segment_sizes();
      size_t numSegments = segmentSizes.size();
      size_t numVariadicSegments = llvm::count_if(segmentSizes, isVariadic);

      if (numVariadicSegments != 1) {
        return op->emitError() << "only exactly one variadic segment supported";
      }

      auto lastSegmentSize = *(segmentSizes.begin() + (numSegments - 1));

      if (!isVariadic(lastSegmentSize)) {
        return op->emitError() << "expected the last segment to be variadic";
      }

      size_t numSpans = lastSegmentSize.getSExtValue();

      // TODO(simon-camp): The generated code would be cleaner if we used the
      // args attribute of the call op to specify the constant.
      auto numSpansOp = rewriter.create<emitc::ConstantOp>(
          /*location=*/loc,
          /*resultType=*/rewriter.getIndexType(),
          /*value=*/rewriter.getIndexAttr(numSpans));

      updatedOperands.push_back(numSpansOp.getResult());
    }

    if (failed(updateOperands(op, op->getOperands(), rewriter, updatedOperands,
                              resultOperands))) {
      return failure();
    }

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, funcName.getValue()),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/updatedOperands, /*vmAnalysisCache=*/vmAnalysisCache);

    if (failed(updateResults(op, rewriter, resultOperands))) {
      return failure();
    }

    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult updateOperands(Operation *op, OperandRange operands,
                               ConversionPatternRewriter &rewriter,
                               SmallVector<Value, 4> &updatedOperands,
                               SmallVector<Value, 4> &resultOperands) const {
    auto ctx = op->getContext();
    auto loc = op->getLoc();

    auto funcOp = op->getParentOfType<mlir::FuncOp>();
    auto ptr = vmAnalysisCache.find(funcOp.getOperation());
    if (ptr == vmAnalysisCache.end()) {
      return op->emitError() << "parent func op not found in cache.";
    }

    for (Value operand : operands) {
      assert(operand.getType() !=
             emitc::OpaqueType::get(ctx, "iree_vm_ref_t*"));

      if (operand.getType().isa<IREE::VM::RefType>()) {
        Optional<Value> operandRef = findRef(funcOp, vmAnalysisCache, operand);

        if (!operandRef.hasValue()) {
          return op->emitError() << "local ref not found";
        }

        auto refOp = rewriter.create<emitc::ConstantOp>(
            /*location=*/loc,
            /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t"),
            /*value=*/emitc::OpaqueAttr::get(ctx, ""));

        auto refPtrOp = rewriter.create<emitc::ApplyOp>(
            /*location=*/loc,
            /*result=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t*"),
            /*applicableOperator=*/StringAttr::get(ctx, "&"),
            /*operand=*/refOp.getResult());

        if (failed(clearStruct(rewriter, refPtrOp.getResult(),
                               /*isPointer=*/true))) {
          return failure();
        }

        rewriter.create<emitc::CallOp>(
            /*location=*/loc,
            /*type=*/TypeRange{},
            /*callee=*/StringAttr::get(ctx, "iree_vm_ref_assign"),
            /*args=*/ArrayAttr{},
            /*templateArgs=*/ArrayAttr{},
            /*operands=*/
            ArrayRef<Value>{operandRef.getValue(), refPtrOp.getResult()});

        updatedOperands.push_back(refPtrOp.getResult());
      } else {
        updatedOperands.push_back(operand);
      }
    }

    // Create a variable for every result and a pointer to it as output
    // parameter to the call.
    for (OpResult result : op->getResults()) {
      if (result.getType().isa<IREE::VM::RefType>()) {
        Optional<Value> ref = findRef(funcOp, vmAnalysisCache, result);

        if (!ref.hasValue()) {
          return op->emitError() << "local ref not found";
        }

        resultOperands.push_back(ref.getValue());
        updatedOperands.push_back(ref.getValue());
      } else {
        auto resultOp = rewriter.create<emitc::ConstantOp>(
            /*location=*/loc,
            /*resultType=*/result.getType(),
            /*value=*/emitc::OpaqueAttr::get(ctx, ""));

        Optional<std::string> cType = getCType(resultOp.getType());
        if (!cType.hasValue()) {
          return op->emitError() << "unable to emit C type";
        }

        std::string cPtrType = cType.getValue() + std::string("*");
        auto resultPtrOp = rewriter.create<emitc::ApplyOp>(
            /*location=*/loc,
            /*type=*/emitc::OpaqueType::get(ctx, cPtrType),
            /*applicableOperator=*/StringAttr::get(ctx, "&"),
            /*operand=*/resultOp.getResult());

        resultOperands.push_back(resultOp.getResult());
        updatedOperands.push_back(resultPtrOp.getResult());
      }
    }
    return success();
  }

  LogicalResult updateResults(Operation *op,
                              ConversionPatternRewriter &rewriter,
                              SmallVector<Value, 4> &resultOperands) const {
    auto funcOp = op->getParentOfType<mlir::FuncOp>();
    auto ptr = vmAnalysisCache.find(funcOp.getOperation());
    if (ptr == vmAnalysisCache.end()) {
      return op->emitError() << "parent func op not found in cache.";
    }

    for (auto &pair : llvm::enumerate(op->getResults())) {
      size_t index = pair.index();
      OpResult result = pair.value();

      if (!result.getType().isa<IREE::VM::RefType>()) {
        result.replaceAllUsesWith(resultOperands[index]);
      }
    }
    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

template <typename CmpOpTy, typename Adaptor = typename CmpOpTy::Adaptor>
class CompareRefOpConversion : public OpConversionPattern<CmpOpTy> {
 public:
  using OpConversionPattern<CmpOpTy>::OpConversionPattern;

  CompareRefOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                         StringRef funcName, VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<CmpOpTy>(typeConverter, context),
        funcName(funcName),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      CmpOpTy cmpOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = cmpOp.getContext();
    auto loc = cmpOp.getLoc();

    auto funcOp =
        cmpOp.getOperation()->template getParentOfType<mlir::FuncOp>();

    auto ptr = vmAnalysisCache.find(funcOp.getOperation());
    if (ptr == vmAnalysisCache.end()) {
      return cmpOp.emitError() << "parent func op not found in cache.";
    }

    bool moveLhs =
        ptr->second.isLastValueUse(cmpOp.lhs(), cmpOp.getOperation()) && false;
    bool moveRhs =
        ptr->second.isLastValueUse(cmpOp.rhs(), cmpOp.getOperation()) && false;

    Optional<Value> refLhs = findRef(funcOp, vmAnalysisCache, cmpOp.lhs());

    if (!refLhs.hasValue()) {
      return cmpOp.emitError() << "local ref not found";
    }

    Optional<Value> refRhs = findRef(funcOp, vmAnalysisCache, cmpOp.rhs());

    if (!refRhs.hasValue()) {
      return cmpOp.emitError() << "local ref not found";
    }

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        /*op=*/cmpOp,
        /*type=*/cmpOp.getType(),
        /*callee=*/StringAttr::get(ctx, funcName),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refLhs.getValue(), refRhs.getValue()});

    if (moveLhs) {
      rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "iree_vm_ref_release"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{refLhs.getValue()});
    }

    // NOTE: If lhs and rhs alias we call release twice on the same
    // argument.
    if (moveRhs) {
      rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "iree_vm_ref_release"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{refRhs.getValue()});
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
  CompareRefNotZeroOpConversion(TypeConverter &typeConverter,
                                MLIRContext *context,
                                VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<IREE::VM::CmpNZRefOp>(typeConverter, context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::CmpNZRefOp cmpOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = cmpOp.getContext();
    auto loc = cmpOp.getLoc();

    auto funcOp = cmpOp.getOperation()->getParentOfType<mlir::FuncOp>();

    auto ptr = vmAnalysisCache.find(funcOp.getOperation());
    if (ptr == vmAnalysisCache.end()) {
      return cmpOp.emitError() << "parent func op not found in cache.";
    }

    bool move =
        ptr->second.isLastValueUse(cmpOp.operand(), cmpOp.getOperation()) &&
        false;

    Optional<Value> ref = findRef(funcOp, vmAnalysisCache, cmpOp.operand());

    if (!ref.hasValue()) {
      return cmpOp.emitError() << "local ref not found";
    }

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        /*op=*/cmpOp,
        /*type=*/cmpOp.getType(),
        /*callee=*/StringAttr::get(ctx, "vm_cmp_nz_ref"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{ref.getValue()});

    if (move) {
      rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "iree_vm_ref_release"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{ref.getValue()});
    }

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

template <typename ConstOpTy>
class ConstOpConversion : public OpConversionPattern<ConstOpTy> {
 public:
  using Adaptor = typename ConstOpTy::Adaptor;
  using OpConversionPattern<ConstOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstOpTy constOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(constOp, constOp.getType(),
                                                   constOp.value());
    return success();
  }
};

template <typename ConstZeroOpTy>
class ConstZeroOpConversion : public OpConversionPattern<ConstZeroOpTy> {
 public:
  using Adaptor = typename ConstZeroOpTy::Adaptor;
  using OpConversionPattern<ConstZeroOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstZeroOpTy constZeroOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
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
    : public OpConversionPattern<IREE::VM::ConstRefZeroOp> {
 public:
  using OpConversionPattern<IREE::VM::ConstRefZeroOp>::OpConversionPattern;

  ConstRefZeroOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                           VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<IREE::VM::ConstRefZeroOp>(typeConverter, context),
        vmAnalysisCache(vmAnalysisCache) {}

  LogicalResult matchAndRewrite(
      IREE::VM::ConstRefZeroOp constRefZeroOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    auto ctx = constRefZeroOp.getContext();
    auto loc = constRefZeroOp.getLoc();

    auto funcOp =
        constRefZeroOp.getOperation()->getParentOfType<mlir::FuncOp>();

    Optional<Value> ref =
        findRef(funcOp, vmAnalysisCache, constRefZeroOp.getResult());

    if (!ref.hasValue()) {
      return constRefZeroOp.emitError() << "local ref not found";
    }

    rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "iree_vm_ref_release"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{ref.getValue()});

    rewriter.replaceOp(constRefZeroOp, ref.getValue());

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

class ConstRefRodataOpConversion
    : public OpConversionPattern<IREE::VM::ConstRefRodataOp> {
 public:
  using OpConversionPattern<IREE::VM::ConstRefRodataOp>::OpConversionPattern;

  ConstRefRodataOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                             VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<IREE::VM::ConstRefRodataOp>(typeConverter, context),
        vmAnalysisCache(vmAnalysisCache) {}

  LogicalResult matchAndRewrite(
      IREE::VM::ConstRefRodataOp constRefRodataOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    auto ctx = constRefRodataOp.getContext();
    auto loc = constRefRodataOp.getLoc();

    auto rodataOp = lookupSymbolRef<IREE::VM::RodataOp>(
        constRefRodataOp.getOperation(), "rodata");
    if (!rodataOp) {
      return constRefRodataOp.emitError() << "Unable to find RodataOp";
    }

    auto funcOp =
        constRefRodataOp.getOperation()->getParentOfType<mlir::FuncOp>();

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

    Optional<Value> ref =
        findRef(funcOp, vmAnalysisCache, constRefRodataOp.getResult());

    if (!ref.hasValue()) {
      return constRefRodataOp.emitError() << "local ref not found";
    }

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_ref_wrap_retain"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{byteBufferPtrOp.getResult(0), typeIdOp.getResult(0),
                        ref.getValue()},
        /*vmAnalysisCache=*/vmAnalysisCache);

    rewriter.replaceOp(constRefRodataOp, ref.getValue());

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

class BranchOpConversion : public OpConversionPattern<IREE::VM::BranchOp> {
  using OpConversionPattern<IREE::VM::BranchOp>::OpConversionPattern;

 public:
  BranchOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                     VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<IREE::VM::BranchOp>(typeConverter, context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::BranchOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    assert(op.getOperands().size() == adaptor.getOperands().size());

    auto isNotRefOperand = [](Value operand) {
      return !operand.getType().isa<IREE::VM::RefType>();
    };

    SmallVector<Value> nonRefOperands;
    for (Value operand : op.getOperands()) {
      if (isNotRefOperand(operand)) {
        nonRefOperands.push_back(operand);
      }
    }

    Block *dest = op.getDest();

    // If we don't have ref block arguments, we can convert the operation
    // directly.
    if (adaptor.getOperands().size() == nonRefOperands.size()) {
      rewriter.replaceOpWithNewOp<mlir::BranchOp>(op, op.dest(),
                                                  op.getOperands());
      return success();
    }

    auto funcOp = op.getOperation()->getParentOfType<mlir::FuncOp>();

    auto ptr = vmAnalysisCache.find(funcOp.getOperation());
    if (ptr == vmAnalysisCache.end()) {
      return op->emitError() << "parent func op not found in cache.";
    }

    Block *destDispatch;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      destDispatch = rewriter.createBlock(dest);

      for (auto pair : llvm::zip(op.getOperands(), dest->getArguments())) {
        Value operand = std::get<0>(pair);
        BlockArgument blockArg = std::get<1>(pair);

        if (isNotRefOperand(operand)) {
          continue;
        }

        assert(operand.getType().isa<IREE::VM::RefType>());
        assert(blockArg.getType().isa<IREE::VM::RefType>());

        Optional<Value> operandRef = findRef(funcOp, vmAnalysisCache, operand);
        Optional<Value> blockArgRef =
            findRef(funcOp, vmAnalysisCache, blockArg);

        if (!operandRef.hasValue()) {
          return op.emitError() << "local ref not found";
        }
        if (!blockArgRef.hasValue()) {
          return op.emitError() << "local ref not found";
        }

        rewriter.create<emitc::CallOp>(
            /*location=*/loc,
            /*type=*/TypeRange{},
            /*callee=*/
            StringAttr::get(ctx, "iree_vm_ref_retain"),
            /*args=*/ArrayAttr{},
            /*templateArgs=*/ArrayAttr{},
            /*operands=*/
            ArrayRef<Value>{operandRef.getValue(), blockArgRef.getValue()});
      }
      rewriter.create<mlir::BranchOp>(loc, op.dest(), op.getOperands());
    }

    rewriter.replaceOpWithNewOp<mlir::BranchOp>(op, destDispatch);

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

// Basic block arguments are emitted as variable assignments in EmitC. Because
// of that we need to treat ref operands separately here. We remove ref
// arguments from the basic blocks and use the ref C API to set the ref
// variables. The generated IR looks roughly as follows:

// clang-format off
// vm.cond_br %cond, ^bb1(%ref : !vm.ref<?>, %int : i32), ^bb2(%ref : !vm.ref<?>, %int : i32)
// ^bb1(%ref_arg_1 : !vm.ref<?>, %int_arg : i32):
//   ...
// ^bb2(%ref_arg_2 : !vm.ref<?>, %int_arg : i32):
//   ...
// =>
// cond_br %cond, ^bb1_dispatch, ^bb2_dispatch
// ^bb1_dispatch:
//   // populate the variable corresponding to ordinal(%ref_arg_1)
//   br ^bb1(%int : i32)
// ^bb2_dispatch:
//   // populate the variable corresponding to ordinal(%ref_arg_2)
//   br ^bb2(%int : i32)
// ^bb1(%int_arg : i32):
//   ...
// ^bb2(%int_arg : i32):
//   ...
// clang-format on
class CondBranchOpConversion
    : public OpConversionPattern<IREE::VM::CondBranchOp> {
  using OpConversionPattern<IREE::VM::CondBranchOp>::OpConversionPattern;

 public:
  CondBranchOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                         VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<IREE::VM::CondBranchOp>(typeConverter, context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::CondBranchOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    assert(op.getOperands().size() == adaptor.getOperands().size());

    auto isNotRefOperand = [](Value operand) {
      return !operand.getType().isa<IREE::VM::RefType>();
    };

    SmallVector<Value> nonRefOperands;
    for (Value operand : op.getOperands()) {
      if (isNotRefOperand(operand)) {
        nonRefOperands.push_back(operand);
      }
    }

    Block *trueDest = op.getTrueDest();
    Block *falseDest = op.getFalseDest();

    Type boolType = rewriter.getI1Type();

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

    // If we don't have ref block arguments, we can convert the operation
    // directly.
    if (adaptor.getOperands().size() == nonRefOperands.size()) {
      rewriter.replaceOpWithNewOp<mlir::CondBranchOp>(
          op, conditionI1.getResult(0), op.trueDest(), op.getTrueOperands(),
          op.falseDest(), op.getFalseOperands());
      return success();
    }

    auto funcOp = op.getOperation()->getParentOfType<mlir::FuncOp>();

    auto ptr = vmAnalysisCache.find(funcOp.getOperation());
    if (ptr == vmAnalysisCache.end()) {
      return op->emitError() << "parent func op not found in cache.";
    }

    Block *trueDestDispatch;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      trueDestDispatch = rewriter.createBlock(trueDest);

      for (auto pair :
           llvm::zip(op.getTrueOperands(), trueDest->getArguments())) {
        Value operand = std::get<0>(pair);
        BlockArgument blockArg = std::get<1>(pair);

        if (isNotRefOperand(operand)) {
          continue;
        }

        assert(operand.getType().isa<IREE::VM::RefType>());
        assert(blockArg.getType().isa<IREE::VM::RefType>());

        Optional<Value> operandRef = findRef(funcOp, vmAnalysisCache, operand);
        Optional<Value> blockArgRef =
            findRef(funcOp, vmAnalysisCache, blockArg);

        if (!operandRef.hasValue()) {
          return op.emitError() << "local ref not found";
        }
        if (!blockArgRef.hasValue()) {
          return op.emitError() << "local ref not found";
        }

        rewriter.create<emitc::CallOp>(
            /*location=*/loc,
            /*type=*/TypeRange{},
            /*callee=*/
            StringAttr::get(ctx, "iree_vm_ref_retain"),
            /*args=*/ArrayAttr{},
            /*templateArgs=*/ArrayAttr{},
            /*operands=*/
            ArrayRef<Value>{operandRef.getValue(), blockArgRef.getValue()});
      }
      rewriter.create<mlir::BranchOp>(loc, op.trueDest(), op.getTrueOperands());
    }

    Block *falseDestDispatch;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      falseDestDispatch = rewriter.createBlock(falseDest);

      for (auto pair :
           llvm::zip(op.getFalseOperands(), falseDest->getArguments())) {
        Value operand = std::get<0>(pair);
        BlockArgument blockArg = std::get<1>(pair);

        if (isNotRefOperand(operand)) {
          continue;
        }

        assert(operand.getType().isa<IREE::VM::RefType>());
        assert(blockArg.getType().isa<IREE::VM::RefType>());

        Optional<Value> operandRef = findRef(funcOp, vmAnalysisCache, operand);
        Optional<Value> blockArgRef =
            findRef(funcOp, vmAnalysisCache, blockArg);

        if (!operandRef.hasValue()) {
          return op.emitError() << "local ref not found";
        }
        if (!blockArgRef.hasValue()) {
          return op.emitError() << "local ref not found";
        }

        rewriter.create<emitc::CallOp>(
            /*location=*/loc,
            /*type=*/TypeRange{},
            /*callee=*/
            StringAttr::get(ctx, "iree_vm_ref_retain"),
            /*args=*/ArrayAttr{},
            /*templateArgs=*/ArrayAttr{},
            /*operands=*/
            ArrayRef<Value>{operandRef.getValue(), blockArgRef.getValue()});
      }
      rewriter.create<mlir::BranchOp>(loc, op.falseDest(),
                                      op.getFalseOperands());
    }

    rewriter.replaceOpWithNewOp<mlir::CondBranchOp>(
        op, conditionI1.getResult(0), trueDestDispatch, falseDestDispatch);

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

class ReturnOpConversion : public OpConversionPattern<IREE::VM::ReturnOp> {
  using OpConversionPattern<IREE::VM::ReturnOp>::OpConversionPattern;

 public:
  ReturnOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                     VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<IREE::VM::ReturnOp>(typeConverter, context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    auto funcOp = op.getOperation()->getParentOfType<mlir::FuncOp>();

    // The result variables are the last N arguments of the function.
    unsigned int firstOutputArgumentIndex =
        funcOp.getNumArguments() - op.getOperands().size();

    for (auto &pair : llvm::enumerate(op.getOperands())) {
      Value operand = pair.value();
      size_t index = pair.index();

      unsigned int argumentIndex = firstOutputArgumentIndex + index;
      BlockArgument resultArgument = funcOp.getArgument(argumentIndex);

      if (operand.getType().isa<IREE::VM::RefType>()) {
        assert(operand.getType() !=
               emitc::OpaqueType::get(ctx, "iree_vm_ref_t*"));

        Optional<Value> operandRef = findRef(funcOp, vmAnalysisCache, operand);

        if (!operandRef.hasValue()) {
          return op->emitError() << "local ref not found";
        }

        rewriter.create<emitc::CallOp>(
            /*location=*/loc,
            /*type=*/TypeRange{},
            /*callee=*/StringAttr::get(ctx, "iree_vm_ref_move"),
            /*args=*/ArrayAttr{},
            /*templateArgs=*/ArrayAttr{},
            /*operands=*/
            ArrayRef<Value>{operandRef.getValue(), resultArgument});
      } else {
        rewriter.create<emitc::CallOp>(
            /*location=*/loc,
            /*type=*/TypeRange{},
            /*callee=*/StringAttr::get(ctx, "EMITC_DEREF_ASSIGN_VALUE"),
            /*args=*/ArrayAttr{},
            /*templateArgs=*/ArrayAttr{},
            /*operands=*/ArrayRef<Value>{resultArgument, operand});
      }
    }

    releaseRefs(rewriter, loc, funcOp, vmAnalysisCache);

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

  VMAnalysisCache &vmAnalysisCache;
};

class FailOpConversion : public OpConversionPattern<IREE::VM::FailOp> {
  using OpConversionPattern<IREE::VM::FailOp>::OpConversionPattern;

 public:
  FailOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<IREE::VM::FailOp>(typeConverter, context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::FailOp op, OpAdaptor adaptor,
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

      releaseRefs(rewriter, loc, funcOp, vmAnalysisCache);

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

      releaseRefs(rewriter, loc, funcOp, vmAnalysisCache);

      std::string message = std::string("\"") +
                            op.message().getValueOr("").str() +
                            std::string("\"");

      auto messageOp = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_string_view_t"),
          /*callee=*/StringAttr::get(ctx, "iree_make_cstring_view"),
          /*args=*/
          ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(ctx, message)}),
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
          ArrayAttr::get(
              ctx,
              {emitc::OpaqueAttr::get(ctx, "IREE_STATUS_FAILED_PRECONDITION"),
               emitc::OpaqueAttr::get(ctx, "\"<vm>\""),
               rewriter.getI32IntegerAttr(0),
               emitc::OpaqueAttr::get(ctx, "\"%.*s\""),
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

  VMAnalysisCache &vmAnalysisCache;
};

template <typename LoadOpTy, typename GlobalOpTy,
          typename Adaptor = typename LoadOpTy::Adaptor>
class GlobalLoadOpConversion : public OpConversionPattern<LoadOpTy> {
  using OpConversionPattern<LoadOpTy>::OpConversionPattern;

 public:
  GlobalLoadOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                         StringRef funcName)
      : OpConversionPattern<LoadOpTy>(typeConverter, context),
        funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      LoadOpTy loadOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = loadOp.getContext();
    auto loc = loadOp.getLoc();

    GlobalOpTy globalOp =
        lookupSymbolRef<GlobalOpTy>(loadOp.getOperation(), "global");
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

template <typename LoadStoreOpTy,
          typename Adaptor = typename LoadStoreOpTy::Adaptor>
class GlobalLoadStoreRefOpConversion
    : public OpConversionPattern<LoadStoreOpTy> {
 public:
  using OpConversionPattern<LoadStoreOpTy>::OpConversionPattern;

  GlobalLoadStoreRefOpConversion(TypeConverter &typeConverter,
                                 MLIRContext *context,
                                 VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<LoadStoreOpTy>(typeConverter, context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      LoadStoreOpTy op, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isa<IREE::VM::GlobalLoadRefOp>(op)) {
      return rewriteOp(op.getOperation(), adaptor, rewriter, true);
    } else if (isa<IREE::VM::GlobalStoreRefOp>(op)) {
      return rewriteOp(op.getOperation(), adaptor, rewriter, false);
    }

    return op.emitError() << "op must be one of `vm.global.load.ref` or "
                             "`vm.global.store.ref`";
  }

  LogicalResult rewriteOp(Operation *op, Adaptor adaptor,
                          ConversionPatternRewriter &rewriter,
                          bool isLoad) const {
    auto ctx = op->getContext();
    auto loc = op->getLoc();

    IREE::VM::GlobalRefOp globalOp =
        lookupSymbolRef<IREE::VM::GlobalRefOp>(op, "global");
    if (!globalOp) {
      return op->emitError() << "Unable to find GlobalOp";
    }

    auto globalOrdinal = globalOp.ordinal().getValue().getZExtValue();

    auto funcOp = op->getParentOfType<mlir::FuncOp>();

    auto ptr = vmAnalysisCache.find(funcOp.getOperation());
    if (ptr == vmAnalysisCache.end()) {
      return op->emitError() << "parent func op not found in cache.";
    }

    Value localValue = isLoad ? op->getResult(0) : op->getOperand(0);

    Optional<Value> localRef = findRef(funcOp, vmAnalysisCache, localValue);

    if (!localRef.hasValue()) {
      return op->emitError() << "local ref not found";
    }

    BlockArgument stateArg = funcOp.getArgument(2);
    auto refs = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t*"),
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, "refs")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{stateArg});

    auto stateRef = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t*"),
        /*callee=*/StringAttr::get(ctx, "EMITC_ARRAY_ELEMENT_ADDRESS"),
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                             rewriter.getUI32IntegerAttr(globalOrdinal)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refs.getResult(0)});

    Type elementType = localValue.getType();

    auto elementTypePtrOp = createVmTypeDefPtr(rewriter, op, elementType);

    if (!elementTypePtrOp.hasValue()) {
      return op->emitError() << "generating iree_vm_type_def_t* failed";
    }

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

    Value srcRef = isLoad ? stateRef.getResult(0) : localRef.getValue();
    Value destRef = isLoad ? localRef.getValue() : stateRef.getResult(0);

    bool move = ptr->second.isLastValueUse(localValue, op) && false;

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_ref_retain_or_move_checked"),
        /*args=*/
        ArrayAttr::get(ctx,
                       {rewriter.getBoolAttr(move), rewriter.getIndexAttr(0),
                        rewriter.getIndexAttr(1), rewriter.getIndexAttr(2)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{srcRef, typedefRefType.getResult(0), destRef},
        /*vmAnalysisCache=*/vmAnalysisCache);

    if (isLoad) {
      rewriter.replaceOp(op, localRef.getValue());
    } else {
      rewriter.eraseOp(op);
    }

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

template <typename StoreOpTy, typename GlobalOpTy,
          typename Adaptor = typename StoreOpTy::Adaptor>
class GlobalStoreOpConversion : public OpConversionPattern<StoreOpTy> {
  using OpConversionPattern<StoreOpTy>::OpConversionPattern;

 public:
  GlobalStoreOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                          StringRef funcName)
      : OpConversionPattern<StoreOpTy>(typeConverter, context),
        funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      StoreOpTy storeOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = storeOp.getContext();
    auto loc = storeOp.getLoc();

    GlobalOpTy globalOp =
        lookupSymbolRef<GlobalOpTy>(storeOp.getOperation(), "global");
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
        /*operands=*/
        ArrayRef<Value>{rwDataPtr.getResult(0), storeOp.value()});

    return success();
  }

  StringRef funcName;
};

// Convert vm list operations to two emitc calls. The wrapping ref pointer
// is first dereferenced and the result is used as the argument of the
// specified function name.
template <typename SrcOpTy, typename Adaptor = typename SrcOpTy::Adaptor>
class ListOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

 public:
  ListOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   StringRef funcName, size_t listArgumentIndex, bool failable,
                   VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<SrcOpTy>(typeConverter, context),
        funcName(funcName),
        listArgumentIndex(listArgumentIndex),
        failable(failable),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      SrcOpTy op, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    if (listArgumentIndex >= adaptor.getOperands().size()) {
      return op.emitError() << " index for list argument out of range";
    }

    Value listOperand = adaptor.getOperands()[listArgumentIndex];

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
        /*operands=*/ArrayRef<Value>{refOp.getResult()},
        /*vmAnalysisCache=*/vmAnalysisCache);

    // Replace the one list argument (which is wrapped in a ref) with the
    // unwrapped list.
    SmallVector<Value, 4> updatedOperands;
    for (auto &operand : llvm::enumerate(adaptor.getOperands())) {
      if (operand.index() == listArgumentIndex) {
        updatedOperands.push_back(listDerefOp.getResult(0));
      } else {
        updatedOperands.push_back(operand.value());
      }
    }

    if (failable) {
      returnIfError(
          /*rewriter=*/rewriter,
          /*location=*/loc,
          /*callee=*/StringAttr::get(ctx, funcName),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>(updatedOperands),
          /*vmAnalysisCache=*/vmAnalysisCache);

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

  VMAnalysisCache &vmAnalysisCache;
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
      IREE::VM::ListAllocOp allocOp, OpAdaptor adaptor,
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
        createVmTypeDefPtr(rewriter, allocOp.getOperation(), elementType);

    if (!elementTypePtrOp.hasValue()) {
      return allocOp.emitError() << "generating iree_vm_type_def_t* failed";
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

    auto funcOp = allocOp.getOperation()->getParentOfType<mlir::FuncOp>();

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
        ArrayRef<Value>{elementTypePtrOp.getValue().getResult(),
                        adaptor.getOperands()[0], allocatorOp.getResult(0),
                        listPtrOp.getResult()},
        /*vmAnalysisCache=*/vmAnalysisCache);

    auto ref = findRef(funcOp, vmAnalysisCache, allocOp.getResult());

    if (!ref.hasValue()) {
      return allocOp.emitError() << "local ref not found";
    }

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
                        ref.getValue()},
        /*vmAnalysisCache=*/vmAnalysisCache);

    rewriter.replaceOp(allocOp, ref.getValue());

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

template <typename GetOpTy, typename Adaptor = typename GetOpTy::Adaptor>
class ListGetOpConversion : public OpConversionPattern<GetOpTy> {
  using OpConversionPattern<GetOpTy>::OpConversionPattern;

 public:
  ListGetOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                      VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<GetOpTy>(typeConverter, context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      GetOpTy getOp, Adaptor adaptor,
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
        /*operand=*/adaptor.getOperands()[0]);

    auto listDerefOp = failListNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_list_t*"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_deref"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refOp.getResult()},
        /*vmAnalysisCache=*/vmAnalysisCache);

    returnIfError(
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
                        valuePtrOp.getResult()},
        /*vmAnalysisCache=*/vmAnalysisCache);

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        /*op=*/getOp,
        /*type=*/getOp.getType(),
        /*callee=*/StringAttr::get(ctx, valueExtractor.getValue()),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{valuePtrOp.getResult()});

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

class ListGetRefOpConversion
    : public OpConversionPattern<IREE::VM::ListGetRefOp> {
 public:
  using OpConversionPattern<IREE::VM::ListGetRefOp>::OpConversionPattern;

  ListGetRefOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                         VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<IREE::VM::ListGetRefOp>(typeConverter, context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::ListGetRefOp getOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getOp.getContext();
    auto loc = getOp.getLoc();

    auto listRefOp = rewriter.create<emitc::ApplyOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t"),
        /*applicableOperator=*/StringAttr::get(ctx, "*"),
        /*operand=*/adaptor.getOperands()[0]);

    auto listDerefOp = failListNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_list_t*"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_deref"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{listRefOp.getResult()},
        /*vmAnalysisCache=*/vmAnalysisCache);

    auto funcOp = getOp.getOperation()->getParentOfType<mlir::FuncOp>();

    auto ref = findRef(funcOp, vmAnalysisCache, getOp.getResult());

    if (!ref.hasValue()) {
      return getOp.emitError() << "local ref not found";
    }

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_get_ref_retain"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{listDerefOp.getResult(0), getOp.index(),
                        ref.getValue()},
        /*vmAnalysisCache=*/vmAnalysisCache);

    Type elementType = getOp.getResult().getType();

    auto elementTypePtrOp =
        createVmTypeDefPtr(rewriter, getOp.getOperation(), elementType);

    if (!elementTypePtrOp.hasValue()) {
      return getOp.emitError() << "generating iree_vm_type_def_t* failed";
    }

    // Build the following expression:
    // (ref->type != IREE_VM_REF_TYPE_NULL &&
    // (iree_vm_type_def_is_value(type_def) || ref->type !=
    // type_def->ref_type))
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
          ArrayRef<Value>{ref.getValue()});

      auto refTypeNull = rewriter.create<emitc::ConstantOp>(
          /*location=*/loc,
          /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
          /*value=*/emitc::OpaqueAttr::get(ctx, "IREE_VM_REF_TYPE_NULL"));

      auto typedefIsValue = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/rewriter.getI1Type(),
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
          /*type=*/rewriter.getI1Type(),
          /*callee=*/StringAttr::get(ctx, "EMITC_NE"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{refType.getResult(0), refTypeNull.getResult()});

      auto refTypesDontMatch = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/rewriter.getI1Type(),
          /*callee=*/StringAttr::get(ctx, "EMITC_NE"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{refType.getResult(0), typedefRefType.getResult(0)});

      auto invalidRefType = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/rewriter.getI1Type(),
          /*callee=*/StringAttr::get(ctx, "EMITC_OR"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{typedefIsValue.getResult(0),
                          refTypesDontMatch.getResult(0)});

      invalidType = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/rewriter.getI1Type(),
          /*callee=*/StringAttr::get(ctx, "EMITC_AND"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{refTypeIsNotNull.getResult(0),
                          invalidRefType.getResult(0)});
    }

    // Start by splitting the block into two. The part before will contain
    // the condition, and the part after will contain the continuation
    // point.
    Block *condBlock = rewriter.getInsertionBlock();
    Block::iterator opPosition = rewriter.getInsertionPoint();
    Block *continuationBlock = condBlock->splitBlock(opPosition);

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
          /*operands=*/ArrayRef<Value>{ref.getValue()});

      rewriter.create<mlir::BranchOp>(loc, continuationBlock);
    }

    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<IREE::VM::CondBranchOp>(loc, invalidType.getResult(0),
                                            failureBlock, continuationBlock);

    rewriter.replaceOp(getOp, ref.getValue());

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

template <typename SetOpTy, typename Adaptor = typename SetOpTy::Adaptor>
class ListSetOpConversion : public OpConversionPattern<SetOpTy> {
  using OpConversionPattern<SetOpTy>::OpConversionPattern;

 public:
  ListSetOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                      VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<SetOpTy>(typeConverter, context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      SetOpTy setOp, Adaptor adaptor,
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
        /*operand=*/adaptor.getOperands()[0]);

    auto listDerefOp = failListNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_list_t*"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_deref"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refOp.getResult()},
        /*vmAnalysisCache=*/vmAnalysisCache);

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_set_value"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{listDerefOp.getResult(0), setOp.index(),
                        valuePtrOp.getResult()},
        /*vmAnalysisCache=*/vmAnalysisCache);

    rewriter.eraseOp(setOp);

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};

class ListSetRefOpConversion
    : public OpConversionPattern<IREE::VM::ListSetRefOp> {
  using OpConversionPattern<IREE::VM::ListSetRefOp>::OpConversionPattern;

 public:
  ListSetRefOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                         VMAnalysisCache &vmAnalysisCache)
      : OpConversionPattern<IREE::VM::ListSetRefOp>(typeConverter, context),
        vmAnalysisCache(vmAnalysisCache) {}

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::ListSetRefOp setOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = setOp.getContext();
    auto loc = setOp.getLoc();

    auto refOp = rewriter.create<emitc::ApplyOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t"),
        /*applicableOperator=*/StringAttr::get(ctx, "*"),
        /*operand=*/adaptor.getOperands()[0]);

    auto listDerefOp = failListNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_list_t*"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_deref"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refOp.getResult()},
        /*vmAnalysisCache=*/vmAnalysisCache);

    auto funcOp = setOp.getOperation()->getParentOfType<mlir::FuncOp>();

    auto ptr = vmAnalysisCache.find(funcOp.getOperation());
    if (ptr == vmAnalysisCache.end()) {
      return setOp.emitError() << "parent func op not found in cache.";
    }

    bool move =
        ptr->second.isLastValueUse(setOp.value(), setOp.getOperation()) &&
        false;

    StringRef callee =
        move ? "iree_vm_list_set_ref_move" : "iree_vm_list_set_ref_retain";

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, callee),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{listDerefOp.getResult(0), setOp.index(),
                        adaptor.getOperands()[2]},
        /*vmAnalysisCache=*/vmAnalysisCache);

    rewriter.eraseOp(setOp);

    return success();
  }

  VMAnalysisCache &vmAnalysisCache;
};
}  // namespace

void populateVMToEmitCPatterns(MLIRContext *context,
                               ConversionTarget &conversionTarget,
                               IREE::VM::EmitCTypeConverter &typeConverter,
                               OwningRewritePatternList &patterns,
                               VMAnalysisCache &vmAnalysisCache) {
  populateUtilConversionPatterns(context, conversionTarget, typeConverter,
                                 patterns);

  // CFG
  patterns.insert<BranchOpConversion>(typeConverter, context, vmAnalysisCache);
  patterns.insert<CallOpConversion<IREE::VM::CallOp>>(typeConverter, context,
                                                      vmAnalysisCache);
  patterns.insert<CallOpConversion<IREE::VM::CallVariadicOp>>(
      typeConverter, context, vmAnalysisCache);
  patterns.insert<CondBranchOpConversion>(typeConverter, context,
                                          vmAnalysisCache);
  patterns.insert<FailOpConversion>(typeConverter, context, vmAnalysisCache);
  patterns.insert<FuncOpConversion>(typeConverter, context, vmAnalysisCache);
  patterns.insert<ReturnOpConversion>(typeConverter, context, vmAnalysisCache);

  // Globals
  patterns.insert<
      GlobalLoadOpConversion<IREE::VM::GlobalLoadI32Op, IREE::VM::GlobalI32Op>>(
      typeConverter, context, "vm_global_load_i32");
  patterns.insert<GlobalStoreOpConversion<IREE::VM::GlobalStoreI32Op,
                                          IREE::VM::GlobalI32Op>>(
      typeConverter, context, "vm_global_store_i32");

  patterns.insert<GlobalLoadStoreRefOpConversion<IREE::VM::GlobalLoadRefOp>>(
      typeConverter, context, vmAnalysisCache);
  patterns.insert<GlobalLoadStoreRefOpConversion<IREE::VM::GlobalStoreRefOp>>(
      typeConverter, context, vmAnalysisCache);

  // Constants
  patterns.insert<ConstOpConversion<IREE::VM::ConstI32Op>>(typeConverter,
                                                           context);
  patterns.insert<ConstZeroOpConversion<IREE::VM::ConstI32ZeroOp>>(
      typeConverter, context);
  patterns.insert<ConstRefZeroOpConversion>(typeConverter, context,
                                            vmAnalysisCache);
  patterns.insert<ConstRefRodataOpConversion>(typeConverter, context,
                                              vmAnalysisCache);

  // List ops
  patterns.insert<ListAllocOpConversion>(typeConverter, context,
                                         vmAnalysisCache);
  patterns.insert<ListOpConversion<IREE::VM::ListReserveOp>>(
      typeConverter, context, "iree_vm_list_reserve", 0, true, vmAnalysisCache);
  patterns.insert<ListOpConversion<IREE::VM::ListResizeOp>>(
      typeConverter, context, "iree_vm_list_resize", 0, true, vmAnalysisCache);
  patterns.insert<ListOpConversion<IREE::VM::ListSizeOp>>(
      typeConverter, context, "iree_vm_list_size", 0, false, vmAnalysisCache);
  patterns.insert<ListGetOpConversion<IREE::VM::ListGetI32Op>>(
      typeConverter, context, vmAnalysisCache);
  patterns.insert<ListGetRefOpConversion>(typeConverter, context,
                                          vmAnalysisCache);
  patterns.insert<ListSetOpConversion<IREE::VM::ListSetI32Op>>(
      typeConverter, context, vmAnalysisCache);
  patterns.insert<ListSetRefOpConversion>(typeConverter, context,
                                          vmAnalysisCache);

  // Conditional assignment ops
  patterns.insert<GenericOpConversion<IREE::VM::SelectI32Op>>(
      typeConverter, context, "vm_select_i32");

  // Native integer arithmetic ops
  patterns.insert<GenericOpConversion<IREE::VM::AddI32Op>>(
      typeConverter, context, "vm_add_i32");
  patterns.insert<GenericOpConversion<IREE::VM::SubI32Op>>(
      typeConverter, context, "vm_sub_i32");
  patterns.insert<GenericOpConversion<IREE::VM::MulI32Op>>(
      typeConverter, context, "vm_mul_i32");
  patterns.insert<GenericOpConversion<IREE::VM::DivI32SOp>>(
      typeConverter, context, "vm_div_i32s");
  patterns.insert<GenericOpConversion<IREE::VM::DivI32UOp>>(
      typeConverter, context, "vm_div_i32u");
  patterns.insert<GenericOpConversion<IREE::VM::RemI32SOp>>(
      typeConverter, context, "vm_rem_i32s");
  patterns.insert<GenericOpConversion<IREE::VM::RemI32UOp>>(
      typeConverter, context, "vm_rem_i32u");
  patterns.insert<GenericOpConversion<IREE::VM::FMAI32Op>>(
      typeConverter, context, "vm_fma_i32");
  patterns.insert<GenericOpConversion<IREE::VM::NotI32Op>>(
      typeConverter, context, "vm_not_i32");
  patterns.insert<GenericOpConversion<IREE::VM::AndI32Op>>(
      typeConverter, context, "vm_and_i32");
  patterns.insert<GenericOpConversion<IREE::VM::OrI32Op>>(typeConverter,
                                                          context, "vm_or_i32");
  patterns.insert<GenericOpConversion<IREE::VM::XorI32Op>>(
      typeConverter, context, "vm_xor_i32");

  // Casting and type conversion/emulation ops
  patterns.insert<GenericOpConversion<IREE::VM::TruncI32I8Op>>(
      typeConverter, context, "vm_trunc_i32i8");
  patterns.insert<GenericOpConversion<IREE::VM::TruncI32I16Op>>(
      typeConverter, context, "vm_trunc_i32i16");
  patterns.insert<GenericOpConversion<IREE::VM::ExtI8I32SOp>>(
      typeConverter, context, "vm_ext_i8i32s");
  patterns.insert<GenericOpConversion<IREE::VM::ExtI8I32UOp>>(
      typeConverter, context, "vm_ext_i8i32u");
  patterns.insert<GenericOpConversion<IREE::VM::ExtI16I32SOp>>(
      typeConverter, context, "vm_ext_i16i32s");
  patterns.insert<GenericOpConversion<IREE::VM::ExtI16I32UOp>>(
      typeConverter, context, "vm_ext_i16i32u");

  // Native bitwise shift and rotate ops
  patterns.insert<GenericOpConversion<IREE::VM::ShlI32Op>>(
      typeConverter, context, "vm_shl_i32");
  patterns.insert<GenericOpConversion<IREE::VM::ShrI32SOp>>(
      typeConverter, context, "vm_shr_i32s");
  patterns.insert<GenericOpConversion<IREE::VM::ShrI32UOp>>(
      typeConverter, context, "vm_shr_i32u");

  // Comparison ops
  patterns.insert<GenericOpConversion<IREE::VM::CmpEQI32Op>>(
      typeConverter, context, "vm_cmp_eq_i32");
  patterns.insert<GenericOpConversion<IREE::VM::CmpNEI32Op>>(
      typeConverter, context, "vm_cmp_ne_i32");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTI32SOp>>(
      typeConverter, context, "vm_cmp_lt_i32s");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTI32UOp>>(
      typeConverter, context, "vm_cmp_lt_i32u");
  patterns.insert<GenericOpConversion<IREE::VM::CmpNZI32Op>>(
      typeConverter, context, "vm_cmp_nz_i32");
  patterns.insert<CompareRefOpConversion<IREE::VM::CmpEQRefOp>>(
      typeConverter, context, "vm_cmp_eq_ref", vmAnalysisCache);
  patterns.insert<CompareRefOpConversion<IREE::VM::CmpNERefOp>>(
      typeConverter, context, "vm_cmp_ne_ref", vmAnalysisCache);
  patterns.insert<CompareRefNotZeroOpConversion>(typeConverter, context,
                                                 vmAnalysisCache);

  // ExtF32: Globals
  patterns.insert<
      GlobalLoadOpConversion<IREE::VM::GlobalLoadF32Op, IREE::VM::GlobalF32Op>>(
      typeConverter, context, "vm_global_load_f32");
  patterns.insert<GlobalStoreOpConversion<IREE::VM::GlobalStoreF32Op,
                                          IREE::VM::GlobalF32Op>>(
      typeConverter, context, "vm_global_store_f32");

  // ExtF32: Native floating-point constants
  patterns.insert<ConstOpConversion<IREE::VM::ConstF32Op>>(typeConverter,
                                                           context);
  patterns.insert<ConstZeroOpConversion<IREE::VM::ConstF32ZeroOp>>(
      typeConverter, context);

  // ExtF32: Conditional assignment
  patterns.insert<GenericOpConversion<IREE::VM::SelectF32Op>>(
      typeConverter, context, "vm_select_f32");

  // ExtF32: Native floating-point arithmetic
  patterns.insert<GenericOpConversion<IREE::VM::AddF32Op>>(
      typeConverter, context, "vm_add_f32");
  patterns.insert<GenericOpConversion<IREE::VM::SubF32Op>>(
      typeConverter, context, "vm_sub_f32");
  patterns.insert<GenericOpConversion<IREE::VM::MulF32Op>>(
      typeConverter, context, "vm_mul_f32");
  patterns.insert<GenericOpConversion<IREE::VM::DivF32Op>>(
      typeConverter, context, "vm_div_f32");
  patterns.insert<GenericOpConversion<IREE::VM::RemF32Op>>(
      typeConverter, context, "vm_rem_f32");
  patterns.insert<GenericOpConversion<IREE::VM::FMAF32Op>>(
      typeConverter, context, "vm_fma_f32");
  patterns.insert<GenericOpConversion<IREE::VM::AbsF32Op>>(
      typeConverter, context, "vm_abs_f32");
  patterns.insert<GenericOpConversion<IREE::VM::NegF32Op>>(
      typeConverter, context, "vm_neg_f32");
  patterns.insert<GenericOpConversion<IREE::VM::CeilF32Op>>(
      typeConverter, context, "vm_ceil_f32");
  patterns.insert<GenericOpConversion<IREE::VM::FloorF32Op>>(
      typeConverter, context, "vm_floor_f32");

  patterns.insert<GenericOpConversion<IREE::VM::AtanF32Op>>(
      typeConverter, context, "vm_atan_f32");
  patterns.insert<GenericOpConversion<IREE::VM::Atan2F32Op>>(
      typeConverter, context, "vm_atan2_f32");
  patterns.insert<GenericOpConversion<IREE::VM::CosF32Op>>(
      typeConverter, context, "vm_cos_f32");
  patterns.insert<GenericOpConversion<IREE::VM::SinF32Op>>(
      typeConverter, context, "vm_sin_f32");
  patterns.insert<GenericOpConversion<IREE::VM::ExpF32Op>>(
      typeConverter, context, "vm_exp_f32");
  patterns.insert<GenericOpConversion<IREE::VM::Exp2F32Op>>(
      typeConverter, context, "vm_exp2_f32");
  patterns.insert<GenericOpConversion<IREE::VM::ExpM1F32Op>>(
      typeConverter, context, "vm_expm1_f32");
  patterns.insert<GenericOpConversion<IREE::VM::LogF32Op>>(
      typeConverter, context, "vm_log_f32");
  patterns.insert<GenericOpConversion<IREE::VM::Log10F32Op>>(
      typeConverter, context, "vm_log10_f32");
  patterns.insert<GenericOpConversion<IREE::VM::Log1pF32Op>>(
      typeConverter, context, "vm_log1p_f32");
  patterns.insert<GenericOpConversion<IREE::VM::Log2F32Op>>(
      typeConverter, context, "vm_log2_f32");
  patterns.insert<GenericOpConversion<IREE::VM::PowF32Op>>(
      typeConverter, context, "vm_pow_f32");
  patterns.insert<GenericOpConversion<IREE::VM::RsqrtF32Op>>(
      typeConverter, context, "vm_rsqrt_f32");
  patterns.insert<GenericOpConversion<IREE::VM::SqrtF32Op>>(
      typeConverter, context, "vm_sqrt_f32");
  patterns.insert<GenericOpConversion<IREE::VM::TanhF32Op>>(
      typeConverter, context, "vm_tanh_f32");

  // ExtF32: Casting and type conversion/emulation
  patterns.insert<GenericOpConversion<IREE::VM::CastSI32F32Op>>(
      typeConverter, context, "vm_cast_si32f32");
  patterns.insert<GenericOpConversion<IREE::VM::CastUI32F32Op>>(
      typeConverter, context, "vm_cast_ui32f32");
  patterns.insert<GenericOpConversion<IREE::VM::CastF32SI32Op>>(
      typeConverter, context, "vm_cast_f32si32");
  patterns.insert<GenericOpConversion<IREE::VM::CastF32UI32Op>>(
      typeConverter, context, "vm_cast_f32ui32");
  patterns.insert<GenericOpConversion<IREE::VM::BitcastI32F32Op>>(
      typeConverter, context, "vm_bitcast_i32f32");
  patterns.insert<GenericOpConversion<IREE::VM::BitcastF32I32Op>>(
      typeConverter, context, "vm_bitcast_f32i32");

  // ExtF32: Comparison ops
  patterns.insert<GenericOpConversion<IREE::VM::CmpEQF32OOp>>(
      typeConverter, context, "vm_cmp_eq_f32o");
  patterns.insert<GenericOpConversion<IREE::VM::CmpEQF32UOp>>(
      typeConverter, context, "vm_cmp_eq_f32u");
  patterns.insert<GenericOpConversion<IREE::VM::CmpNEF32OOp>>(
      typeConverter, context, "vm_cmp_ne_f32o");
  patterns.insert<GenericOpConversion<IREE::VM::CmpNEF32UOp>>(
      typeConverter, context, "vm_cmp_ne_f32u");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTF32OOp>>(
      typeConverter, context, "vm_cmp_lt_f32o");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTF32UOp>>(
      typeConverter, context, "vm_cmp_lt_f32u");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTEF32OOp>>(
      typeConverter, context, "vm_cmp_lte_f32o");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTEF32UOp>>(
      typeConverter, context, "vm_cmp_lte_f32u");
  patterns.insert<GenericOpConversion<IREE::VM::CmpNaNF32Op>>(
      typeConverter, context, "vm_cmp_nan_f32");

  // ExtI64: Globals
  patterns.insert<
      GlobalLoadOpConversion<IREE::VM::GlobalLoadI64Op, IREE::VM::GlobalI64Op>>(
      typeConverter, context, "vm_global_load_i64");
  patterns.insert<GlobalStoreOpConversion<IREE::VM::GlobalStoreI64Op,
                                          IREE::VM::GlobalI64Op>>(
      typeConverter, context, "vm_global_store_i64");

  // ExtI64: Constants
  patterns.insert<ConstOpConversion<IREE::VM::ConstI64Op>>(typeConverter,
                                                           context);
  patterns.insert<ConstZeroOpConversion<IREE::VM::ConstI64ZeroOp>>(
      typeConverter, context);

  // ExtI64: List ops
  patterns.insert<ListGetOpConversion<IREE::VM::ListGetI64Op>>(
      typeConverter, context, vmAnalysisCache);
  patterns.insert<ListSetOpConversion<IREE::VM::ListSetI64Op>>(
      typeConverter, context, vmAnalysisCache);

  // ExtI64: Conditional assignment ops
  patterns.insert<GenericOpConversion<IREE::VM::SelectI64Op>>(
      typeConverter, context, "vm_select_i64");
  // ExtI64: Native integer arithmetic ops
  patterns.insert<GenericOpConversion<IREE::VM::AddI64Op>>(
      typeConverter, context, "vm_add_i64");
  patterns.insert<GenericOpConversion<IREE::VM::SubI64Op>>(
      typeConverter, context, "vm_sub_i64");
  patterns.insert<GenericOpConversion<IREE::VM::MulI64Op>>(
      typeConverter, context, "vm_mul_i64");
  patterns.insert<GenericOpConversion<IREE::VM::DivI64SOp>>(
      typeConverter, context, "vm_div_i64s");
  patterns.insert<GenericOpConversion<IREE::VM::DivI64UOp>>(
      typeConverter, context, "vm_div_i64u");
  patterns.insert<GenericOpConversion<IREE::VM::RemI64SOp>>(
      typeConverter, context, "vm_rem_i64s");
  patterns.insert<GenericOpConversion<IREE::VM::RemI64UOp>>(
      typeConverter, context, "vm_rem_i64u");
  patterns.insert<GenericOpConversion<IREE::VM::FMAI64Op>>(
      typeConverter, context, "vm_fma_i64");
  patterns.insert<GenericOpConversion<IREE::VM::NotI64Op>>(
      typeConverter, context, "vm_not_i64");
  patterns.insert<GenericOpConversion<IREE::VM::AndI64Op>>(
      typeConverter, context, "vm_and_i64");
  patterns.insert<GenericOpConversion<IREE::VM::OrI64Op>>(typeConverter,
                                                          context, "vm_or_i64");
  patterns.insert<GenericOpConversion<IREE::VM::XorI64Op>>(
      typeConverter, context, "vm_xor_i64");

  // ExtI64: Casting and type conversion/emulation ops
  patterns.insert<GenericOpConversion<IREE::VM::TruncI64I32Op>>(
      typeConverter, context, "vm_trunc_i64i32");
  patterns.insert<GenericOpConversion<IREE::VM::ExtI32I64SOp>>(
      typeConverter, context, "vm_ext_i32i64s");
  patterns.insert<GenericOpConversion<IREE::VM::ExtI32I64UOp>>(
      typeConverter, context, "vm_ext_i32i64u");

  // ExtI64: Native bitwise shift and rotate ops
  patterns.insert<GenericOpConversion<IREE::VM::ShlI64Op>>(
      typeConverter, context, "vm_shl_i64");
  patterns.insert<GenericOpConversion<IREE::VM::ShrI64SOp>>(
      typeConverter, context, "vm_shr_i64s");
  patterns.insert<GenericOpConversion<IREE::VM::ShrI64UOp>>(
      typeConverter, context, "vm_shr_i64u");

  // ExtI64: Comparison ops
  patterns.insert<GenericOpConversion<IREE::VM::CmpEQI64Op>>(
      typeConverter, context, "vm_cmp_eq_i64");
  patterns.insert<GenericOpConversion<IREE::VM::CmpNEI64Op>>(
      typeConverter, context, "vm_cmp_ne_i64");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTI64SOp>>(
      typeConverter, context, "vm_cmp_lt_i64s");
  patterns.insert<GenericOpConversion<IREE::VM::CmpLTI64UOp>>(
      typeConverter, context, "vm_cmp_lt_i64u");
  patterns.insert<GenericOpConversion<IREE::VM::CmpNZI64Op>>(
      typeConverter, context, "vm_cmp_nz_i64");
}

namespace IREE {
namespace VM {

namespace {

// A pass converting IREE VM operations into the EmitC dialect.
// vm.func ops get converted to std.func with the calling convention used by
// EmitC. Each function gets three additional arguments a `iree_vm_stack_t*` as
// well as two module specific struct pointers (`{module_name}_t*` and
// `{module_name}_state_t`). These are followed by the original function
// arguments and out arguments for the vm.func results. The result type of the
// function is `iree_status_t`. Ref types are always passed as pointers.
//
// Examples:
//   () -> () => (iree_vm_stack_t*, module_t*, module_state_t*) -> iree_status_t
//
//   (i) -> () => (iree_vm_stack_t*, module_t*, module_state_t*, int32_t) ->
//                  iree_status_t
//
//   (r) -> () => (iree_vm_stack_t*, module_t*, module_state_t*, iree_vm_ref_t*)
//                  -> iree_status_t
//
//   () -> (r) => (iree_vm_stack_t*, module_t*, module_state_t*, iree_vm_ref_t*)
//                  -> iree_status_t
//
//   (iir) -> (ri) => (iree_vm_stack_t*, module_t*, module_state_t*, int32_t,
//                      int32_t, iree_vm_ref_t*, iree_vm_ref_t*, int32_t*) ->
//                      iree_status_t
class ConvertVMToEmitCPass
    : public PassWrapper<ConvertVMToEmitCPass,
                         OperationPass<IREE::VM::ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::emitc::EmitCDialect, mlir::BuiltinDialect,
                    mlir::StandardOpsDialect, mlir::arith::ArithmeticDialect,
                    mlir::math::MathDialect, IREE::Util::UtilDialect>();
  }

  StringRef getArgument() const override { return "iree-convert-vm-to-emitc"; }

  StringRef getDescription() const override {
    return "Convert VM Ops to the EmitC dialect";
  }

  void runOnOperation() override {
    IREE::VM::ModuleOp module = getOperation();

    // Replace illegal characters in symbol names and replace them with "_"
    if (failed(sanitizeSymbolNames(module))) return signalPassFailure();

    ConversionTarget target(getContext());
    EmitCTypeConverter typeConverter;

    // Run analysis passes
    VMAnalysisCache vmAnalysisCache;

    // Convert vm.func ops to std.func with the calling convention used by
    // EmitC. We convert these upfront to make sure vm.call ops always
    // reference std.func ops with the correct calling convention during the
    // conversion.
    SmallVector<IREE::VM::FuncOp, 4> funcsToRemove;
    SmallVector<BlockArgument, 4> blockArgsToRemove;
    for (auto funcOp : module.getOps<IREE::VM::FuncOp>()) {
      Operation *op = funcOp.getOperation();
      vmAnalysisCache.insert(std::make_pair(
          op, VMAnalysis{RegisterAllocation(op), ValueLiveness(op)}));

      if (failed(convertFuncOp(funcOp, typeConverter, vmAnalysisCache,
                               blockArgsToRemove))) {
        return signalPassFailure();
      }
      funcsToRemove.push_back(funcOp);
    }

    for (auto &funcOp : funcsToRemove) {
      funcOp.erase();
    }

    // Generate func ops that implement the C API.
    if (failed(createAPIFunctions(module, vmAnalysisCache))) {
      return signalPassFailure();
    }

    OwningRewritePatternList patterns(&getContext());
    populateVMToEmitCPatterns(&getContext(), target, typeConverter, patterns,
                              vmAnalysisCache);

    target.addLegalDialect<
        emitc::EmitCDialect, mlir::BuiltinDialect, mlir::StandardOpsDialect,
        mlir::arith::ArithmeticDialect, mlir::math::MathDialect>();

    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });

    // Structural ops
    target.addLegalOp<IREE::VM::ModuleOp>();
    target.addLegalOp<IREE::VM::ModuleTerminatorOp>();
    // These ops are needed to build arrays for the module descriptor. There is
    // no way to generate this directly with the EmitC dialect at the moment.
    target.addLegalOp<IREE::VM::ExportOp>();
    target.addLegalOp<IREE::VM::ImportOp>();

    // Global ops
    // The global ops are dead after the conversion and will get removed.
    target.addLegalOp<IREE::VM::GlobalI32Op>();
    target.addLegalOp<IREE::VM::GlobalI64Op>();
    target.addLegalOp<IREE::VM::GlobalF32Op>();
    target.addLegalOp<IREE::VM::GlobalRefOp>();

    // This op is needed in the printer to emit an array holding the data.
    target.addLegalOp<IREE::VM::RodataOp>();

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Remove unused block arguments from refs
    if (failed(removeBlockArguments(module, blockArgsToRemove))) {
      return signalPassFailure();
    }

    SetVector<Operation *> &materializations =
        typeConverter.sourceMaterializations;

    module.walk([&materializations](Operation *op) {
      // Global ops are dead now
      if (isa<IREE::VM::GlobalI32Op, IREE::VM::GlobalI64Op,
              IREE::VM::GlobalF32Op, IREE::VM::GlobalRefOp>(op)) {
        op->erase();
        return;
      }
      // Remove dead basic block arguments
      if (materializations.contains(op)) {
        assert(isa<emitc::ConstantOp>(op));
        assert(op->use_empty());

        materializations.remove(op);
        op->erase();
        return;
      }
    });
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
