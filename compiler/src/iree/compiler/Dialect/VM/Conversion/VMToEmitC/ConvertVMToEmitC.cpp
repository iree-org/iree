// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/ConvertVMToEmitC.h"

#include <optional>

#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCBuilders.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/VMAnalysis.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

enum {
  SHIM_ARGUMENT_STACK = 0,
  SHIM_ARGUMENT_FLAGS,
  SHIM_ARGUMENT_ARGS_STORAGE,
  SHIM_ARGUMENT_RETS_STORAGE,
  SHIM_ARGUMENT_MODULE,
  SHIM_ARGUMENT_MODULE_STATE,
};

enum {
  CCONV_ARGUMENT_STACK = 0,
  CCONV_ARGUMENT_MODULE,
  CCONV_ARGUMENT_MODULE_STATE,
};

LogicalResult convertFuncOp(IREE::VM::FuncOp funcOp,
                            const IREE::VM::EmitCTypeConverter &typeConverter) {
  auto ctx = funcOp.getContext();
  auto loc = funcOp.getLoc();

  OpBuilder builder(funcOp);

  auto moduleOp = funcOp.getOperation()->getParentOfType<IREE::VM::ModuleOp>();

  FunctionType funcType = funcOp.getFunctionType();
  std::string name = moduleOp.getName().str() + "_" + funcOp.getName().str();
  std::string moduleTypeName =
      std::string("struct ") + moduleOp.getName().str() + "_t";
  std::string moduleStateTypeName =
      std::string("struct ") + moduleOp.getName().str() + "_state_t";

  Type stackType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_stack_t"));
  Type moduleType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, moduleTypeName));
  Type moduleStateType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, moduleStateTypeName));

  SmallVector<Type, 3> inputTypes = {stackType, moduleType, moduleStateType};
  SmallVector<Type, 1> outputTypes;

  for (auto &inputType : funcType.getInputs()) {
    inputTypes.push_back(inputType);
  }

  for (auto &resultType : funcType.getResults()) {
    // We pass refs as iree_vm_ref_t* regardless of whether it is an in or out
    // parameter
    Type type = typeConverter.convertTypeAsPointer(resultType);

    inputTypes.push_back(type);
    outputTypes.push_back(type);
  }

  auto newFuncType = mlir::FunctionType::get(
      ctx, {inputTypes}, {emitc::OpaqueType::get(ctx, "iree_status_t")});

  auto newFuncOp = builder.create<mlir::emitc::FuncOp>(loc, name, newFuncType);
  newFuncOp.setSpecifiersAttr(
      builder.getArrayAttr({builder.getStringAttr("static")}));
  newFuncOp.setPrivate();

  // This call shold be equivalent to rewriter.inlineRegionBefore()
  newFuncOp.getFunctionBody().getBlocks().splice(
      newFuncOp.end(), funcOp.getFunctionBody().getBlocks());

  Block &entryBlock = newFuncOp.getBlocks().front();

  if (!entryBlock.hasNoPredecessors()) {
    return funcOp.emitError()
           << "branches to the entry block are not supported for now.";
  }

  entryBlock.insertArgument(static_cast<unsigned>(0), stackType, loc);
  entryBlock.insertArgument(static_cast<unsigned>(1), moduleType, loc);
  entryBlock.insertArgument(static_cast<unsigned>(2), moduleStateType, loc);

  SmallVector<Location> locs(outputTypes.size(), loc);
  entryBlock.addArguments(outputTypes, locs);

  typeConverter.analysis.move(newFuncOp, funcOp);

  auto &funcAnalysis = typeConverter.analysis.lookupFunction(newFuncOp);

  // Add variable ops for local refs
  const int numRefArgs = funcAnalysis.getNumRefArguments();
  const int numLocalRefs = funcAnalysis.getNumLocalRefs();

  builder.setInsertionPointToStart(&entryBlock);

  for (int i = 0; i < numLocalRefs; i++) {
    auto [ref, refPtr] = emitc_builders::allocZeroInitializedVar(
        builder, loc, emitc::OpaqueType::get(ctx, "iree_vm_ref_t"));

    // Cache local refs so that we can release them before a return operation.
    // Here we rely on the fact that the register allocation maps arguments in
    // the first slots.
    funcAnalysis.cacheLocalRef(i + numRefArgs, refPtr);
  }

  if (failed(
          funcOp.replaceAllSymbolUses(builder.getStringAttr(name), moduleOp)))
    return funcOp.emitError() << "unable to update symbol name in module";

  return success();
}

std::optional<std::string> buildFunctionName(IREE::VM::ModuleOp &moduleOp,
                                             IREE::VM::ImportOp &importOp) {
  auto callingConvention = makeImportCallingConventionString(importOp);
  if (!callingConvention.has_value()) {
    return std::nullopt;
  }
  return moduleOp.getName().str() + "_call_" + callingConvention.value() +
         "_import_shim";
}

std::optional<std::string>
buildVariadicFunctionName(IREE::VM::ModuleOp &moduleOp,
                          IREE::VM::ImportOp &importOp,
                          DenseIntElementsAttr segmentSizes) {
  auto callingConvention = makeImportCallingConventionString(importOp);
  if (!callingConvention.has_value()) {
    return std::nullopt;
  }
  std::string result(moduleOp.getName());
  result += "_call_";
  result += callingConvention.value();
  for (int i = 0; i < importOp.getNumArguments(); i++) {
    if (importOp.isFuncArgumentVariadic(i)) {
      APInt size = *(segmentSizes.begin() + i);
      result += "_";
      result += std::to_string(size.getSExtValue());
    }
  }
  result += "_import_shim";
  return result;
}

std::optional<Value>
createVmTypeDefPtr(ConversionPatternRewriter &rewriter, Location loc,
                   const IREE::VM::ModuleAnalysis &moduleAnalysis,
                   IREE::VM::ModuleOp moduleOp, BlockArgument moduleArg,
                   Type elementType) {
  auto ctx = rewriter.getContext();

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
  };
  Value elementTypeValue;

  auto ptr = valueTypeMap.find((elementType));
  if (ptr != valueTypeMap.end()) {
    elementTypeValue =
        rewriter
            .create<emitc::CallOpaqueOp>(
                /*location=*/loc,
                /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_type_def_t"),
                /*callee=*/"iree_vm_make_value_type_def",
                /*operands=*/ArrayRef<Value>{},
                /*args=*/
                ArrayAttr::get(
                    ctx, {emitc::OpaqueAttr::get(ctx, ptr->second.first)}))
            .getResult(0);
  } else if (llvm::isa<IREE::VM::RefType>(elementType)) {
    Type objType = llvm::cast<IREE::VM::RefType>(elementType).getObjectType();

    Type typeRefType = emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t");
    Type typeRefArrayType = emitc::PointerType::get(typeRefType);
    std::optional<size_t> typeIndex = moduleAnalysis.lookupType(objType);
    if (!typeIndex.has_value()) {
      moduleOp.emitError("type index lookup failed");
      return std::nullopt;
    }

    TypedValue<emitc::LValueType> moduleArgLValue =
        emitc_builders::asLValue(rewriter, loc, moduleArg);

    TypedValue<emitc::PointerType> typeArray =
        cast<TypedValue<emitc::PointerType>>(emitc_builders::structPtrMember(
            rewriter, loc, typeRefArrayType, "types", moduleArgLValue));
    Value refType = emitc_builders::arrayElement(rewriter, loc,
                                                 typeIndex.value(), typeArray);

    elementTypeValue =
        rewriter
            .create<emitc::CallOpaqueOp>(
                /*location=*/loc,
                /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_type_def_t"),
                /*callee=*/"iree_vm_make_ref_type_def",
                /*operands=*/ArrayRef<Value>{refType})
            .getResult(0);
  } else if (llvm::isa<IREE::VM::OpaqueType>(elementType)) {
    elementTypeValue =
        rewriter
            .create<emitc::CallOpaqueOp>(
                /*location=*/loc,
                /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_type_def_t"),
                /*callee=*/"iree_vm_make_undefined_type_def",
                /*operands=*/ArrayRef<Value>{})
            .getResult(0);
  }

  return elementTypeValue;
}

/// Move multiple refs from one set of variables to another set. As these two
/// sets may alias we move the source variables into temporaries first.
/// The generated code works as follows:
/// `isMove` == true:
///    move(src_i, tmp_i); for all i
///    move(tmp_i, dest_i); for all i
/// `isMove` == false:
///    retain(src_i, tmp_i); for all i
///    assign(tmp_i, dest_i); for all i
LogicalResult retainOrMoveRefs(OpBuilder &builder, Location location,
                               IRMapping mapping, bool isMove) {
  auto ctx = builder.getContext();

  IRMapping tmpMapping;
  for (auto &[srcRef, destRef] : mapping.getValueMap()) {
    assert(srcRef.getType() == emitc::PointerType::get(emitc::OpaqueType::get(
                                   ctx, "iree_vm_ref_t")));

    auto [tmpRef, tmpPtr] = emitc_builders::allocZeroInitializedVar(
        builder, location, emitc::OpaqueType::get(ctx, "iree_vm_ref_t"));

    StringRef callee = isMove ? "iree_vm_ref_move" : "iree_vm_ref_retain";
    builder.create<emitc::CallOpaqueOp>(
        /*location=*/location,
        /*type=*/TypeRange{},
        /*callee=*/callee,
        /*operands=*/ArrayRef<Value>{srcRef, tmpPtr});

    tmpMapping.map(srcRef, tmpPtr);
  }

  for (const auto &[srcRef, destRef] : mapping.getValueMap()) {
    Value tmpRef = tmpMapping.lookup(srcRef);

    StringRef callee = isMove ? "iree_vm_ref_move" : "iree_vm_ref_assign";

    builder.create<emitc::CallOpaqueOp>(
        /*location=*/location,
        /*type=*/TypeRange{},
        /*callee=*/callee,
        /*operands=*/ArrayRef<Value>{tmpRef, destRef});
  }

  return success();
}

/// Releases refs which are local to the function as well as ref arguments.
void releaseRefs(OpBuilder &builder, Location location,
                 mlir::emitc::FuncOp funcOp,
                 IREE::VM::ModuleAnalysis &moduleAnalysis) {
  auto ctx = builder.getContext();

  auto &funcAnalysis = moduleAnalysis.lookupFunction(funcOp);

  if (funcAnalysis.hasLocalRefs()) {

    for (auto &[key, localRef] : funcAnalysis.localRefs()) {
      emitc_builders::ireeVmRefRelease(builder, location, localRef);
    }
  }

  // We only release the original arguments not the results which were appended
  // as further operands.
  size_t refArgumentsReleased = 0;
  for (auto arg : funcOp.getArguments()) {
    if (arg.getType() ==
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_ref_t"))) {
      if (funcAnalysis.getNumRefArguments() <= refArgumentsReleased++) {
        break;
      }
      emitc_builders::ireeVmRefRelease(builder, location, arg);
    }
  }
}

/// Generate an emitc.call_opaque op with one result and split the current block
/// into a continuation and failure block based on the truthiness of the result
/// value, i.e. a truthy value branches to the continuation block when
/// `negateCondition` is false.
emitc::CallOpaqueOp failableCall(
    OpBuilder &builder, Location location, Type type, StringRef callee,
    ArrayAttr args, ArrayRef<Value> operands,
    const std::function<void(emitc::CallOpaqueOp &)> &failureBlockBuilder,
    bool negateCondition = false) {
  auto callOp = builder.create<emitc::CallOpaqueOp>(
      /*location=*/location,
      /*type=*/type,
      /*callee=*/callee,
      /*operands=*/operands,
      /*args=*/args);

  Type boolType = builder.getIntegerType(1);

  auto conditionI1 = builder.create<emitc::CastOp>(
      /*location=*/location,
      /*type=*/boolType,
      /*operand=*/callOp.getResult(0));

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
  builder.create<mlir::cf::CondBranchOp>(
      location, conditionI1.getResult(),
      negateCondition ? failureBlock : continuationBlock,
      negateCondition ? continuationBlock : failureBlock);

  builder.setInsertionPointToStart(continuationBlock);

  return callOp;
}

emitc::CallOpaqueOp returnIfError(OpBuilder &builder, Location location,
                                  StringRef callee, ArrayAttr args,
                                  ArrayRef<Value> operands,
                                  IREE::VM::ModuleAnalysis &moduleAnalysis) {
  auto blockBuilder = [&builder, &location,
                       &moduleAnalysis](emitc::CallOpaqueOp &callOp) {
    Block *block = builder.getBlock();
    mlir::emitc::FuncOp funcOp =
        cast<mlir::emitc::FuncOp>(block->getParentOp());

    releaseRefs(builder, location, funcOp, moduleAnalysis);

    builder.create<mlir::emitc::ReturnOp>(location, callOp.getResult(0));
  };

  auto ctx = builder.getContext();
  Type type = emitc::OpaqueType::get(ctx, "iree_status_t");
  return failableCall(builder, location, type, callee, args, operands,
                      blockBuilder, /*negateCondition=*/true);
}

emitc::CallOpaqueOp
failContainerNull(OpBuilder &builder, Location location, Type type,
                  StringRef callee, ArrayAttr args, ArrayRef<Value> operands,
                  IREE::VM::ModuleAnalysis &moduleAnalysis) {
  auto blockBuilder = [&builder, &location,
                       &moduleAnalysis](emitc::CallOpaqueOp &callOp) {
    auto ctx = builder.getContext();

    Block *block = builder.getBlock();
    mlir::emitc::FuncOp funcOp =
        cast<mlir::emitc::FuncOp>(block->getParentOp());

    releaseRefs(builder, location, funcOp, moduleAnalysis);

    auto statusOp = builder.create<emitc::CallOpaqueOp>(
        /*location=*/location,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
        /*callee=*/"iree_make_status",
        /*operands=*/ArrayRef<Value>{},
        /*args=*/
        ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(
                                ctx, "IREE_STATUS_INVALID_ARGUMENT")}));

    builder.create<mlir::emitc::ReturnOp>(location, statusOp.getResult(0));
  };

  return failableCall(builder, location, type, callee, args, operands,
                      blockBuilder);
}

/// Generate a emitc.call op with one result and split the current block into a
/// continuation and failure block based on the truthiness of the result
/// value, i.e. a truthy value branches to the continuation block when
/// `negateCondition` is false.
mlir::emitc::CallOp failableCall(
    OpBuilder &builder, Location location, mlir::emitc::FuncOp &callee,
    ArrayRef<Value> operands,
    const std::function<void(mlir::emitc::CallOp &)> &failureBlockBuilder,
    bool negateCondition = false) {
  auto callOp = builder.create<mlir::emitc::CallOp>(
      /*location=*/location,
      /*callee=*/callee,
      /*operands=*/operands);

  Type boolType = builder.getIntegerType(1);

  auto conditionI1 = builder.create<emitc::CastOp>(
      /*location=*/location,
      /*type=*/boolType,
      /*operand=*/callOp.getResult(0));

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
  builder.create<mlir::cf::CondBranchOp>(
      location, conditionI1.getResult(),
      negateCondition ? failureBlock : continuationBlock,
      negateCondition ? continuationBlock : failureBlock);

  builder.setInsertionPointToStart(continuationBlock);

  return callOp;
}

mlir::emitc::CallOp returnIfError(OpBuilder &builder, Location location,
                                  mlir::emitc::FuncOp &callee,
                                  ArrayRef<Value> operands,
                                  IREE::VM::ModuleAnalysis &moduleAnalysis) {
  auto blockBuilder = [&builder, &location,
                       &moduleAnalysis](mlir::emitc::CallOp &callOp) {
    Block *block = builder.getBlock();
    mlir::emitc::FuncOp funcOp =
        cast<mlir::emitc::FuncOp>(block->getParentOp());

    releaseRefs(builder, location, funcOp, moduleAnalysis);

    builder.create<mlir::emitc::ReturnOp>(location, callOp.getResult(0));
  };

  return failableCall(builder, location, callee, operands, blockBuilder,
                      /*negateCondition=*/true);
}

LogicalResult createAPIFunctions(IREE::VM::ModuleOp moduleOp,
                                 IREE::VM::ModuleAnalysis &moduleAnalysis) {
  auto ctx = moduleOp.getContext();
  auto loc = moduleOp.getLoc();

  OpBuilder builder(moduleOp);
  builder.setInsertionPoint(moduleOp.getBlock().getTerminator());

  std::string moduleName{moduleOp.getName()};

  // void destroy(void*)
  {
    OpBuilder::InsertionGuard guard(builder);

    const int moduleArgIndex = 0;

    auto funcType = mlir::FunctionType::get(
        ctx, {emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"))},
        {});

    auto funcOp = builder.create<mlir::emitc::FuncOp>(
        loc, moduleName + "_destroy", funcType);
    funcOp.setSpecifiersAttr(
        builder.getArrayAttr({builder.getStringAttr("static")}));
    funcOp.setPrivate();

    moduleAnalysis.addDummy(funcOp, /*emitAtEnd=*/false);

    Block *entryBlock = funcOp.addEntryBlock();
    const BlockArgument moduleArg = funcOp.getArgument(moduleArgIndex);

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleTypeName = std::string("struct ") + moduleName + "_t";

    auto castedModuleOp = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, moduleTypeName)),
        /*operand=*/moduleArg);

    auto castedModuleOpLValue =
        emitc_builders::asLValue(builder, loc, castedModuleOp.getResult());

    auto allocatorOp = emitc_builders::structPtrMember(
        builder, loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_allocator_t"),
        /*memberName=*/"allocator",
        /*operand=*/castedModuleOpLValue);

    builder.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/"iree_allocator_free",
        /*operands=*/
        ArrayRef<Value>{allocatorOp, castedModuleOp.getResult()});

    builder.create<mlir::emitc::ReturnOp>(loc, nullptr);
  }

  // iree_status_t alloc_state(void*, iree_allocator_t,
  // iree_vm_module_state_t**)
  {
    OpBuilder::InsertionGuard guard(builder);

    const int allocatorArgIndex = 1;
    const int moduleStateArgIndex = 2;

    auto funcType = mlir::FunctionType::get(
        ctx,
        {emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void")),
         emitc::OpaqueType::get(ctx, "iree_allocator_t"),
         emitc::PointerType::get(emitc::PointerType::get(
             emitc::OpaqueType::get(ctx, "iree_vm_module_state_t")))},
        {emitc::OpaqueType::get(ctx, "iree_status_t")});

    auto funcOp = builder.create<mlir::emitc::FuncOp>(
        loc, moduleName + "_alloc_state", funcType);
    funcOp.setSpecifiersAttr(
        builder.getArrayAttr({builder.getStringAttr("static")}));
    funcOp.setPrivate();

    moduleAnalysis.addDummy(funcOp, /*emitAtEnd=*/false);

    Block *entryBlock = funcOp.addEntryBlock();

    const BlockArgument allocatorArg = funcOp.getArgument(allocatorArgIndex);
    const BlockArgument moduleStateArg =
        funcOp.getArgument(moduleStateArgIndex);

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleStateTypeName =
        std::string("struct ") + moduleName + "_state_t";

    auto state = emitc_builders::allocateVariable(
        builder, loc,
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, moduleStateTypeName)),
        {"NULL"});

    Value stateSize = emitc_builders::sizeOf(
        builder, loc, emitc::OpaqueAttr::get(ctx, moduleStateTypeName));

    Value statePtr = emitc_builders::addressOf(builder, loc, state);

    auto voidPtr = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"))),
        /*operand=*/statePtr);

    returnIfError(builder, loc, "iree_allocator_malloc", {},
                  {allocatorArg, stateSize, voidPtr.getResult()},
                  moduleAnalysis);

    auto stateRValue = emitc_builders::asRValue(builder, loc, state);
    emitc_builders::memset(builder, loc, stateRValue, 0, stateSize);

    emitc_builders::structPtrMemberAssign(builder, loc,
                                          /*memberName=*/"allocator",
                                          /*operand=*/state,
                                          /*value=*/allocatorArg);

    // Initialize buffers
    for (auto rodataOp : moduleOp.getOps<IREE::VM::RodataOp>()) {
      auto ordinal = rodataOp.getOrdinal()->getZExtValue();

      std::string bufferName = moduleName + "_" + rodataOp.getName().str();

      Type type =
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));

      auto rodataPointer =
          emitc_builders::allocateVariable(builder, loc, type, {bufferName});
      auto rodataPointerRValue =
          emitc_builders::asRValue(builder, loc, rodataPointer);

      auto bufferVoid = builder.create<emitc::CastOp>(
          /*location=*/loc,
          /*type=*/emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void")),
          /*operand=*/rodataPointerRValue);

      Value bufferSize = emitc_builders::sizeOf(
          builder, loc, emitc::OpaqueAttr::get(ctx, bufferName));

      auto byteSpan = builder.create<emitc::CallOpaqueOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_byte_span_t"),
          /*callee=*/"iree_make_byte_span",
          /*operands=*/ArrayRef<Value>{bufferVoid.getResult(), bufferSize});

      auto allocator = builder.create<emitc::CallOpaqueOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_allocator_t"),
          /*callee=*/"iree_allocator_null",
          /*operands=*/ArrayRef<Value>{});

      auto buffers =
          cast<TypedValue<emitc::PointerType>>(emitc_builders::structPtrMember(
              builder, loc, /*type=*/
              emitc::PointerType::get(
                  emitc::OpaqueType::get(ctx, "iree_vm_buffer_t")),
              /*memberName=*/"rodata_buffers", /*operand=*/state));

      auto buffer = emitc_builders::arrayElementAddress(
          builder, loc, /*index=*/ordinal, /*operand=*/buffers);

      builder.create<emitc::CallOpaqueOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/"iree_vm_buffer_initialize",
          /*operands=*/
          ArrayRef<Value>{byteSpan.getResult(0), allocator.getResult(0),
                          buffer},
          /*args=*/
          ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(
                                   ctx, "IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE"),
                               builder.getIndexAttr(0), builder.getIndexAttr(1),
                               builder.getIndexAttr(2)}));
    }

    auto baseStateOp = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_module_state_t")),
        /*operand=*/stateRValue);

    builder.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/"EMITC_DEREF_ASSIGN_VALUE",
        /*operands=*/ArrayRef<Value>{moduleStateArg, baseStateOp.getResult()});

    auto status = emitc_builders::ireeOkStatus(builder, loc);

    builder.create<mlir::emitc::ReturnOp>(loc, status);
  }

  // void free_state(void*, iree_vm_module_state_t*)
  {
    OpBuilder::InsertionGuard guard(builder);

    const int moduleStateArgIndex = 1;

    auto funcType = mlir::FunctionType::get(
        ctx,
        {emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void")),
         emitc::PointerType::get(
             emitc::OpaqueType::get(ctx, "iree_vm_module_state_t"))},
        {});

    auto funcOp = builder.create<mlir::emitc::FuncOp>(
        loc, moduleName + "_free_state", funcType);
    funcOp.setSpecifiersAttr(
        builder.getArrayAttr({builder.getStringAttr("static")}));
    funcOp.setPrivate();

    moduleAnalysis.addDummy(funcOp, /*emitAtEnd=*/false);

    Block *entryBlock = funcOp.addEntryBlock();

    const BlockArgument moduleStateArg =
        funcOp.getArgument(moduleStateArgIndex);

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleStateTypeName =
        std::string("struct ") + moduleName + "_state_t";

    auto stateOp = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, moduleStateTypeName)),
        /*operand=*/moduleStateArg);

    auto stateOpLValue =
        emitc_builders::asLValue(builder, loc, stateOp.getResult());

    // Release refs from state struct.
    auto ordinalCounts = moduleOp.getOrdinalCountsAttr();
    if (!ordinalCounts) {
      return moduleOp.emitError()
             << "ordinal_counts attribute not found. The OrdinalAllocationPass "
                "must be run before.";
    }
    const int numGlobalRefs = ordinalCounts.getGlobalRefs();

    if (numGlobalRefs > 0) {
      auto refs =
          cast<TypedValue<emitc::PointerType>>(emitc_builders::structPtrMember(
              builder, loc,
              /*type=*/
              emitc::PointerType::get(
                  emitc::OpaqueType::get(ctx, "iree_vm_ref_t")),
              /*memberName=*/"refs", /*operand=*/stateOpLValue));

      for (int i = 0; i < numGlobalRefs; i++) {
        auto refPtrOp = emitc_builders::arrayElementAddress(
            builder, loc, /*index=*/i, /*operand=*/refs);

        emitc_builders::ireeVmRefRelease(builder, loc, refPtrOp);
      }
    }

    auto allocatorOp = emitc_builders::structPtrMember(
        builder, loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_allocator_t"),
        /*memberName=*/"allocator",
        /*operand=*/stateOpLValue);

    builder.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/"iree_allocator_free",
        /*operands=*/ArrayRef<Value>{allocatorOp, stateOp.getResult()});

    builder.create<mlir::emitc::ReturnOp>(loc, nullptr);
  }

  // iree_status_t fork_state(
  //   void* self,
  //   iree_vm_module_state_t* parent_state,
  //   iree_allocator_t allocator,
  //   iree_vm_module_state_t** out_child_state
  // )
  {
    OpBuilder::InsertionGuard guard(builder);

    // const int parentStateArgIndex = 1;
    // const int allocatorArgIndex = 2;
    // const int childStateArgIndex = 3;

    auto funcType = mlir::FunctionType::get(
        ctx,
        {emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void")),
         emitc::PointerType::get(
             emitc::OpaqueType::get(ctx, "iree_vm_module_state_t")),
         emitc::OpaqueType::get(ctx, "iree_allocator_t"),
         emitc::PointerType::get(emitc::PointerType::get(
             emitc::OpaqueType::get(ctx, "iree_vm_module_state_t")))},
        {emitc::OpaqueType::get(ctx, "iree_status_t")});

    auto funcOp = builder.create<mlir::emitc::FuncOp>(
        loc, moduleName + "_fork_state", funcType);
    funcOp.setSpecifiersAttr(
        builder.getArrayAttr({builder.getStringAttr("static")}));
    funcOp.setPrivate();

    moduleAnalysis.addDummy(funcOp, /*emitAtEnd=*/false);

    Block *entryBlock = funcOp.addEntryBlock();

    builder.setInsertionPointToStart(entryBlock);

    // TODO: someone will need to do what the bytecode module does in order to
    // support forking. For now we don't support forking emitc contexts.
    auto statusOp = builder.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
        /*callee=*/"iree_make_status",
        /*operands=*/ArrayRef<Value>{},
        /*args=*/
        ArrayAttr::get(
            ctx, {emitc::OpaqueAttr::get(ctx, "IREE_STATUS_UNIMPLEMENTED")}));

    builder.create<mlir::emitc::ReturnOp>(loc, statusOp.getResult(0));
  }

  // iree_status_t resolve_import(
  //   void*,
  //   iree_vm_module_state_t*,
  //   iree_host_size_t,
  //   const iree_vm_function_t*,
  //   const iree_vm_function_signature_t*
  // )
  {
    OpBuilder::InsertionGuard guard(builder);

    const int moduleStateArgIndex = 1;
    const int ordinalArgIndex = 2;
    const int functionArgIndex = 3;

    auto funcType = mlir::FunctionType::get(
        ctx,
        {
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void")),
            emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "iree_vm_module_state_t")),
            emitc::OpaqueType::get(ctx, "iree_host_size_t"),
            emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "const iree_vm_function_t")),
            emitc::PointerType::get(emitc::OpaqueType::get(
                ctx, "const iree_vm_function_signature_t")),
        },
        {emitc::OpaqueType::get(ctx, "iree_status_t")});

    auto funcOp = builder.create<mlir::emitc::FuncOp>(
        loc, moduleName + "_resolve_import", funcType);
    funcOp.setSpecifiersAttr(
        builder.getArrayAttr({builder.getStringAttr("static")}));
    funcOp.setPrivate();

    moduleAnalysis.addDummy(funcOp, /*emitAtEnd=*/false);

    Block *entryBlock = funcOp.addEntryBlock();

    const BlockArgument moduleStateArg =
        funcOp.getArgument(moduleStateArgIndex);
    const BlockArgument ordinalArg = funcOp.getArgument(ordinalArgIndex);
    const BlockArgument functionArg = funcOp.getArgument(functionArgIndex);

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleStateTypeName =
        std::string("struct ") + moduleName + "_state_t";

    auto stateOp = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, moduleStateTypeName)),
        /*operand=*/moduleStateArg);

    auto stateOpLValue =
        emitc_builders::asLValue(builder, loc, stateOp.getResult());

    auto imports =
        cast<TypedValue<emitc::PointerType>>(emitc_builders::structPtrMember(
            builder, loc,
            /*type=*/
            emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "iree_vm_function_t")),
            /*memberName=*/"imports",
            /*operand=*/stateOpLValue));

    auto import = emitc_builders::arrayElementAddress(
        builder, loc, /*index=*/ordinalArg, /*operand=*/imports);

    builder.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/"EMITC_DEREF_ASSIGN_PTR",
        /*operands=*/ArrayRef<Value>{import, functionArg});

    auto status = emitc_builders::ireeOkStatus(builder, loc);

    builder.create<mlir::emitc::ReturnOp>(loc, status);
  }

  // iree_status_t create(
  //   iree_vm_instance_t*,
  //   iree_allocator_t,
  //   iree_vm_module_t**
  // );
  {
    OpBuilder::InsertionGuard guard(builder);

    const int instanceArgIndex = 0;
    const int allocatorArgIndex = 1;
    const int moduleArgIndex = 2;

    auto funcType = mlir::FunctionType::get(
        ctx,
        {
            emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "iree_vm_instance_t")),
            emitc::OpaqueType::get(ctx, "iree_allocator_t"),
            emitc::PointerType::get(emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "iree_vm_module_t"))),
        },
        {
            emitc::OpaqueType::get(ctx, "iree_status_t"),
        });

    auto funcOp = builder.create<mlir::emitc::FuncOp>(
        loc, moduleName + "_create", funcType);
    funcOp.setPublic();

    // This function needs an iree_vm_native_module_descriptor_t that is emitted
    // by the CModuleTarget at the moment. So we add a marker to this function
    // and delay the printing of it.
    moduleAnalysis.addDummy(funcOp, /*emitAtEnd=*/true);

    Block *entryBlock = funcOp.addEntryBlock();

    const BlockArgument instanceArg = funcOp.getArgument(instanceArgIndex);
    const BlockArgument allocatorArg = funcOp.getArgument(allocatorArgIndex);
    const BlockArgument moduleArg = funcOp.getArgument(moduleArgIndex);

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleTypeName = std::string("struct ") + moduleName + "_t";

    auto module = emitc_builders::allocateVariable(
        builder, loc,
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, moduleTypeName)),
        {"NULL"});

    Value moduleSize = emitc_builders::sizeOf(
        builder, loc, emitc::OpaqueAttr::get(ctx, moduleTypeName));

    Value modulePtr = emitc_builders::addressOf(builder, loc, module);

    auto voidPtr = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"))),
        /*operand=*/modulePtr);

    returnIfError(builder, loc, "iree_allocator_malloc", {},
                  {allocatorArg, moduleSize, voidPtr.getResult()},
                  moduleAnalysis);

    auto moduleRValue = emitc_builders::asRValue(builder, loc, module);
    emitc_builders::memset(builder, loc, moduleRValue, 0, moduleSize);

    emitc_builders::structPtrMemberAssign(builder, loc,
                                          /*memberName=*/"allocator",
                                          /*operand=*/module,
                                          /*value=*/allocatorArg);

    auto &typeTable = moduleAnalysis.typeTable;
    if (!typeTable.empty()) {
      Type typeRefType = emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t");
      Type typeRefArrayType = emitc::PointerType::get(typeRefType);
      auto moduleTypes =
          cast<TypedValue<emitc::PointerType>>(emitc_builders::structPtrMember(
              builder, loc, typeRefArrayType, "types", module));

      std::string listType = "!vm.list";
      for (auto [index, typeDef] : llvm::enumerate(typeTable)) {
        std::string typeName = typeDef.full_name;
        std::string listPrefix = typeName.substr(0, listType.size());
        if (listType == listPrefix) {
          typeName = listPrefix;
        }

        // Remove leading '!' and wrap in quotes
        if (typeName[0] == '!') {
          typeName = typeName.substr(1);
        }

        Value stringView =
            emitc_builders::ireeMakeCstringView(builder, loc, typeName);
        Value refType = emitc_builders::ireeVmInstanceLookupType(
            builder, loc, instanceArg, stringView);
        emitc_builders::arrayElementAssign(builder, loc, moduleTypes, index,
                                           refType);
      }
    }

    auto vmModule = emitc_builders::allocateVariable(
        builder, loc, emitc::OpaqueType::get(ctx, "iree_vm_module_t"));

    Value vmModulePtr = emitc_builders::addressOf(builder, loc, vmModule);

    auto vmInitializeStatus = builder.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
        /*callee=*/"iree_vm_module_initialize",
        /*operands=*/
        ArrayRef<Value>{vmModulePtr,
                        emitc_builders::asRValue(builder, loc, module)});

    Type boolType = builder.getIntegerType(1);

    auto vmInitializeIsOk = builder.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/boolType,
        /*callee=*/"iree_status_is_ok",
        /*operands=*/ArrayRef<Value>{vmInitializeStatus.getResult(0)});

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

      builder.create<emitc::CallOpaqueOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/"iree_allocator_free",
          /*operands=*/
          ArrayRef<Value>{allocatorArg,
                          emitc_builders::asRValue(builder, loc, module)});

      builder.create<mlir::emitc::ReturnOp>(loc,
                                            vmInitializeStatus.getResult(0));
    }

    builder.setInsertionPointToEnd(condBlock);

    builder.create<mlir::cf::CondBranchOp>(loc, vmInitializeIsOk.getResult(0),
                                           continuationBlock, failureBlock);

    builder.setInsertionPointToStart(continuationBlock);

    // Set function pointers
    for (std::string funcName : {"destroy", "alloc_state", "free_state",
                                 "fork_state", "resolve_import"}) {
      // The type doesn't matter, the result gets inlined into it's uses anyway.
      Type type = emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"));

      Value funcPtr = builder.create<emitc::LiteralOp>(
          loc, type, moduleName + "_" + funcName);
      emitc_builders::structMemberAssign(builder, loc,
                                         /*memberName=*/funcName,
                                         /*operand=*/vmModule,
                                         /*value=*/funcPtr);
    }

    std::string descriptorPtr = "&" + moduleName + "_descriptor_";

    auto status = builder.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
        /*callee=*/"iree_vm_native_module_create",
        /*operands=*/
        ArrayRef<Value>{vmModulePtr, instanceArg, allocatorArg, moduleArg},
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, descriptorPtr),
                             builder.getIndexAttr(1), builder.getIndexAttr(2),
                             builder.getIndexAttr(3)}));

    builder.create<mlir::emitc::ReturnOp>(loc, status.getResult(0));
  }

  return success();
}

/// Generate boilerplate code like includes for the IREE C API, include guards,
/// structures to hold the module state, functions and global variables to
/// create a module instance etc.
LogicalResult
createModuleStructure(IREE::VM::ModuleOp moduleOp,
                      IREE::VM::EmitCTypeConverter &typeConverter) {
  if (failed(createAPIFunctions(moduleOp, typeConverter.analysis))) {
    return failure();
  }

  auto loc = moduleOp.getLoc();

  OpBuilder builder(moduleOp);

  SmallVector<Operation *> opsToRemove;
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&moduleOp.getBlock());

    std::string includeGuard = moduleOp.getName().upper() + "_H_";

    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::IFNDEF,
                                          includeGuard);

    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::DEFINE,
                                          includeGuard);

    builder.create<emitc::IncludeOp>(loc, "iree/vm/api.h");

    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::IFDEF,
                                          "__cplusplus");
    builder.create<emitc::VerbatimOp>(loc, "extern \"C\" {");
    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::ENDIF,
                                          "//  __cplusplus");

    // Emit declarations for public functions.
    for (auto funcOp : moduleOp.getOps<mlir::emitc::FuncOp>()) {
      if (funcOp.isPublic()) {
        builder.create<emitc::DeclareFuncOp>(loc, funcOp.getName());
      }
    }

    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::IFDEF,
                                          "__cplusplus");
    builder.create<emitc::VerbatimOp>(loc, "}  // extern \"C\"");
    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::ENDIF,
                                          "//  __cplusplus");
    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::ENDIF,
                                          std::string("//  ") + includeGuard);
    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::IF,
                                          "defined(EMITC_IMPLEMENTATION)");
    builder.create<emitc::IncludeOp>(loc, "iree/vm/ops.h");
    builder.create<emitc::IncludeOp>(loc, "iree/vm/ops_emitc.h");
    builder.create<emitc::IncludeOp>(loc, "iree/vm/shims_emitc.h");

    // Rodata ops.
    for (auto rodataOp : moduleOp.getOps<IREE::VM::RodataOp>()) {
      auto value = llvm::dyn_cast<IREE::Util::SerializableAttrInterface>(
          rodataOp.getValue());
      assert(value && "expected a serializable rodata value");
      SmallVector<char> byteBuffer;
      if (failed(value.serializeToVector(
              rodataOp.getLoc(), llvm::endianness::little, byteBuffer))) {
        return rodataOp.emitError() << "error during serialization";
      }

      constexpr size_t kDefaultRodataAlignment = 16;
      size_t alignment =
          rodataOp.getAlignment()
              ? static_cast<size_t>(rodataOp.getAlignment().value())
              : 0;
      if (alignment == 0)
        alignment = kDefaultRodataAlignment;

      std::string bufferName =
          moduleOp.getName().str() + "_" + rodataOp.getName().str();

      std::string stmt = "iree_alignas(" + std::to_string(alignment) +
                         ") static const uint8_t " + bufferName + "[] = {";
      size_t index = 0;
      for (char value : byteBuffer) {
        if (index++ > 0)
          stmt += ", ";
        stmt += std::to_string(
            static_cast<unsigned int>(static_cast<unsigned char>(value)));
      }
      stmt += "};";
      builder.create<emitc::VerbatimOp>(loc, stmt);
      opsToRemove.push_back(rodataOp.getOperation());
    }

    // structs
    // Returns |count| or 1 if |count| == 0.
    // Some compilers (MSVC) don't support zero-length struct fields on the
    // interior of structs (just VLA at the tail).
    auto countOrEmpty = [](uint32_t count) { return count ? count : 1; };

    const int64_t numTypes = typeConverter.analysis.typeTable.size();

    std::string moduleStructName = moduleOp.getName().str() + "_t";
    SmallVector<emitc_builders::StructField> moduleStructFields{
        {"iree_allocator_t", "allocator"},
        {"iree_vm_ref_type_t", "types", countOrEmpty(numTypes)}};

    emitc_builders::structDefinition(builder, loc, moduleStructName,
                                     moduleStructFields);

    auto ordinalCounts = moduleOp.getOrdinalCountsAttr();

    std::string moduleStructStateName = moduleOp.getName().str() + "_state_t";
    SmallVector<emitc_builders::StructField> moduleStructStateFields{
        {"iree_allocator_t", "allocator"},
        {"uint8_t", "rwdata", countOrEmpty(ordinalCounts.getGlobalBytes())},
        {"iree_vm_ref_t", "refs", countOrEmpty(ordinalCounts.getGlobalRefs())},
        {"iree_vm_buffer_t", "rodata_buffers",
         countOrEmpty(ordinalCounts.getRodatas())},
        {"iree_vm_function_t", "imports",
         countOrEmpty(ordinalCounts.getImportFuncs())},
    };

    emitc_builders::structDefinition(builder, loc, moduleStructStateName,
                                     moduleStructStateFields);

    // Create a typedef for the begin_call member of the `iree_vm_module_t`
    // type. See `vm/module.h` for the definition. The EmitC dialect doesn't
    // handle function pointer types currently. Neither member lookups nor
    // calls can be fused into one expression at the moment.
    StringRef beginCallTypedef =
        "typedef iree_status_t(*begin_call_t)(void*, iree_vm_stack_t*, "
        "iree_vm_function_call_t);";
    builder.create<emitc::VerbatimOp>(loc, beginCallTypedef);

    // Emit declarations for private functions.
    for (auto funcOp : moduleOp.getOps<mlir::emitc::FuncOp>()) {
      if (funcOp.isPrivate()) {
        builder.create<emitc::DeclareFuncOp>(loc, funcOp.getName());
      }
    }

    // TODO(simon-camp): Move these to a structured helper
    // global descriptors
    //   - define structs for each entity etc.
    auto printStringView = [](StringRef s) -> std::string {
      // We can't use iree_make_string_view because function calls are not
      // allowed for constant expressions in C.
      // TODO(#7605): Switch to IREE_SVL. We can't use IREE_SVL today because it
      // uses designated initializers, which cause issues when compiled as C++.
      return ("{\"" + s + "\", " + std::to_string(s.size()) + "}").str();
    };

    // dependencies
    std::string dependenciesName = moduleOp.getName().str() + "_dependencies_";
    std::string deps;
    deps += "static const iree_vm_module_dependency_t " + dependenciesName +
            "[] = {";
    auto dependencies = moduleOp.getDependencies();
    if (dependencies.empty()) {
      // Empty list placeholder.
      deps += "{{0}},";
    } else {
      for (auto &dependency : dependencies) {
        deps += "{" + printStringView(dependency.name) + ", " +
                std::to_string(dependency.minimumVersion) + ", " +
                (dependency.isOptional
                     ? "IREE_VM_MODULE_DEPENDENCY_FLAG_OPTIONAL"
                     : "IREE_VM_MODULE_DEPENDENCY_FLAG_REQUIRED") +
                "},";
      }
    }
    deps += "};";
    builder.create<emitc::VerbatimOp>(loc, deps);

    // Imports.
    SmallVector<IREE::VM::ImportOp> importOps(
        moduleOp.getOps<IREE::VM::ImportOp>());
    std::string importName = moduleOp.getName().str() + "_imports_";
    std::string imports;
    imports += "static const iree_vm_native_import_descriptor_t " + importName +
               "[] = {";
    if (importOps.empty()) {
      // Empty list placeholder.
      imports += "{0},";
    } else {
      // Sort import ops by ordinal.
      llvm::sort(importOps, [](auto &lhs, auto &rhs) {
        return lhs.getOrdinal()->getZExtValue() <
               rhs.getOrdinal()->getZExtValue();
      });
      for (auto importOp : importOps) {
        imports +=
            std::string("{") +
            (importOp.getIsOptional() ? "IREE_VM_NATIVE_IMPORT_OPTIONAL"
                                      : "IREE_VM_NATIVE_IMPORT_REQUIRED") +
            ", " + printStringView(importOp.getName()) + "},";
      }
    }
    imports += "};";
    builder.create<emitc::VerbatimOp>(loc, imports);

    for (auto op : moduleOp.getOps<IREE::VM::ImportOp>()) {
      opsToRemove.push_back(op);
    }

    // Exports.
    SmallVector<emitc::FuncOp> exportedFunctions;
    for (auto func : moduleOp.getOps<emitc::FuncOp>()) {
      if (typeConverter.analysis.lookupFunction(func).isExported()) {
        exportedFunctions.push_back(func);
      }
    }
    auto extractExportName = [&typeConverter](emitc::FuncOp funcOp) {
      return typeConverter.analysis.lookupFunction(funcOp).getExportName();
    };
    std::string exportName = moduleOp.getName().str() + "_exports_";
    std::string exports;
    exports += "static const iree_vm_native_export_descriptor_t " + exportName +
               "[] = {";
    if (exportedFunctions.empty()) {
      // Empty list placeholder.
      exports += "{{0}},";
    } else {
      // Sort export ops.
      llvm::sort(exportedFunctions, [&extractExportName](auto &lhs, auto &rhs) {
        return extractExportName(lhs).compare(extractExportName(rhs)) < 0;
      });
      for (auto funcOp : exportedFunctions) {
        StringRef exportName = extractExportName(funcOp);
        StringRef callingConvention =
            typeConverter.analysis.lookupFunction(funcOp)
                .getCallingConvention();

        // TODO(simon-camp): support function-level reflection attributes
        exports += "{" + printStringView(exportName) + ", " +
                   printStringView(callingConvention) + ", 0, NULL},";
      }
    }
    exports += "};";
    builder.create<emitc::VerbatimOp>(loc, exports);

    // Functions.
    std::string functionName = moduleOp.getName().str() + "_funcs_";
    std::string functions;
    functions +=
        "static const iree_vm_native_function_ptr_t " + functionName + "[] = {";
    if (exportedFunctions.empty()) {
      // Empty list placeholder.
      functions += "{0},";
    } else {
      // We only add exported functions to the table, as calls to internal
      // functions are directly mapped to C function calls of the generated
      // implementation.
      for (auto funcOp : exportedFunctions) {
        auto funcName = funcOp.getName();
        functions += std::string("{") +
                     "(iree_vm_native_function_shim_t)iree_emitc_shim, " +
                     "(iree_vm_native_function_target_t)" + funcName.str() +
                     "},";
      }
    }
    functions += "};";
    builder.create<emitc::VerbatimOp>(loc, functions);

    // Module descriptor.
    // TODO(simon-camp): support module-level reflection attributes
    std::string descriptorName = moduleOp.getName().str() + "_descriptor_";
    std::string descriptor;
    descriptor +=
        "static const iree_vm_native_module_descriptor_t " + descriptorName +
        " = {"
        // name:
        + printStringView(moduleOp.getName()) +
        ","
        // version:
        + std::to_string(moduleOp.getVersion().value_or(0u)) +
        ","
        // attrs:
        + "0," +
        "NULL,"
        // dependencies:
        + std::to_string(dependencies.size()) + "," + dependenciesName +
        ","
        // imports:
        + std::to_string(importOps.size()) + "," + importName +
        ","
        // exports:
        + std::to_string(exportedFunctions.size()) + "," + exportName +
        ","
        // functions:
        + std::to_string(exportedFunctions.size()) + "," + functionName + "," +
        "};";

    builder.create<emitc::VerbatimOp>(loc, descriptor);

    // Move functions marked as `emitAtEnd` to the end of the module.
    auto funcs =
        SmallVector<emitc::FuncOp>(moduleOp.getOps<mlir::emitc::FuncOp>());
    for (auto func : funcs) {
      if (typeConverter.analysis.lookupFunction(func).shouldEmitAtEnd()) {
        func->moveBefore(moduleOp.getBlock().getTerminator());
      }
    }

    builder.setInsertionPoint(moduleOp.getBlock().getTerminator());
    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::ENDIF,
                                          "  // EMITC_IMPLEMENTATION");
  }

  for (auto op : opsToRemove) {
    op->erase();
  }

  // TODO(simon-camp): The Cpp Emitter expects a builtin.module as the
  // outer container of the supported operations. Instead of nesting a
  // builtin.module inside the vm.module as in the current implementation,
  // the conversion pass should be changed to replace the vm.module
  // with a builtin.module.
  builder.setInsertionPointToStart(&moduleOp.getBlock());
  auto innerModule = builder.create<mlir::ModuleOp>(loc);

  IRRewriter rewriter(moduleOp.getContext());

  for (Operation &op : llvm::make_early_inc_range(moduleOp.getBlock())) {
    if (isa<IREE::VM::ModuleTerminatorOp>(op) ||
        &op == innerModule.getOperation()) {
      continue;
    }
    rewriter.moveOpBefore(&op, innerModule.getBody(),
                          innerModule.getBody()->end());
  }

  return success();
}

SmallVector<Attribute> indexSequence(int64_t n, MLIRContext *ctx) {
  return llvm::map_to_vector(llvm::seq<int64_t>(0, n),
                             [&ctx](int64_t i) -> Attribute {
                               return IntegerAttr::get(IndexType::get(ctx), i);
                             });
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

template <typename OpTy>
class EmitCConversionPattern : public OpConversionPattern<OpTy> {
public:
  using Adaptor = typename OpTy::Adaptor;
  using OpConversionPattern<OpTy>::OpConversionPattern;

  EmitCConversionPattern(const TypeConverter &typeConverter,
                         MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(typeConverter, context, benefit) {}

protected:
  IREE::VM::ModuleAnalysis &getModuleAnalysis() const {
    return this->template getTypeConverter<IREE::VM::EmitCTypeConverter>()
        ->analysis;
  }
};

// Convert vm operations to emitc opaque_calls. The resultiong opaque_call has
// the ops operands as arguments followed by an argument for every attribute.
template <typename OpTy>
class GenericOpConversion : public EmitCConversionPattern<OpTy> {
  using Adaptor = typename OpTy::Adaptor;
  using EmitCConversionPattern<OpTy>::EmitCConversionPattern;

public:
  GenericOpConversion(const TypeConverter &typeConverter, MLIRContext *context,
                      StringRef funcName)
      : EmitCConversionPattern<OpTy>(typeConverter, context),
        funcName(funcName) {}

private:
  LogicalResult
  matchAndRewrite(OpTy op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();

    // Default to an empty args attribute, which results in the operands being
    // printed as the arguments to the function call.
    SmallVector<Attribute> args_;

    // If the operation has attributes, we need to explicitely build the args
    // attribute of the emitc opaque_call op. This consists of index attributes
    // for the operands, followed by the source op attributes themselves.
    if (op->getAttrs().size() > 0) {
      args_ = indexSequence(adaptor.getOperands().size(), op.getContext());

      for (NamedAttribute attr : op->getAttrs()) {
        args_.push_back(attr.getValue());
      }
    }

    ArrayAttr args =
        args_.size() > 0 ? ArrayAttr::get(ctx, args_) : ArrayAttr{};

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        /*op=*/op,
        /*type=*/op.getOperation()->getResultTypes(),
        /*callee=*/funcName,
        /*operands=*/adaptor.getOperands(),
        /*args=*/args);

    return success();
  }

  StringRef funcName;
};

template <typename OpTy>
class DeleteOpConversion : public EmitCConversionPattern<OpTy> {
  using Adaptor = typename OpTy::Adaptor;
  using EmitCConversionPattern<OpTy>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class FuncOpConversion : public EmitCConversionPattern<mlir::emitc::FuncOp> {
  using Adaptor = mlir::emitc::FuncOp::Adaptor;
  using EmitCConversionPattern<mlir::emitc::FuncOp>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::emitc::FuncOp funcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Entry block arguments, i.e. function arguments get converted 1:1.
    // VM::RefType arguments get replaced by iree_vm_ref_t*.
    {
      Block &block = funcOp.getBlocks().front();
      TypeConverter::SignatureConversion signatureConversion(
          block.getNumArguments());

      for (const auto &[index, arg] : llvm::enumerate(block.getArguments())) {
        Type convertedType = getTypeConverter()->convertType(arg.getType());
        signatureConversion.addInputs(index, convertedType);
      }

      rewriter.applySignatureConversion(&block, signatureConversion);

      rewriter.modifyOpInPlace(funcOp, [&] {
        funcOp.setType(
            rewriter.getFunctionType(signatureConversion.getConvertedTypes(),
                                     funcOp.getFunctionType().getResults()));
      });
    }

    // Non-entry block arguments are handled differently between numeric types
    // and VM::RefType.
    {
      for (Block &block : llvm::make_early_inc_range(
               llvm::drop_begin(funcOp.getBlocks(), 1))) {
        TypeConverter::SignatureConversion signatureConversion(
            block.getNumArguments());

        for (const auto &[index, arg] : llvm::enumerate(block.getArguments())) {
          if (isa<IREE::VM::RefType>(arg.getType())) {
            // VM::RefType arguments are dropped and their uses are replaced.
            // The replacement values are determined by the register allocation
            // pass.
            Value ref = getModuleAnalysis().lookupRef(arg);
            signatureConversion.remapInput(index, ref);
          } else {
            // Numerically typed arguments are kept as block arguments. These
            // are automatically handled later in the emitter.
            signatureConversion.addInputs(index, arg.getType());
          }
        }

        Block *newBlock =
            rewriter.applySignatureConversion(&block, signatureConversion);

        // The signatureConversion stores a mapping from the original block
        // argument index to the replacement value. This information is needed
        // in the conversion of branch ops to correctly map from branch operands
        // to the replacement values.
        getModuleAnalysis().lookupFunction(funcOp).cacheBlockConversion(
            newBlock, signatureConversion);
      }
    }

    return success();
  }
};

class ExportOpConversion : public EmitCConversionPattern<IREE::VM::ExportOp> {
  using Adaptor = IREE::VM::ExportOp::Adaptor;
  using EmitCConversionPattern<IREE::VM::ExportOp>::EmitCConversionPattern;

  struct GeneratedStruct {
    std::optional<TypedValue<emitc::LValueType>> value = std::nullopt;
    std::optional<std::string> name = std::nullopt;
    SmallVector<Value> callArguments;
  };

  LogicalResult
  matchAndRewrite(IREE::VM::ExportOp exportOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = exportOp.getContext();
    auto loc = exportOp.getLoc();

    mlir::emitc::FuncOp funcOp = lookupSymbolRef<mlir::emitc::FuncOp>(
        exportOp.getOperation(), "function_ref");

    std::string newFuncName = funcOp.getName().str() + "_export_shim";

    Type stackType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_stack_t"));
    Type flagsType = emitc::OpaqueType::get(ctx, "uint32_t");
    Type spanType = emitc::OpaqueType::get(ctx, "iree_byte_span_t");
    Type moduleType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"));
    Type moduleStateType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"));

    SmallVector<Type> inputTypes = {
        stackType,       // SHIM_ARGUMENT_STACK
        flagsType,       // SHIM_ARGUMENT_FLAGS
        spanType,        // SHIM_ARGUMENT_ARGS_STORAGE
        spanType,        // SHIM_ARGUMENT_RETS_STORAGE
        moduleType,      // SHIM_ARGUMENT_MODULE
        moduleStateType, // SHIM_ARGUMENT_MODULE_STATE
    };

    auto newFuncType = mlir::FunctionType::get(
        ctx, {inputTypes}, {emitc::OpaqueType::get(ctx, "iree_status_t")});

    auto newFuncOp =
        rewriter.create<mlir::emitc::FuncOp>(loc, newFuncName, newFuncType);
    newFuncOp.setSpecifiersAttr(
        rewriter.getArrayAttr({rewriter.getStringAttr("static")}));
    newFuncOp.setPrivate();

    getModuleAnalysis().addFromExport(newFuncOp, exportOp);

    // Populate newly generated function.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *block = rewriter.createBlock(&newFuncOp.getFunctionBody(),
                                          newFuncOp.getFunctionBody().end());

      // Insert arguments into block.
      block->addArgument(stackType, loc);       // SHIM_ARGUMENT_STACK
      block->addArgument(flagsType, loc);       // SHIM_ARGUMENT_FLAGS
      block->addArgument(spanType, loc);        // SHIM_ARGUMENT_ARGS_STORAGE
      block->addArgument(spanType, loc);        // SHIM_ARGUMENT_RETS_STORAGE
      block->addArgument(moduleType, loc);      // SHIM_ARGUMENT_MODULE
      block->addArgument(moduleStateType, loc); // SHIM_ARGUMENT_MODULE_STATE

      rewriter.setInsertionPointToStart(block);

      // Create typedefs for argument and result structs.
      auto typedefs =
          typedefArgumentAndResultStructs(rewriter, exportOp, newFuncOp);

      if (failed(typedefs)) {
        return exportOp.emitError() << "struct typedef failed.";
      }

      GeneratedStruct argumentStruct;
      GeneratedStruct resultStruct;

      std::tie(argumentStruct, resultStruct) = typedefs.value();

      // Cast module and module state structs.
      auto moduleStructs =
          castModuleAndStateStructs(rewriter, exportOp, newFuncOp);

      if (failed(moduleStructs)) {
        return exportOp.emitError() << "module struct casting failed.";
      }

      Value moduleStruct;
      Value moduleStateStruct;

      std::tie(moduleStruct, moduleStateStruct) = moduleStructs.value();

      // Cast argument and result structs.
      castArgumentAndResultStructs(rewriter, exportOp, newFuncOp,
                                   argumentStruct, resultStruct);

      // Unpack arguments from struct.
      auto arguments = unpackArguments(rewriter, exportOp, argumentStruct);

      if (failed(arguments)) {
        return exportOp.emitError() << "failed to unpack arguments.";
      }

      // Unpack result pointers from struct.
      auto results = unpackResults(rewriter, exportOp, resultStruct);

      if (failed(results)) {
        return exportOp.emitError() << "failed to unpack results.";
      }

      // Call internal function and return on error.
      SmallVector<Value> operands{block->getArgument(SHIM_ARGUMENT_STACK),
                                  moduleStruct, moduleStateStruct};

      for (auto &argument : argumentStruct.callArguments) {
        operands.push_back(argument);
      }
      for (auto &result : resultStruct.callArguments) {
        operands.push_back(result);
      }

      returnIfError(rewriter, loc, funcOp, operands, getModuleAnalysis());

      auto status = emitc_builders::ireeOkStatus(rewriter, loc);

      rewriter.create<mlir::emitc::ReturnOp>(loc, status);
    }

    rewriter.eraseOp(exportOp);

    return success();
  }

  FailureOr<std::pair<Value, Value>>
  castModuleAndStateStructs(ConversionPatternRewriter &rewriter,
                            IREE::VM::ExportOp &exportOp,
                            mlir::emitc::FuncOp &newFuncOp) const {
    auto ctx = exportOp.getContext();
    auto loc = exportOp.getLoc();

    auto module = newFuncOp.getArgument(SHIM_ARGUMENT_MODULE);
    auto moduleState = newFuncOp.getArgument(SHIM_ARGUMENT_MODULE_STATE);

    auto moduleOp =
        newFuncOp.getOperation()->getParentOfType<IREE::VM::ModuleOp>();

    std::string moduleTypeName =
        std::string("struct ") + moduleOp.getName().str() + "_t";
    std::string moduleStateTypeName =
        std::string("struct ") + moduleOp.getName().str() + "_state_t";

    auto moduleCasted = rewriter.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, moduleTypeName)),
        /*operand=*/module);

    auto moduleStateCasted = rewriter.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, moduleStateTypeName)),
        /*operand=*/moduleState);

    return {{moduleCasted.getResult(), moduleStateCasted.getResult()}};
  }

  FailureOr<std::pair<GeneratedStruct, GeneratedStruct>>
  typedefArgumentAndResultStructs(ConversionPatternRewriter &rewriter,
                                  IREE::VM::ExportOp &exportOp,
                                  mlir::emitc::FuncOp &newFuncOp) const {
    auto loc = exportOp.getLoc();

    mlir::emitc::FuncOp funcOp = lookupSymbolRef<mlir::emitc::FuncOp>(
        exportOp.getOperation(), "function_ref");

    auto &funcAnalysis = getModuleAnalysis().lookupFunction(funcOp);

    auto generateStructFields = [this](ArrayRef<Type> types, StringRef prefix)
        -> FailureOr<SmallVector<emitc_builders::StructField>> {
      SmallVector<emitc_builders::StructField> result;

      for (auto pair : llvm::enumerate(types)) {
        emitc::OpaqueType cType =
            getTypeConverter<IREE::VM::EmitCTypeConverter>()
                ->convertTypeAsCType(pair.value());

        if (!cType) {
          return failure();
        }

        auto fieldName = prefix.str() + std::to_string(pair.index());
        result.push_back({cType.getValue().str(), fieldName});
      }

      return result;
    };

    // TODO(simon-camp): Clean up; We generate calls to a macro that defines
    // a struct. As we declare all variables at the start of the function,
    // the macro call cannot be inlined into the function.

    // To prevent scoping issues we prefix the struct name with module and
    // function name.
    auto typedefStruct = [&rewriter, &newFuncOp,
                          &loc](std::string structName,
                                ArrayRef<emitc_builders::StructField> fields) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(newFuncOp.getOperation());

      emitc_builders::structDefinition(/*builder=*/rewriter, /*location=*/loc,
                                       /*structName=*/structName,
                                       /*fields=*/fields);
    };

    FunctionType funcType = funcAnalysis.getFunctionType();

    GeneratedStruct argumentStruct;
    GeneratedStruct resultStruct;

    const bool needArgumentStruct = funcType.getNumInputs() > 0;

    if (needArgumentStruct) {
      auto structBody = generateStructFields(funcType.getInputs(), "arg");
      if (failed(structBody)) {
        return exportOp.emitError()
               << "failed to emit C type for struct definition";
      }

      std::string structName = funcOp.getName().str() + "_args_t";
      argumentStruct.name = structName;
      typedefStruct(structName, structBody.value());
    }

    const bool needResultStruct = funcType.getNumResults() > 0;

    if (needResultStruct) {
      auto structBody = generateStructFields(funcType.getResults(), "res");

      if (failed(structBody)) {
        return failure();
      }

      std::string structName = funcOp.getName().str() + "_result_t";
      resultStruct.name = structName;
      typedefStruct(structName, structBody.value());
    }

    return {{argumentStruct, resultStruct}};
  }

  void castArgumentAndResultStructs(ConversionPatternRewriter &rewriter,
                                    IREE::VM::ExportOp &exportOp,
                                    mlir::emitc::FuncOp &newFuncOp,
                                    GeneratedStruct &argumentStruct,
                                    GeneratedStruct &resultStruct) const {
    auto ctx = exportOp.getContext();
    auto loc = exportOp.getLoc();

    const bool haveArgumentStruct = argumentStruct.name.has_value();

    if (haveArgumentStruct) {
      auto argumentsLValue = emitc_builders::asLValue(
          rewriter, loc, newFuncOp.getArgument(SHIM_ARGUMENT_ARGS_STORAGE));

      // args_t* args = (args_t*)call->arguments.data;

      // arguments.data
      auto argumentsData = emitc_builders::structMember(
          rewriter, loc,
          /*type=*/emitc::PointerType::get(rewriter.getIntegerType(8, false)),
          /*memberName=*/"data",
          /*operand=*/argumentsLValue);

      // cast
      std::string argumentsType =
          std::string("struct ") + argumentStruct.name.value();
      auto arguments = rewriter.create<emitc::CastOp>(
          /*location=*/loc,
          /*type=*/
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, argumentsType)),
          /*operand=*/argumentsData);

      argumentsLValue =
          emitc_builders::asLValue(rewriter, loc, arguments.getResult());

      argumentStruct.value = argumentsLValue;
    }

    const bool haveResultStruct = resultStruct.name.has_value();
    if (haveResultStruct) {
      auto resultsLValue = emitc_builders::asLValue(
          rewriter, loc, newFuncOp.getArgument(SHIM_ARGUMENT_RETS_STORAGE));
      // results_t* results = (results_t*)call->results.data;

      // results.data
      auto resultsData = cast<
          TypedValue<emitc::PointerType>>(emitc_builders::structMember(
          rewriter, loc,
          /*type=*/emitc::PointerType::get(rewriter.getIntegerType(8, false)),
          /*memberName=*/"data", /*operand=*/resultsLValue));

      // cast
      std::string resultType =
          std::string("struct ") + resultStruct.name.value();
      auto results = rewriter.create<emitc::CastOp>(
          /*location=*/loc,
          /*type=*/
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, resultType)),
          /*operand=*/resultsData);

      resultsLValue =
          emitc_builders::asLValue(rewriter, loc, results.getResult());

      resultStruct.value = resultsLValue;
    }
  }

  LogicalResult unpackArguments(ConversionPatternRewriter &rewriter,
                                IREE::VM::ExportOp &exportOp,
                                GeneratedStruct &argumentStruct) const {
    auto ctx = exportOp.getContext();
    auto loc = exportOp.getLoc();

    // The struct is empty, nothing to do.
    if (!argumentStruct.value.has_value()) {
      return success();
    }

    mlir::emitc::FuncOp funcOp = lookupSymbolRef<mlir::emitc::FuncOp>(
        exportOp.getOperation(), "function_ref");

    auto &funcAnalysis = getModuleAnalysis().lookupFunction(funcOp);

    FunctionType funcType = funcAnalysis.getFunctionType();

    for (const auto &input : llvm::enumerate(funcType.getInputs())) {
      assert(argumentStruct.value.has_value());
      auto value = argumentStruct.value.value();

      if (llvm::isa<IREE::VM::RefType>(input.value())) {
        auto ptrType = emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_ref_t"));
        std::string memberName = "arg" + std::to_string(input.index());
        auto memberPtr = emitc_builders::structPtrMemberAddress(
            rewriter, loc, ptrType, memberName, value);
        rewriter.create<emitc::CallOpaqueOp>(
            /*location=*/memberPtr.getLoc(),
            /*type=*/TypeRange{},
            /*callee=*/"iree_vm_ref_retain_inplace",
            /*operands=*/ArrayRef<Value>{memberPtr});
        argumentStruct.callArguments.push_back(memberPtr);
      } else {
        Type memberType = input.value();
        std::string memberName = "arg" + std::to_string(input.index());
        auto member = emitc_builders::structPtrMember(rewriter, loc,
                                                      /*type=*/memberType,
                                                      /*memberName=*/memberName,
                                                      /*operand=*/value);
        argumentStruct.callArguments.push_back(member);
      }
    }

    return success();
  }

  LogicalResult unpackResults(ConversionPatternRewriter &rewriter,
                              IREE::VM::ExportOp &exportOp,
                              GeneratedStruct &resultStruct) const {
    auto loc = exportOp.getLoc();

    // The struct is empty, nothing to do.
    if (!resultStruct.value.has_value()) {
      return success();
    }

    const auto typeConverter = getTypeConverter<IREE::VM::EmitCTypeConverter>();

    mlir::emitc::FuncOp funcOp = lookupSymbolRef<mlir::emitc::FuncOp>(
        exportOp.getOperation(), "function_ref");

    auto &funcAnalysis = getModuleAnalysis().lookupFunction(funcOp);

    FunctionType funcType = funcAnalysis.getFunctionType();

    for (const auto &result : llvm::enumerate(funcType.getResults())) {
      assert(resultStruct.value.has_value());
      auto value = resultStruct.value.value();

      auto ptrType = typeConverter->convertTypeAsPointer(result.value());

      std::string memberName = "res" + std::to_string(result.index());
      Value memberPtr = emitc_builders::structPtrMemberAddress(
          rewriter, loc, ptrType, memberName, value);
      resultStruct.callArguments.push_back(memberPtr);
    }

    return success();
  }
};

class ImportOpConverter {
public:
  ImportOpConverter(IREE::VM::EmitCTypeConverter &typeConverter,
                    SmallVector<std::string> &importShims)
      : typeConverter(typeConverter), importShims(importShims) {}

  LogicalResult operator()(IREE::VM::ImportOp importOp) const {
    OpBuilder builder(importOp);

    auto key = makeImportCallingConventionString(importOp);
    if (!key.has_value()) {
      return importOp.emitError()
             << "Failed to build key for import shim cache.";
    }

    // The needed shim already exists.
    if (llvm::find(importShims, key) != std::end(importShims)) {
      return success();
    }

    if (importOp.isVariadic()) {
      if (failed(createVariadicImportShims(importOp, builder))) {
        return failure();
      }
    } else {
      if (failed(createImportShim(importOp, nullptr, builder))) {
        return failure();
      }
    }

    importShims.push_back(key.value());
    return success();
  }

private:
  struct MaybeZeroValue {
    Value value;
    bool isZero;
  };

  LogicalResult createVariadicImportShims(IREE::VM::ImportOp &importOp,
                                          OpBuilder &builder) const {
    SetVector<const void *> arities;

    for (auto caller : getCallers(importOp)) {
      DenseIntElementsAttr segmentSizes = caller.getSegmentSizes();
      const void *p = segmentSizes.getAsOpaquePointer();
      if (arities.insert(p)) {
        if (failed(createImportShim(importOp, segmentSizes, builder))) {
          return failure();
        }
      }
    }
    return success();
  }

  void failIfImportUnresolved(OpBuilder &builder, Location location,
                              TypedValue<emitc::LValueType> import) const {
    auto *ctx = builder.getContext();
    Type boolType = builder.getIntegerType(1);

    // (iree_vm_function_t*)->module
    auto importModule = emitc_builders::structPtrMember(
        builder, location,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_module_t")),
        /*memberName=*/"module",
        /*operand=*/import);

    auto conditionI1 = builder
                           .create<emitc::LogicalNotOp>(
                               /*location=*/location, /*type=*/boolType,
                               /*operands=*/importModule)
                           .getResult();

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

      mlir::emitc::FuncOp funcOp =
          cast<mlir::emitc::FuncOp>(failureBlock->getParentOp());
      releaseRefs(builder, location, funcOp, typeConverter.analysis);

      auto statusOp = builder.create<emitc::CallOpaqueOp>(
          /*location=*/location,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
          /*callee=*/"iree_make_status",
          /*operands=*/ArrayRef<Value>{},
          /*args=*/
          ArrayAttr::get(
              ctx, {emitc::OpaqueAttr::get(ctx, "IREE_STATUS_NOT_FOUND")}));
      builder.create<mlir::emitc::ReturnOp>(location, statusOp.getResult(0));
    }

    builder.setInsertionPointToEnd(condBlock);
    builder.create<cf::CondBranchOp>(location, conditionI1, failureBlock,
                                     continuationBlock);

    builder.setInsertionPointToStart(continuationBlock);
  }

  LogicalResult createImportShim(IREE::VM::ImportOp &importOp,
                                 DenseIntElementsAttr segmentSizes,
                                 OpBuilder &builder) const {
    auto loc = importOp.getLoc();

    auto moduleOp =
        importOp.getOperation()->getParentOfType<IREE::VM::ModuleOp>();

    auto newFuncName =
        importOp.isVariadic()
            ? buildVariadicFunctionName(moduleOp, importOp, segmentSizes)
            : buildFunctionName(moduleOp, importOp);

    if (!newFuncName.has_value()) {
      return importOp.emitError() << "failed to build import shim name.";
    }

    auto newFuncType = buildFuncType(importOp, segmentSizes, builder, loc);

    if (failed(newFuncType)) {
      return importOp.emitError()
             << "Failed to build function type for wrapper";
    }

    auto newFuncOp = builder.create<mlir::emitc::FuncOp>(
        loc, newFuncName.value(), newFuncType.value());
    newFuncOp.setSpecifiersAttr(
        builder.getArrayAttr({builder.getStringAttr("static")}));
    newFuncOp.setPrivate();

    typeConverter.analysis.addFromImport(newFuncOp, importOp);

    // Populate newly generated function.
    {
      OpBuilder::InsertionGuard guard(builder);
      Block *block = builder.createBlock(&newFuncOp.getFunctionBody(),
                                         newFuncOp.getFunctionBody().end());

      for (Type type : newFuncOp.getFunctionType().getInputs()) {
        block->addArgument(type, loc);
      }

      builder.setInsertionPointToStart(block);

      MaybeZeroValue argumentSize = buildSizeExpression(
          flattenInputTypes(importOp, segmentSizes, builder), builder, loc);
      MaybeZeroValue resultSize =
          buildSizeExpression(importOp.getResultTypes(), builder, loc);

      const int importArgIndex = 1;
      const BlockArgument importArg = newFuncOp.getArgument(importArgIndex);
      auto importArgLValue = emitc_builders::asLValue(builder, loc, importArg);
      failIfImportUnresolved(builder, loc, importArgLValue);

      auto call = buildIreeVmFunctionCallStruct(importArg, argumentSize,
                                                resultSize, builder, loc);

      if (failed(call)) {
        return importOp.emitError() << "failed to create call struct";
      }

      if (failed(packArgumentBuffer(
              flattenInputTypes(importOp, segmentSizes, builder), newFuncOp,
              call.value(), builder, loc))) {
        return importOp.emitError() << "failed to pack argument struct";
      }

      const BlockArgument stackArg =
          newFuncOp.getArgument(CCONV_ARGUMENT_STACK);
      if (failed(createCall(call.value(), importArgLValue, stackArg, builder,
                            loc))) {
        return importOp.emitError() << "failed to create call";
      }

      if (failed(unpackResultBuffer(importOp.getResultTypes(), newFuncOp,
                                    call.value(), builder, loc))) {
        return importOp.emitError() << "failed to unpack result struct";
      }

      auto status = emitc_builders::ireeOkStatus(builder, loc);

      builder.create<mlir::emitc::ReturnOp>(loc, status);
    }

    return success();
  }

  FailureOr<FunctionType> buildFuncType(IREE::VM::ImportOp importOp,
                                        DenseIntElementsAttr segmentSizes,
                                        OpBuilder &builder,
                                        Location loc) const {
    auto ctx = builder.getContext();

    Type stackType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_stack_t"));
    Type funcType = emitc::PointerType::get(
        emitc::OpaqueType::get(ctx, "iree_vm_function_t"));

    SmallVector<Type> types{stackType, funcType};

    for (Type type : flattenInputTypes(importOp, segmentSizes, builder)) {
      auto convertedType = typeConverter.convertType(type);
      types.push_back(convertedType);
    }

    for (auto &resultType : importOp.getResultTypes()) {
      Type ptrType = typeConverter.convertTypeAsPointer(resultType);

      types.push_back(ptrType);
    }

    FunctionType result = mlir::FunctionType::get(
        ctx, {types}, {emitc::OpaqueType::get(ctx, "iree_status_t")});

    return {result};
  }

  MaybeZeroValue buildSizeExpression(ArrayRef<Type> types, OpBuilder &builder,
                                     Location loc) const {
    auto ctx = builder.getContext();

    Type hostSizeType = emitc::OpaqueType::get(ctx, "iree_host_size_t");

    Value result = builder
                       .create<emitc::ConstantOp>(
                           /*location=*/loc,
                           /*resultType=*/hostSizeType,
                           /*value=*/emitc::OpaqueAttr::get(ctx, "0"))
                       .getResult();
    bool isZero = true;
    for (Type type : types) {
      Type valueType = typeConverter.convertTypeAsNonPointer(type);
      Value size =
          emitc_builders::sizeOf(builder, loc, TypeAttr::get(valueType));

      result = builder
                   .create<emitc::AddOp>(
                       /*location=*/loc,
                       /*type=*/hostSizeType,
                       /*operands=*/ArrayRef<Value>{result, size})
                   .getResult();
      isZero = false;
    }

    return MaybeZeroValue{result, isZero};
  }

  FailureOr<TypedValue<emitc::LValueType>>
  buildIreeVmFunctionCallStruct(Value import, MaybeZeroValue argumentSize,
                                MaybeZeroValue resultSize, OpBuilder &builder,
                                Location loc) const {
    auto ctx = builder.getContext();

    // iree_vm_function_call_t call;
    auto call = emitc_builders::allocateVariable(
        builder, loc, emitc::OpaqueType::get(ctx, "iree_vm_function_call_t"));

    // importValue = *import;
    auto importValue = emitc_builders::contentsOf(builder, loc, import);

    // call.function = importValue;
    emitc_builders::structMemberAssign(builder, loc,
                                       /*memberName=*/"function",
                                       /*operand=*/call,
                                       /*value=*/importValue);

    allocateByteSpan(call, argumentSize, "arguments", builder, loc);
    allocateByteSpan(call, resultSize, "results", builder, loc);

    return {call};
  }

  Value allocateByteSpan(TypedValue<emitc::LValueType> call,
                         MaybeZeroValue size, StringRef memberName,
                         OpBuilder &builder, Location loc) const {

    auto ctx = builder.getContext();

    // byteSpan = call.<memberName>;
    auto byteSpan = builder
                        .create<emitc::MemberOp>(
                            loc,
                            emitc::LValueType::get(emitc::OpaqueType::get(
                                ctx, "iree_byte_span_t")),
                            memberName, call)
                        .getResult();

    // alloca_(0) returns NULL in some configurations on Windows. Make sure to
    // allocate at least one byte to get a valid pointer.
    Value allocaSize;
    if (size.isZero) {
      Type hostSizeType = emitc::OpaqueType::get(ctx, "iree_host_size_t");

      allocaSize = builder
                       .create<emitc::ConstantOp>(
                           /*location=*/loc,
                           /*resultType=*/hostSizeType,
                           /*value=*/emitc::OpaqueAttr::get(ctx, "1"))
                       .getResult();
    } else {
      allocaSize = size.value;
    }

    // void *byteSpan_data_void = iree_alloca(size);
    auto byteSpanDataVoid =
        builder
            .create<emitc::CallOpaqueOp>(
                /*location=*/loc,
                /*type=*/
                emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void")),
                /*callee=*/"iree_alloca",
                /*operands=*/ArrayRef<Value>{allocaSize})
            .getResult(0);

    // uint8_t *byteSpan_data = (uint8_t*)byteSpan_data_void;
    Type bytePtr = emitc::PointerType::get(builder.getIntegerType(8, false));
    auto byteSpanData = builder
                            .create<emitc::CastOp>(
                                /*location=*/loc,
                                /*type=*/bytePtr,
                                /*operand=*/byteSpanDataVoid)
                            .getResult();

    // byteSpan.data_length = SIZE;
    emitc_builders::structMemberAssign(builder, loc,
                                       /*memberName=*/"data_length",
                                       /*operand=*/byteSpan,
                                       /*value=*/size.value);

    // byteSpan.data = byteSpan_data
    emitc_builders::structMemberAssign(builder, loc,
                                       /*memberName=*/"data",
                                       /*operand=*/byteSpan,
                                       /*value=*/byteSpanData);

    // memset(byteSpanData, 0, SIZE);
    emitc_builders::memset(builder, loc, byteSpanData, 0, allocaSize);

    return byteSpan;
  }

  LogicalResult packArgumentBuffer(ArrayRef<Type> inputTypes,
                                   mlir::emitc::FuncOp &funcOp,
                                   TypedValue<emitc::LValueType> call,
                                   OpBuilder &builder, Location loc) const {
    if (inputTypes.empty()) {
      return success();
    }

    auto ctx = builder.getContext();

    size_t inputOffset = 2;

    auto arguments = builder
                         .create<emitc::MemberOp>(
                             loc,
                             /*type=*/
                             emitc::LValueType::get(emitc::OpaqueType::get(
                                 ctx, "iree_byte_span_t")),
                             /*memberName=*/"arguments",
                             /*operand=*/call)
                         .getResult();

    Type bytePtrType =
        emitc::PointerType::get(builder.getIntegerType(8, false));
    auto uint8Ptr = emitc_builders::structMember(builder, loc,
                                                 /*type=*/bytePtrType,
                                                 /*memberName=*/"data",
                                                 /*operand=*/arguments);
    for (size_t i = 0; i < inputTypes.size(); i++) {
      BlockArgument arg = funcOp.getArgument(i + inputOffset);

      Type argType = arg.getType();
      assert(!isa<IREE::VM::RefType>(argType));

      if (argType == emitc::PointerType::get(
                         emitc::OpaqueType::get(ctx, "iree_vm_ref_t"))) {
        Type refPtrType = emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_ref_t"));
        Value refPtr = builder
                           .create<emitc::CastOp>(
                               /*location=*/loc,
                               /*type=*/refPtrType,
                               /*operand=*/uint8Ptr)
                           .getResult();

        builder.create<emitc::CallOpaqueOp>(
            /*location=*/loc,
            /*type=*/TypeRange{},
            /*callee=*/"iree_vm_ref_assign",
            /*operands=*/ArrayRef<Value>{arg, refPtr});
      } else {
        auto argLValue = emitc_builders::asLValue(builder, loc, arg);
        assert(!isa<emitc::PointerType>(argType));
        Value size =
            emitc_builders::sizeOf(builder, loc, TypeAttr::get(argType));

        // memcpy(uint8Ptr, &arg, size);
        Value argPtr = emitc_builders::addressOf(builder, loc, argLValue);
        emitc_builders::memcpy(builder, loc, uint8Ptr, argPtr, size);
      }

      // Skip the addition in the last iteration.
      if (i < inputTypes.size() - 1) {
        Type valueType = typeConverter.convertTypeAsNonPointer(argType);
        Value size =
            emitc_builders::sizeOf(builder, loc, TypeAttr::get(valueType));

        uint8Ptr = builder
                       .create<emitc::AddOp>(
                           /*location=*/loc, /*type=*/bytePtrType,
                           /*operands=*/ArrayRef<Value>{uint8Ptr, size})
                       .getResult();
      }
    }
    return success();
  }

  LogicalResult unpackResultBuffer(ArrayRef<Type> resultTypes,
                                   mlir::emitc::FuncOp &funcOp,
                                   TypedValue<emitc::LValueType> call,
                                   OpBuilder &builder, Location loc) const {
    if (resultTypes.empty()) {
      return success();
    }

    auto ctx = builder.getContext();

    // The last N arguments are the results.
    size_t resultOffset = funcOp.getNumArguments() - resultTypes.size();

    auto results = builder.create<emitc::MemberOp>(
        loc,
        /*type=*/
        emitc::LValueType::get(emitc::OpaqueType::get(ctx, "iree_byte_span_t")),
        /*memberName=*/"results",
        /*operand=*/call);

    Type bytePtrType =
        emitc::PointerType::get(builder.getIntegerType(8, false));
    auto uint8Ptr = emitc_builders::structMember(builder, loc,
                                                 /*type=*/bytePtrType,
                                                 /*memberName=*/"data",
                                                 /*operand=*/results);

    for (size_t i = 0; i < resultTypes.size(); i++) {
      BlockArgument arg = funcOp.getArgument(i + resultOffset);
      Type argType = arg.getType();
      assert(!isa<IREE::VM::RefType>(argType));

      if (argType == emitc::PointerType::get(
                         emitc::OpaqueType::get(ctx, "iree_vm_ref_t"))) {
        Type refPtrType = emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_ref_t"));
        Value refPtr = builder
                           .create<emitc::CastOp>(
                               /*location=*/loc,
                               /*type=*/refPtrType,
                               /*operand=*/uint8Ptr)
                           .getResult();

        builder.create<emitc::CallOpaqueOp>(
            /*location=*/loc,
            /*type=*/TypeRange{},
            /*callee=*/"iree_vm_ref_move",
            /*operands=*/ArrayRef<Value>{refPtr, arg});
      } else {
        Type valueType = llvm::cast<emitc::PointerType>(argType).getPointee();
        Value size =
            emitc_builders::sizeOf(builder, loc, TypeAttr::get(valueType));

        // memcpy(arg, uint8Ptr, size);
        emitc_builders::memcpy(builder, loc, arg, uint8Ptr, size);
      }

      // Skip the addition in the last iteration.
      if (i < resultTypes.size() - 1) {
        Type valueType = llvm::cast<emitc::PointerType>(argType).getPointee();
        Value size =
            emitc_builders::sizeOf(builder, loc, TypeAttr::get(valueType));
        uint8Ptr = builder
                       .create<emitc::AddOp>(
                           /*location=*/loc,
                           /*type=*/bytePtrType,
                           /*operands=*/ArrayRef<Value>{uint8Ptr, size})
                       .getResult();
      }
    }
    return success();
  }

  LogicalResult createCall(TypedValue<emitc::LValueType> call,
                           TypedValue<emitc::LValueType> import, Value stack,
                           OpBuilder &builder, Location loc) const {
    auto ctx = builder.getContext();

    // RETURN_IF_ERROR(import->module->begin_call(import->module, stack, call));
    auto importModule = builder.create<emitc::MemberOfPtrOp>(
        loc,
        /*type=*/
        emitc::LValueType::get(emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_module_t"))),
        /*memberName=*/"module",
        /*operand=*/import);

    // EmitC can't emit function pointers, so we need to fallback to a typedef
    // currently. This and the `EMITC_CALL_INDIRECT` macro can be replaced with
    // a new `emitc.call_indirect` op once it has been added upstream.
    emitc::OpaqueType type = emitc::OpaqueType::get(ctx, "begin_call_t");

    auto beginCall =
        builder
            .create<emitc::MemberOfPtrOp>(loc, emitc::LValueType::get(type),
                                          "begin_call", importModule)
            .getResult();

    returnIfError(
        /*rewriter=*/builder,
        /*location=*/loc,
        /*callee=*/"EMITC_CALL_INDIRECT",
        /*args=*/{},
        /*operands=*/
        ArrayRef<Value>{emitc_builders::asRValue(builder, loc, beginCall),
                        emitc_builders::asRValue(builder, loc, importModule),
                        stack, emitc_builders::asRValue(builder, loc, call)},
        typeConverter.analysis);

    return success();
  }

  // A span count of -1 means a non variadic call
  SmallVector<Type> flattenInputTypes(IREE::VM::ImportOp importOp,
                                      DenseIntElementsAttr segmentSizes,
                                      OpBuilder &builder) const {
    assert(!segmentSizes ||
           (importOp.getNumArguments() == segmentSizes.size()));

    SmallVector<Type> result;
    auto expandType = [&result](Type type) {
      if (auto tupleType = llvm::dyn_cast<TupleType>(type)) {
        for (Type inner : tupleType) {
          result.push_back(inner);
        }
      } else {
        result.push_back(type);
      }
    };

    for (size_t i = 0; i < importOp.getNumArguments(); i++) {
      Type type = importOp.getFunctionType().getInput(i);

      if (importOp.isFuncArgumentVariadic(i)) {
        assert(segmentSizes && "segmentSizes must not be nullptr");
        APInt segmentSize = *(segmentSizes.begin() + i);
        int64_t size = segmentSize.getSExtValue();
        result.push_back(builder.getI32Type());

        assert(size >= 0);
        for (int j = 0; j < size; j++) {
          expandType(type);
        }
      } else {
        expandType(type);
      }
    }

    return result;
  }

  SmallVector<IREE::VM::CallVariadicOp>
  getCallers(IREE::VM::ImportOp &importOp) const {
    SmallVector<IREE::VM::CallVariadicOp> result;

    auto moduleOp =
        importOp.getOperation()->getParentOfType<IREE::VM::ModuleOp>();

    moduleOp.walk([&result, &importOp](Operation *op) {
      if (auto callOp = dyn_cast<IREE::VM::CallVariadicOp>(op)) {
        if (importOp == lookupSymbolRef<IREE::VM::ImportOp>(
                            callOp.getOperation(), "callee")) {
          result.push_back(callOp);
        }
      }
    });

    return result;
  }

  IREE::VM::EmitCTypeConverter &typeConverter;
  SmallVector<std::string> &importShims;
};

template <typename OpTy>
class CallOpConversion : public EmitCConversionPattern<OpTy> {
  using Adaptor = typename OpTy::Adaptor;
  using EmitCConversionPattern<OpTy>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::emitc::FuncOp funcOp =
        lookupSymbolRef<mlir::emitc::FuncOp>(op.getOperation(), "callee");
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
                                    mlir::emitc::FuncOp funcOp) const {
    auto loc = op->getLoc();

    SmallVector<Value> updatedOperands;
    SmallVector<Value> resultOperands;

    auto parentFuncOp = op->getParentOfType<mlir::emitc::FuncOp>();

    const BlockArgument stackArg =
        parentFuncOp.getArgument(CCONV_ARGUMENT_STACK);
    const BlockArgument moduleArg =
        parentFuncOp.getArgument(CCONV_ARGUMENT_MODULE);
    const BlockArgument moduleStateArg =
        parentFuncOp.getArgument(CCONV_ARGUMENT_MODULE_STATE);

    updatedOperands = {stackArg, moduleArg, moduleStateArg};

    if (failed(updateOperands(op, nullptr, rewriter, updatedOperands,
                              resultOperands))) {
      return failure();
    };

    returnIfError(
        /*rewriter=*/rewriter, /*location=*/loc, /*callee=*/funcOp,
        /*operands=*/updatedOperands, this->getModuleAnalysis());

    emitc_builders::asRValues(rewriter, loc, resultOperands);
    rewriter.replaceOp(op, resultOperands);

    return success();
  }

  LogicalResult rewriteImportedCall(Operation *op, Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    IREE::VM::ImportOp importOp) const {
    auto ctx = op->getContext();
    auto loc = op->getLoc();

    SmallVector<Value> updatedOperands;
    SmallVector<Value> resultOperands;

    auto moduleOp =
        importOp.getOperation()->getParentOfType<IREE::VM::ModuleOp>();

    int importOrdinal = importOp.getOrdinal()->getZExtValue();

    auto funcOp = op->getParentOfType<mlir::emitc::FuncOp>();

    const BlockArgument stackArg = funcOp.getArgument(CCONV_ARGUMENT_STACK);
    const BlockArgument stateArg =
        funcOp.getArgument(CCONV_ARGUMENT_MODULE_STATE);
    auto stateArgLValue = emitc_builders::asLValue(rewriter, loc, stateArg);

    auto imports =
        cast<TypedValue<emitc::PointerType>>(emitc_builders::structPtrMember(
            rewriter, loc,
            /*type=*/
            emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "iree_vm_function_t")),
            /*memberName=*/"imports",
            /*operand=*/stateArgLValue));

    auto import = emitc_builders::arrayElementAddress(
        rewriter, loc, /*index=*/importOrdinal, /*operand=*/imports);

    updatedOperands = {stackArg, import};

    std::optional<std::string> funcName;
    if (auto variadicOp = dyn_cast<IREE::VM::CallVariadicOp>(op)) {
      funcName = buildVariadicFunctionName(moduleOp, importOp,
                                           variadicOp.getSegmentSizes());
    } else {
      funcName = buildFunctionName(moduleOp, importOp);
    }

    if (failed(updateOperands(op, importOp, rewriter, updatedOperands,
                              resultOperands))) {
      return failure();
    }

    if (!funcName.has_value())
      return op->emitError() << "Couldn't build name to imported function";

    auto callee = moduleOp.lookupSymbol<mlir::emitc::FuncOp>(funcName.value());
    if (callee == nullptr) {
      return op->emitError()
             << "Couldn't find function with name `" << funcName.value() << "`";
    }

    returnIfError(rewriter, loc, callee, updatedOperands,
                  this->getModuleAnalysis());

    emitc_builders::asRValues(rewriter, loc, resultOperands);
    rewriter.replaceOp(op, resultOperands);

    return success();
  }

  LogicalResult updateOperands(Operation *op, IREE::VM::ImportOp importOp,
                               ConversionPatternRewriter &rewriter,
                               SmallVector<Value> &updatedOperands,
                               SmallVector<Value> &resultOperands) const {
    auto loc = op->getLoc();

    OperandRange operands = op->getOperands();

    int operandIndex = 0;
    int numInputs =
        importOp ? importOp.getFunctionType().getNumInputs() : operands.size();
    for (int i = 0; i < numInputs; i++) {
      if (importOp && importOp.isFuncArgumentVariadic(i)) {
        assert(isa<IREE::VM::CallVariadicOp>(op));
        auto variadicCallOp = cast<IREE::VM::CallVariadicOp>(op);
        APInt segment = *(variadicCallOp.getSegmentSizes().begin() + i);
        int64_t size = segment.getSExtValue();

        Value segmentSize = rewriter
                                .create<emitc::ConstantOp>(
                                    /*location=*/loc,
                                    /*resultType=*/rewriter.getI32Type(),
                                    /*value=*/rewriter.getI32IntegerAttr(size))
                                .getResult();
        updatedOperands.push_back(segmentSize);

        Type type = importOp.getFunctionType().getInput(i);
        int numOps =
            llvm::isa<TupleType>(type) ? llvm::cast<TupleType>(type).size() : 1;
        for (int i = 0; i < size; i++) {
          for (int j = 0; j < numOps; j++) {
            FailureOr<Value> updatedOperand =
                updateOperand(operands[operandIndex], rewriter, loc);
            if (failed(updatedOperand)) {
              return failure();
            }
            updatedOperands.push_back(updatedOperand.value());
            operandIndex++;
          }
        }
      } else {
        FailureOr<Value> updatedOperand =
            updateOperand(operands[operandIndex], rewriter, loc);
        if (failed(updatedOperand)) {
          return failure();
        }
        updatedOperands.push_back(updatedOperand.value());
        operandIndex++;
      }
    }

    // Create a variable for every result and a pointer to it as output
    // parameter to the call.
    for (OpResult result : op->getResults()) {
      if (llvm::isa<IREE::VM::RefType>(result.getType())) {
        Value ref = this->getModuleAnalysis().lookupRef(result);

        resultOperands.push_back(ref);
        updatedOperands.push_back(ref);
      } else {
        auto resultValue =
            emitc_builders::allocateVariable(rewriter, loc, result.getType());
        Value resultPtr = emitc_builders::addressOf(rewriter, loc, resultValue);

        resultOperands.push_back(resultValue);
        updatedOperands.push_back(resultPtr);
      }
    }
    return success();
  }

  FailureOr<Value> updateOperand(Value operand, OpBuilder &builder,
                                 Location loc) const {
    auto ctx = builder.getContext();

    assert(operand.getType() != emitc::PointerType::get(emitc::OpaqueType::get(
                                    ctx, "iree_vm_ref_t")));
    if (!llvm::isa<IREE::VM::RefType>(operand.getType())) {
      return operand;
    }

    Value operandRef = this->getModuleAnalysis().lookupRef(operand);

    auto [ref, refPtr] = emitc_builders::allocZeroInitializedVar(
        builder, loc, emitc::OpaqueType::get(ctx, "iree_vm_ref_t"));

    builder.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/"iree_vm_ref_assign",
        /*operands=*/ArrayRef<Value>{operandRef, refPtr});

    return refPtr;
  }
};

template <typename OpTy>
class CompareRefOpConversion : public EmitCConversionPattern<OpTy> {
  using Adaptor = typename OpTy::Adaptor;
  using EmitCConversionPattern<OpTy>::EmitCConversionPattern;

public:
  CompareRefOpConversion(const TypeConverter &typeConverter,
                         MLIRContext *context, StringRef funcName)
      : EmitCConversionPattern<OpTy>(typeConverter, context),
        funcName(funcName) {}

private:
  LogicalResult
  matchAndRewrite(OpTy cmpOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = cmpOp.getContext();
    auto loc = cmpOp.getLoc();

    auto funcOp =
        cmpOp.getOperation()->template getParentOfType<mlir::emitc::FuncOp>();

    const auto typeConverter =
        this->template getTypeConverter<IREE::VM::EmitCTypeConverter>();

    auto analysis = this->getModuleAnalysis().lookupFunction(funcOp);
    if (failed(analysis)) {
      return cmpOp.emitError() << "parent func op not found in cache.";
    }

    bool moveLhs =
        analysis.value().get().isMove(cmpOp.getLhs(), cmpOp.getOperation());
    bool moveRhs =
        analysis.value().get().isMove(cmpOp.getRhs(), cmpOp.getOperation());

    Value refLhs = this->getModuleAnalysis().lookupRef(cmpOp.getLhs());
    Value refRhs = this->getModuleAnalysis().lookupRef(cmpOp.getRhs());

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        /*op=*/cmpOp,
        /*type=*/cmpOp.getType(),
        /*callee=*/funcName,
        /*operands=*/ArrayRef<Value>{refLhs, refRhs});

    if (moveLhs) {
      emitc_builders::ireeVmRefRelease(rewriter, loc, refLhs);
    }

    // NOTE: If lhs and rhs alias we call release twice on the same
    // argument.
    if (moveRhs) {
      emitc_builders::ireeVmRefRelease(rewriter, loc, refRhs);
    }

    return success();
  }

  StringRef funcName;
};

class CompareRefNotZeroOpConversion
    : public EmitCConversionPattern<IREE::VM::CmpNZRefOp> {
  using Adaptor = IREE::VM::CmpNZRefOp::Adaptor;
  using EmitCConversionPattern<IREE::VM::CmpNZRefOp>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::VM::CmpNZRefOp cmpOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = cmpOp.getLoc();

    auto funcOp = cmpOp.getOperation()->getParentOfType<mlir::emitc::FuncOp>();

    auto &funcAnalysis = getModuleAnalysis().lookupFunction(funcOp);

    bool move = funcAnalysis.isMove(cmpOp.getOperand(), cmpOp.getOperation());

    Value ref = getModuleAnalysis().lookupRef(cmpOp.getOperand());

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        /*op=*/cmpOp,
        /*type=*/cmpOp.getType(),
        /*callee=*/"vm_cmp_nz_ref",
        /*operands=*/ArrayRef<Value>{ref});

    if (move) {
      emitc_builders::ireeVmRefRelease(rewriter, loc, ref);
    }

    return success();
  }
};

class SelectRefOpConversion
    : public EmitCConversionPattern<IREE::VM::SelectRefOp> {
  using Adaptor = typename IREE::VM::SelectRefOp::Adaptor;
  using EmitCConversionPattern<IREE::VM::SelectRefOp>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::VM::SelectRefOp selectOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = selectOp.getContext();
    auto loc = selectOp.getLoc();

    auto moduleOp =
        selectOp.getOperation()->template getParentOfType<IREE::VM::ModuleOp>();
    auto funcOp = selectOp.getOperation()
                      ->template getParentOfType<mlir::emitc::FuncOp>();
    auto &funcAnalysis = getModuleAnalysis().lookupFunction(funcOp);

    const BlockArgument moduleArg = funcOp.getArgument(CCONV_ARGUMENT_MODULE);

    auto resultTypePtr =
        createVmTypeDefPtr(rewriter, loc, this->getModuleAnalysis(), moduleOp,
                           moduleArg, selectOp.getType());
    if (!resultTypePtr.has_value()) {
      return selectOp->emitError() << "generating iree_vm_type_def_t* failed";
    }
    auto resultTypeAsRef =
        rewriter
            .create<emitc::CallOpaqueOp>(
                /*location=*/loc,
                /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
                /*callee=*/"iree_vm_type_def_as_ref",
                /*operands=*/ArrayRef<Value>{resultTypePtr.value()})
            .getResult(0);

    bool moveTrue =
        funcAnalysis.isMove(selectOp.getTrueValue(), selectOp.getOperation());
    bool moveFalse =
        funcAnalysis.isMove(selectOp.getFalseValue(), selectOp.getOperation());

    Value refTrue =
        this->getModuleAnalysis().lookupRef(selectOp.getTrueValue());
    Value refFalse =
        this->getModuleAnalysis().lookupRef(selectOp.getFalseValue());
    Value refResult = this->getModuleAnalysis().lookupRef(selectOp.getResult());

    Type boolType = rewriter.getI1Type();
    auto condition = rewriter.create<IREE::VM::CmpNZI32Op>(
        loc, rewriter.getI32Type(), selectOp.getCondition());
    auto conditionI1 = rewriter.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/boolType,
        /*operand=*/condition.getResult());

    auto *continueBlock =
        rewriter.splitBlock(selectOp->getBlock(), Block::iterator(selectOp));

    Block *trueBlock = nullptr;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      trueBlock = rewriter.createBlock(continueBlock);
      returnIfError(
          /*rewriter=*/rewriter,
          /*location=*/loc,
          /*callee=*/"iree_vm_ref_retain_or_move_checked",
          /*args=*/
          ArrayAttr::get(
              ctx, {rewriter.getBoolAttr(moveTrue), rewriter.getIndexAttr(0),
                    rewriter.getIndexAttr(1), rewriter.getIndexAttr(2)}),
          /*operands=*/ArrayRef<Value>{refTrue, resultTypeAsRef, refResult},
          this->getModuleAnalysis());
      rewriter.create<IREE::VM::BranchOp>(loc, continueBlock);
    }

    Block *falseBlock = nullptr;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      falseBlock = rewriter.createBlock(continueBlock);
      returnIfError(
          /*rewriter=*/rewriter,
          /*location=*/loc,
          /*callee=*/"iree_vm_ref_retain_or_move_checked",
          /*args=*/
          ArrayAttr::get(
              ctx, {rewriter.getBoolAttr(moveFalse), rewriter.getIndexAttr(0),
                    rewriter.getIndexAttr(1), rewriter.getIndexAttr(2)}),
          /*operands=*/ArrayRef<Value>{refFalse, resultTypeAsRef, refResult},
          this->getModuleAnalysis());
      rewriter.create<IREE::VM::BranchOp>(loc, continueBlock);
    }

    rewriter.setInsertionPointAfterValue(conditionI1);
    rewriter.create<mlir::cf::CondBranchOp>(loc, conditionI1.getResult(),
                                            trueBlock, falseBlock);
    rewriter.replaceOp(selectOp, refResult);

    return success();
  }
};

template <typename OpTy>
class ConstOpConversion : public EmitCConversionPattern<OpTy> {
  using Adaptor = typename OpTy::Adaptor;
  using EmitCConversionPattern<OpTy>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy constOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(constOp, constOp.getType(),
                                                   constOp.getValue());
    return success();
  }
};

template <typename OpTy>
class ConstZeroOpConversion : public EmitCConversionPattern<OpTy> {
  using Adaptor = typename OpTy::Adaptor;
  using EmitCConversionPattern<OpTy>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy constZeroOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = constZeroOp.getType();

    Attribute value = rewriter.getZeroAttr(type);

    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(constZeroOp, type, value);
    return success();
  }
};

class ConstRefZeroOpConversion
    : public EmitCConversionPattern<IREE::VM::ConstRefZeroOp> {
  using Adaptor = IREE::VM::ConstRefZeroOp::Adaptor;
  using EmitCConversionPattern<
      IREE::VM::ConstRefZeroOp>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::VM::ConstRefZeroOp constRefZeroOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = constRefZeroOp.getLoc();

    Value ref = getModuleAnalysis().lookupRef(constRefZeroOp.getResult());

    emitc_builders::ireeVmRefRelease(rewriter, loc, ref);

    rewriter.replaceOp(constRefZeroOp, ref);

    return success();
  }
};

class ConstRefRodataOpConversion
    : public EmitCConversionPattern<IREE::VM::ConstRefRodataOp> {
  using Adaptor = IREE::VM::ConstRefRodataOp::Adaptor;
  using EmitCConversionPattern<
      IREE::VM::ConstRefRodataOp>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::VM::ConstRefRodataOp constRefRodataOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = constRefRodataOp.getContext();
    auto loc = constRefRodataOp.getLoc();

    auto rodataOp = lookupSymbolRef<IREE::VM::RodataOp>(
        constRefRodataOp.getOperation(), "rodata");
    if (!rodataOp) {
      return constRefRodataOp.emitError() << "Unable to find RodataOp";
    }

    auto funcOp =
        constRefRodataOp.getOperation()->getParentOfType<mlir::emitc::FuncOp>();

    const BlockArgument stateArg =
        funcOp.getArgument(CCONV_ARGUMENT_MODULE_STATE);
    auto stateArgLValue = emitc_builders::asLValue(rewriter, loc, stateArg);

    auto rodataBuffersPtr =
        cast<TypedValue<emitc::PointerType>>(emitc_builders::structPtrMember(
            rewriter, loc,
            /*type=*/
            emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "iree_vm_buffer_t")),
            /*memberName=*/"rodata_buffers",
            /*operand=*/stateArgLValue));

    auto byteBufferPtrOp = emitc_builders::arrayElementAddress(
        rewriter, loc, /*index=*/rodataOp.getOrdinal()->getZExtValue(),
        /*operand=*/rodataBuffersPtr);

    auto typeIdOp = rewriter.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
        /*callee=*/"iree_vm_buffer_type",
        /*operands=*/ArrayRef<Value>{});

    Value ref = getModuleAnalysis().lookupRef(constRefRodataOp.getResult());

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/"iree_vm_ref_wrap_retain",
        /*args=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{byteBufferPtrOp, typeIdOp.getResult(0), ref},
        getModuleAnalysis());

    rewriter.replaceOp(constRefRodataOp, ref);

    return success();
  }
};

class BranchOpConversion : public EmitCConversionPattern<IREE::VM::BranchOp> {
  using Adaptor = IREE::VM::BranchOp::Adaptor;
  using EmitCConversionPattern<IREE::VM::BranchOp>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::VM::BranchOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    assert(op.getOperands().size() == adaptor.getOperands().size());

    auto isNotRefOperand = [](Value operand) {
      return !llvm::isa<IREE::VM::RefType>(operand.getType());
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
      rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.getDest(),
                                                      op.getOperands());
      return success();
    }

    Block *destDispatch;
    {
      auto funcOp =
          op.getOperation()->template getParentOfType<mlir::emitc::FuncOp>();
      auto &funcAnalysis = getModuleAnalysis().lookupFunction(funcOp);
      auto &signatureConversion = funcAnalysis.lookupBlockConversion(dest);

      OpBuilder::InsertionGuard guard(rewriter);
      destDispatch = rewriter.createBlock(dest);

      IRMapping refMapping;
      for (auto [index, operand] : llvm::enumerate(op.getOperands())) {
        if (isNotRefOperand(operand)) {
          continue;
        }

        Value blockArgRef =
            signatureConversion.getInputMapping(index)->replacementValue;

        assert(isa<IREE::VM::RefType>(operand.getType()));
        assert(isa<emitc::PointerType>(blockArgRef.getType()));

        Value operandRef = getModuleAnalysis().lookupRef(operand);

        refMapping.map(operandRef, blockArgRef);
      }
      if (failed(retainOrMoveRefs(rewriter, loc, refMapping,
                                  /*isMove=*/false))) {
        return op.emitError() << "moving of multiple refs failed";
      }
      rewriter.create<mlir::cf::BranchOp>(loc, op.getDest(), nonRefOperands);
    }

    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, destDispatch);

    return success();
  }
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
    : public EmitCConversionPattern<IREE::VM::CondBranchOp> {
  using Adaptor = IREE::VM::CondBranchOp::Adaptor;
  using EmitCConversionPattern<IREE::VM::CondBranchOp>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::VM::CondBranchOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    assert(op.getOperands().size() == adaptor.getOperands().size());

    auto isNotRefOperand = [](Value operand) {
      return !llvm::isa<IREE::VM::RefType>(operand.getType());
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
        loc, rewriter.getI32Type(), op.getCondition());
    auto conditionI1 = rewriter.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/boolType,
        /*operand=*/condition.getResult());

    // If we don't have ref block arguments, we can convert the operation
    // directly.
    if (adaptor.getOperands().size() == nonRefOperands.size()) {
      rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
          op, conditionI1.getResult(), op.getTrueDest(), op.getTrueOperands(),
          op.getFalseDest(), op.getFalseOperands());
      return success();
    }

    Block *trueDestDispatch;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      trueDestDispatch = rewriter.createBlock(trueDest);

      // Let the BranchOpConversion handle ref block arguments.
      rewriter.create<IREE::VM::BranchOp>(loc, op.getTrueDest(),
                                          op.getTrueOperands());
    }

    Block *falseDestDispatch;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      falseDestDispatch = rewriter.createBlock(falseDest);

      // Let the BranchOpConversion handle ref block arguments.
      rewriter.create<IREE::VM::BranchOp>(loc, op.getFalseDest(),
                                          op.getFalseOperands());
    }

    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        op, conditionI1.getResult(), trueDestDispatch, falseDestDispatch);

    return success();
  }
};

// EmitC does not support cf.switch so we turn the branch table into a long
// sequence of conditional branches.
class BranchTableOpConversion
    : public EmitCConversionPattern<IREE::VM::BranchTableOp> {
  using Adaptor = IREE::VM::BranchTableOp::Adaptor;
  using EmitCConversionPattern<IREE::VM::BranchTableOp>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::VM::BranchTableOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto caseDestinations = op.getCaseDestinations();
    SmallVector<Block *> caseBlocks;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto *nextBlock = rewriter.getInsertionBlock()->getNextNode();
      for (size_t i = 0; i < caseDestinations.size(); ++i)
        caseBlocks.push_back(rewriter.createBlock(nextBlock));
      caseBlocks.push_back(rewriter.createBlock(nextBlock)); // default
    }
    rewriter.create<IREE::VM::BranchOp>(op.getLoc(), caseBlocks.front());
    for (size_t i = 0; i < caseDestinations.size(); ++i) {
      rewriter.setInsertionPointToStart(caseBlocks[i]);
      Value cmp = rewriter.create<IREE::VM::CmpEQI32Op>(
          op.getLoc(), rewriter.getI32Type(), adaptor.getIndex(),
          rewriter.create<IREE::VM::ConstI32Op>(op.getLoc(), i));
      auto caseOperands = adaptor.getCaseOperands();
      rewriter.create<IREE::VM::CondBranchOp>(
          op.getLoc(), cmp, caseDestinations[i], caseOperands[i],
          caseBlocks[i + 1], ValueRange{});
    }
    rewriter.setInsertionPointToStart(caseBlocks.back());
    rewriter.replaceOpWithNewOp<IREE::VM::BranchOp>(
        op, op.getDefaultDestination(), adaptor.getDefaultOperands());
    return success();
  }
};

class ReturnOpConversion : public EmitCConversionPattern<IREE::VM::ReturnOp> {
  using Adaptor = IREE::VM::ReturnOp::Adaptor;
  using EmitCConversionPattern<IREE::VM::ReturnOp>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::VM::ReturnOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto funcOp = op.getOperation()->getParentOfType<mlir::emitc::FuncOp>();

    // The result variables are the last N arguments of the function.
    unsigned int firstOutputArgumentIndex =
        funcOp.getNumArguments() - op.getOperands().size();

    IRMapping refMapping;
    for (const auto &pair : llvm::enumerate(op.getOperands())) {
      Value operand = pair.value();
      size_t index = pair.index();

      unsigned int argumentIndex = firstOutputArgumentIndex + index;
      BlockArgument resultArgument = funcOp.getArgument(argumentIndex);

      if (llvm::isa<IREE::VM::RefType>(operand.getType())) {
        assert(operand.getType() !=
               emitc::PointerType::get(
                   emitc::OpaqueType::get(op.getContext(), "iree_vm_ref_t")));

        Value operandRef = getModuleAnalysis().lookupRef(operand);

        refMapping.map(operandRef, resultArgument);
      } else {
        rewriter.create<emitc::CallOpaqueOp>(
            /*location=*/loc,
            /*type=*/TypeRange{},
            /*callee=*/"EMITC_DEREF_ASSIGN_VALUE",
            /*operands=*/ArrayRef<Value>{resultArgument, operand});
      }
    }

    if (failed(retainOrMoveRefs(rewriter, loc, refMapping, /*isMove=*/true))) {
      return op.emitError() << "moving of multiple refs failed";
    }
    releaseRefs(rewriter, loc, funcOp, getModuleAnalysis());

    auto status = emitc_builders::ireeOkStatus(rewriter, loc);

    rewriter.replaceOpWithNewOp<mlir::emitc::ReturnOp>(op, status);

    return success();
  }
};

class ImportResolvedOpConversion
    : public EmitCConversionPattern<IREE::VM::ImportResolvedOp> {
  using Adaptor = IREE::VM::ImportResolvedOp::Adaptor;
  using EmitCConversionPattern<
      IREE::VM::ImportResolvedOp>::EmitCConversionPattern;

private:
  LogicalResult
  matchAndRewrite(IREE::VM::ImportResolvedOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    IREE::VM::ImportOp importOp =
        lookupSymbolRef<IREE::VM::ImportOp>(op.getOperation(), "import");
    int importOrdinal = importOp.getOrdinal()->getZExtValue();

    auto funcOp = op->getParentOfType<mlir::emitc::FuncOp>();

    const BlockArgument stateArg =
        funcOp.getArgument(CCONV_ARGUMENT_MODULE_STATE);
    auto stateArgLValue = emitc_builders::asLValue(rewriter, loc, stateArg);

    auto imports =
        cast<TypedValue<emitc::PointerType>>(emitc_builders::structPtrMember(
            rewriter, loc,
            /*type=*/
            emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "iree_vm_function_t")),
            /*memberName=*/"imports",
            /*operand=*/stateArgLValue));

    auto import = emitc_builders::arrayElementAddress(
        rewriter, loc, /*index=*/importOrdinal, /*operand=*/imports);
    auto importLValue = emitc_builders::asLValue(rewriter, loc, import);

    // (iree_vm_function_t*)->module
    auto importModule = emitc_builders::structPtrMember(
        rewriter, loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_module_t")),
        /*memberName=*/"module",
        /*operand=*/importLValue);

    Type boolType = rewriter.getIntegerType(1);
    auto conditionI1 = rewriter
                           .create<emitc::LogicalNotOp>(
                               /*location=*/loc,
                               /*type=*/boolType,
                               /*operands=*/importModule)
                           .getResult();
    auto invConditionI1 = rewriter
                              .create<emitc::LogicalNotOp>(
                                  /*location=*/loc,
                                  /*type=*/boolType,
                                  /*operands=*/conditionI1)
                              .getResult();

    auto i32Type = rewriter.getIntegerType(32);
    auto conditionI32 = rewriter.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/i32Type,
        /*operand=*/invConditionI1);

    rewriter.replaceOp(op, {conditionI32.getResult()});

    return success();
  }
};

class FailOpConversion : public EmitCConversionPattern<IREE::VM::FailOp> {
  using Adaptor = IREE::VM::FailOp::Adaptor;
  using EmitCConversionPattern<IREE::VM::FailOp>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::VM::FailOp op, Adaptor adaptor,
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

      auto funcOp = op.getOperation()->getParentOfType<mlir::emitc::FuncOp>();

      releaseRefs(rewriter, loc, funcOp, getModuleAnalysis());

      auto status = emitc_builders::ireeOkStatus(rewriter, loc);

      rewriter.create<mlir::emitc::ReturnOp>(loc, status);
    }
    Block *failureBlock;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      failureBlock = rewriter.createBlock(parentRegion, parentRegion->end());

      auto funcOp = op.getOperation()->getParentOfType<mlir::emitc::FuncOp>();

      releaseRefs(rewriter, loc, funcOp, getModuleAnalysis());

      Value message = emitc_builders::ireeMakeCstringView(
          rewriter, loc, op.getMessage().value_or("").str());
      auto messageLValue = emitc_builders::asLValue(rewriter, loc, message);
      Type type = emitc::OpaqueType::get(ctx, "iree_host_size_t");
      Value messageSize =
          emitc_builders::structMember(rewriter, loc,
                                       /*type=*/type,
                                       /*memberName=*/"size",
                                       /*operand=*/messageLValue);

      auto messageSizeIntOp = rewriter.create<emitc::CastOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "int"),
          /*operand=*/messageSize);

      Type charPtr =
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const char"));
      auto messageDataOp =
          emitc_builders::structMember(rewriter, loc,
                                       /*type=*/charPtr,
                                       /*memberName=*/"data",
                                       /*operand=*/messageLValue);

      auto status = rewriter.create<emitc::CallOpaqueOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
          /*callee=*/"iree_status_allocate_f",
          /*operands=*/
          ArrayRef<Value>{messageSizeIntOp.getResult(), messageDataOp},
          /*args=*/
          ArrayAttr::get(
              ctx,
              {emitc::OpaqueAttr::get(ctx, "IREE_STATUS_FAILED_PRECONDITION"),
               emitc::OpaqueAttr::get(ctx, "\"<vm>\""),
               rewriter.getI32IntegerAttr(0),
               emitc::OpaqueAttr::get(ctx, "\"%.*s\""),
               rewriter.getIndexAttr(0), rewriter.getIndexAttr(1)}));

      rewriter.create<mlir::emitc::ReturnOp>(loc, status.getResult(0));
    }

    Type boolType = rewriter.getIntegerType(1);
    auto condition = rewriter.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/boolType,
        /*operand=*/op.getStatus());

    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        op, condition.getResult(), failureBlock, passthroughBlock);

    return success();
  }
};

template <typename OpTy, typename GlobalOpTy>
class GlobalLoadOpConversion : public EmitCConversionPattern<OpTy> {
  using Adaptor = typename OpTy::Adaptor;
  using EmitCConversionPattern<OpTy>::EmitCConversionPattern;

public:
  GlobalLoadOpConversion(const TypeConverter &typeConverter,
                         MLIRContext *context, StringRef funcName)
      : EmitCConversionPattern<OpTy>(typeConverter, context),
        funcName(funcName) {}

private:
  LogicalResult
  matchAndRewrite(OpTy loadOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();

    GlobalOpTy globalOp =
        lookupSymbolRef<GlobalOpTy>(loadOp.getOperation(), "global");
    if (!globalOp) {
      return loadOp.emitError() << "Unable to find GlobalOp";
    }

    auto funcOp =
        loadOp.getOperation()->template getParentOfType<mlir::emitc::FuncOp>();

    const BlockArgument stateArg =
        funcOp.getArgument(CCONV_ARGUMENT_MODULE_STATE);
    auto stateArgLValue = emitc_builders::asLValue(rewriter, loc, stateArg);

    auto rwDataPtr = emitc_builders::structPtrMember(
        rewriter, loc,
        /*type=*/emitc::PointerType::get(rewriter.getIntegerType(8, false)),
        /*memberName=*/"rwdata",
        /*operand=*/stateArgLValue);

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        /*op=*/loadOp,
        /*type=*/loadOp.getOperation()->getResultTypes(),
        /*callee=*/funcName,
        /*operands=*/ArrayRef<Value>{rwDataPtr},
        /*args=*/
        rewriter.getArrayAttr(
            {rewriter.getIndexAttr(0),
             rewriter.getUI32IntegerAttr(static_cast<uint32_t>(
                 globalOp.getOrdinal()->getZExtValue()))}));

    return success();
  }

  StringRef funcName;
};

template <typename OpTy>
class GlobalLoadStoreRefOpConversion : public EmitCConversionPattern<OpTy> {
  using Adaptor = typename OpTy::Adaptor;
  using EmitCConversionPattern<OpTy>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, Adaptor adaptor,
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

    auto globalOrdinal = globalOp.getOrdinal()->getZExtValue();

    auto funcOp = op->getParentOfType<mlir::emitc::FuncOp>();

    auto &funcAnalysis = this->getModuleAnalysis().lookupFunction(funcOp);

    Value localValue = isLoad ? op->getResult(0) : op->getOperand(0);

    Value localRef = this->getModuleAnalysis().lookupRef(localValue);

    const BlockArgument stateArg =
        funcOp.getArgument(CCONV_ARGUMENT_MODULE_STATE);
    auto stateArgLValue = emitc_builders::asLValue(rewriter, loc, stateArg);

    auto refs =
        cast<TypedValue<emitc::PointerType>>(emitc_builders::structPtrMember(
            rewriter, loc,
            /*type=*/
            emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "iree_vm_ref_t")),
            /*memberName=*/"refs",
            /*operand=*/stateArgLValue));

    auto stateRef = emitc_builders::arrayElementAddress(
        rewriter, loc, /*index=*/globalOrdinal, /*operand=*/refs);

    auto moduleOp = op->getParentOfType<IREE::VM::ModuleOp>();
    auto parentFuncOp = op->getParentOfType<mlir::emitc::FuncOp>();
    const BlockArgument moduleArg =
        parentFuncOp.getArgument(CCONV_ARGUMENT_MODULE);

    Type elementType = localValue.getType();

    auto elementTypePtr =
        createVmTypeDefPtr(rewriter, op->getLoc(), this->getModuleAnalysis(),
                           moduleOp, moduleArg, elementType);

    if (!elementTypePtr.has_value()) {
      return op->emitError() << "generating iree_vm_type_def_t* failed";
    }

    auto typedefAsRef =
        rewriter
            .create<emitc::CallOpaqueOp>(
                /*location=*/loc,
                /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
                /*callee=*/"iree_vm_type_def_as_ref",
                /*operands=*/ArrayRef<Value>{elementTypePtr.value()})
            .getResult(0);

    Value srcRef = isLoad ? stateRef : localRef;
    Value destRef = isLoad ? localRef : stateRef;

    bool move = funcAnalysis.isMove(localValue, op);

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/"iree_vm_ref_retain_or_move_checked",
        /*args=*/
        ArrayAttr::get(ctx,
                       {rewriter.getBoolAttr(move), rewriter.getIndexAttr(0),
                        rewriter.getIndexAttr(1), rewriter.getIndexAttr(2)}),
        /*operands=*/ArrayRef<Value>{srcRef, typedefAsRef, destRef},
        this->getModuleAnalysis());

    if (isLoad) {
      rewriter.replaceOp(op, localRef);
    } else {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

template <typename OpTy, typename GlobalOpTy>
class GlobalStoreOpConversion : public EmitCConversionPattern<OpTy> {
  using Adaptor = typename OpTy::Adaptor;
  using EmitCConversionPattern<OpTy>::EmitCConversionPattern;

public:
  GlobalStoreOpConversion(const TypeConverter &typeConverter,
                          MLIRContext *context, StringRef funcName)
      : EmitCConversionPattern<OpTy>(typeConverter, context),
        funcName(funcName) {}

private:
  LogicalResult
  matchAndRewrite(OpTy storeOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = storeOp.getLoc();

    GlobalOpTy globalOp =
        lookupSymbolRef<GlobalOpTy>(storeOp.getOperation(), "global");
    if (!globalOp) {
      return storeOp.emitError() << "Unable to find GlobalOp";
    }

    auto funcOp =
        storeOp.getOperation()->template getParentOfType<mlir::emitc::FuncOp>();

    const BlockArgument stateArg =
        funcOp.getArgument(CCONV_ARGUMENT_MODULE_STATE);
    auto stateArgLValue = emitc_builders::asLValue(rewriter, loc, stateArg);

    auto rwDataPtr = emitc_builders::structPtrMember(
        rewriter, loc,
        /*type=*/emitc::PointerType::get(rewriter.getIntegerType(8, false)),
        /*memberName=*/"rwdata", /*operand=*/stateArgLValue);

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        /*op=*/storeOp,
        /*type=*/storeOp.getOperation()->getResultTypes(),
        /*callee=*/funcName,
        /*operands=*/ArrayRef<Value>{rwDataPtr, storeOp.getValue()},
        /*args=*/
        rewriter.getArrayAttr(
            {rewriter.getIndexAttr(0),
             rewriter.getUI32IntegerAttr(
                 static_cast<uint32_t>(globalOp.getOrdinal()->getZExtValue())),
             rewriter.getIndexAttr(1)}));

    return success();
  }

  StringRef funcName;
};

// Convert vm operations with wrapped containers to multiple emitc opaque_calls.
// The wrapping ref pointers are first dereferenced and the results are used as
// the arguments of the specified function name.
template <typename OpTy>
class ContainerOpConversion : public EmitCConversionPattern<OpTy> {
  using Adaptor = typename OpTy::Adaptor;
  using EmitCConversionPattern<OpTy>::EmitCConversionPattern;

public:
  ContainerOpConversion(const TypeConverter &typeConverter,
                        MLIRContext *context, StringRef funcName,
                        DenseSet<size_t> refArgumentIndices, bool failable)
      : EmitCConversionPattern<OpTy>(typeConverter, context),
        funcName(funcName), refArgumentIndices(refArgumentIndices),
        failable(failable) {}

private:
  LogicalResult
  matchAndRewrite(OpTy op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    SmallVector<Value> unwrappedOperands;
    for (const auto &operand : llvm::enumerate(adaptor.getOperands())) {
      if (refArgumentIndices.contains(operand.index())) {
        Type originalType =
            op.getOperation()->getOperand(operand.index()).getType();

        assert(isa<IREE::VM::RefType>(originalType) && "expected ref type");

        Type objectType =
            llvm::cast<IREE::VM::RefType>(originalType).getObjectType();

        std::optional<std::pair<StringRef, StringRef>> vmNames =
            TypeSwitch<Type, std::optional<std::pair<StringRef, StringRef>>>(
                objectType)
                .Case<IREE::VM::ListType>([&](auto t) {
                  return std::make_pair(StringRef("iree_vm_list_t"),
                                        StringRef("iree_vm_list_deref"));
                })
                .template Case<IREE::VM::BufferType>([&](auto t) {
                  return std::make_pair(StringRef("iree_vm_buffer_t"),
                                        StringRef("iree_vm_buffer_deref"));
                })
                .Default([](Type) { return std::nullopt; });

        if (!vmNames.has_value()) {
          return op.emitOpError() << "object type not handled";
        }

        StringRef vmType = std::get<0>(vmNames.value());
        StringRef vmDerefCallee = std::get<1>(vmNames.value());

        Value refValue =
            emitc_builders::contentsOf(rewriter, loc, operand.value());
        auto derefOp = failContainerNull(
            /*rewriter=*/rewriter,
            /*location=*/loc,
            /*type=*/
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, vmType)),
            /*callee=*/vmDerefCallee,
            /*args=*/ArrayAttr{},
            /*operands=*/ArrayRef<Value>{refValue}, this->getModuleAnalysis());
        unwrappedOperands.push_back(derefOp.getResult(0));
      } else {
        unwrappedOperands.push_back(operand.value());
      }
    }

    SmallVector<Value> resultOperands;
    if (failed(patchOperands(op, adaptor, rewriter, unwrappedOperands,
                             resultOperands))) {
      return op.emitError() << "failed to patch operands";
    }

    if (failable) {
      returnIfError(
          /*rewriter=*/rewriter,
          /*location=*/loc,
          /*callee=*/funcName,
          /*args=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>(unwrappedOperands),
          this->getModuleAnalysis());

      emitc_builders::asRValues(rewriter, loc, resultOperands);
      rewriter.replaceOp(op, resultOperands);
    } else {
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          /*op=*/op,
          /*type=*/op.getOperation()->getResultTypes(),
          /*callee=*/funcName,
          /*operands=*/ArrayRef<Value>(unwrappedOperands));
    }

    return success();
  }

  LogicalResult patchOperands(Operation *op, Adaptor adaptor,
                              ConversionPatternRewriter &rewriter,
                              SmallVector<Value> &operands,
                              SmallVector<Value> &results) const {
    if (failable) {
      if (failed(createOutOperands(op, rewriter, operands, results))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult createOutOperands(Operation *op,
                                  ConversionPatternRewriter &rewriter,
                                  SmallVector<Value> &operands,
                                  SmallVector<Value> &results) const {
    auto loc = op->getLoc();

    for (OpResult result : op->getResults()) {
      if (llvm::isa<IREE::VM::RefType>(result.getType())) {
        Value ref = this->getModuleAnalysis().lookupRef(result);

        results.push_back(ref);
        operands.push_back(ref);
      } else {
        Type type = result.getType();
        auto resultValue = emitc_builders::allocateVariable(
            rewriter, loc, type, rewriter.getZeroAttr(type));
        Value resultPtr = emitc_builders::addressOf(rewriter, loc, resultValue);

        results.push_back(resultValue);
        operands.push_back(resultPtr);
      }
    }
    return success();
  }

  StringRef funcName;

  // The indices of the wrapped arguments.
  DenseSet<size_t> refArgumentIndices;

  // Whether the function call can fail, i.e. it returns an iree_status_t.
  bool failable;
};

template <typename OpTy>
class ContainerAllocOpConversion : public EmitCConversionPattern<OpTy> {
  using Adaptor = typename OpTy::Adaptor;
  using EmitCConversionPattern<OpTy>::EmitCConversionPattern;

  // Bundle function and type names based on the container type
  struct CNames {
    std::string type;
    std::string typeId;
    std::string constructor;
  };

  LogicalResult
  matchAndRewrite(OpTy op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    Type objectType =
        llvm::cast<IREE::VM::RefType>(op.getType()).getObjectType();
    std::optional<Type> elementType = extractElementType(ctx, objectType);
    std::optional<CNames> cNames = extractCNames(op);

    if (!elementType.has_value() || !cNames.has_value()) {
      return op.emitError() << "unknown container type";
    }

    auto container = emitc_builders::allocateVariable(
        rewriter, loc,
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, cNames.value().type)),
        {"NULL"});

    Value containerPtr = emitc_builders::addressOf(rewriter, loc, container);

    auto funcOp =
        op.getOperation()->template getParentOfType<mlir::emitc::FuncOp>();

    const BlockArgument stateArg =
        funcOp.getArgument(CCONV_ARGUMENT_MODULE_STATE);
    auto stateArgLValue = emitc_builders::asLValue(rewriter, loc, stateArg);

    auto allocator = emitc_builders::structPtrMember(
        rewriter, loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_allocator_t"),
        /*memberName=*/"allocator",
        /*operand=*/stateArgLValue);

    std::optional<SmallVector<Value>> operands = getOperands(
        op, adaptor, rewriter, elementType.value(), containerPtr, allocator);

    if (!operands.has_value()) {
      return op.emitError() << "failed to build operands";
    }

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/cNames.value().constructor,
        /*args=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{operands.value()},
        this->getModuleAnalysis());

    Value ref = this->getModuleAnalysis().lookupRef(op.getResult());

    Value refType =
        rewriter
            .create<emitc::CallOpaqueOp>(
                /*location=*/loc,
                /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
                /*callee=*/cNames.value().typeId,
                /*operands=*/ArrayRef<Value>{})
            .getResult(0);

    auto containerRValue = emitc_builders::asRValue(rewriter, loc, container);
    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/"iree_vm_ref_wrap_assign",
        /*args=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{containerRValue, refType, ref},
        this->getModuleAnalysis());

    rewriter.replaceOp(op, ref);

    return success();
  }

  std::optional<Type> extractElementType(MLIRContext *ctx, Type t) const {
    if (auto listType = llvm::dyn_cast<IREE::VM::ListType>(t)) {
      return listType.getElementType();
    } else if (auto bufferType = llvm::dyn_cast<IREE::VM::BufferType>(t)) {
      return NoneType::get(ctx);
    }
    return std::nullopt;
  }

  std::optional<CNames> extractCNames(OpTy op) const {
    if (isa<IREE::VM::ListAllocOp>(op)) {
      return CNames{"iree_vm_list_t", "iree_vm_list_type",
                    "iree_vm_list_create"};
    } else if (isa<IREE::VM::BufferAllocOp>(op)) {
      return CNames{"iree_vm_buffer_t", "iree_vm_buffer_type",
                    "iree_vm_buffer_create"};
    } else if (isa<IREE::VM::BufferCloneOp>(op)) {
      return CNames{"iree_vm_buffer_t", "iree_vm_buffer_type",
                    "iree_vm_buffer_clone"};
    }
    return std::nullopt;
  }

  std::optional<SmallVector<Value>>
  getOperands(IREE::VM::ListAllocOp op, Adaptor adaptor,
              ConversionPatternRewriter &rewriter, Type elementType,
              Value containerPtr, Value allocator) const {

    auto moduleOp = op.getOperation()->getParentOfType<IREE::VM::ModuleOp>();
    auto parentFuncOp =
        op.getOperation()->getParentOfType<mlir::emitc::FuncOp>();
    const BlockArgument moduleArg =
        parentFuncOp.getArgument(CCONV_ARGUMENT_MODULE);
    auto elementTypePtr =
        createVmTypeDefPtr(rewriter, op.getLoc(), this->getModuleAnalysis(),
                           moduleOp, moduleArg, elementType);

    if (!elementTypePtr.has_value()) {
      return std::nullopt;
    }

    Value capacity = adaptor.getOperands()[0];
    SmallVector<Value> result = {elementTypePtr.value(), capacity, allocator,
                                 containerPtr};
    return result;
  }

  std::optional<SmallVector<Value>>
  getOperands(IREE::VM::BufferAllocOp op, Adaptor adaptor,
              ConversionPatternRewriter &rewriter, Type elementType,
              Value containerPtr, Value allocator) const {
    auto ctx = op.getContext();
    auto loc = op.getLoc();
    Value access =
        rewriter
            .create<emitc::ConstantOp>(
                /*location=*/loc,
                /*resultType=*/
                emitc::OpaqueType::get(ctx, "iree_vm_buffer_access_t"),
                /*value=*/
                emitc::OpaqueAttr::get(ctx,
                                       "IREE_VM_BUFFER_ACCESS_MUTABLE | "
                                       "IREE_VM_BUFFER_ACCESS_ORIGIN_GUEST"))
            .getResult();
    Value length = adaptor.getOperands()[0];
    Value alignment = adaptor.getOperands()[1];

    SmallVector<Value> result = {access, length, alignment, allocator,
                                 containerPtr};
    return result;
  }

  std::optional<SmallVector<Value>>
  getOperands(IREE::VM::BufferCloneOp op, Adaptor adaptor,
              ConversionPatternRewriter &rewriter, Type elementType,
              Value containerPtr, Value allocator) const {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    Value access =
        rewriter
            .create<emitc::ConstantOp>(
                /*location=*/loc,
                /*resultType=*/
                emitc::OpaqueType::get(ctx, "iree_vm_buffer_access_t"),
                /*value=*/
                emitc::OpaqueAttr::get(ctx,
                                       "IREE_VM_BUFFER_ACCESS_MUTABLE | "
                                       "IREE_VM_BUFFER_ACCESS_ORIGIN_GUEST"))
            .getResult();

    Value refPtr = adaptor.getOperands()[0];
    Value refValue = emitc_builders::contentsOf(rewriter, loc, refPtr);

    Value source =
        failContainerNull(
            /*rewriter=*/rewriter,
            /*location=*/loc,
            /*type=*/
            emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "iree_vm_buffer_t")),
            /*callee=*/"iree_vm_buffer_deref",
            /*args=*/ArrayAttr{},
            /*operands=*/ArrayRef<Value>{refValue}, this->getModuleAnalysis())
            .getResult(0);

    Value offset = adaptor.getOperands()[1];
    Value length = adaptor.getOperands()[2];
    Value alignment = adaptor.getOperands()[3];

    SmallVector<Value> result = {access,    source,    offset,      length,
                                 alignment, allocator, containerPtr};
    return result;
  }
};

template <typename OpTy>
class ListGetOpConversion : public EmitCConversionPattern<OpTy> {
  using Adaptor = typename OpTy::Adaptor;
  using EmitCConversionPattern<OpTy>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy getOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = getOp.getContext();
    auto loc = getOp.getLoc();

    std::optional<StringRef> valueTypeEnum;
    std::optional<StringRef> valueExtractor;

    std::tie(valueTypeEnum, valueExtractor) =
        TypeSwitch<Operation *, std::pair<std::optional<StringRef>,
                                          std::optional<StringRef>>>(
            getOp.getOperation())
            .Case<IREE::VM::ListGetI32Op>([&](auto op) {
              return std::make_pair(StringRef("IREE_VM_VALUE_TYPE_I32"),
                                    StringRef("iree_vm_value_get_i32"));
            })
            .template Case<IREE::VM::ListGetI64Op>([&](auto op) {
              return std::make_pair(StringRef("IREE_VM_VALUE_TYPE_I64"),
                                    StringRef("iree_vm_value_get_i64"));
            })
            .Default([](Operation *) {
              return std::make_pair(std::nullopt, std::nullopt);
            });

    if (!valueTypeEnum.has_value() || !valueExtractor.has_value()) {
      return getOp.emitOpError() << "element type not handled";
    }

    auto value = emitc_builders::allocateVariable(
        rewriter, loc, emitc::OpaqueType::get(ctx, "iree_vm_value_t"));

    Value valuePtr = emitc_builders::addressOf(rewriter, loc, value);

    Value refValue =
        emitc_builders::contentsOf(rewriter, loc, adaptor.getOperands()[0]);

    auto listDerefOp = failContainerNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_list_t")),
        /*callee=*/"iree_vm_list_deref",
        /*args=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refValue}, this->getModuleAnalysis());

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/"iree_vm_list_get_value_as",
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
                             emitc::OpaqueAttr::get(ctx, valueTypeEnum.value()),
                             rewriter.getIndexAttr(2)}),
        /*operands=*/
        ArrayRef<Value>{listDerefOp.getResult(0), getOp.getIndex(), valuePtr},
        this->getModuleAnalysis());

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        /*op=*/getOp,
        /*type=*/getOp.getType(),
        /*callee=*/valueExtractor.value(),
        /*operands=*/ArrayRef<Value>{valuePtr});

    return success();
  }
};

class ListGetRefOpConversion
    : public EmitCConversionPattern<IREE::VM::ListGetRefOp> {
  using Adaptor = IREE::VM::ListGetRefOp::Adaptor;
  using EmitCConversionPattern<IREE::VM::ListGetRefOp>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::VM::ListGetRefOp getOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = getOp.getContext();
    auto loc = getOp.getLoc();

    Value listRefValue =
        emitc_builders::contentsOf(rewriter, loc, adaptor.getOperands()[0]);

    auto listDerefOp = failContainerNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_list_t")),
        /*callee=*/"iree_vm_list_deref",
        /*args=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{listRefValue}, getModuleAnalysis());

    Value ref = getModuleAnalysis().lookupRef(getOp.getResult());

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/"iree_vm_list_get_ref_retain",
        /*args=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{listDerefOp.getResult(0), getOp.getIndex(), ref},
        getModuleAnalysis());

    auto moduleOp = getOp.getOperation()->getParentOfType<IREE::VM::ModuleOp>();
    auto parentFuncOp =
        getOp.getOperation()->getParentOfType<mlir::emitc::FuncOp>();
    const BlockArgument moduleArg =
        parentFuncOp.getArgument(CCONV_ARGUMENT_MODULE);

    Type elementType = getOp.getResult().getType();

    auto elementTypePtr =
        createVmTypeDefPtr(rewriter, getOp.getLoc(), getModuleAnalysis(),
                           moduleOp, moduleArg, elementType);

    if (!elementTypePtr.has_value()) {
      return getOp.emitError() << "generating iree_vm_type_def_t* failed";
    }

    // Build the following expression:
    // (ref->type != IREE_VM_REF_TYPE_NULL &&
    // (iree_vm_type_def_is_value(type_def) || ref->type !=
    // iree_vm_type_def_as_ref(type_def)))
    Value invalidType;
    {
      auto refLValue = emitc_builders::asLValue(rewriter, loc, ref);
      // ref->type
      auto refType = emitc_builders::structPtrMember(
          rewriter, loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
          /*memberName=*/"type",
          /*operand=*/refLValue);

      // IREE_VM_REF_TYPE_NULL
      auto refTypeNull =
          rewriter
              .create<emitc::ConstantOp>(
                  /*location=*/loc,
                  /*resultType=*/
                  emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
                  /*value=*/
                  emitc::OpaqueAttr::get(ctx, "IREE_VM_REF_TYPE_NULL"))
              .getResult();

      // ref->type != IREE_VM_REF_TYPE_NULL
      auto refTypeIsNotNull = rewriter
                                  .create<emitc::CmpOp>(
                                      /*location=*/loc,
                                      /*type=*/rewriter.getI1Type(),
                                      /*predicate*/ emitc::CmpPredicate::ne,
                                      /*lhs*/ refType,
                                      /*rhs*/ refTypeNull)
                                  .getResult();

      // (iree_vm_type_def_is_value(type_def)
      auto typedefIsValue =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  /*location=*/loc,
                  /*type=*/rewriter.getI1Type(),
                  /*callee=*/"iree_vm_type_def_is_value",
                  /*operands=*/ArrayRef<Value>{elementTypePtr.value()})
              .getResult(0);

      // iree_vm_type_def_as_ref(type_def)
      auto typedefAsRef =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  /*location=*/loc,
                  /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
                  /*callee=*/"iree_vm_type_def_as_ref",
                  /*operands=*/ArrayRef<Value>{elementTypePtr.value()})
              .getResult(0);

      // ref->type != iree_vm_type_def_as_ref(type_def)
      auto refTypesDontMatch = rewriter
                                   .create<emitc::CmpOp>(
                                       /*location=*/loc,
                                       /*type=*/rewriter.getI1Type(),
                                       /*predicate*/ emitc::CmpPredicate::ne,
                                       /*lhs*/ refType,
                                       /*rhs*/ typedefAsRef)
                                   .getResult();

      auto invalidRefType = rewriter
                                .create<emitc::LogicalOrOp>(
                                    /*location=*/loc,
                                    /*type=*/rewriter.getI1Type(),
                                    /*lhs*/ typedefIsValue,
                                    /*rhs*/ refTypesDontMatch)
                                .getResult();

      invalidType = rewriter
                        .create<emitc::LogicalAndOp>(
                            /*location=*/loc,
                            /*type=*/rewriter.getI1Type(),
                            /*lhs*/ refTypeIsNotNull,
                            /*rhs*/ invalidRefType)
                        .getResult();
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

      emitc_builders::ireeVmRefRelease(rewriter, loc, ref);

      rewriter.create<mlir::cf::BranchOp>(loc, continuationBlock);
    }

    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<cf::CondBranchOp>(loc, invalidType, failureBlock,
                                      continuationBlock);

    rewriter.replaceOp(getOp, ref);

    return success();
  }
};

template <typename OpTy>
class ListSetOpConversion : public EmitCConversionPattern<OpTy> {
  using Adaptor = typename OpTy::Adaptor;
  using EmitCConversionPattern<OpTy>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy setOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = setOp.getContext();
    auto loc = setOp.getLoc();

    std::optional<StringRef> valueConstructor =
        TypeSwitch<Operation *, std::optional<StringRef>>(setOp.getOperation())
            .Case<IREE::VM::ListSetI32Op>(
                [&](auto op) { return StringRef("iree_vm_value_make_i32"); })
            .template Case<IREE::VM::ListSetI64Op>(
                [&](auto op) { return StringRef("iree_vm_value_make_i64"); })
            .Default([](Operation *) { return std::nullopt; });

    if (!valueConstructor.has_value()) {
      return setOp.emitOpError() << " not handled";
    }

    auto valueOp = rewriter.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_value_t"),
        /*callee=*/valueConstructor.value(),
        /*operands=*/ArrayRef<Value>{setOp.getValue()});

    auto value = emitc_builders::asLValue(rewriter, loc, valueOp.getResult(0));

    Value valuePtr = emitc_builders::addressOf(rewriter, loc, value);

    Value refValue =
        emitc_builders::contentsOf(rewriter, loc, adaptor.getOperands()[0]);

    auto listDerefOp = failContainerNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_list_t")),
        /*callee=*/"iree_vm_list_deref",
        /*args=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refValue}, this->getModuleAnalysis());

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/"iree_vm_list_set_value",
        /*args=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{listDerefOp.getResult(0), setOp.getIndex(), valuePtr},
        this->getModuleAnalysis());

    rewriter.eraseOp(setOp);

    return success();
  }
};

class ListSetRefOpConversion
    : public EmitCConversionPattern<IREE::VM::ListSetRefOp> {
  using Adaptor = IREE::VM::ListSetRefOp::Adaptor;
  using EmitCConversionPattern<IREE::VM::ListSetRefOp>::EmitCConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::VM::ListSetRefOp setOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = setOp.getContext();
    auto loc = setOp.getLoc();

    Value refValue =
        emitc_builders::contentsOf(rewriter, loc, adaptor.getOperands()[0]);

    auto listDerefOp = failContainerNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_list_t")),
        /*callee=*/"iree_vm_list_deref",
        /*args=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refValue}, getModuleAnalysis());

    auto funcOp = setOp.getOperation()->getParentOfType<mlir::emitc::FuncOp>();

    auto &funcAnalysis = getModuleAnalysis().lookupFunction(funcOp);

    bool move = funcAnalysis.isMove(setOp.getValue(), setOp.getOperation());

    StringRef callee =
        move ? "iree_vm_list_set_ref_move" : "iree_vm_list_set_ref_retain";

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/callee,
        /*args=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{listDerefOp.getResult(0), setOp.getIndex(),
                        adaptor.getValue()},
        getModuleAnalysis());

    rewriter.eraseOp(setOp);

    return success();
  }
};
} // namespace

void populateVMToEmitCPatterns(ConversionTarget &conversionTarget,
                               IREE::VM::EmitCTypeConverter &typeConverter,
                               RewritePatternSet &patterns) {
  auto context = patterns.getContext();
  populateUtilConversionPatterns(context, conversionTarget, typeConverter,
                                 patterns);

  // Patterns
#define ADD_GENERIC_PATTERN(Op, FuncName)                                      \
  patterns.add<GenericOpConversion<Op>>(typeConverter, context, FuncName)
#define ADD_CONTAINER_PATTERN(Op, FuncName, IndexSet, Failable)                \
  patterns.add<ContainerOpConversion<Op>>(typeConverter, context, FuncName,    \
                                          IndexSet, Failable);
#define ADD_GLOBAL_LOAD_PATTERN(Op, GlobalOp, FuncName)                        \
  patterns.add<GlobalLoadOpConversion<Op, GlobalOp>>(typeConverter, context,   \
                                                     FuncName);
#define ADD_GLOBAL_STORE_PATTERN(Op, GlobalOp, FuncName)                       \
  patterns.add<GlobalStoreOpConversion<Op, GlobalOp>>(typeConverter, context,  \
                                                      FuncName);

  // argument free patterns
  // clang-format off
  patterns.add<
    BranchOpConversion,
    CallOpConversion<IREE::VM::CallOp>,
    CallOpConversion<IREE::VM::CallVariadicOp>,
    CompareRefNotZeroOpConversion,
    SelectRefOpConversion,
    CondBranchOpConversion,
    BranchTableOpConversion,
    ConstOpConversion<IREE::VM::ConstF32Op>,
    ConstOpConversion<IREE::VM::ConstF64Op>,
    ConstOpConversion<IREE::VM::ConstI32Op>,
    ConstOpConversion<IREE::VM::ConstI64Op>,
    ConstRefRodataOpConversion,
    ConstRefZeroOpConversion,
    ConstZeroOpConversion<IREE::VM::ConstF32ZeroOp>,
    ConstZeroOpConversion<IREE::VM::ConstF64ZeroOp>,
    ConstZeroOpConversion<IREE::VM::ConstI32ZeroOp>,
    ConstZeroOpConversion<IREE::VM::ConstI64ZeroOp>,
    ContainerAllocOpConversion<IREE::VM::BufferAllocOp>,
    ContainerAllocOpConversion<IREE::VM::BufferCloneOp>,
    ContainerAllocOpConversion<IREE::VM::ListAllocOp>,
    DeleteOpConversion<IREE::VM::GlobalF32Op>,
    DeleteOpConversion<IREE::VM::GlobalI32Op>,
    DeleteOpConversion<IREE::VM::GlobalI64Op>,
    DeleteOpConversion<IREE::VM::GlobalRefOp>,
    ExportOpConversion,
    FailOpConversion,
    FuncOpConversion,
    GlobalLoadStoreRefOpConversion<IREE::VM::GlobalLoadRefOp>,
    GlobalLoadStoreRefOpConversion<IREE::VM::GlobalStoreRefOp>,
    ImportResolvedOpConversion,
    ListGetOpConversion<IREE::VM::ListGetI32Op>,
    ListGetOpConversion<IREE::VM::ListGetI64Op>,
    ListGetRefOpConversion,
    ListSetOpConversion<IREE::VM::ListSetI32Op>,
    ListSetOpConversion<IREE::VM::ListSetI64Op>,
    ListSetRefOpConversion,
    ReturnOpConversion
  >(typeConverter, context);
  // clang-format on

  // generic conversions
  ADD_GENERIC_PATTERN(IREE::VM::AbsF32Op, "vm_abs_f32");
  ADD_GENERIC_PATTERN(IREE::VM::AbsI32Op, "vm_abs_i32");
  ADD_GENERIC_PATTERN(IREE::VM::AbsI64Op, "vm_abs_i64");
  ADD_GENERIC_PATTERN(IREE::VM::AddF32Op, "vm_add_f32");
  ADD_GENERIC_PATTERN(IREE::VM::AddI32Op, "vm_add_i32");
  ADD_GENERIC_PATTERN(IREE::VM::AddI64Op, "vm_add_i64");
  ADD_GENERIC_PATTERN(IREE::VM::AndI32Op, "vm_and_i32");
  ADD_GENERIC_PATTERN(IREE::VM::AndI64Op, "vm_and_i64");
  ADD_GENERIC_PATTERN(IREE::VM::Atan2F32Op, "vm_atan2_f32");
  ADD_GENERIC_PATTERN(IREE::VM::AtanF32Op, "vm_atan_f32");
  ADD_GENERIC_PATTERN(IREE::VM::BitcastF32I32Op, "vm_bitcast_f32i32");
  ADD_GENERIC_PATTERN(IREE::VM::BitcastI32F32Op, "vm_bitcast_i32f32");
  ADD_GENERIC_PATTERN(IREE::VM::CastF32SI32Op, "vm_cast_f32si32");
  ADD_GENERIC_PATTERN(IREE::VM::CastF32SI64Op, "vm_cast_f32si64");
  ADD_GENERIC_PATTERN(IREE::VM::CastF32UI32Op, "vm_cast_f32ui32");
  ADD_GENERIC_PATTERN(IREE::VM::CastF32UI64Op, "vm_cast_f32ui64");
  ADD_GENERIC_PATTERN(IREE::VM::CastSI32F32Op, "vm_cast_si32f32");
  ADD_GENERIC_PATTERN(IREE::VM::CastSI64F32Op, "vm_cast_si64f32");
  ADD_GENERIC_PATTERN(IREE::VM::CastUI32F32Op, "vm_cast_ui32f32");
  ADD_GENERIC_PATTERN(IREE::VM::CastUI64F32Op, "vm_cast_ui64f32");
  ADD_GENERIC_PATTERN(IREE::VM::CeilF32Op, "vm_ceil_f32");
  ADD_GENERIC_PATTERN(IREE::VM::CmpEQF32OOp, "vm_cmp_eq_f32o");
  ADD_GENERIC_PATTERN(IREE::VM::CmpEQF32UOp, "vm_cmp_eq_f32u");
  ADD_GENERIC_PATTERN(IREE::VM::CmpEQI32Op, "vm_cmp_eq_i32");
  ADD_GENERIC_PATTERN(IREE::VM::CmpEQI64Op, "vm_cmp_eq_i64");
  ADD_GENERIC_PATTERN(IREE::VM::CmpEQRefOp, "vm_cmp_eq_ref");
  ADD_GENERIC_PATTERN(IREE::VM::CmpLTEF32OOp, "vm_cmp_lte_f32o");
  ADD_GENERIC_PATTERN(IREE::VM::CmpLTEF32UOp, "vm_cmp_lte_f32u");
  ADD_GENERIC_PATTERN(IREE::VM::CmpLTF32OOp, "vm_cmp_lt_f32o");
  ADD_GENERIC_PATTERN(IREE::VM::CmpLTF32UOp, "vm_cmp_lt_f32u");
  ADD_GENERIC_PATTERN(IREE::VM::CmpLTI32SOp, "vm_cmp_lt_i32s");
  ADD_GENERIC_PATTERN(IREE::VM::CmpLTI32UOp, "vm_cmp_lt_i32u");
  ADD_GENERIC_PATTERN(IREE::VM::CmpLTI64SOp, "vm_cmp_lt_i64s");
  ADD_GENERIC_PATTERN(IREE::VM::CmpLTI64UOp, "vm_cmp_lt_i64u");
  ADD_GENERIC_PATTERN(IREE::VM::CmpNaNF32Op, "vm_cmp_nan_f32");
  ADD_GENERIC_PATTERN(IREE::VM::CmpNEF32OOp, "vm_cmp_ne_f32o");
  ADD_GENERIC_PATTERN(IREE::VM::CmpNEF32UOp, "vm_cmp_ne_f32u");
  ADD_GENERIC_PATTERN(IREE::VM::CmpNEI32Op, "vm_cmp_ne_i32");
  ADD_GENERIC_PATTERN(IREE::VM::CmpNEI64Op, "vm_cmp_ne_i64");
  ADD_GENERIC_PATTERN(IREE::VM::CmpNERefOp, "vm_cmp_ne_ref");
  ADD_GENERIC_PATTERN(IREE::VM::CmpNZI32Op, "vm_cmp_nz_i32");
  ADD_GENERIC_PATTERN(IREE::VM::CmpNZI64Op, "vm_cmp_nz_i64");
  ADD_GENERIC_PATTERN(IREE::VM::CosF32Op, "vm_cos_f32");
  ADD_GENERIC_PATTERN(IREE::VM::CtlzI32Op, "vm_ctlz_i32");
  ADD_GENERIC_PATTERN(IREE::VM::CtlzI64Op, "vm_ctlz_i64");
  ADD_GENERIC_PATTERN(IREE::VM::DivF32Op, "vm_div_f32");
  ADD_GENERIC_PATTERN(IREE::VM::DivI32SOp, "vm_div_i32s");
  ADD_GENERIC_PATTERN(IREE::VM::DivI32UOp, "vm_div_i32u");
  ADD_GENERIC_PATTERN(IREE::VM::DivI64SOp, "vm_div_i64s");
  ADD_GENERIC_PATTERN(IREE::VM::DivI64UOp, "vm_div_i64u");
  ADD_GENERIC_PATTERN(IREE::VM::ErfF32Op, "vm_erf_f32");
  ADD_GENERIC_PATTERN(IREE::VM::Exp2F32Op, "vm_exp2_f32");
  ADD_GENERIC_PATTERN(IREE::VM::ExpF32Op, "vm_exp_f32");
  ADD_GENERIC_PATTERN(IREE::VM::ExpM1F32Op, "vm_expm1_f32");
  ADD_GENERIC_PATTERN(IREE::VM::ExtI8I32SOp, "vm_ext_i8i32s");
  ADD_GENERIC_PATTERN(IREE::VM::ExtI8I32UOp, "vm_ext_i8i32u");
  ADD_GENERIC_PATTERN(IREE::VM::ExtI16I32SOp, "vm_ext_i16i32s");
  ADD_GENERIC_PATTERN(IREE::VM::ExtI16I32UOp, "vm_ext_i16i32u");
  ADD_GENERIC_PATTERN(IREE::VM::ExtI32I64SOp, "vm_ext_i32i64s");
  ADD_GENERIC_PATTERN(IREE::VM::ExtI32I64UOp, "vm_ext_i32i64u");
  ADD_GENERIC_PATTERN(IREE::VM::FloorF32Op, "vm_floor_f32");
  ADD_GENERIC_PATTERN(IREE::VM::FMAF32Op, "vm_fma_f32");
  ADD_GENERIC_PATTERN(IREE::VM::FMAI32Op, "vm_fma_i32");
  ADD_GENERIC_PATTERN(IREE::VM::FMAI64Op, "vm_fma_i64");
  ADD_GENERIC_PATTERN(IREE::VM::Log1pF32Op, "vm_log1p_f32");
  ADD_GENERIC_PATTERN(IREE::VM::Log2F32Op, "vm_log2_f32");
  ADD_GENERIC_PATTERN(IREE::VM::Log10F32Op, "vm_log10_f32");
  ADD_GENERIC_PATTERN(IREE::VM::LogF32Op, "vm_log_f32");
  ADD_GENERIC_PATTERN(IREE::VM::MaxF32Op, "vm_max_f32");
  ADD_GENERIC_PATTERN(IREE::VM::MaxI32SOp, "vm_max_i32s");
  ADD_GENERIC_PATTERN(IREE::VM::MaxI32UOp, "vm_max_i32u");
  ADD_GENERIC_PATTERN(IREE::VM::MaxI64SOp, "vm_max_i64s");
  ADD_GENERIC_PATTERN(IREE::VM::MaxI64UOp, "vm_max_i64u");
  ADD_GENERIC_PATTERN(IREE::VM::MinF32Op, "vm_min_f32");
  ADD_GENERIC_PATTERN(IREE::VM::MinI32SOp, "vm_min_i32s");
  ADD_GENERIC_PATTERN(IREE::VM::MinI32UOp, "vm_min_i32u");
  ADD_GENERIC_PATTERN(IREE::VM::MinI64SOp, "vm_min_i64s");
  ADD_GENERIC_PATTERN(IREE::VM::MinI64UOp, "vm_min_i64u");
  ADD_GENERIC_PATTERN(IREE::VM::MulF32Op, "vm_mul_f32");
  ADD_GENERIC_PATTERN(IREE::VM::MulI32Op, "vm_mul_i32");
  ADD_GENERIC_PATTERN(IREE::VM::MulI64Op, "vm_mul_i64");
  ADD_GENERIC_PATTERN(IREE::VM::NegF32Op, "vm_neg_f32");
  ADD_GENERIC_PATTERN(IREE::VM::NotI32Op, "vm_not_i32");
  ADD_GENERIC_PATTERN(IREE::VM::NotI64Op, "vm_not_i64");
  ADD_GENERIC_PATTERN(IREE::VM::OrI32Op, "vm_or_i32");
  ADD_GENERIC_PATTERN(IREE::VM::OrI64Op, "vm_or_i64");
  ADD_GENERIC_PATTERN(IREE::VM::PowF32Op, "vm_pow_f32");
  ADD_GENERIC_PATTERN(IREE::VM::RemF32Op, "vm_rem_f32");
  ADD_GENERIC_PATTERN(IREE::VM::RemI32SOp, "vm_rem_i32s");
  ADD_GENERIC_PATTERN(IREE::VM::RemI32UOp, "vm_rem_i32u");
  ADD_GENERIC_PATTERN(IREE::VM::RemI64SOp, "vm_rem_i64s");
  ADD_GENERIC_PATTERN(IREE::VM::RemI64UOp, "vm_rem_i64u");
  ADD_GENERIC_PATTERN(IREE::VM::RoundF32EvenOp, "vm_round_f32_even");
  ADD_GENERIC_PATTERN(IREE::VM::RoundF32Op, "vm_round_f32");
  ADD_GENERIC_PATTERN(IREE::VM::RsqrtF32Op, "vm_rsqrt_f32");
  ADD_GENERIC_PATTERN(IREE::VM::SelectF32Op, "vm_select_f32");
  ADD_GENERIC_PATTERN(IREE::VM::SelectI32Op, "vm_select_i32");
  ADD_GENERIC_PATTERN(IREE::VM::SelectI64Op, "vm_select_i64");
  ADD_GENERIC_PATTERN(IREE::VM::ShlI32Op, "vm_shl_i32");
  ADD_GENERIC_PATTERN(IREE::VM::ShlI64Op, "vm_shl_i64");
  ADD_GENERIC_PATTERN(IREE::VM::ShrI32SOp, "vm_shr_i32s");
  ADD_GENERIC_PATTERN(IREE::VM::ShrI32UOp, "vm_shr_i32u");
  ADD_GENERIC_PATTERN(IREE::VM::ShrI64SOp, "vm_shr_i64s");
  ADD_GENERIC_PATTERN(IREE::VM::ShrI64UOp, "vm_shr_i64u");
  ADD_GENERIC_PATTERN(IREE::VM::SinF32Op, "vm_sin_f32");
  ADD_GENERIC_PATTERN(IREE::VM::SqrtF32Op, "vm_sqrt_f32");
  ADD_GENERIC_PATTERN(IREE::VM::SubF32Op, "vm_sub_f32");
  ADD_GENERIC_PATTERN(IREE::VM::SubI32Op, "vm_sub_i32");
  ADD_GENERIC_PATTERN(IREE::VM::SubI64Op, "vm_sub_i64");
  ADD_GENERIC_PATTERN(IREE::VM::TanhF32Op, "vm_tanh_f32");
  ADD_GENERIC_PATTERN(IREE::VM::TruncI32I8Op, "vm_trunc_i32i8");
  ADD_GENERIC_PATTERN(IREE::VM::TruncI32I16Op, "vm_trunc_i32i16");
  ADD_GENERIC_PATTERN(IREE::VM::TruncI64I32Op, "vm_trunc_i64i32");
  ADD_GENERIC_PATTERN(IREE::VM::XorI32Op, "vm_xor_i32");
  ADD_GENERIC_PATTERN(IREE::VM::XorI64Op, "vm_xor_i64");

  // containers wrapped in ref types
  ADD_CONTAINER_PATTERN(IREE::VM::BufferCompareOp, "vm_buffer_compare",
                        DenseSet<size_t>({0, 2}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferCopyOp, "iree_vm_buffer_copy_bytes",
                        DenseSet<size_t>({0, 2}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferFillF32Op, "vm_buffer_fill_f32",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferFillF64Op, "vm_buffer_fill_f64",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferFillI8Op, "vm_buffer_fill_i8",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferFillI16Op, "vm_buffer_fill_i16",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferFillI32Op, "vm_buffer_fill_i32",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferFillI64Op, "vm_buffer_fill_i64",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferLengthOp, "iree_vm_buffer_length",
                        DenseSet<size_t>({0}), false);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferLoadF32Op, "vm_buffer_load_f32",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferLoadF64Op, "vm_buffer_load_f64",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferLoadI8SOp, "vm_buffer_load_i8s",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferLoadI8UOp, "vm_buffer_load_i8u",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferLoadI16SOp, "vm_buffer_load_i16s",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferLoadI16UOp, "vm_buffer_load_i16u",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferLoadI32Op, "vm_buffer_load_i32",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferLoadI64Op, "vm_buffer_load_i64",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferStoreF32Op, "vm_buffer_store_f32",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferStoreF64Op, "vm_buffer_store_f64",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferStoreI8Op, "vm_buffer_store_i8",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferStoreI16Op, "vm_buffer_store_i16",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferStoreI32Op, "vm_buffer_store_i32",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferStoreI64Op, "vm_buffer_store_i64",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::BufferHashOp, "iree_vm_buffer_hash",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::ListReserveOp, "iree_vm_list_reserve",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::ListResizeOp, "iree_vm_list_resize",
                        DenseSet<size_t>({0}), true);
  ADD_CONTAINER_PATTERN(IREE::VM::ListSizeOp, "iree_vm_list_size",
                        DenseSet<size_t>({0}), false);
  // global patterns
  ADD_GLOBAL_LOAD_PATTERN(IREE::VM::GlobalLoadF32Op, IREE::VM::GlobalF32Op,
                          "vm_global_load_f32");
  ADD_GLOBAL_LOAD_PATTERN(IREE::VM::GlobalLoadI32Op, IREE::VM::GlobalI32Op,
                          "vm_global_load_i32");
  ADD_GLOBAL_LOAD_PATTERN(IREE::VM::GlobalLoadI64Op, IREE::VM::GlobalI64Op,
                          "vm_global_load_i64");
  ADD_GLOBAL_STORE_PATTERN(IREE::VM::GlobalStoreF32Op, IREE::VM::GlobalF32Op,
                           "vm_global_store_f32");
  ADD_GLOBAL_STORE_PATTERN(IREE::VM::GlobalStoreI32Op, IREE::VM::GlobalI32Op,
                           "vm_global_store_i32");
  ADD_GLOBAL_STORE_PATTERN(IREE::VM::GlobalStoreI64Op, IREE::VM::GlobalI64Op,
                           "vm_global_store_i64");

#undef ADD_GENERIC_PATTERN
#undef ADD_CONTAINER_PATTERN
#undef ADD_GLOBAL_LOAD_PATTERN
#undef ADD_GLOBAL_STORE_PATTERN
}

namespace IREE::VM {

namespace {

// A pass converting IREE VM operations into the EmitC dialect.
// vm.func ops get converted to emitc.func with the calling convention used by
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
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVMToEmitCPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::BuiltinDialect, mlir::cf::ControlFlowDialect,
                    mlir::emitc::EmitCDialect, IREE::Util::UtilDialect>();
  }

  StringRef getArgument() const override { return "iree-convert-vm-to-emitc"; }

  StringRef getDescription() const override {
    return "Convert VM Ops to the EmitC dialect";
  }

  void runOnOperation() override {
    IREE::VM::ModuleOp module = getOperation();

    ConversionTarget target(getContext());
    EmitCTypeConverter typeConverter(module);

    // Convert vm.func ops to emitc.func with the calling convention used by
    // EmitC. We convert these upfront to make sure vm.call ops always
    // reference emitc.func ops with the correct calling convention during the
    // conversion.
    SmallVector<IREE::VM::FuncOp> funcsToRemove;
    for (auto funcOp : module.getOps<IREE::VM::FuncOp>()) {
      if (failed(convertFuncOp(funcOp, typeConverter))) {
        return signalPassFailure();
      }
      funcsToRemove.push_back(funcOp);
    }

    for (auto &funcOp : funcsToRemove) {
      funcOp.erase();
    }

    SmallVector<std::string> importShims;

    // The conversion of `call/call.variadic` ops on imported functions expects
    // import ops to be rewritten to compiler generated shim functions. To
    // ensure this we only rewrite `import` ops first.
    ImportOpConverter importOpConverter(typeConverter, importShims);
    for (auto importOp : module.getOps<IREE::VM::ImportOp>()) {
      if (failed(importOpConverter(importOp))) {
        return signalPassFailure();
      }
    }

    RewritePatternSet patterns(&getContext());
    populateVMToEmitCPatterns(target, typeConverter, patterns);

    target.addLegalDialect<emitc::EmitCDialect, mlir::BuiltinDialect,
                           mlir::cf::ControlFlowDialect>();

    target.addDynamicallyLegalOp<mlir::emitc::FuncOp>(
        [&](mlir::emitc::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter.isLegal(&op.getFunctionBody());
        });

    // Structural ops
    target.addLegalOp<IREE::VM::ModuleOp>();
    target.addLegalOp<IREE::VM::ModuleTerminatorOp>();
    target.addLegalOp<IREE::VM::ImportOp>();

    // This op is needed in the printer to emit an array holding the data.
    target.addLegalOp<IREE::VM::RodataOp>();

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    if (failed(createModuleStructure(module, typeConverter))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createConvertVMToEmitCPass() {
  return std::make_unique<ConvertVMToEmitCPass>();
}

} // namespace IREE::VM

static PassRegistration<IREE::VM::ConvertVMToEmitCPass> pass;

} // namespace mlir::iree_compiler
