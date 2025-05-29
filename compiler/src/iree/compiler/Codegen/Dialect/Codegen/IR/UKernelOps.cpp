// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::Codegen {

//===---------------------------------------------------------------------===//
// Helpers
//===---------------------------------------------------------------------===//

/// Helper method to generate a function declaration at a module scope,
/// and a call to that function
static FailureOr<func::CallOp>
createFunctionCall(RewriterBase &rewriter, Operation *op, StringRef fnName,
                   TypeRange callArgumentTypes, TypeRange callReturnTypes,
                   ValueRange callOperands,
                   ArrayRef<NamedAttribute> fnDefAttrs) {
  FunctionType functionType =
      rewriter.getFunctionType(callArgumentTypes, callReturnTypes);

  // Create a declaration for the function type.
  Location loc = op->getLoc();
  auto moduleOp = SymbolTable::getNearestSymbolTable(op);
  // Check for duplicates.
  auto fnDecl = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(moduleOp, fnName));
  if (!fnDecl) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(&moduleOp->getRegion(0).front());
    fnDecl = rewriter.create<func::FuncOp>(loc, fnName, functionType);
    SymbolTable::setSymbolVisibility(fnDecl, SymbolTable::Visibility::Private);
    for (auto attr : fnDefAttrs) {
      fnDecl->setAttr(attr.getName(), attr.getValue());
    }
    fnDecl->setAttr("llvm.bareptr", rewriter.getBoolAttr(true));
  } else if (fnDecl.getFunctionType() != functionType) {
    return rewriter.notifyMatchFailure(
        op, llvm::formatv("mismatch in function type computed during lowering "
                          "({0}) and already declared function ({1})",
                          functionType, fnDecl.getFunctionType()));
  }

  // Insert the function call.
  auto callOp = rewriter.create<func::CallOp>(loc, fnDecl, callOperands);
  if (op->hasAttr("hal.executable.objects")) {
    callOp->setAttr("hal.executable.objects",
                    op->getAttr("hal.executable.objects"));
  }
  return callOp;
}

//===---------------------------------------------------------------------===//
// UKernelGenericOp
//===---------------------------------------------------------------------===//

/// Map type of operand of a `iree_codegen.ukernel.generic` operation to
/// the type(s) of the function call arguments(s) it lowers to.
static LogicalResult getCallOpType(MLIRContext *context,
                                   Type microKernelOpOperandType,
                                   IntegerAttr stridedOuterDimsAttr,
                                   SmallVector<Type> &callOperandTypes) {
  return TypeSwitch<Type, LogicalResult>(microKernelOpOperandType)
      .Case<FloatType, IndexType, IntegerType>([&](auto scalarType) {
        callOperandTypes.push_back(scalarType);
        return success();
      })
      .Case<MemRefType>([&](MemRefType memrefType) {
        // Base ptr.
        callOperandTypes.push_back(MemRefType::get(
            ArrayRef<int64_t>{}, memrefType.getElementType(),
            MemRefLayoutAttrInterface{}, memrefType.getMemorySpace()));
        // Offset
        auto indexType = IndexType::get(context);
        callOperandTypes.push_back(indexType);
        // Strides.
        int stridedOuterDims = stridedOuterDimsAttr
                                   ? stridedOuterDimsAttr.getInt()
                                   : memrefType.getRank();
        callOperandTypes.resize(callOperandTypes.size() + stridedOuterDims,
                                indexType);
        return success();
      })
      .Case<NullPointerType>([&](NullPointerType nullPointerType) {
        callOperandTypes.push_back(nullPointerType);
        callOperandTypes.push_back(IndexType::get(context));
        return success();
      })
      .Default([&](Type t) { return failure(); });
}

/// Map `operand` of a `ukernel.generic` operation to the operand(s) of
/// the function call it lowers to.
static LogicalResult lowerToCallOperands(Location loc, RewriterBase &rewriter,
                                         Value operand,
                                         IntegerAttr stridedOuterDimsAttr,
                                         SmallVector<Value> &callOperands) {
  return TypeSwitch<Type, LogicalResult>(operand.getType())
      .Case<FloatType, IndexType, IntegerType>([&](auto scalarType) {
        callOperands.push_back(operand);
        return success();
      })
      .Case<MemRefType>([&](MemRefType memrefType) {
        auto extractStridedMetadataOp =
            rewriter.create<memref::ExtractStridedMetadataOp>(loc, operand);
        // Base ptr.
        callOperands.push_back(extractStridedMetadataOp.getBaseBuffer());
        // Offset.
        callOperands.push_back(extractStridedMetadataOp.getOffset());
        // Strides.
        int stridedOuterDims = stridedOuterDimsAttr
                                   ? stridedOuterDimsAttr.getInt()
                                   : memrefType.getRank();
        auto strides = extractStridedMetadataOp.getStrides();
        for (int i = 0; i < stridedOuterDims; ++i) {
          callOperands.push_back(strides[i]);
        }
        return success();
      })
      .Case<NullPointerType>([&](NullPointerType /*unused*/) {
        callOperands.push_back(operand);
        callOperands.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
        return success();
      })
      .Default([](Type) { return failure(); });
}

static FailureOr<func::CallOp> lowerUKernelGenericToFunctionCall(
    RewriterBase &rewriter, IREE::Codegen::UKernelGenericOp op,
    StringRef fnName, IntegerAttr stridedOuterDimsAttr) {
  // Create the function type based on the operands and results.
  SmallVector<Type> callArgumentTypes;
  for (auto microKernelOpOperandType : op->getOperandTypes()) {
    if (failed(getCallOpType(rewriter.getContext(), microKernelOpOperandType,
                             stridedOuterDimsAttr, callArgumentTypes))) {
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("failed to lower operand type {}",
                            microKernelOpOperandType));
    }
  }
  SmallVector<Type> callResultTypes;
  for (auto resultType : op->getResultTypes()) {
    if (llvm::isa<ShapedType>(resultType)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower a `ShapedType` return value to function call");
    }
    if (failed(getCallOpType(rewriter.getContext(), resultType,
                             stridedOuterDimsAttr, callResultTypes))) {
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("failed to lower result type {}", resultType));
    }
  }

  // Get the operands for the function call.
  SmallVector<Value> callOperands;
  for (auto operand : op->getOperands()) {
    if (failed(lowerToCallOperands(op->getLoc(), rewriter, operand,
                                   stridedOuterDimsAttr, callOperands))) {
      return rewriter.notifyMatchFailure(
          op, "failed to lower operands to function call operands");
    }
  }
  ArrayRef<NamedAttribute> fnDefAttrs = {};
  if (auto specifiedfnDefAttrs = op.getFnDefAttrs()) {
    fnDefAttrs = specifiedfnDefAttrs->getValue();
  }
  return createFunctionCall(rewriter, op, fnName, callArgumentTypes,
                            callResultTypes, callOperands, fnDefAttrs);
}

MutableOperandRange UKernelGenericOp::getDpsInitsMutable() {
  return getOutputsMutable();
}

FailureOr<mlir::CallOpInterface>
UKernelGenericOp::lowerToFunctionCall(RewriterBase &rewriter) {
  return lowerUKernelGenericToFunctionCall(rewriter, *this, getUKernelFnName(),
                                           getStridedOuterDimsAttr());
}

void UKernelGenericOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  SmallVector<OpOperand *> readOnlyOperands = getDpsInputOperands();
  llvm::append_range(readOnlyOperands,
                     llvm::make_pointer_range(getOtherOperandsMutable()));
  for (OpOperand *operand : readOnlyOperands) {
    if (!llvm::isa<MemRefType>(operand->get().getType())) {
      continue;
    }
    effects.emplace_back(MemoryEffects::Read::get(), operand,
                         SideEffects::DefaultResource::get());
  }
  for (OpOperand &operand : getDpsInitsMutable()) {
    if (!llvm::isa<MemRefType>(operand.get().getType())) {
      continue;
    }
    effects.emplace_back(MemoryEffects::Read::get(), &operand,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), &operand,
                         SideEffects::DefaultResource::get());
  }
}

} // namespace mlir::iree_compiler::IREE::Codegen

namespace mlir::iree_compiler {

//===---------------------------------------------------------------------===//
// Register bufferization interface.
//===---------------------------------------------------------------------===//

namespace {

template <typename OpTy>
struct UKernelOpsBufferizationInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          UKernelOpsBufferizationInterface<OpTy>, OpTy> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    // TODO: Handle operations with regions if needed.
    if (op->getNumRegions() != 0) {
      op->emitOpError(
          "unhandled bufferization of micro kernel op with regions");
    }
    SmallVector<Value> bufferOpOperands;

    // Replace all `tensor` operands with corresponding `memref` operands.
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      // For `tensor` type operands, replace with `memref` type operand.
      if (llvm::isa<RankedTensorType>(operand.getType())) {
        FailureOr<Value> memrefOperand =
            getBuffer(rewriter, operand, options, state);
        if (failed(memrefOperand)) {
          return op->emitOpError(
              llvm::formatv("failed to bufferize operand {} ", index));
        }
        bufferOpOperands.push_back(memrefOperand.value());
        continue;
      }

      // For all other operand types, just use the same value.
      bufferOpOperands.push_back(operand);
    }

    SmallVector<Type> nonTensorResultTypes;
    SmallVector<Value> nonTensorResultValues;
    for (OpResult result : op->getResults()) {
      Type resultType = result.getType();
      if (llvm::isa<RankedTensorType>(resultType))
        continue;
      nonTensorResultTypes.push_back(resultType);
      nonTensorResultValues.push_back(result);
    }

    auto bufferOp = rewriter.create<OpTy>(op->getLoc(), nonTensorResultTypes,
                                          bufferOpOperands, op->getAttrs());
    SmallVector<Value> bufferizedResults =
        cast<DestinationStyleOpInterface>(bufferOp.getOperation())
            .getDpsInits();
    bufferizedResults.append(nonTensorResultValues);
    bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                 bufferizedResults);
    return success();
  }
};

template <typename... Ops>
struct RegisterUKernelOpsBufferizationInterface {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<UKernelOpsBufferizationInterface<Ops>>(
         *context),
     ...);
  }
};

} // namespace

void registerUKernelBufferizationInterface(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *context, IREE::Codegen::IREECodegenDialect *dialect) {
        RegisterUKernelOpsBufferizationInterface<
#define GET_OP_LIST
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.cpp.inc"
            >::registerOpInterface(context);
      });
}

} // namespace mlir::iree_compiler
