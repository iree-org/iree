// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/UKernelOps.h"

#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Utils/EncodingInfo.h"
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
#include "iree/compiler/Codegen/Dialect/UKernelOps.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Codegen {

/// Helper method to generate a function declaration at a module scope,
/// and a call to that function
static FailureOr<func::CallOp> createFunctionCall(RewriterBase &rewriter,
                                                  Operation *op,
                                                  StringRef fnName,
                                                  TypeRange callReturnTypes,
                                                  ValueRange callArguments) {
  SmallVector<Type> callArgumentTypes = llvm::to_vector(llvm::map_range(
      callArguments, [](Value v) -> Type { return v.getType(); }));
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
    // TODO(#12327): Based on description in the issue, add an attribute
    // `vm.import.module` and set it to `vmvx`. This only works on `vmvx`
    // backend (obviously), but is enough to unblock while the proper fix lands.
    fnDecl->setAttr("vm.import.module", rewriter.getStringAttr("vmvx"));
  } else if (fnDecl.getFunctionType() != functionType) {
    return rewriter.notifyMatchFailure(
        op, llvm::formatv("mismatch in function type computed during lowering "
                          "({0}) and already declared function ({1})",
                          functionType, fnDecl.getFunctionType()));
  }

  // Insert the function call.
  return rewriter.create<func::CallOp>(loc, fnDecl, callArguments);
}

//===---------------------------------------------------------------------===//
// UKernelGenericOp
//===---------------------------------------------------------------------===//

std::pair<int64_t, int64_t> UKernelGenericOp::getDpsInitsPositionRange() {
  auto [pos, size] = getODSOperandIndexAndLength(1);
  return {static_cast<int64_t>(pos), static_cast<int64_t>(pos + size)};
}

LogicalResult appendUkernelGenericBufferAndOffsetArgs(
    RewriterBase &rewriter, Location loc, ValueRange values,
    SmallVector<Value> &outArgs) {
  for (Value value : values) {
    Type type = value.getType();
    if (type.isa<FloatType>() || type.isa<IndexType>() ||
        type.isa<IntegerType>()) {
      // do nothing.
    } else if (MemRefType memrefType = type.dyn_cast<MemRefType>()) {
      auto extractStridedMetadataOp =
          rewriter.create<memref::ExtractStridedMetadataOp>(loc, value);
      // Base ptr.
      outArgs.push_back(extractStridedMetadataOp.getBaseBuffer());
      // Offset.
      outArgs.push_back(extractStridedMetadataOp.getOffset());
    } else {
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unhandled operand type {0}", type));
    }
  }
  return success();
}

LogicalResult appendUkernelGenericNonBufferAndOffsetArgs(
    RewriterBase &rewriter, Location loc, ValueRange values,
    SmallVector<Value> &outArgs) {
  for (Value value : values) {
    Type type = value.getType();
    if (type.isa<FloatType>() || type.isa<IndexType>() ||
        type.isa<IntegerType>()) {
      outArgs.push_back(value);
    } else if (type.isa<MemRefType>()) {
      auto extractStridedMetadataOp =
          rewriter.create<memref::ExtractStridedMetadataOp>(loc, value);
      // Strides
      const auto &strides = extractStridedMetadataOp.getStrides();
      if (strides.size() >= 1) {
        for (unsigned i = 0; i < strides.size() - 1; ++i) {
          outArgs.push_back(strides[i]);
        }
      }
    } else {
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unhandled operand type {0}", type));
    }
  }
  return success();
}

LogicalResult appendUkernelGenericReturnTypes(RewriterBase &rewriter,
                                              Location loc, TypeRange types,
                                              SmallVector<Type> &outTypes) {
  for (Type type : types) {
    if (type.isa<FloatType>() || type.isa<IndexType>() ||
        type.isa<IntegerType>()) {
      outTypes.push_back(type);
    } else {
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unhandled return type {0}", type));
    }
  }
  return success();
}

FailureOr<func::CallOp> UKernelGenericOp::lowerToFunctionCall(
    RewriterBase &rewriter) {
  Location loc = getLoc();
  SmallVector<Value> callArguments;
  if (failed(appendUkernelGenericBufferAndOffsetArgs(
          rewriter, loc, getOperands(), callArguments))) {
    return rewriter.notifyMatchFailure(getOperation(),
                                       "failed to lower operands");
  }
  if (failed(appendUkernelGenericNonBufferAndOffsetArgs(
          rewriter, loc, getOperands(), callArguments))) {
    return rewriter.notifyMatchFailure(getOperation(),
                                       "failed to lower operands");
  }
  SmallVector<Type> callResultTypes;
  if (failed(appendUkernelGenericReturnTypes(rewriter, loc, getResultTypes(),
                                             callResultTypes))) {
    return rewriter.notifyMatchFailure(getOperation(),
                                       "failed to lower result types");
  }
  return createFunctionCall(rewriter, getOperation(), getUKernelFnName(),
                            callResultTypes, callArguments);
}

//===---------------------------------------------------------------------===//
// UKernelMmt4DOp
//===---------------------------------------------------------------------===//

std::pair<int64_t, int64_t> UKernelMmt4DOp::getDpsInitsPositionRange() {
  auto [pos, size] = getODSOperandIndexAndLength(2);
  return {static_cast<int64_t>(pos), static_cast<int64_t>(pos + size)};
}

LogicalResult appendUkernelMmt4DNonBufferAndOffsetArgs(
    RewriterBase &rewriter, Location loc, ValueRange values,
    SmallVector<Value> &outArgs) {
  for (Value value : values) {
    Type type = value.getType();
    if (type.isa<FloatType>() || type.isa<IndexType>() ||
        type.isa<IntegerType>()) {
      outArgs.push_back(value);
    } else if (type.isa<MemRefType>()) {
      auto extractStridedMetadataOp =
          rewriter.create<memref::ExtractStridedMetadataOp>(loc, value);
      // Strides[0]
      const auto &strides = extractStridedMetadataOp.getStrides();
      if (strides.size() >= 1) {
        outArgs.push_back(strides[0]);
      }
    } else {
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unhandled operand type {0}", type));
    }
  }
  return success();
}

FailureOr<func::CallOp> UKernelMmt4DOp::lowerToFunctionCall(
    RewriterBase &rewriter) {
  // TODO: handle op with return values if they are scalar.
  if (getNumResults() != 0) {
    return rewriter.notifyMatchFailure(
        getOperation(), "cannot lower to function call operation with results");
  }

  std::optional<MatmulType> matmulType = getMatmulType(
      getLhsElementType(), getRhsElementType(), getOutputElementType());
  if (!matmulType) {
    return emitOpError(
        "unhandled element types of operands for lowering to micro kernel "
        "function call");
  }

  // Function name.
  std::string fnName = "vmvx.mmt4d.";
  switch (matmulType.value()) {
    case MatmulType::I8I8I32:
      fnName.append("i8i8i32");
      break;
    case MatmulType::F32F32F32:
      fnName.append("f32f32f32");
      break;
  }

  SmallVector<Value> callArguments;
  Location loc = getLoc();
  if (failed(appendUkernelGenericBufferAndOffsetArgs(
          rewriter, loc, getOperands(), callArguments))) {
    return rewriter.notifyMatchFailure(getOperation(),
                                       "failed to lower operands");
  }
  if (failed(appendUkernelMmt4DNonBufferAndOffsetArgs(
          rewriter, loc, {getLhs(), getRhs(), getOutput()}, callArguments))) {
    return rewriter.notifyMatchFailure(getOperation(),
                                       "failed to lower operands");
  }
  // M
  callArguments.push_back(rewriter.create<memref::DimOp>(loc, getLhs(), 0));
  // N
  callArguments.push_back(rewriter.create<memref::DimOp>(loc, getRhs(), 0));
  // K
  callArguments.push_back(rewriter.create<memref::DimOp>(loc, getLhs(), 1));
  auto getDimAsI32 = [](RewriterBase &rewriter, Location loc, Value value,
                        int dim) -> Value {
    auto dimValue = rewriter.create<memref::DimOp>(loc, value, dim);
    return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                               dimValue);
  };
  // M0
  callArguments.push_back(getDimAsI32(rewriter, loc, getLhs(), 2));
  // N0
  callArguments.push_back(getDimAsI32(rewriter, loc, getRhs(), 2));
  // K0
  callArguments.push_back(getDimAsI32(rewriter, loc, getLhs(), 3));
  // Flags;
  int flags = 0;
  if (getAccumulate()) {
    flags |= IREE_UK_FLAG_ACCUMULATE;
  }
  callArguments.push_back(rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(flags)));
  return createFunctionCall(rewriter, getOperation(), fnName,
                            /*callReturnTypes=*/TypeRange{}, callArguments);
}
}  // namespace Codegen
}  // namespace IREE

//===---------------------------------------------------------------------===//
// Register bufferization interface.
//===---------------------------------------------------------------------===//

namespace {
template <typename OpTy>
struct UKernelOpsBufferizationInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          UKernelOpsBufferizationInterface<OpTy>, OpTy> {
  LogicalResult bufferize(
      Operation *op, RewriterBase &rewriter,
      const bufferization::BufferizationOptions &options) const {
    // TODO: Handle operations with regions if needed.
    if (op->getNumRegions() != 0) {
      op->emitOpError(
          "unhandled bufferization of micro kernel op with regions");
    }
    SmallVector<Value> bufferOpOperands;

    // Replace all `tensor` operands with corresponding `memref` operands.
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      // For `tensor` type operands, replace with `memref` type operand.
      if (operand.getType().template isa<RankedTensorType>()) {
        FailureOr<Value> memrefOperand = getBuffer(rewriter, operand, options);
        if (failed(memrefOperand)) {
          return op->emitOpError(
              llvm::formatv("failed to bufferize operand {0} ", index));
        }
        bufferOpOperands.push_back(memrefOperand.value());
        continue;
      }

      // For all other operand types, just use the same value.
      bufferOpOperands.push_back(operand);
    }

    // Ignore all result types that are tensor types.
    SmallVector<Type> resultTypes;
    for (auto resultType : op->getResultTypes()) {
      if (resultType.isa<RankedTensorType>()) continue;
      resultTypes.push_back(resultType);
    }

    auto bufferOp = rewriter.create<OpTy>(op->getLoc(), resultTypes,
                                          bufferOpOperands, op->getAttrs());
    SmallVector<Value> dpsInits =
        cast<DestinationStyleOpInterface>(bufferOp.getOperation())
            .getDpsInitOperands();
    bufferization::replaceOpWithBufferizedValues(rewriter, op, dpsInits);
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
}  // namespace

void registerUKernelBufferizationInterface(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *context, IREE::Codegen::IREECodegenDialect *dialect) {
        RegisterUKernelOpsBufferizationInterface<
#define GET_OP_LIST
#include "iree/compiler/Codegen/Dialect/UKernelOps.cpp.inc"
            >::registerOpInterface(context);
      });
}

}  // namespace iree_compiler
}  // namespace mlir
