// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/Torch/InputConversion/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

namespace Torch = mlir::torch::Torch;
namespace TorchConversion = mlir::torch::TorchConversion;

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_DEF_SYNCFUNCCONVERSIONPASS
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Synchronous-only variant of FuncConversion (see FuncConversion.cpp for the
// coarse-fences ABI it is an alternative to). Functions are rewritten in
// place to a plain builtin tensor ABI intended for synchronous invocation
// (e.g. via the inline HAL): immutable tensors become builtin tensors and
// torch primitives their builtin scalar equivalents, with torch_c conversion
// ops materialized at the edges. No fences, no HAL imports/exports, and no
// $async variant are created. Mutable tensors require in-place aliasing
// through the HAL and are rejected.
//===----------------------------------------------------------------------===//

TensorType convertToBuiltinTensorType(OpBuilder &builder,
                                      Torch::ValueTensorType vtensorType) {
  TensorType builtinType = vtensorType.toBuiltinTensor();
  if (auto intTy = dyn_cast<IntegerType>(builtinType.getElementType())) {
    builtinType = builtinType.clone(
        builder.getIntegerType(intTy.getIntOrFloatBitWidth()));
  }
  return builtinType;
}

Value convertToBuiltinTensor(OpBuilder &builder, Value possibleTorchTensor) {
  Type ty = possibleTorchTensor.getType();
  if (isa<TensorType>(ty)) {
    return possibleTorchTensor;
  }

  if (auto defining = dyn_cast_if_present<TorchConversion::FromBuiltinTensorOp>(
          possibleTorchTensor.getDefiningOp())) {
    return defining.getOperand();
  }

  TensorType builtinType =
      convertToBuiltinTensorType(builder, cast<Torch::ValueTensorType>(ty));
  return TorchConversion::ToBuiltinTensorOp::create(
      builder, possibleTorchTensor.getLoc(), builtinType, possibleTorchTensor);
}

enum class TypeDisposition {
  IMMUTABLE_TENSOR,
  TORCH_PRIMITIVE,
  PASSTHROUGH,
};

struct ConvertedSyncFunctionInfo {
  IREE::Util::FuncOp funcOp;
  SmallVector<IREE::Util::ReturnOp> returnOps;
  SmallVector<Type> torchInputTypes;
  SmallVector<Type> torchResultTypes;
  SmallVector<TypeDisposition> inputDispositions;
  SmallVector<TypeDisposition> resultDispositions;

  LogicalResult postProcess();
  LogicalResult convertImmutableTensorArg(BlockArgument argValue,
                                          Type torchType, OpBuilder &builder);
};

LogicalResult ConvertedSyncFunctionInfo::postProcess() {
  if (funcOp.isExternal()) {
    return success();
  }

  if (returnOps.size() != 1) {
    // Multi-exit funcs could be supported by materializing the result
    // conversions at each return; restricted to single-exit for parity with
    // the coarse-fences conversion.
    return emitError(funcOp.getLoc())
           << "currently only single exit torch funcs are supported";
  }

  Block *entryBlock = &funcOp.getBlocks().front();

  // Materialize argument conversions.
  OpBuilder preambleBuilder = OpBuilder::atBlockBegin(entryBlock);
  auto entryArgs = entryBlock->getArguments();
  for (auto [disp, argValue, torchType] :
       llvm::zip_equal(inputDispositions, entryArgs, torchInputTypes)) {
    switch (disp) {
    case TypeDisposition::IMMUTABLE_TENSOR: {
      if (failed(convertImmutableTensorArg(argValue, torchType,
                                           preambleBuilder))) {
        return failure();
      }
      break;
    }
    case TypeDisposition::TORCH_PRIMITIVE: {
      Location loc = argValue.getLoc();
      Operation *convertUser = nullptr;
      Value convertResult;
      if (isa<Torch::BoolType>(torchType)) {
        convertUser =
            TorchConversion::FromI1Op::create(preambleBuilder, loc, argValue);
        convertResult = convertUser->getResult(0);
      } else if (isa<Torch::FloatType>(torchType)) {
        convertUser =
            TorchConversion::FromF64Op::create(preambleBuilder, loc, argValue);
        convertResult = convertUser->getResult(0);
      } else if (isa<Torch::IntType>(torchType)) {
        convertUser =
            TorchConversion::FromI64Op::create(preambleBuilder, loc, argValue);
        convertResult = convertUser->getResult(0);
      } else {
        emitError(loc) << "unhandled torch primitive materialization: "
                       << torchType;
        return failure();
      }
      argValue.replaceAllUsesExcept(convertResult, convertUser);
      break;
    }
    case TypeDisposition::PASSTHROUGH:
      // Do nothing.
      break;
    }
  }

  // Materialize result conversions.
  IREE::Util::ReturnOp returnOp = returnOps.front();
  SmallVector<Value> newReturnOperands;
  OpBuilder postambleBuilder(returnOp);
  for (auto [disp, returnValue, torchType] : llvm::zip_equal(
           resultDispositions, returnOp.getOperands(), torchResultTypes)) {
    newReturnOperands.emplace_back(returnValue);
    switch (disp) {
    case TypeDisposition::IMMUTABLE_TENSOR: {
      newReturnOperands.back() =
          convertToBuiltinTensor(postambleBuilder, returnValue);
      break;
    }
    case TypeDisposition::TORCH_PRIMITIVE: {
      Location loc = returnValue.getLoc();
      if (isa<Torch::BoolType>(torchType)) {
        newReturnOperands.back() =
            TorchConversion::ToI1Op::create(postambleBuilder, loc, returnValue);
      } else if (isa<Torch::FloatType>(torchType)) {
        newReturnOperands.back() = TorchConversion::ToF64Op::create(
            postambleBuilder, loc, returnValue);
      } else if (isa<Torch::IntType>(torchType)) {
        newReturnOperands.back() = TorchConversion::ToI64Op::create(
            postambleBuilder, loc, returnValue);
      } else if (isa<Torch::GeneratorType>(torchType)) {
        newReturnOperands.back() = TorchConversion::GeneratorToI64Op::create(
            postambleBuilder, loc, returnValue);
      } else {
        emitError(loc) << "unhandled torch primitive materialization: "
                       << torchType;
        return failure();
      }
      break;
    }
    case TypeDisposition::PASSTHROUGH:
      // Do nothing.
      break;
    }
  }
  returnOp->setOperands(newReturnOperands);

  return success();
}

LogicalResult ConvertedSyncFunctionInfo::convertImmutableTensorArg(
    BlockArgument argValue, Type torchType, OpBuilder &builder) {
  // Already a builtin tensor: nothing to materialize.
  if (isa<TensorType>(torchType)) {
    return success();
  }

  if (!isa<Torch::ValueTensorType>(torchType)) {
    return emitError(argValue.getLoc())
           << "unsupported immutable tensor argument: " << torchType;
  }

  // If the arg is just directly returned, then don't do anything special with
  // it: the postamble passes it through unconverted.
  bool hasNonTrivialUse = false;
  for (auto *userOp : argValue.getUsers()) {
    if (isa<IREE::Util::ReturnOp>(userOp)) {
      continue;
    }
    hasNonTrivialUse = true;
  }
  if (!hasNonTrivialUse) {
    return success();
  }
  Value converted = TorchConversion::FromBuiltinTensorOp::create(
      builder, argValue.getLoc(), torchType, argValue);
  argValue.replaceAllUsesExcept(converted, converted.getDefiningOp());
  return success();
}

void retainFunctionAttributes(Operation *srcOp, IREE::Util::FuncOp destOp) {
  // Allowlist of function attributes to retain when importing funcs.
  constexpr const char *kRetainedAttributes[] = {
      "iree.reflection",
  };
  for (const char *attrName : kRetainedAttributes) {
    if (Attribute attr = srcOp->getAttr(attrName)) {
      destOp->setAttr(attrName, attr);
    }
  }
}

class SyncFuncConversionPass final
    : public impl::SyncFuncConversionPassBase<SyncFuncConversionPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
    registry.insert<TorchConversion::TorchConversionDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Convert all functions in the module to IREE funcs. In this stage,
    // we convert contained return ops and argument/result types, but we have
    // not yet converted anything "on the inside". Therefore, it is pretty
    // likely the functions are still illegal.
    SmallVector<Operation *> eraseFuncOps;
    std::vector<ConvertedSyncFunctionInfo> convertedFuncInfos;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (!shouldConvertFunc(funcOp)) {
        continue;
      }
      ConvertedSyncFunctionInfo &convertedFuncInfo =
          convertedFuncInfos.emplace_back();
      if (failed(convertFuncOp(funcOp, convertedFuncInfo))) {
        signalPassFailure();
        return;
      }
      eraseFuncOps.push_back(funcOp);
    }
    for (auto op : eraseFuncOps) {
      op->erase();
    }

    for (auto &info : convertedFuncInfos) {
      if (failed(info.postProcess())) {
        signalPassFailure();
        return;
      }
    }
  }

  bool shouldConvertFunc(func::FuncOp torchFunc) {
    // For now, we don't touch externals and assume they are in the proper
    // calling convention. In the future, we may support "torch externals"
    // which we convert to mate up with a torch module. We can remove/adapt
    // this when that is elaborated.
    if (torchFunc.isExternal()) {
      return false;
    }

    // Something has already converted this and told us not to touch it.
    if (torchFunc->hasAttr("iree.abi.stub")) {
      return false;
    }

    return true;
  }

  LogicalResult convertFuncOp(func::FuncOp torchFunc,
                              ConvertedSyncFunctionInfo &convertedFuncInfo) {
    IRRewriter rewriter(torchFunc.getContext());
    rewriter.setInsertionPoint(torchFunc);
    Location loc = torchFunc.getLoc();

    // Convert function signature.
    FunctionType torchFuncType = torchFunc.getFunctionType();
    convertedFuncInfo.torchInputTypes.append(torchFuncType.getInputs().begin(),
                                             torchFuncType.getInputs().end());
    convertedFuncInfo.torchResultTypes.append(
        torchFuncType.getResults().begin(), torchFuncType.getResults().end());
    SmallVector<Type> ireeInputTypes(convertedFuncInfo.torchInputTypes);
    SmallVector<Type> ireeResultTypes(convertedFuncInfo.torchResultTypes);
    convertedFuncInfo.inputDispositions.resize(ireeInputTypes.size());
    convertedFuncInfo.resultDispositions.resize(ireeResultTypes.size());

    for (size_t i = 0; i < convertedFuncInfo.torchInputTypes.size(); ++i) {
      if (failed(convertType(
              rewriter, loc, convertedFuncInfo.torchInputTypes[i],
              ireeInputTypes[i], convertedFuncInfo.inputDispositions[i]))) {
        return failure();
      }
    }
    for (size_t i = 0; i < convertedFuncInfo.torchResultTypes.size(); ++i) {
      if (failed(convertType(
              rewriter, loc, convertedFuncInfo.torchResultTypes[i],
              ireeResultTypes[i], convertedFuncInfo.resultDispositions[i]))) {
        return failure();
      }
    }

    // Build tied operands index mapping results back to operands.
    SmallVector<int64_t> tiedOperands;
    bool anyTiedOperands = false;
    for (unsigned i = 0; i < torchFuncType.getNumResults(); ++i) {
      auto tiedAttr =
          torchFunc.getResultAttrOfType<IntegerAttr>(i, "iree.abi.tied");
      if (tiedAttr) {
        tiedOperands.push_back(tiedAttr.getInt());
        anyTiedOperands = true;
      } else {
        tiedOperands.push_back(-1);
      }
    }
    auto tiedOperandsAttr = anyTiedOperands
                                ? rewriter.getIndexArrayAttr(tiedOperands)
                                : ArrayAttr{};

    // Create new func with the original name.
    FunctionType syncFuncType =
        FunctionType::get(loc.getContext(), ireeInputTypes, ireeResultTypes);
    auto syncFuncOp = IREE::Util::FuncOp::create(
        rewriter, torchFunc.getLoc(), torchFunc.getName(), syncFuncType,
        tiedOperandsAttr);
    convertedFuncInfo.funcOp = syncFuncOp;
    syncFuncOp.setSymVisibilityAttr(torchFunc.getSymVisibilityAttr());
    retainFunctionAttributes(torchFunc, syncFuncOp);
    if (auto affinityAttr = torchFunc->getAttr("iree.abi.affinity")) {
      syncFuncOp->setAttr("iree.abi.affinity", affinityAttr);
    }
    rewriter.inlineRegionBefore(torchFunc.getBody(),
                                syncFuncOp.getFunctionBody(), syncFuncOp.end());

    // Convert block arguments.
    Block *entryBlock = &syncFuncOp.getBlocks().front();
    for (size_t i = 0; i < ireeInputTypes.size(); ++i) {
      entryBlock->getArgument(i).setType(ireeInputTypes[i]);
    }

    // Replace return ops.
    syncFuncOp->walk([&](func::ReturnOp returnOp) {
      rewriter.setInsertionPoint(returnOp);
      auto ireeReturnOp = rewriter.replaceOpWithNewOp<IREE::Util::ReturnOp>(
          returnOp, returnOp.getOperands());
      convertedFuncInfo.returnOps.push_back(ireeReturnOp);
    });
    return success();
  }

  LogicalResult convertType(OpBuilder &builder, Location loc, Type torchType,
                            Type &ireeType, TypeDisposition &disp) {
    if (isa<TensorType>(torchType)) {
      ireeType = torchType;
      disp = TypeDisposition::IMMUTABLE_TENSOR;
      return success();
    }

    if (auto vtType = dyn_cast<Torch::ValueTensorType>(torchType)) {
      ireeType = convertToBuiltinTensorType(builder, vtType);
      disp = TypeDisposition::IMMUTABLE_TENSOR;
      return success();
    }

    if (isa<Torch::NonValueTensorType>(torchType)) {
      return emitError(loc)
             << "mutable tensors are not supported by sync func conversion: "
             << torchType;
    }

    if (isa<Torch::BoolType>(torchType)) {
      ireeType = IntegerType::get(torchType.getContext(), 1);
      disp = TypeDisposition::TORCH_PRIMITIVE;
      return success();
    }

    if (isa<Torch::IntType, Torch::GeneratorType>(torchType)) {
      ireeType = IntegerType::get(torchType.getContext(), 64);
      disp = TypeDisposition::TORCH_PRIMITIVE;
      return success();
    }

    if (isa<Torch::FloatType>(torchType)) {
      ireeType = Float64Type::get(torchType.getContext());
      disp = TypeDisposition::TORCH_PRIMITIVE;
      return success();
    }

    if (isa<IntegerType, FloatType, IndexType>(torchType)) {
      ireeType = torchType;
      disp = TypeDisposition::PASSTHROUGH;
      return success();
    }

    return emitError(loc) << "unhandled torch type: " << torchType;
  }
};

} // namespace

} // namespace mlir::iree_compiler::TorchInput
