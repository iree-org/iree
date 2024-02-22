// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torch-iree/InputConversion/Passes.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "torch-iree/InputConversion/PassDetail.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

namespace Torch = mlir::torch::Torch;
namespace TorchConversion = mlir::torch::TorchConversion;

namespace mlir::iree_compiler::TorchInput {

namespace {

//===----------------------------------------------------------------------===//
// Overall Approach
// ----------------
// This pass converts from the "torch" programming model to the "iree"
// programming model by rewriting all functions and calls to operate on native
// IREE types. In the process, synchronization is added as appropriate for any
// mutable and immutable torch-level arguments.
//
// Currently, the result of this pass is that every torch-func is augmented
// to be implemented in terms of IREE's "coarse fences" ABI. In this ABI,
// there is a (wait, signal) fence pair added to the end of every function.
// Since torch functions are single-exit, practically, this involves:
//   * Adding preambles to convert function arguments to `tensor` (and native
//     torch types via `torch_c`), adding coarse synchronization on import
//     (presently for all buffer arguments but in the future could only be
//     those which are not tied to fine grained fences).
//   * Adding a postamble with a synchronization barrier on any produced
//     or mutated tensors and appropriate exports/in-place tieing to buffers.
//   * Generation of a synchronous wrapper function with the original name
//     (the async function is named with an `$async` suffix) which internally
//     sets up/waits on fences while delegating to the async function.
//
// Immutable tensor types
// ----------------------
//
// Immutable tensor types are mapped to a buffer_view and subject to
// `hal.tensor.import` on use. On return, they will be placed behind a
// synchronization barrier and exported.
//
// Mutable types
// -------------
// Here we rely on the characteristic that at the torch level, conversion to
// and from the value domain is only legal at certain well defined points in
// the program (currently at graph edges but potentially in the future at
// various control flow points). These conversions are modeled by:
//   * `torch.copy.to_vtensor`: Copy from a mutable tensor (torch.tensor) to
//     an immutable value (torch.vtensor).
//   * `torch.copy.to_tensor`: Allocates a new mutable tensor and initializes it
//     with the value of the given immutable tensor. Presently un-used.
//   * `torch.overwrite.tensor.contents`: Updates the contents of a mutable
//     tensor from a given immutable tensor.
//
// Note that when importing from Torch, these ops cannot just be added at will,
// and they are only created as a result of structural conversions. Therefore,
// we can rely on these invariants and assume that usage outside of this is an
// invalid program.
//===----------------------------------------------------------------------===//

std::optional<std::pair<Value, Value>>
getEnclosingWaitSignalFences(Operation *op) {
  auto parentFuncOp = dyn_cast<IREE::Util::FuncOp>(op);
  if (!parentFuncOp) {
    parentFuncOp = parentFuncOp->getParentOfType<IREE::Util::FuncOp>();
    if (!parentFuncOp)
      return {};
  }
  Block *entryBlock = &parentFuncOp.front();
  auto numArguments = entryBlock->getNumArguments();
  Value coarseWaitFence = entryBlock->getArgument(numArguments - 2);
  Value coarseSignalFence = entryBlock->getArgument(numArguments - 1);
  return std::make_pair(coarseWaitFence, coarseSignalFence);
}

std::optional<std::pair<Value, Value>>
getEnclosingWaitSignalFences(Value value) {
  return getEnclosingWaitSignalFences(value.getParentRegion()->getParentOp());
}

Value convertToBuiltinTensor(OpBuilder &builder, Value possibleTorchTensor) {
  Type ty = possibleTorchTensor.getType();
  if (isa<TensorType>(ty))
    return possibleTorchTensor;

  Torch::ValueTensorType vtensorType = cast<Torch::ValueTensorType>(ty);
  TensorType builtinTy = vtensorType.toBuiltinTensor();
  return builder.create<TorchConversion::ToBuiltinTensorOp>(
      possibleTorchTensor.getLoc(), builtinTy, possibleTorchTensor);
}

enum class TypeDisposition {
  IMMUTABLE_TENSOR,
  MUTABLE_TENSOR,
  TORCH_PRIMITIVE,
  PASSTHROUGH,
  FENCE,
};

struct ConvertedAsyncFunctionInfo {
  IREE::Util::FuncOp funcOp;
  SmallVector<IREE::Util::ReturnOp> returnOps;
  SmallVector<Type> torchInputTypes;
  SmallVector<Type> torchResultTypes;
  SmallVector<TypeDisposition> inputDispositions;
  SmallVector<TypeDisposition> resultDispositions;

  // Post processing state.
  // Values that must be captured in the coarse barrier.
  SmallVector<Value> barrierInputs;
  // Meta data per barrier input: storage, torchType, returnIndex (or -1)
  SmallVector<std::tuple<Value, Type, int>> barrierResultMeta;

  LogicalResult postProcess();
  LogicalResult convertImmutableTensorArg(BlockArgument argValue,
                                          Type torchType, OpBuilder &builder);
  LogicalResult convertMutableTensorArg(BlockArgument argValue, Type torchType,
                                        OpBuilder &builder);

  void addBarrierInput(Value inputTensor, Value storage, Type torchType,
                       int returnIndex) {
    barrierInputs.push_back(inputTensor);
    barrierResultMeta.emplace_back(storage, torchType, returnIndex);
  }
};

LogicalResult ConvertedAsyncFunctionInfo::postProcess() {
  if (funcOp.isExternal())
    return success();

  if (returnOps.size() != 1) {
    // Multi-exit/CFG could be supported but requires more complicated dominance
    // analysis with respect to where the exit happens relative to mutated
    // buffers.
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
      if (failed(
              convertImmutableTensorArg(argValue, torchType, preambleBuilder)))
        return failure();
      break;
    }
    case TypeDisposition::MUTABLE_TENSOR: {
      if (failed(convertMutableTensorArg(argValue, torchType, preambleBuilder)))
        return failure();
      break;
    }
    case TypeDisposition::TORCH_PRIMITIVE: {
      Location loc = argValue.getLoc();
      Operation *convertUser = nullptr;
      Value convertResult;
      if (isa<Torch::BoolType>(torchType)) {
        convertUser =
            preambleBuilder.create<TorchConversion::FromI1Op>(loc, argValue);
        convertResult = convertUser->getResult(0);
      } else if (isa<Torch::FloatType>(torchType)) {
        convertUser =
            preambleBuilder.create<TorchConversion::FromF64Op>(loc, argValue);
        convertResult = convertUser->getResult(0);
      } else if (isa<Torch::IntType>(torchType)) {
        convertUser =
            preambleBuilder.create<TorchConversion::FromI64Op>(loc, argValue);
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
    case TypeDisposition::FENCE:
      // Do nothing.
      break;
    }
  }

  // Materialize synchronization postamble and conversions.
  IREE::Util::ReturnOp returnOp = returnOps.front();
  SmallVector<Value> newReturnOperands;
  OpBuilder postambleBuilder(returnOp);
  for (auto [disp, returnValue, torchType] : llvm::zip_equal(
           resultDispositions, returnOp.getOperands(), torchResultTypes)) {
    size_t returnIndex = newReturnOperands.size();
    newReturnOperands.emplace_back(returnValue);
    switch (disp) {
    case TypeDisposition::IMMUTABLE_TENSOR: {
      bool needsBarrier = true;
      if (auto blockArg = dyn_cast<BlockArgument>(returnValue)) {
        // Trivial return of input. Just pass it through.
        needsBarrier = blockArg.getOwner() != entryBlock;
      }
      if (needsBarrier) {
        Value source = convertToBuiltinTensor(postambleBuilder, returnValue);
        addBarrierInput(source, /*storage=*/Value{}, torchType, returnIndex);
      }
      break;
    }
    case TypeDisposition::TORCH_PRIMITIVE: {
      Location loc = returnValue.getLoc();
      if (isa<Torch::BoolType>(torchType)) {
        newReturnOperands.back() =
            postambleBuilder.create<TorchConversion::ToI1Op>(loc, returnValue);
      } else if (isa<Torch::FloatType>(torchType)) {
        newReturnOperands.back() =
            postambleBuilder.create<TorchConversion::ToF64Op>(loc, returnValue);
      } else if (isa<Torch::IntType>(torchType)) {
        newReturnOperands.back() =
            postambleBuilder.create<TorchConversion::ToI64Op>(loc, returnValue);
      } else if (isa<Torch::GeneratorType>(torchType)) {
        newReturnOperands.back() =
            postambleBuilder.create<TorchConversion::GeneratorToI64Op>(
                loc, returnValue);
      } else {
        emitError(loc) << "unhandled torch primitive materialization: "
                       << torchType;
        return failure();
      }
      break;
    }
    default: {
      // Non-tensor/converting. Just preserve.
    }
    }
  }

  // Emit the barrier and exports.
  Value coarseSignalFence =
      entryBlock->getArgument(entryBlock->getNumArguments() - 1);
  if (barrierInputs.empty()) {
    postambleBuilder.create<IREE::HAL::FenceSignalOp>(funcOp.getLoc(),
                                                      coarseSignalFence);
  } else {
    auto barrierOp = postambleBuilder.create<IREE::HAL::TensorBarrierOp>(
        funcOp.getLoc(), barrierInputs, coarseSignalFence);
    for (auto [barrierResult, meta] :
         llvm::zip_equal(barrierOp.getResults(), barrierResultMeta)) {
      Value exportStorage;
      Type torchType;
      int returnIndex;
      std::tie(exportStorage, torchType, returnIndex) = meta;
      Value exportedValue = postambleBuilder.create<IREE::HAL::TensorExportOp>(
          funcOp.getLoc(),
          postambleBuilder.getType<IREE::HAL::BufferViewType>(), barrierResult,
          TypeAttr::get(barrierResult.getType()), exportStorage, StringAttr());
      if (returnIndex >= 0) {
        newReturnOperands[returnIndex] = exportedValue;
      } else {
        // Don't drop it.
        postambleBuilder.create<IREE::Util::OptimizationBarrierOp>(
            funcOp.getLoc(), exportedValue);
      }
    }
  }

  // New return operands are all collected.
  returnOp->setOperands(newReturnOperands);

  return success();
}

class OriginalUses {
public:
  OriginalUses(Value value) {
    for (auto &use : value.getUses()) {
      originalUses.push_back(&use);
    }
  }

  void assign(Value newValue) {
    for (OpOperand *originalUse : originalUses) {
      originalUse->assign(newValue);
    }
  }

private:
  SmallVector<OpOperand *> originalUses;
};

LogicalResult ConvertedAsyncFunctionInfo::convertImmutableTensorArg(
    BlockArgument argValue, Type torchType, OpBuilder &builder) {
  Location loc = argValue.getLoc();

  // If the arg is just directly returned, then don't do anything special with
  // it.
  bool hasNonTrivialUse = false;
  for (auto *userOp : argValue.getUsers()) {
    if (isa<IREE::Util::ReturnOp>(userOp))
      continue;
    hasNonTrivialUse = true;
  }
  if (!hasNonTrivialUse)
    return success();

  // Remember original uses so we can redirect them.
  OriginalUses originalUses(argValue);

  // The type can either be a builtin TensorType or a Torch::ValueTensorType.
  // OpBuilder
  TensorType builtinTensorType;
  if (auto tType = dyn_cast<TensorType>(torchType)) {
    builtinTensorType = tType;
  } else if (auto vtType = dyn_cast<Torch::ValueTensorType>(torchType)) {
    builtinTensorType = vtType.toBuiltinTensor();
  } else {
    return emitError(loc) << "unsupported immutable tensor argument: "
                          << torchType;
  }

  auto waitSignalFences = getEnclosingWaitSignalFences(argValue);
  assert(waitSignalFences && "async function missing fences");
  Value waitFence = waitSignalFences->first;
  Value importedTensor = builder.create<IREE::HAL::TensorImportOp>(
      loc, builtinTensorType, argValue, TypeAttr::get(builtinTensorType),
      waitFence,
      /*name=*/StringAttr());
  if (builtinTensorType != torchType) {
    importedTensor = builder.create<TorchConversion::FromBuiltinTensorOp>(
        loc, torchType, importedTensor);
  }

  originalUses.assign(importedTensor);
  return success();
}

LogicalResult ConvertedAsyncFunctionInfo::convertMutableTensorArg(
    BlockArgument argValue, Type torchType, OpBuilder &builder) {
  Location loc = argValue.getLoc();
  auto fences = getEnclosingWaitSignalFences(argValue);
  assert(fences && "could not find async fences on func");
  TensorType builtinTensorType;
  if (auto t = dyn_cast<TensorType>(torchType)) {
    builtinTensorType = t;
  } else {
    builtinTensorType = cast<Torch::NonValueTensorType>(torchType)
                            .getWithValueSemantics()
                            .toBuiltinTensor();
  }

  // There are only a small set of possible users of a mutable tensor.
  // Handle them by operation here.
  SmallVector<Operation *> users(argValue.getUsers());
  for (auto *userOp : users) {
    IRRewriter rewriter(loc.getContext());
    rewriter.setInsertionPoint(userOp);
    if (auto copyToVtOp = dyn_cast<Torch::CopyToValueTensorOp>(userOp)) {
      Value imported = rewriter.create<IREE::HAL::TensorImportOp>(
          loc, builtinTensorType, argValue,
          /*target_encoding=*/TypeAttr::get(builtinTensorType),
          /*wait_fence*/ fences->first,
          /*name=*/StringAttr());
      rewriter.replaceOpWithNewOp<TorchConversion::FromBuiltinTensorOp>(
          userOp, copyToVtOp.getResult().getType(), imported);
    } else if (auto overwriteOp =
                   dyn_cast<Torch::OverwriteTensorContentsOp>(userOp)) {
      Value overwriteValue =
          convertToBuiltinTensor(rewriter, overwriteOp.getValue());
      addBarrierInput(overwriteValue, /*storage=*/argValue, torchType,
                      /*returnIndex=*/-1);
      rewriter.eraseOp(overwriteOp);
    } else {
      return emitError(userOp->getLoc())
             << "unsupported operation on coarse signaling mutable tensor: "
             << *userOp;
    }
  }

  return success();
}

void retainFunctionAttributes(Operation *srcOp, IREE::Util::FuncOp destOp) {
  // Allowlist of function attributes to retain when importing funcs.
  constexpr const char *kRetainedAttributes[] = {
      "iree.reflection",
  };
  auto retainedAttributes = ArrayRef<const char *>(
      kRetainedAttributes,
      sizeof(kRetainedAttributes) / sizeof(kRetainedAttributes[0]));
  for (auto retainAttrName : retainedAttributes) {
    StringRef attrName(retainAttrName);
    Attribute attr = srcOp->getAttr(attrName);
    if (attr)
      destOp->setAttr(attrName, attr);
  }
}

void createCoarseFencesSyncWrapper(StringRef syncFunctionName,
                                   IREE::Util::FuncOp asyncFuncOp,
                                   IRRewriter &rewriter) {
  Location loc = asyncFuncOp.getLoc();
  // The coarse fences wrapper has the same signature as the async variant
  // but with the last two inputs (wait, signal fence) sliced off.
  FunctionType asyncFuncType = asyncFuncOp.getFunctionType();
  SmallVector<Type> inputTypes(asyncFuncType.getInputs().begin(),
                               asyncFuncType.getInputs().end() - 2);

  // Create the function.
  auto syncFuncType = rewriter.getType<mlir::FunctionType>(
      inputTypes, asyncFuncType.getResults());
  auto syncFuncOp =
      rewriter.create<IREE::Util::FuncOp>(loc, syncFunctionName, syncFuncType,
                                          /*tiedOperandsAttr=*/nullptr);
  syncFuncOp.setSymVisibilityAttr(asyncFuncOp.getSymVisibilityAttr());
  retainFunctionAttributes(asyncFuncOp, syncFuncOp);
  syncFuncOp->setAttr("iree.abi.stub", rewriter.getUnitAttr());
  Block *entryBlock = syncFuncOp.addEntryBlock();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(entryBlock);

  // HACK: this is relying on the fact that there's only one HAL device.
  // We should instead have a way of creating fences on the device that
  // is used to produce the tensors we're wrapping.
  //
  // TODO(multi-device): emit get with derived ordinal or lookup with attr. We
  // could always say device 0 for now but could instead look for an
  // iree.abi.affinity/iree.abi.device/etc.
  Value timeoutMillis = rewriter.create<arith::ConstantIntOp>(loc, -1, 32);
  Value device = IREE::HAL::DeviceType::resolveAny(loc, rewriter);
  Value waitFence = rewriter.create<IREE::Util::NullOp>(
      loc, rewriter.getType<IREE::HAL::FenceType>());
  Value signalFence = rewriter.create<IREE::HAL::FenceCreateOp>(
      loc, rewriter.getType<IREE::HAL::FenceType>(), device,
      IREE::HAL::FenceFlagBitfield::None);

  SmallVector<Value> callOperands(entryBlock->getArguments());
  callOperands.push_back(waitFence);
  callOperands.push_back(signalFence);
  std::optional<ArrayAttr> targetTiedOperands = asyncFuncOp.getTiedOperands();
  auto callResults =
      rewriter
          .create<IREE::Util::CallOp>(loc, asyncFuncOp, callOperands,
                                      targetTiedOperands ? *targetTiedOperands
                                                         : ArrayAttr{})
          .getResults();

  // Wait forever for signal.
  rewriter.create<IREE::HAL::FenceAwaitOp>(loc, rewriter.getI32Type(),
                                           timeoutMillis, signalFence);

  rewriter.create<IREE::Util::ReturnOp>(loc, callResults);
}

} // namespace

struct FuncConversionPass : public FuncConversionBase<FuncConversionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<IREE::HAL::HALDialect>();
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
    std::vector<ConvertedAsyncFunctionInfo> convertedFuncInfos;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (!shouldConvertFunc(funcOp))
        continue;
      ConvertedAsyncFunctionInfo &convertedFuncInfo =
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

    // Now post-process async functions.
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
    if (torchFunc.isExternal())
      return false;

    // Something has already converted this and told us not to touch it.
    if (torchFunc->hasAttr("iree.abi.stub"))
      return false;

    return true;
  }

  LogicalResult convertFuncOp(func::FuncOp torchFunc,
                              ConvertedAsyncFunctionInfo &convertedFuncInfo) {
    IRRewriter rewriter(torchFunc.getContext());
    rewriter.setInsertionPoint(torchFunc);
    Location loc = torchFunc.getLoc();
    // Determine whether to build pure async or async + sync wrapper.
    bool generateSyncWrapper = true;
    StringRef originalName = torchFunc.getName();
    std::string asyncFunctionName = originalName.str();
    if (generateSyncWrapper) {
      asyncFunctionName.append("$async");
    }

    // Convert function signature.
    Type fenceType = rewriter.getType<IREE::HAL::FenceType>();
    FunctionType torchFuncType = torchFunc.getFunctionType();
    convertedFuncInfo.torchInputTypes.append(torchFuncType.getInputs().begin(),
                                             torchFuncType.getInputs().end());
    convertedFuncInfo.torchResultTypes.append(
        torchFuncType.getResults().begin(), torchFuncType.getResults().end());
    // For the coarse-fences ABI, we add two fences to the end. Treat these as
    // original types so that the lists line up.
    convertedFuncInfo.torchInputTypes.push_back(fenceType);
    convertedFuncInfo.torchInputTypes.push_back(fenceType);
    SmallVector<Type> ireeInputTypes(convertedFuncInfo.torchInputTypes);
    SmallVector<Type> ireeResultTypes(convertedFuncInfo.torchResultTypes);
    convertedFuncInfo.inputDispositions.resize(ireeInputTypes.size());
    convertedFuncInfo.resultDispositions.resize(ireeResultTypes.size());

    for (size_t i = 0; i < convertedFuncInfo.torchInputTypes.size(); ++i) {
      if (failed(convertType(loc, convertedFuncInfo.torchInputTypes[i],
                             ireeInputTypes[i],
                             convertedFuncInfo.inputDispositions[i])))
        return failure();
    }
    for (size_t i = 0; i < convertedFuncInfo.torchResultTypes.size(); ++i) {
      if (failed(convertType(loc, convertedFuncInfo.torchResultTypes[i],
                             ireeResultTypes[i],
                             convertedFuncInfo.resultDispositions[i])))
        return failure();
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

    // Create new func.
    FunctionType asyncFuncType =
        FunctionType::get(loc.getContext(), ireeInputTypes, ireeResultTypes);
    auto asyncFuncOp = rewriter.create<IREE::Util::FuncOp>(
        torchFunc.getLoc(), asyncFunctionName, asyncFuncType, tiedOperandsAttr);
    convertedFuncInfo.funcOp = asyncFuncOp;
    asyncFuncOp.setSymVisibilityAttr(torchFunc.getSymVisibilityAttr());
    // Handle defacto attrs to specialized ones.
    if (torchFunc->hasAttr("noinline")) {
      asyncFuncOp.setInliningPolicyAttr(
          rewriter.getAttr<IREE::Util::InlineNeverAttr>());
    }
    retainFunctionAttributes(torchFunc, asyncFuncOp);
    asyncFuncOp->setAttr("iree.abi.stub", rewriter.getUnitAttr());
    asyncFuncOp->setAttr("iree.abi.model",
                         rewriter.getStringAttr("coarse-fences"));
    rewriter.inlineRegionBefore(
        torchFunc.getBody(), asyncFuncOp.getFunctionBody(), asyncFuncOp.end());

    // Convert block arguments.
    Block *entryBlock = &asyncFuncOp.getBlocks().front();
    for (size_t i = 0; i < ireeInputTypes.size(); ++i) {
      // Add if we have extended the list.
      if (i >= entryBlock->getNumArguments()) {
        entryBlock->addArgument(ireeInputTypes[i], loc);
        continue;
      }
      // Convert.
      entryBlock->getArgument(i).setType(ireeInputTypes[i]);
    }

    // Replace return ops.
    asyncFuncOp->walk([&](func::ReturnOp returnOp) {
      rewriter.setInsertionPoint(returnOp);
      auto ireeReturnOp = rewriter.replaceOpWithNewOp<IREE::Util::ReturnOp>(
          returnOp, returnOp.getOperands());
      convertedFuncInfo.returnOps.push_back(ireeReturnOp);
    });

    // Create the sync variant.
    rewriter.setInsertionPoint(torchFunc);
    createCoarseFencesSyncWrapper(originalName, asyncFuncOp, rewriter);
    return success();
  }

  LogicalResult convertType(Location loc, Type torchType, Type &ireeType,
                            TypeDisposition &disp) {
    if (isa<TensorType, Torch::ValueTensorType>(torchType)) {
      ireeType = IREE::HAL::BufferViewType::get(torchType.getContext());
      disp = TypeDisposition::IMMUTABLE_TENSOR;
      return success();
    }

    if (isa<Torch::NonValueTensorType>(torchType)) {
      ireeType = IREE::HAL::BufferViewType::get(torchType.getContext());
      disp = TypeDisposition::MUTABLE_TENSOR;
      return success();
    }

    if (isa<IREE::HAL::FenceType>(torchType)) {
      ireeType = torchType;
      disp = TypeDisposition::FENCE;
      return success();
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

    if (isa<IntegerType, FloatType>(torchType)) {
      ireeType = torchType;
      disp = TypeDisposition::PASSTHROUGH;
      return success();
    }

    return emitError(loc) << "unhandled torch type: " << torchType;
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createFuncConversionPass() {
  return std::make_unique<FuncConversionPass>();
}

} // namespace mlir::iree_compiler::TorchInput
