// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/Torch/InputConversion/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/SmallSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

namespace Torch = mlir::torch::Torch;
namespace TorchConversion = mlir::torch::TorchConversion;

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_DEF_FUNCCONVERSIONPASS
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc"

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
//     or mutated tensors and appropriate exports/in-place tying to buffers.
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
    if (!parentFuncOp) {
      return {};
    }
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
  if (isa<TensorType>(ty)) {
    return possibleTorchTensor;
  }

  if (auto defining = dyn_cast_if_present<TorchConversion::FromBuiltinTensorOp>(
          possibleTorchTensor.getDefiningOp())) {
    return defining.getOperand();
  }

  Torch::ValueTensorType vtensorType = cast<Torch::ValueTensorType>(ty);
  TensorType builtinTy = vtensorType.toBuiltinTensor();
  if (auto intTy = dyn_cast<IntegerType>(builtinTy.getElementType())) {
    builtinTy =
        builtinTy.clone(builder.getIntegerType(intTy.getIntOrFloatBitWidth()));
  }

  return TorchConversion::ToBuiltinTensorOp::create(
      builder, possibleTorchTensor.getLoc(), builtinTy, possibleTorchTensor);
}

enum class TypeDisposition {
  IMMUTABLE_TENSOR,
  MUTABLE_TENSOR,
  TORCH_PRIMITIVE,
  PASSTHROUGH,
  TRANSIENT_BUFFER,
  FENCE,
};

struct BarrierResult {
  BlockArgument storage;
  Type torchType;
  int returnIndex = -1;
};

struct ConvertedAsyncFunctionInfo {
  IREE::Util::FuncOp funcOp;
  SmallVector<IREE::Util::ReturnOp> returnOps;
  SmallVector<DictionaryAttr> torchArgAttrs;
  SmallVector<DictionaryAttr> torchResultAttrs;
  SmallVector<Type> torchInputTypes;
  SmallVector<Type> torchResultTypes;
  SmallVector<TypeDisposition> inputDispositions;
  SmallVector<TypeDisposition> resultDispositions;

  // Post processing state.
  // Values that must be captured in the coarse barrier.
  SmallVector<Value> barrierInputs;
  // Meta data per barrier input: storage, torchType, returnIndex (or -1)
  SmallVector<BarrierResult> barrierResultMeta;
  // Transient buffer
  BlockArgument transientBuffer;

  LogicalResult postProcess();
  LogicalResult convertImmutableTensorArg(BlockArgument argValue,
                                          Type torchType, OpBuilder &builder);
  LogicalResult convertMutableTensorArg(BlockArgument argValue, Type torchType,
                                        OpBuilder &builder);

  void addBarrierInput(Value inputTensor, BlockArgument storage, Type torchType,
                       int returnIndex) {
    barrierInputs.push_back(inputTensor);
    barrierResultMeta.emplace_back(BarrierResult{
        storage,
        torchType,
        returnIndex,
    });
  }

  void setTransientBuffer(BlockArgument bufferArg) {
    assert(!transientBuffer && "Cannot reassign existing transient buffer.");
    transientBuffer = bufferArg;
  }

  Attribute getTorchArgAttr(BlockArgument argValue, StringRef attrName) {
    return torchArgAttrs.empty()
               ? Attribute{}
               : torchArgAttrs[argValue.getArgNumber()].get(attrName);
  }
  Attribute getTorchResultAttr(int returnIndex, StringRef attrName) {
    return torchResultAttrs.empty()
               ? Attribute{}
               : torchResultAttrs[returnIndex].get(attrName);
  }
};

LogicalResult ConvertedAsyncFunctionInfo::postProcess() {
  if (funcOp.isExternal()) {
    return success();
  }

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
      if (failed(convertImmutableTensorArg(argValue, torchType,
                                           preambleBuilder))) {
        return failure();
      }
      break;
    }
    case TypeDisposition::MUTABLE_TENSOR: {
      if (failed(
              convertMutableTensorArg(argValue, torchType, preambleBuilder))) {
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
    case TypeDisposition::TRANSIENT_BUFFER:
      setTransientBuffer(argValue);
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
        addBarrierInput(source, /*storage=*/BlockArgument{}, torchType,
                        returnIndex);
      }
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
    default: {
      // Non-tensor/converting. Just preserve.
    }
    }
  }

  // Emit the barrier and exports.
  // If any of the exports are in-place we need to alias their storage to the
  // provided buffers.
  Value coarseSignalFence =
      entryBlock->getArgument(entryBlock->getNumArguments() - 1);
  if (barrierInputs.empty()) {
    IREE::HAL::FenceSignalOp::create(postambleBuilder, funcOp.getLoc(),
                                     coarseSignalFence);
  } else {
    SmallVector<Value> aliasedResults;
    for (auto [barrierInput, meta] :
         llvm::zip_equal(barrierInputs, barrierResultMeta)) {
      Value aliasResult;
      if (meta.storage) {
        // Use the wait fence indicating when the storage is available for
        // mutation. We need to ensure that no writes are made to the storage
        // until it indicates it's safe to do so.
        auto storageAffinityAttr =
            getTorchArgAttr(meta.storage, "iree.abi.affinity");
        auto waitSignalFences = getEnclosingWaitSignalFences(meta.storage);
        assert(waitSignalFences && "async function missing fences");
        Value waitFence = waitSignalFences->first;
        auto barrierInputDims = IREE::Util::buildDynamicDimsForValue(
            barrierInput.getLoc(), barrierInput, postambleBuilder);
        aliasResult = IREE::HAL::TensorAliasOp::create(
            postambleBuilder, barrierInput.getLoc(), barrierInput.getType(),
            barrierInput, barrierInputDims, meta.storage, waitFence,
            storageAffinityAttr);
      } else {
        aliasResult = barrierInput;
      }
      if (transientBuffer) {
        auto sourceDims = IREE::Util::buildDynamicDimsForValue(
            aliasResult.getLoc(), aliasResult, postambleBuilder);
        aliasResult = IREE::HAL::TensorTransientsOp::create(
            postambleBuilder, aliasResult.getLoc(), aliasResult.getType(),
            aliasResult, sourceDims, transientBuffer, Attribute{});
      }
      aliasedResults.push_back(aliasResult);
    }
    auto barrierOp = IREE::HAL::TensorBarrierOp::create(
        postambleBuilder, funcOp.getLoc(), aliasedResults, coarseSignalFence);
    for (auto [barrierResult, meta] :
         llvm::zip_equal(barrierOp.getResults(), barrierResultMeta)) {
      Attribute exportAffinityAttr;
      if (meta.storage) {
        exportAffinityAttr = getTorchArgAttr(meta.storage, "iree.abi.affinity");
      } else if (meta.returnIndex >= 0) {
        exportAffinityAttr =
            getTorchResultAttr(meta.returnIndex, "iree.abi.affinity");
      }
      Value exportedValue = IREE::HAL::TensorExportOp::create(
          postambleBuilder, funcOp.getLoc(),
          postambleBuilder.getType<IREE::HAL::BufferViewType>(), barrierResult,
          TypeAttr::get(barrierResult.getType()), /*name=*/nullptr,
          exportAffinityAttr);
      if (meta.returnIndex >= 0) {
        newReturnOperands[meta.returnIndex] = exportedValue;
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
    if (isa<IREE::Util::ReturnOp>(userOp)) {
      continue;
    }
    hasNonTrivialUse = true;
  }
  if (!hasNonTrivialUse) {
    return success();
  }

  // Remember original uses so we can redirect them.
  OriginalUses originalUses(argValue);

  // The type can either be a builtin TensorType or a Torch::ValueTensorType.
  // OpBuilder
  TensorType builtinTensorType;
  if (auto tType = dyn_cast<TensorType>(torchType)) {
    builtinTensorType = tType;
  } else if (auto vtType = dyn_cast<Torch::ValueTensorType>(torchType)) {
    builtinTensorType = vtType.toBuiltinTensor();
    if (auto intTy =
            dyn_cast<IntegerType>(builtinTensorType.getElementType())) {
      builtinTensorType = builtinTensorType.clone(
          builder.getIntegerType(intTy.getIntOrFloatBitWidth()));
    }
  } else {
    return emitError(loc) << "unsupported immutable tensor argument: "
                          << torchType;
  }

  // Propagate explicit affinities and ABI behavior to the read.
  bool consume = getTorchArgAttr(argValue, "iree.abi.consume") ? true : false;
  auto affinityAttr = getTorchArgAttr(argValue, "iree.abi.affinity");

  auto waitSignalFences = getEnclosingWaitSignalFences(argValue);
  assert(waitSignalFences && "async function missing fences");
  Value waitFence = waitSignalFences->first;
  Value importedTensor = IREE::HAL::TensorImportOp::create(
      builder, loc, builtinTensorType, argValue,
      TypeAttr::get(builtinTensorType), consume, waitFence,
      /*name=*/nullptr, affinityAttr);
  if (builtinTensorType != torchType) {
    importedTensor = TorchConversion::FromBuiltinTensorOp::create(
        builder, loc, torchType, importedTensor);
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
    // Strip signedness from integer element types so that the import side
    // matches the overwrite/alias side (which goes through
    // `convertToBuiltinTensor` and folds to signless). Without this,
    // signed-integer storage args produce a `tensor<NxsiNN>` import that
    // can't legalize against the signless tensor flowing into the
    // shape-query companion's `IREE::HAL::TensorAliasOp`. Mirrors
    // convertImmutableTensorArg.
    if (auto intTy =
            dyn_cast<IntegerType>(builtinTensorType.getElementType())) {
      builtinTensorType = builtinTensorType.clone(
          builder.getIntegerType(intTy.getIntOrFloatBitWidth()));
    }
  }

  // Propagate explicit affinities and ABI behavior to the read and write.
  auto affinityAttr = getTorchArgAttr(argValue, "iree.abi.affinity");

  // There are only a small set of possible users of a mutable tensor.
  // Handle them by operation here.
  SmallVector<Operation *> users(argValue.getUsers());
  for (auto *userOp : users) {
    IRRewriter rewriter(loc.getContext());
    rewriter.setInsertionPoint(userOp);
    if (auto copyToVtOp = dyn_cast<Torch::CopyToValueTensorOp>(userOp)) {
      Value imported = IREE::HAL::TensorImportOp::create(
          rewriter, loc, builtinTensorType, argValue,
          /*target_encoding=*/TypeAttr::get(builtinTensorType),
          /*consume=*/false,
          /*wait_fence*/ fences->first,
          /*name=*/nullptr, affinityAttr);
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
    if (attr) {
      destOp->setAttr(attrName, attr);
    }
  }
}

//===----------------------------------------------------------------------===//
// Output Shape Query Companion
//===----------------------------------------------------------------------===//
// A sibling public function `<name>$shape_query` -- the single source of
// truth for querying the dynamic output shapes.
// Runtime callers find it via the reflection attribute on the main func.
//
// ABI: data args from the main func, in original order, followed by one
// extra mutable `!torch.tensor<[rank],si64>` arg per dynamic-shape result,
// in result-index order.
// The shape-query companion writes only the dynamic-slot indices with
// runtime-resolved dim values; static slots are untouched. Therefore the
// caller should pre-fill the static slots with the declared shape.
//
// Storage-arg handling: a mutable `!torch.tensor` arg ("storage arg") is
// dropped from the companion's signature only if it is a pure write-only
// output binding - every read of it (`Torch::CopyToValueTensorOp`) follows
// every write (`Torch::OverwriteTensorContentsOp`) in source order. The
// body rewrite then short-circuits each such read to the cloned write
// source. Read-before-write storage args are kept in the signature; the
// cloned read resolves to a load from the companion's own block arg, and
// runtime callers must pass the storage buffer view alongside the regular
// data inputs at companion-call time.
//===----------------------------------------------------------------------===//

/// Returns true if any vtensor result of `f` has at least one dynamic dim.
static bool hasDynamicShapeResult(func::FuncOp f) {
  for (Type t : f.getFunctionType().getResults()) {
    auto vtType = dyn_cast<Torch::ValueTensorType>(t);
    if (!vtType || !vtType.hasSizes()) {
      continue;
    }
    for (int64_t s : vtType.getSizes()) {
      if (s == Torch::kUnknownSize) {
        return true;
      }
    }
  }
  return false;
}

/// Returns the indices of mutable !torch.tensor args that the shape-query
/// companion can drop from its signature.
///
/// An arg is droppable if it is written, and if there is any read, the read
/// is after the write. Equivalently: every read can be replaced with the
/// just-written value, so the arg itself is no longer needed.
static llvm::SmallSet<unsigned, 4>
getDroppableStorageArgIndices(func::FuncOp f) {
  Block &entry = f.getBody().front();
  // First-occurrence source-order positions of overwrites and copies, keyed
  // by the storage arg.
  llvm::DenseMap<Value, unsigned> firstOverwritePos;
  llvm::DenseMap<Value, unsigned> firstCopyPos;
  unsigned pos = 0;
  for (Operation &op : entry.without_terminator()) {
    if (auto ov = dyn_cast<Torch::OverwriteTensorContentsOp>(&op)) {
      firstOverwritePos.try_emplace(ov.getOverwritten(), pos);
    } else if (auto cp = dyn_cast<Torch::CopyToValueTensorOp>(&op)) {
      firstCopyPos.try_emplace(cp.getOperand(), pos);
    }
    ++pos;
  }
  llvm::SmallSet<unsigned, 4> result;
  for (unsigned i = 0; i < f.getNumArguments(); ++i) {
    Value arg = f.getArgument(i);
    if (!isa<Torch::NonValueTensorType>(arg.getType())) {
      continue;
    }
    auto ovIt = firstOverwritePos.find(arg);
    if (ovIt == firstOverwritePos.end()) {
      continue;
    }
    auto cpIt = firstCopyPos.find(arg);
    if (cpIt != firstCopyPos.end() && cpIt->second < ovIt->second) {
      // Copy precedes overwrite -> the body's read can't be short-circuited;
      // keep the arg so the companion's clone of the read remains in-region.
      continue;
    }
    result.insert(i);
  }
  return result;
}

/// One dynamic-shape result of the main func: its position in the result
/// list and the indices of its dynamic dims.
struct DynResult {
  unsigned resultIdx;
  SmallVector<unsigned> dynamicDims;
};

/// The shape-query companion's signature pieces, derived from the main func.
struct ShapeQueryCompanionSignature {
  FunctionType type;
  SmallVector<DictionaryAttr> argAttrs;
  /// Indices into `mainFunc`'s arg list that map to shape-query companion
  /// data args.
  SmallVector<unsigned> mainDataArgIndices;
};

/// Returns the dynamic-shape results of `mainFunc`, in result-index order.
static SmallVector<DynResult>
collectDynamicShapeResults(func::FuncOp mainFunc) {
  SmallVector<DynResult> dynResults;
  FunctionType ft = mainFunc.getFunctionType();
  for (unsigned r = 0; r < ft.getNumResults(); ++r) {
    auto vtType = dyn_cast<Torch::ValueTensorType>(ft.getResult(r));
    if (!vtType || !vtType.hasSizes()) {
      continue;
    }
    SmallVector<unsigned> dynamicDims;
    auto sizes = vtType.getSizes();
    for (unsigned d = 0; d < sizes.size(); ++d) {
      if (sizes[d] == Torch::kUnknownSize) {
        dynamicDims.push_back(d);
      }
    }
    if (!dynamicDims.empty()) {
      dynResults.push_back({r, std::move(dynamicDims)});
    }
  }
  return dynResults;
}

/// Builds the shape-query companion's FunctionType + arg attrs: main func's
/// data args (droppable storage args) followed by one mutable
/// `!torch.tensor<[rank],si64>` per dynamic-shape result. No return values.
static ShapeQueryCompanionSignature
buildShapeQueryCompanionSignature(func::FuncOp mainFunc,
                                  ArrayRef<DynResult> dynResults) {
  MLIRContext *ctx = mainFunc.getContext();
  llvm::SmallSet<unsigned, 4> droppable =
      getDroppableStorageArgIndices(mainFunc);

  ShapeQueryCompanionSignature sig;
  SmallVector<Type> inputTypes;
  SmallVector<DictionaryAttr> mainArgAttrs;
  mainFunc.getAllArgAttrs(mainArgAttrs);

  for (unsigned i = 0; i < mainFunc.getNumArguments(); ++i) {
    if (droppable.contains(i)) {
      continue;
    }
    inputTypes.push_back(mainFunc.getArgument(i).getType());
    if (!mainArgAttrs.empty()) {
      sig.argAttrs.push_back(mainArgAttrs[i]);
    }
    sig.mainDataArgIndices.push_back(i);
  }

  Type si64 = IntegerType::get(ctx, 64, IntegerType::Signed);
  FunctionType ft = mainFunc.getFunctionType();
  for (const DynResult &dr : dynResults) {
    auto resVt = cast<Torch::ValueTensorType>(ft.getResult(dr.resultIdx));
    int64_t rank = static_cast<int64_t>(resVt.getSizes().size());
    inputTypes.push_back(
        Torch::NonValueTensorType::get(ctx, ArrayRef<int64_t>{rank}, si64));
    if (!sig.argAttrs.empty()) {
      sig.argAttrs.push_back(DictionaryAttr::get(ctx, {}));
    }
  }

  sig.type = FunctionType::get(ctx, inputTypes, /*results=*/{});
  return sig;
}

/// Clones every op in `srcEntry` into the companion body, in source order,
/// while peeling away the in-place output binding pattern:
///   - `Torch::OverwriteTensorContentsOp` is dropped; we remember which
///     value was just written into which storage arg.
///   - A later `Torch::CopyToValueTensorOp` of that same storage is also
///     skipped, and its result is mapped to the remembered value.
/// Reads with no matching prior overwrite clone normally; they resolve to the
/// companion's own block arg via `mapping`.
static void cloneBodyShortCircuitingStorage(Block *srcEntry, OpBuilder &builder,
                                            IRMapping &mapping) {
  llvm::DenseMap<Value, Value> storageToSrc;
  for (Operation &op : srcEntry->without_terminator()) {
    if (auto overwriteOp = dyn_cast<Torch::OverwriteTensorContentsOp>(&op)) {
      storageToSrc[overwriteOp.getOverwritten()] =
          mapping.lookupOrDefault(overwriteOp.getValue());
      continue;
    }
    if (auto copyOp = dyn_cast<Torch::CopyToValueTensorOp>(&op)) {
      auto it = storageToSrc.find(copyOp.getOperand());
      if (it != storageToSrc.end()) {
        // %src and the copy's result share a vtensor type by construction
        // of `Torch::OverwriteTensorContentsOp`, so this substitution is
        // type-safe.
        mapping.map(copyOp.getResult(), it->second);
        continue;
      }
    }
    builder.clone(op, mapping);
  }
}

/// Emits the partial-write sequence for one dynamic-shape result:
///   read:  shape_buf -> vtensor -> builtin tensor
///   per dynamic dim d: `tensor::InsertOp` (i64 of `tensor::DimOp`) at d
///   write: builtin -> vtensor -> `Torch::OverwriteTensorContentsOp`
/// Static slots are untouched.
static void emitShapeBufferWriteback(OpBuilder &builder, Location loc,
                                     const DynResult &dr, Value resultClone,
                                     Value shapeArg) {
  Type i64Type = builder.getI64Type();
  auto shapeTorchTy = cast<Torch::NonValueTensorType>(shapeArg.getType());

  // Bridge the cloned result tensor to a builtin tensor for tensor.dim.
  auto vtType = cast<Torch::ValueTensorType>(resultClone.getType());
  TensorType resBuiltinTy = vtType.toBuiltinTensor();
  if (auto intTy = dyn_cast<IntegerType>(resBuiltinTy.getElementType())) {
    resBuiltinTy = resBuiltinTy.clone(
        builder.getIntegerType(intTy.getIntOrFloatBitWidth()));
  }
  Value resultBuiltin = TorchConversion::ToBuiltinTensorOp::create(
      builder, loc, resBuiltinTy, resultClone);

  Type shapeVtType = shapeTorchTy.getWithValueSemantics();
  Value shapeVt =
      Torch::CopyToValueTensorOp::create(builder, loc, shapeVtType, shapeArg);
  int64_t rank = shapeTorchTy.getSizes()[0];
  auto shapeBuiltinTy = RankedTensorType::get({rank}, i64Type);
  Value shapeBuiltin = TorchConversion::ToBuiltinTensorOp::create(
      builder, loc, shapeBuiltinTy, shapeVt);

  Value cur = shapeBuiltin;
  for (unsigned d : dr.dynamicDims) {
    Value idx = arith::ConstantIndexOp::create(builder, loc, d);
    Value dim = tensor::DimOp::create(builder, loc, resultBuiltin, idx);
    Value dimI64 = arith::IndexCastOp::create(builder, loc, i64Type, dim);
    cur = tensor::InsertOp::create(builder, loc, dimI64, cur, ValueRange{idx});
  }

  Value newShapeVt = TorchConversion::FromBuiltinTensorOp::create(
      builder, loc, shapeVtType, cur);
  Torch::OverwriteTensorContentsOp::create(builder, loc, newShapeVt, shapeArg);
}

/// Stamps `iree.abi.output_shape_query="<shapeQueryCompanionName>"` on
/// `mainFunc`'s iree.reflection dict so runtime callers can discover the
/// shape-query companion. Merges into any existing iree.reflection entries.
static void stampOutputShapeQueryReflection(func::FuncOp mainFunc,
                                            StringRef shapeQueryCompanionName,
                                            OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  SmallVector<NamedAttribute> entries;
  if (auto existing =
          mainFunc->getAttrOfType<DictionaryAttr>("iree.reflection")) {
    llvm::append_range(entries, existing.getValue());
  }
  entries.emplace_back(builder.getStringAttr("iree.abi.output_shape_query"),
                       builder.getStringAttr(shapeQueryCompanionName));
  mainFunc->setAttr("iree.reflection", DictionaryAttr::get(ctx, entries));
}

/// Builds and inserts `<name>$shape_query` next to `mainFunc`, and stamps
/// the `iree.abi.output_shape_query` reflection attribute on `mainFunc` so
/// runtime callers can discover the shape-query companion.
static func::FuncOp createTorchShapeQueryCompanion(func::FuncOp mainFunc,
                                                   OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  Location loc = mainFunc.getLoc();

  SmallVector<DynResult> dynResults = collectDynamicShapeResults(mainFunc);
  ShapeQueryCompanionSignature sig =
      buildShapeQueryCompanionSignature(mainFunc, dynResults);

  std::string shapeQueryCompanionName =
      mainFunc.getName().str() + "$shape_query";
  auto shapeQueryCompanion =
      func::FuncOp::create(builder, loc, shapeQueryCompanionName, sig.type);
  shapeQueryCompanion.setPublic();
  if (!sig.argAttrs.empty()) {
    shapeQueryCompanion.setAllArgAttrs(sig.argAttrs);
  }
  Block *newEntry = shapeQueryCompanion.addEntryBlock();

  // Map main func's data args to the shape-query companion's data args.
  Block *srcEntry = &mainFunc.getBody().front();
  IRMapping mapping;
  for (unsigned k = 0; k < sig.mainDataArgIndices.size(); ++k) {
    mapping.map(srcEntry->getArgument(sig.mainDataArgIndices[k]),
                newEntry->getArgument(k));
  }

  // Phase A: clone the main func's body, short-circuiting the storage
  // write-back path.
  OpBuilder body(ctx);
  body.setInsertionPointToEnd(newEntry);
  cloneBodyShortCircuitingStorage(srcEntry, body, mapping);

  // Phase B: per dynamic-shape result, partial-write its runtime dim values
  // into the matching shape buffer arg.
  Operation *srcReturn = srcEntry->getTerminator();
  for (unsigned k = 0; k < dynResults.size(); ++k) {
    Value resultClone =
        mapping.lookupOrDefault(srcReturn->getOperand(dynResults[k].resultIdx));
    Value shapeArg = newEntry->getArgument(sig.mainDataArgIndices.size() + k);
    emitShapeBufferWriteback(body, loc, dynResults[k], resultClone, shapeArg);
  }

  func::ReturnOp::create(body, loc, ValueRange{});

  stampOutputShapeQueryReflection(mainFunc, shapeQueryCompanion.getName(),
                                  builder);
  return shapeQueryCompanion;
}

void createCoarseFencesSyncWrapper(StringRef syncFunctionName,
                                   IREE::Util::FuncOp asyncFuncOp,
                                   IRRewriter &rewriter) {
  Location loc = asyncFuncOp.getLoc();
  // The coarse fences wrapper has the same signature as the async variant
  // but with the last two inputs (wait, signal fence) sliced off.
  FunctionType asyncFuncType = asyncFuncOp.getFunctionType();
  // Note: If we externalize a transient buffer, we are currently including it
  // as the final input for the sync wrapper.
  SmallVector<Type> inputTypes(asyncFuncType.getInputs().begin(),
                               asyncFuncType.getInputs().end() - 2);

  // Create the function.
  auto syncFuncType = rewriter.getType<mlir::FunctionType>(
      inputTypes, asyncFuncType.getResults());
  auto syncFuncOp =
      IREE::Util::FuncOp::create(rewriter, loc, syncFunctionName, syncFuncType,
                                 /*tiedOperandsAttr=*/nullptr);
  syncFuncOp.setSymVisibilityAttr(asyncFuncOp.getSymVisibilityAttr());
  retainFunctionAttributes(asyncFuncOp, syncFuncOp);
  syncFuncOp->setAttr("iree.abi.stub", rewriter.getUnitAttr());
  if (auto affinityAttr = asyncFuncOp->getAttr("iree.abi.affinity")) {
    syncFuncOp->setAttr("iree.abi.affinity", affinityAttr);
  }
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
  Value timeoutMillis = arith::ConstantIntOp::create(rewriter, loc, -1, 32);
  Value device = IREE::HAL::DeviceType::resolveAny(loc, rewriter);
  Value waitFence = IREE::Util::NullOp::create(
      rewriter, loc, rewriter.getType<IREE::HAL::FenceType>());
  Value signalFence = IREE::HAL::FenceCreateOp::create(
      rewriter, loc, rewriter.getType<IREE::HAL::FenceType>(), device,
      IREE::HAL::FenceFlagBitfield::None);

  SmallVector<Value> callOperands(entryBlock->getArguments());
  callOperands.push_back(waitFence);
  callOperands.push_back(signalFence);
  std::optional<ArrayAttr> targetTiedOperands = asyncFuncOp.getTiedOperands();
  auto callResults = IREE::Util::CallOp::create(
                         rewriter, loc, asyncFuncOp, callOperands,
                         targetTiedOperands ? *targetTiedOperands : ArrayAttr{})
                         .getResults();

  // Wait forever for signal.
  IREE::HAL::FenceAwaitOp::create(
      rewriter, loc, rewriter.getI32Type(), timeoutMillis,
      IREE::HAL::WaitFlagBitfield::None, signalFence);

  IREE::Util::ReturnOp::create(rewriter, loc, callResults);
}

} // namespace

class FuncConversionPass final
    : public impl::FuncConversionPassBase<FuncConversionPass> {
public:
  using Base::Base;

  FuncConversionPass(bool externalizeTransients) {
    this->externalizeTransients = externalizeTransients;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<IREE::HAL::HALDialect>();
    registry.insert<IREE::Util::UtilDialect>();
    registry.insert<TorchConversion::TorchConversionDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Pre-pass: for every public torch func with a dynamic-shape result,
    // synthesize a sibling `<name>$shape_query` companion and stamp
    // `iree.abi.output_shape_query="<name>$shape_query"` on the main func's
    // iree.reflection dict. We do this *before* the main conversion loop so
    // the shape-query companion rides the same coarse-fences pipeline and
    // emerges as a regular public sync wrapper, and so
    // retainFunctionAttributes carries the reflection entry onto both the
    // async and sync wrappers.
    SmallVector<func::FuncOp> mainFuncsForShapeQueryCompanion;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (!shouldConvertFunc(funcOp)) {
        continue;
      }
      // Don't recurse if this pass has already placed shape-query companions
      // in the module from a prior run.
      if (funcOp.getName().ends_with("$shape_query")) {
        continue;
      }
      if (hasDynamicShapeResult(funcOp)) {
        mainFuncsForShapeQueryCompanion.push_back(funcOp);
      }
    }
    for (auto mainFunc : mainFuncsForShapeQueryCompanion) {
      OpBuilder builder(mainFunc);
      createTorchShapeQueryCompanion(mainFunc, builder);
    }

    // Convert all functions in the module to IREE funcs. In this stage,
    // we convert contained return ops and argument/result types, but we have
    // not yet converted anything "on the inside". Therefore, it is pretty
    // likely the functions are still illegal.
    SmallVector<Operation *> eraseFuncOps;
    std::vector<ConvertedAsyncFunctionInfo> convertedFuncInfos;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (!shouldConvertFunc(funcOp)) {
        continue;
      }
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

    // Stash arg/result attrs so they can be referenced during conversion.
    torchFunc.getAllArgAttrs(convertedFuncInfo.torchArgAttrs);
    torchFunc.getAllResultAttrs(convertedFuncInfo.torchResultAttrs);

    // Convert function signature.
    Type bufferType = rewriter.getType<IREE::HAL::BufferType>();
    Type fenceType = rewriter.getType<IREE::HAL::FenceType>();
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
      if (failed(convertType(loc, convertedFuncInfo.torchInputTypes[i],
                             ireeInputTypes[i],
                             convertedFuncInfo.inputDispositions[i]))) {
        return failure();
      }
    }
    for (size_t i = 0; i < convertedFuncInfo.torchResultTypes.size(); ++i) {
      if (failed(convertType(loc, convertedFuncInfo.torchResultTypes[i],
                             ireeResultTypes[i],
                             convertedFuncInfo.resultDispositions[i]))) {
        return failure();
      }
    }

    // When externalizing transient memory, put the input buffer before fences.
    if (externalizeTransients) {
      convertedFuncInfo.torchInputTypes.push_back(bufferType);
      ireeInputTypes.push_back(bufferType);
      convertedFuncInfo.inputDispositions.push_back(
          TypeDisposition::TRANSIENT_BUFFER);
    }
    // For the coarse-fences ABI, we add two fences to the end. Treat these as
    // original types so that the lists line up.
    convertedFuncInfo.torchInputTypes.append({fenceType, fenceType});
    ireeInputTypes.append({fenceType, fenceType});
    convertedFuncInfo.inputDispositions.append(
        {TypeDisposition::FENCE, TypeDisposition::FENCE});
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
    auto asyncFuncOp = IREE::Util::FuncOp::create(
        rewriter, torchFunc.getLoc(), asyncFunctionName, asyncFuncType,
        tiedOperandsAttr);
    convertedFuncInfo.funcOp = asyncFuncOp;
    asyncFuncOp.setSymVisibilityAttr(torchFunc.getSymVisibilityAttr());
    // Handle defacto attrs to specialized ones.
    asyncFuncOp.setInliningPolicyAttr(
        rewriter.getAttr<IREE::Util::InlineNeverAttr>());
    retainFunctionAttributes(torchFunc, asyncFuncOp);
    asyncFuncOp->setAttr("iree.abi.stub", rewriter.getUnitAttr());
    asyncFuncOp->setAttr("iree.abi.model",
                         rewriter.getStringAttr("coarse-fences"));
    if (auto affinityAttr = torchFunc->getAttr("iree.abi.affinity")) {
      asyncFuncOp->setAttr("iree.abi.affinity", affinityAttr);
    }
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

    if (isa<IREE::HAL::BufferType>(torchType)) {
      // Are there other situations where we might have buffer inputs?
      ireeType = torchType;
      disp = TypeDisposition::TRANSIENT_BUFFER;
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

} // namespace mlir::iree_compiler::TorchInput
