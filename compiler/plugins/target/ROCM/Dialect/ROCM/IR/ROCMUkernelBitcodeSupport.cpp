// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMUkernelBitcodeSupport.h"
#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPUTileSwizzleUtils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AsmState.h"

namespace mlir::iree_compiler::IREE::ROCM {

//===----------------------------------------------------------------------===//
// Utilities for bitcode ukernel
//===----------------------------------------------------------------------===//

constexpr char executableObjectsAttrName[] = "hal.executable.objects";

// Walks parents ops from `op` to return the nearest hal.executable.objects
// array attribute. If the parent hal.executable.variant is reached, its objects
// attribute is returned.
// Adapted from ExecutableTargetAttr::lookup.
static ArrayAttr lookUpExecutableObjects(Operation *op) {
  MLIRContext *context = op->getContext();
  auto attrId = StringAttr::get(context, executableObjectsAttrName);
  while (op) {
    // Take directly from the enclosing variant.
    if (auto variantOp = dyn_cast<IREE::HAL::ExecutableVariantOp>(op)) {
      if (std::optional<ArrayAttr> objects = variantOp.getObjects()) {
        return *objects;
      }
    }
    // Take from op attributes.
    if (auto attr = op->getAttrOfType<ArrayAttr>(attrId)) {
      return attr;
    }
    // Continue walk.
    op = op->getParentOp();
  }
  return {};
}

static Value createSharedMemory(RewriterBase &rewriter, Location loc,
                                int64_t sharedMemoryBytes) {
  RankedTensorType tensorType =
      RankedTensorType::get({sharedMemoryBytes}, rewriter.getI8Type());
  ValueRange dynSizes{};
  if (!sharedMemoryBytes) {
    IREE::Codegen::NullPointerType nullPointerType =
        IREE::Codegen::NullPointerType::get(rewriter.getContext());
    return IREE::Codegen::NullPointerOp::create(rewriter, loc, nullPointerType);
  }
  auto allocOp =
      bufferization::AllocTensorOp::create(rewriter, loc, tensorType, dynSizes);
  Attribute sharedAddrSpace = gpu::AddressSpaceAttr::get(
      rewriter.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  allocOp.setMemorySpaceAttr(sharedAddrSpace);
  return allocOp;
}

// Returns the index of the innermost CrossIntrinsic dimension of the C matrix,
// if it is static, and std::nullopt if it is dynamic or if there are no
// CrossIntrinsic dims.
static std::optional<unsigned>
getCInnermostStaticCrossIntrinsicDim(IREE::Codegen::InnerTiledOp op) {
  auto outputType = dyn_cast<ShapedType>(op.getResultTypes()[0]);
  if (!outputType) {
    return std::nullopt;
  }
  auto mma = cast<IREE::GPU::DataTiledMMAAttr>(op.getKind());
  IREE::Codegen::TileSwizzle accSwizzle =
      getSwizzle(mma, IREE::GPU::MMAFragment::Acc);
  SmallVector<IREE::Codegen::TileSwizzle::Dim> swizzleDims;
  for (IREE::Codegen::TileSwizzle::ExpandShapeDimVectorType group :
       accSwizzle.expandShape) {
    swizzleDims.append(group);
  }
  applyPermutationToVector(swizzleDims, accSwizzle.permutation);
  int rankDiff = outputType.getRank() - swizzleDims.size();
  auto crossIntrinsic = IREE::Codegen::TileSwizzle::Dim::Kind::CrossIntrinsic;
  for (size_t e = swizzleDims.size(), swizzleIdx = e - 1; swizzleIdx < e;
       --swizzleIdx) {
    if (swizzleDims[swizzleIdx].kind != crossIntrinsic) {
      continue;
    }
    int outputIdx = swizzleIdx + rankDiff;
    if (outputType.isDynamicDim(outputIdx)) {
      return std::nullopt;
    }
    return outputIdx;
  }
  return std::nullopt;
}

static int64_t getSharedMemoryBytes(IREE::GPU::TargetAttr gpuTarget) {
  if (!gpuTarget) {
    return 0;
  }
  IREE::GPU::TargetWgpAttr wgp = gpuTarget.getWgp();
  if (!wgp) {
    return 0;
  }
  return wgp.getMaxWorkgroupMemoryBytes();
}

// Returns a ExecutableObjectAttr carrying the bitcode for the given ukernel.
//
// First tries finding the bitcode in the input `sourceExecutableObjects`, which
// must be an array of ExecutableObjectAttr's and is typically coming from a
// hal.executable.objects array attribute in the source IR, which is the
// mechanism by which source programs may provide their own ukernel bitcode.
//
// If no matching bitcode was found in `sourceExecutableObjects`, this function
// will then search in bitcode files that we have embedded as static data.
static IREE::HAL::ExecutableObjectAttr
getUKernelBitcode(MLIRContext *context,
                  IREE::HAL::ExecutableTargetAttr execTarget,
                  ArrayAttr sourceExecutableObjects, StringRef filename) {
  // Early-return if the source executable.objects already contain an object
  // with the expected file name. This happens with user-provided bitcode in the
  // source IR.
  if (sourceExecutableObjects) {
    for (Attribute a : sourceExecutableObjects) {
      if (auto object = dyn_cast<IREE::HAL::ExecutableObjectAttr>(a)) {
        if (object.getPath() == filename) {
          return object;
        }
      }
    }
  }

  // No user-provided bitcode, so we search our embedded bitcode files in the
  // EmbeddedDataDirectory singleton.
  std::optional<StringRef> bitcode;
  EmbeddedDataDirectory::withGlobal(
      [&](EmbeddedDataDirectory &dir) { bitcode = dir.getFile(filename); });
  if (!bitcode) {
    return {};
  }
  AsmResourceBlob blob = HeapAsmResourceBlob::allocateAndCopyInferAlign(
      ArrayRef<char>(bitcode->data(), bitcode->size()));
  auto bitcodeDenseAttr = DenseI8ResourceElementsAttr::get(
      VectorType::get({static_cast<int64_t>(bitcode->size())},
                      IntegerType::get(context, 8)),
      filename, std::move(blob));
  return IREE::HAL::ExecutableObjectAttr::get(
      context, StringAttr::get(context, filename),
      cast<IREE::Util::SerializableAttrInterface>(bitcodeDenseAttr));
}

static std::string getBitcodeFilename(IREE::GPU::TargetAttr gpuTarget,
                                      StringRef name) {
  return llvm::formatv("{}.{}.bc", name, gpuTarget.getArch());
}

// Helper for getSharedMemoryBytes. Typical latency: 2 ms.
// Evaluates the shared memory size required by the multi_mma microkernel by
// interpreting a bitcode function with a specific name.
// On failure, an op warning is emitted and {} is returned.
static std::optional<int64_t> expensivelyEvaluateSharedMemoryBytes(
    MLIRContext *context, IREE::Codegen::InnerTiledOp op, StringRef ukernelName,
    IREE::GPU::TargetAttr gpuTarget) {
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(op);
  std::string filename = getBitcodeFilename(gpuTarget, ukernelName);
  ArrayAttr sourceExecutableObjects = lookUpExecutableObjects(op);
  IREE::HAL::ExecutableObjectAttr bitcodeObject =
      getUKernelBitcode(context, target, sourceExecutableObjects, filename);

  auto mma = dyn_cast<IREE::GPU::DataTiledMMAAttr>(op.getKind());

  IREE::Util::SerializableAttrInterface bitcodeData = bitcodeObject.getData();
  std::string buffer;
  buffer.resize(bitcodeData.getStorageSize());
  if (failed(bitcodeObject.getData().serializeToBuffer(
          op->getLoc(), llvm::endianness::native,
          ArrayRef<char>{buffer.data(), buffer.size()}))) {
    op.emitWarning("Failed to serialize bitcode.");
    return {};
  }
  llvm::LLVMContext llvmContext;
  llvm::Expected<std::unique_ptr<llvm::Module>> moduleOp =
      llvm::getLazyBitcodeModule(llvm::MemoryBufferRef{buffer, ukernelName},
                                 llvmContext,
                                 /*ShouldLazyLoadMetadata=*/true);
  if (!moduleOp) {
    op.emitWarning("Failed to parse bitcode module.");
    return {};
  }
  llvm::EngineBuilder builder(std::move(moduleOp.get()));
  std::string builderError;
  builder.setEngineKind(llvm::EngineKind::Interpreter)
      .setErrorStr(&builderError);
  std::unique_ptr<llvm::ExecutionEngine> interpreter{builder.create()};
  if (!interpreter) {
    op.emitWarning("Failed to create the interpreter.");
    return {};
  }
  std::string queryFuncName =
      llvm::formatv("{}_query_shared_memory_bytes", ukernelName);
  llvm::Function *func = interpreter->FindFunctionNamed(queryFuncName);
  if (!func) {
    op.emitWarning(llvm::formatv(
        "Bitcode does not contain a function named {}.", queryFuncName));
    return {};
  }
  auto constI32 = [](int32_t val) {
    llvm::GenericValue v;
    v.IntVal = APInt(32, val);
    return v;
  };
  llvm::GenericValue args[] = {
      constI32(mma.getIntrinsicsM()), constI32(mma.getSubgroupsM()),
      constI32(mma.getIntrinsicsN()), constI32(mma.getSubgroupsN()),
      constI32(mma.getIntrinsicsK())};
  if (func->arg_size() != /*total elements in 'args'=*/5) {
    op.emitWarning(llvm::formatv(
        "Bitcode function {} takes {} arguments. Expected {}.", queryFuncName,
        func->arg_size(), /*total elements in 'args'=*/5));
    return {};
  }
  llvm::GenericValue interpreterResult = interpreter->runFunction(func, args);
  if (interpreter->hasError()) {
    op.emitWarning(llvm::formatv("Error while interpreting bitcode: {}.",
                                 interpreter->getErrorMessage()));
    return {};
  }
  int64_t sharedMemoryBytes = interpreterResult.IntVal.getSExtValue();

  // Reject a ukernel that would consume too much shared memory, which we need
  // to save for other purposes. This threshold can always be adjusted but we
  // default to a low threshold to get an early signal.
  int64_t maxSharedMemoryBytes = getSharedMemoryBytes(gpuTarget) / 4;
  if (sharedMemoryBytes > maxSharedMemoryBytes) {
    op.emitWarning(llvm::formatv("The shared memory size {} required by the "
                                 "ukernel exceeds the maximum allowed size {}.",
                                 sharedMemoryBytes, maxSharedMemoryBytes));
    return {};
  }
  return sharedMemoryBytes;
}

// Returns the shared memory size required by the multi_mma ukernel.
// On failure, an op warning is emitted and {} is returned.
// Uses a static cache to avoid calling expensivelyEvaluateSharedMemoryBytes
// more than once per DataTiledMMAAttr value.
static std::optional<int64_t>
getSharedMemoryBytes(MLIRContext *context, IREE::Codegen::InnerTiledOp op,
                     StringRef ukernelName,
                     DictionaryAttr targetConfiguration) {
  auto mma = dyn_cast<IREE::GPU::DataTiledMMAAttr>(op.getKind());
  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(targetConfiguration);
  if (!gpuTarget) {
    return {};
  }
  // We use the stringification of the attributes, rather than the
  // attributes themselves, as the key, to ensure it's self-contained and does
  // not contain pointers to other objects, such as a `MLIRContext*`, which
  // could go dangling.
  std::string key = llvm::formatv("mma = {}, gpuTarget = {}", mma, gpuTarget);

  // The cache and the mutex guarding it.
  static llvm::StringMap<std::optional<int64_t>> cache;
  static std::mutex cacheMutex;
  std::lock_guard<std::mutex> lock(cacheMutex);

  auto iter = cache.find(key);
  if (iter != cache.end()) {
    // Cache hit. Early return.
    return iter->second;
  }
  // Cache miss. Do the expensive evaluation and insert into cache.
  auto sharedMemoryBytes =
      expensivelyEvaluateSharedMemoryBytes(context, op, ukernelName, gpuTarget);
  cache.insert({key, sharedMemoryBytes});

  return sharedMemoryBytes;
}

//===----------------------------------------------------------------------===//
// Argmax Ukernel
//===----------------------------------------------------------------------===//

/// Utility function to help create and replace argmax linalg with a ukernel.
LogicalResult handleArgmaxUkernel(RewriterBase &rewriter, StringRef name,
                                  DictionaryAttr targetConfiguration,
                                  Operation *contextualOp,
                                  ArrayRef<Value> inputs,
                                  ArrayRef<Value> outputs,
                                  SmallVectorImpl<Value> &otherOperands) {
  auto genericOp = dyn_cast<linalg::GenericOp>(contextualOp);
  if (!genericOp) {
    return rewriter.notifyMatchFailure(
        genericOp, "expected a linalg.generic op for argmax");
  }
  // Currently only support 1D reduction, where reduction is on fastest dim.
  // Tiling argmax ukernel is also set to enforce this structure.
  const int kReductionDim = genericOp.getNumLoops() - 1;
  Location loc = genericOp.getLoc();
  Value reductionDimSize = tensor::DimOp::create(
      rewriter, loc, genericOp.getDpsInputOperand(0)->get(), kReductionDim);
  // `returnsMaxValue` differentiates between the two argmax versions :-
  // 1. Returns only the index of the max value (returnsMaxValue == true)
  // 2. Returns both the max value as well as the corresponding index.
  bool returnsMaxValue = genericOp.getResults()[0].use_empty();
  Value writeMaxValueFlag =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI1Type(),
                                rewriter.getBoolAttr(!returnsMaxValue));
  llvm::append_values(otherOperands, reductionDimSize, writeMaxValueFlag);
  MLIRContext *context = rewriter.getContext();
  auto fnDefAttrs = DictionaryAttr::get(
      context, {{"vm.import.module", StringAttr::get(context, "rocm")}});
  auto ukernelOp = IREE::Codegen::UKernelGenericOp::create(
      rewriter, loc, contextualOp->getResultTypes(), name, inputs, outputs,
      otherOperands, fnDefAttrs, /*num_strided_outer_dims=*/0);
  if (returnsMaxValue) {
    rewriter.replaceAllUsesWith(genericOp.getResults()[1],
                                ukernelOp.getResults()[1]);
    return success();
  }
  ResultRange origResults = genericOp.getResults();
  ResultRange newResults = ukernelOp.getResults();
  if (origResults.size() != newResults.size()) {
    return rewriter.notifyMatchFailure(genericOp, "result count mismatch");
  }
  rewriter.replaceAllUsesWith(genericOp.getResults()[0],
                              ukernelOp.getResults()[0]);
  rewriter.replaceAllUsesWith(genericOp.getResults()[1],
                              ukernelOp.getResults()[1]);
  return success();
}

//===----------------------------------------------------------------------===//
// Inner Tiled Mma Ukernel
//===----------------------------------------------------------------------===//

/// Utility function to help create and replace inner_tiled with a ukernel.
LogicalResult handleInnerTiledMmaUkernel(
    RewriterBase &rewriter, StringRef name, DictionaryAttr targetConfiguration,
    Operation *contextualOp, ArrayRef<Value> inputs, ArrayRef<Value> outputs,
    SmallVectorImpl<Value> &otherOperands) {
  auto op = dyn_cast<IREE::Codegen::InnerTiledOp>(contextualOp);
  if (!op) {
    return rewriter.notifyMatchFailure(
        contextualOp, "expected a codegen.inner_tiled op for multi_mma");
  }
  auto mma = dyn_cast<IREE::GPU::DataTiledMMAAttr>(op.getKind());
  if (!mma) {
    return rewriter.notifyMatchFailure(op, "unhandled MMAInterfaceAttr");
  }
  std::optional<int64_t> innerCrossIntrinsicDim =
      getCInnermostStaticCrossIntrinsicDim(op);
  if (!innerCrossIntrinsicDim) {
    return rewriter.notifyMatchFailure(
        op, "inner cross-intrinsic dim is dynamic or not found");
  }
  Location loc = op->getLoc();
  Type I32Type = rewriter.getI32Type();
  auto castIndexToI32 = [&](Value val) {
    return arith::IndexCastOp::create(rewriter, loc, I32Type, val);
  };
  auto constI32 = [&](int val) {
    return arith::ConstantIntOp::create(rewriter, loc, I32Type, val);
  };
  MLIRContext *context = rewriter.getContext();
  std::optional<int64_t> maybeSharedMemoryBytes =
      getSharedMemoryBytes(context, op, name, targetConfiguration);
  int64_t sharedMemoryBytes =
      (!maybeSharedMemoryBytes) ? 0 : maybeSharedMemoryBytes.value();
  Value sharedMemory = createSharedMemory(rewriter, loc, sharedMemoryBytes);
  Value k = castIndexToI32(
      tensor::DimOp::create(rewriter, op.getLoc(), op.getInputs()[0], 1));
  Value intrinsicsM = constI32(mma.getIntrinsicsM());
  Value subgroupsM = constI32(mma.getSubgroupsM());
  Value intrinsicsN = constI32(mma.getIntrinsicsN());
  Value subgroupsN = constI32(mma.getSubgroupsN());
  Value intrinsicsK = constI32(mma.getIntrinsicsK());
  // There are 3 shaped input/output operands (A/B/C matrices).
  SmallVector<SmallVector<int64_t>> stridedDims(3, {});
  // Only the C matrix gets strides, and we pass the stride of the innermost
  // CrossIntrinsic dim, because the ukernel needs to know where to store the
  // result vector from each unrolled intrinsic. Offsets into all other
  // dimensions are handled by the compiler, and passed as part of the base
  // pointer + offset. The A and B matrices don't get strides, because we
  // expect them to always be passed as global memory pointers, and the
  // strides can be inferred by the ukernel implementation.
  stridedDims[2].push_back(innerCrossIntrinsicDim.value());
  // The only additional shaped operand is the shared memory buffer. Only
  // create a stride list for it if we have shared memory. Otherwise, the
  // operand is an iree_codegen.null_pointer op.
  if (sharedMemoryBytes != 0) {
    // Shared memory does not need strides.
    stridedDims.push_back({});
  }
  auto fnDefAttrs = DictionaryAttr::get(
      context, {{"vm.import.module", StringAttr::get(context, "rocm")}});
  rewriter.replaceOpWithNewOp<IREE::Codegen::UKernelGenericOp>(
      op, op.getOutputs().getTypes(), name, inputs, outputs,
      ValueRange{sharedMemory, constI32(sharedMemoryBytes), k, intrinsicsM,
                 subgroupsM, intrinsicsN, subgroupsN, intrinsicsK},
      fnDefAttrs, stridedDims);
  return success();
}

} // namespace mlir::iree_compiler::IREE::ROCM
