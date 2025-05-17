// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUSelectUKernels.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Utils/EmbeddedDataDirectory.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/Interpreter.h" // Performs registration.
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler {

namespace {

struct UKernelNameAndSuffix {
  std::string name;
  std::string suffix;
};

// Returns ukernel name and suffix for argmax. Empty name = no ukernel.
static UKernelNameAndSuffix
getUKernelNameAndSuffixForArgmax(linalg::GenericOp op) {
  Value input = op.getDpsInputOperand(0)->get();
  auto inputType = cast<ShapedType>(input.getType());
  Value index = op.getDpsInitOperand(1)->get();
  auto indexType = cast<ShapedType>(index.getType());
  return {"argmax", llvm::formatv("{}{}", inputType.getElementType(),
                                  indexType.getElementType())};
}

// Returns ukernel name and suffix for multi_mma. Empty name = no ukernel.
static UKernelNameAndSuffix
getUKernelNameAndSuffixForMultiMma(IREE::GPU::MultiMmaOp op) {
  auto mma = dyn_cast<IREE::GPU::DataTiledMMAAttr>(op.getKind());
  if (!mma) {
    return {}; // Only handling DataTiledMMAAttr for now.
  }
  return {"multi_mma", stringifyMMAIntrinsic(mma.getIntrinsic()).lower()};
}

// Returns ukernel name and suffix for any op. Empty name = no ukernel.
static UKernelNameAndSuffix getUKernelNameAndSuffix(Operation *op) {
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    if (IREE::LinalgExt::isArgmaxOp(genericOp)) {
      return getUKernelNameAndSuffixForArgmax(genericOp);
    }
  } else if (auto multiMmaOp = dyn_cast<IREE::GPU::MultiMmaOp>(op)) {
    return getUKernelNameAndSuffixForMultiMma(multiMmaOp);
  }
  return {};
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

// Returns an initial UKernelConfigAttr containing the ukernel name and
// def_attrs. Does not yet contain bitcode-dependent fields such as shared
// memory size. Returns {} if no ukernel.
static IREE::GPU::UKernelConfigAttr getInitialUKernelConfig(Operation *op) {
  MLIRContext *context = op->getContext();
  auto [name, suffix] = getUKernelNameAndSuffix(op);
  if (name.empty()) {
    return {};
  }
  auto execTarget = IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (!hasUkernel(execTarget, name)) {
    return {};
  }
  if (isROCMBackend(execTarget)) {
    auto nameAttr = StringAttr::get(
        context, llvm::formatv("iree_uk_amdgpu_{}_{}", name, suffix));
    auto defsAttr = DictionaryAttr::get(
        context, {{StringAttr::get(context, "vm.import.module"),
                   StringAttr::get(context, "rocm")}});
    return IREE::GPU::UKernelConfigAttr::get(context, nameAttr, defsAttr,
                                             /*shared_memory_bytes=*/0);
  }
  return {};
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
  auto blob = HeapAsmResourceBlob::allocateAndCopyInferAlign(
      ArrayRef<char>(bitcode->data(), bitcode->size()));
  auto bitcodeDenseAttr = DenseI8ResourceElementsAttr::get(
      VectorType::get({static_cast<int64_t>(bitcode->size())},
                      IntegerType::get(context, 8)),
      filename, std::move(blob));
  return IREE::HAL::ExecutableObjectAttr::get(
      context, StringAttr::get(context, filename),
      cast<IREE::Util::SerializableAttrInterface>(bitcodeDenseAttr));
}

static constexpr char executableObjectsAttrName[] = "hal.executable.objects";

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

static std::string getBitcodeFilename(IREE::GPU::TargetAttr gpuTarget,
                                      StringRef name) {
  return llvm::formatv("{}.{}.bc", name, gpuTarget.getArch());
}

// Helper for getSharedMemoryBytes. Typical latency: 2 ms.
// Evaluates the shared memory size required by the multi_mma microkernel by
// interpreting a bitcode function with a specific name.
// On failure, an op warning is emitted and {} is returned.
static std::optional<int> expensivelyEvaluateSharedMemoryBytes(
    IREE::GPU::MultiMmaOp op, IREE::GPU::UKernelConfigAttr ukernelConfig,
    IREE::HAL::ExecutableObjectAttr bitcodeObject,
    IREE::GPU::TargetAttr gpuTarget) {
  auto mma = dyn_cast<IREE::GPU::DataTiledMMAAttr>(op.getKind());

  auto bitcodeData = bitcodeObject.getData();
  std::string buffer;
  buffer.resize(bitcodeData.getStorageSize());
  if (failed(bitcodeObject.getData().serializeToBuffer(
          op->getLoc(), llvm::endianness::native,
          ArrayRef<char>{buffer.data(), buffer.size()}))) {
    op.emitWarning("Failed to serialize bitcode.");
    return {};
  }
  llvm::LLVMContext llvmContext;
  llvm::Expected<std::unique_ptr<llvm::Module>> module =
      llvm::getLazyBitcodeModule(
          llvm::MemoryBufferRef{buffer, ukernelConfig.getName()}, llvmContext,
          /*ShouldLazyLoadMetadata=*/true);
  if (!module) {
    op.emitWarning("Failed to parse bitcode module.");
    return {};
  }
  llvm::EngineBuilder builder(std::move(module.get()));
  std::string builderError;
  builder.setEngineKind(llvm::EngineKind::Interpreter)
      .setErrorStr(&builderError);
  std::unique_ptr<llvm::ExecutionEngine> interpreter{builder.create()};
  if (!interpreter) {
    op.emitWarning("Failed to create the interpreter.");
    return {};
  }
  std::string queryFuncName =
      llvm::formatv("{}_query_shared_memory_bytes", ukernelConfig.getName());
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
  SmallVector<llvm::GenericValue> args{
      constI32(mma.getIntrinsicsM()), constI32(mma.getSubgroupsM()),
      constI32(mma.getIntrinsicsN()), constI32(mma.getSubgroupsN()),
      constI32(mma.getIntrinsicsK())};
  if (func->arg_size() != args.size()) {
    op.emitWarning(
        llvm::formatv("Bitcode function {} takes {} arguments. Expected {}.",
                      queryFuncName, func->arg_size(), args.size()));
    return {};
  }
  llvm::GenericValue interpreterResult = interpreter->runFunction(func, args);
  if (interpreter->hasError()) {
    op.emitWarning(llvm::formatv("Error while interpreting bitcode: {}.",
                                 interpreter->getErrorMessage()));
    return {};
  }
  int sharedMemoryBytes = interpreterResult.IntVal.getSExtValue();

  // Reject a ukernel that would consume too much shared memory, which we need
  // to save for other purposes. This threshold can always be adjusted but we
  // default to a low threshold to get an early signal.
  int maxSharedMemoryBytes = getSharedMemoryBytes(gpuTarget) / 4;
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
static std::optional<int>
getSharedMemoryBytes(IREE::GPU::MultiMmaOp op,
                     IREE::GPU::UKernelConfigAttr ukernelConfig,
                     IREE::HAL::ExecutableObjectAttr bitcodeObject,
                     IREE::GPU::TargetAttr gpuTarget) {
  auto mma = dyn_cast<IREE::GPU::DataTiledMMAAttr>(op.getKind());

  // We use the stringification of the attributes, rather than the
  // attributes themselves, as the key, to ensure it's self-contained and does
  // not contain pointers to other objects, such as a `MLIRContext*`, which
  // could go dangling.
  std::string key = llvm::formatv("mma = {}, gpuTarget = {}", mma, gpuTarget);

  struct CacheEntry {
    std::optional<int> sharedMemoryBytes;
    std::mutex mutex;
    bool evaluated = false;
  };

  // The cache and the mutex guarding it.
  // We store the CacheEntry's by pointers, so that we don't need to worry about
  // entryPtr being invalidated.
  static llvm::StringMap<std::unique_ptr<CacheEntry>> cache;
  static std::mutex cacheMutex;

  CacheEntry *entryPtr = nullptr;

  {
    // Critical section on `cacheMutex`. This is the only place where we
    // access `cache`. When we will later update a cache entry, that will be
    // through `entryPtr`, independently of `cache`.
    std::lock_guard<std::mutex> lock(cacheMutex);
    auto iter = cache.find(key);
    if (iter != cache.end()) {
      // Cache hit. Early return.
      return iter->second->sharedMemoryBytes;
    }
    // Cache miss. Create a new cache entry and acquire its mutex.
    entryPtr =
        cache.insert({key, std::make_unique<CacheEntry>()}).first->second.get();
    entryPtr->mutex.lock();
  }

  // If the entry still isn't evaluated after we have acquired its mutex,
  // perform the evaluation now.
  if (!entryPtr->evaluated) {
    entryPtr->sharedMemoryBytes = expensivelyEvaluateSharedMemoryBytes(
        op, ukernelConfig, bitcodeObject, gpuTarget);
    entryPtr->evaluated = true;
  }

  entryPtr->mutex.unlock();
  return entryPtr->sharedMemoryBytes;
}

// Returns the finalized UKernelConfigAttr to use for `op`, or {} if `op` should
// not use a ukernel.
static IREE::GPU::UKernelConfigAttr
finalizeConfig(IREE::GPU::MultiMmaOp op,
               IREE::GPU::UKernelConfigAttr ukernelConfig,
               IREE::HAL::ExecutableObjectAttr bitcodeObject,
               IREE::GPU::TargetAttr gpuTarget) {
  std::optional<int> sharedMemoryBytes =
      getSharedMemoryBytes(op, ukernelConfig, bitcodeObject, gpuTarget);
  if (!sharedMemoryBytes) {
    // Could not evaluate sharedMemoryBytes. Prevent the ukernel selection.
    return {};
  }
  return IREE::GPU::UKernelConfigAttr::get(
      op->getContext(), ukernelConfig.getName(), ukernelConfig.getDefAttrs(),
      *sharedMemoryBytes);
}

// Returns the finalized UKernelConfigAttr to use for `op`, or {} if `op` should
// not use a ukernel.
static IREE::GPU::UKernelConfigAttr
finalizeConfig(Operation *op, IREE::GPU::UKernelConfigAttr ukernelConfig,
               IREE::HAL::ExecutableObjectAttr bitcodeObject,
               IREE::GPU::TargetAttr gpuTarget) {
  if (auto multiMmaOp = dyn_cast<IREE::GPU::MultiMmaOp>(op)) {
    return finalizeConfig(multiMmaOp, ukernelConfig, bitcodeObject, gpuTarget);
  }
  return ukernelConfig;
}

// Ensures that the op has ukernel bitcode as a hal.executable.object, stored
// as a hal.executable.objects attribute on the op itself, ready to be hoisted
// by the HoistExecutableObjects pass, and returns the finalized config attr
// with the remaining bitcode-dependent fields populated.
// Returns {} if no bitcode was found for the configured ukernel, of if an error
// occurred trying to infer bitcode-dependent config fields (which may require
// interpreting bitcode).
static IREE::GPU::UKernelConfigAttr ensureUKernelBitcodeAndFinalizeConfig(
    Operation *op, IREE::GPU::UKernelConfigAttr ukernelConfig) {
  MLIRContext *context = op->getContext();
  if (!ukernelConfig) {
    return {};
  }
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(op);
  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(target);
  if (!gpuTarget) {
    return {};
  }
  std::string filename = getBitcodeFilename(gpuTarget, ukernelConfig.getName());

  ArrayAttr sourceExecutableObjects = lookUpExecutableObjects(op);
  IREE::HAL::ExecutableObjectAttr bitcodeObject =
      getUKernelBitcode(context, target, sourceExecutableObjects, filename);
  if (!bitcodeObject) {
    return {};
  }
  op->setAttr(executableObjectsAttrName,
              ArrayAttr::get(context, bitcodeObject));
  return finalizeConfig(op, ukernelConfig, bitcodeObject, gpuTarget);
}

} // namespace

IREE::GPU::UKernelConfigAttr selectUKernel(Operation *op) {
  IREE::GPU::UKernelConfigAttr initialConfig = getInitialUKernelConfig(op);
  return ensureUKernelBitcodeAndFinalizeConfig(op, initialConfig);
}

} // namespace mlir::iree_compiler
