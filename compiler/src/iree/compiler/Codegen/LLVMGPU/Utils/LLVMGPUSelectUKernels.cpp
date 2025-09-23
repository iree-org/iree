// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUSelectUKernels.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
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

// Returns ukernel name and suffix for inner_tiled operations. Empty name = no
// ukernel.
static UKernelNameAndSuffix
getUKernelNameAndSuffixForInnerTiled(IREE::Codegen::InnerTiledOp op) {
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
  } else if (auto innerTiledOp = dyn_cast<IREE::Codegen::InnerTiledOp>(op)) {
    return getUKernelNameAndSuffixForInnerTiled(innerTiledOp);
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

// Returns an initial UKernelDescriptorAttr containing the ukernel name and
// ukernel kind. Returns nullptr if no ukernel.
static IREE::Codegen::UKernelDescriptorAttr
getInitialUKernelConfig(Operation *op) {
  MLIRContext *context = op->getContext();
  auto [name, suffix] = getUKernelNameAndSuffix(op);
  if (name.empty()) {
    return {};
  }
  auto execTarget = IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (!execTarget || !hasUkernel(execTarget.getConfiguration(), name)) {
    return {};
  }
  if (isROCMBackend(execTarget)) {
    auto nameAttr = StringAttr::get(
        context, llvm::formatv("iree_uk_amdgpu_{}_{}", name, suffix));
    return IREE::Codegen::UKernelDescriptorAttr::get(
        context, nameAttr, IREE::Codegen::UKernelArgumentKind::Bitcode);
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

static IREE::Codegen::UKernelDescriptorAttr
ensureUKernelBitcodeAndFinalizeConfig(
    Operation *op, IREE::Codegen::UKernelDescriptorAttr ukernelConfig) {
  if (!ukernelConfig) {
    return {};
  }
  MLIRContext *context = op->getContext();
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(op);
  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(context, target);
  if (!gpuTarget) {
    return {};
  }
  std::string filename =
      getBitcodeFilename(gpuTarget, ukernelConfig.getUkernelName());

  ArrayAttr sourceExecutableObjects = lookUpExecutableObjects(op);
  IREE::HAL::ExecutableObjectAttr bitcodeObject =
      getUKernelBitcode(context, target, sourceExecutableObjects, filename);
  if (!bitcodeObject) {
    return {};
  }
  op->setAttr(executableObjectsAttrName,
              ArrayAttr::get(context, bitcodeObject));
  return ukernelConfig;
}

} // namespace

IREE::Codegen::UKernelDescriptorAttr selectUKernel(Operation *op) {
  IREE::Codegen::UKernelDescriptorAttr initialConfig =
      getInitialUKernelConfig(op);
  return ensureUKernelBitcodeAndFinalizeConfig(op, initialConfig);
}

} // namespace mlir::iree_compiler
