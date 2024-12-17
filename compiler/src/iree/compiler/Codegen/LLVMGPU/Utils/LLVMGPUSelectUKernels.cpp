// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUSelectUKernels.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Utils/EmbeddedDataDirectory.h"
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
  std::string suffix{
      stringifyMMAIntrinsic(mma.getIntrinsic().getValue()).lower()};
  if (mma.getUnrollM() != 1 || mma.getUnrollN() != 1 || mma.getUnrollK() != 1) {
    suffix += llvm::formatv("_unroll{}x{}x{}", mma.getUnrollM(),
                            mma.getUnrollN(), mma.getUnrollK());
  }
  if (mma.getSubgroupsM() != 1 || mma.getSubgroupsN() != 1) {
    suffix += llvm::formatv("_subgroups{}x{}", mma.getSubgroupsM(),
                            mma.getSubgroupsN());
  }
  return {"multi_mma", suffix};
}

// Returns ukernel name and suffix for any op. Empty name = no ukernel.
static UKernelNameAndSuffix getUKernelNameAndSuffix(Operation *op) {
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    if (succeeded(isArgmaxOp(genericOp))) {
      return getUKernelNameAndSuffixForArgmax(genericOp);
    }
  } else if (auto multiMmaOp = dyn_cast<IREE::GPU::MultiMmaOp>(op)) {
    return getUKernelNameAndSuffixForMultiMma(multiMmaOp);
  }
  return {};
}

// Returns the UKernelConfigAttr for any op. Returns {} if no ukernel.
static IREE::GPU::UKernelConfigAttr getUKernelConfig(Operation *op) {
  MLIRContext *context = op->getContext();
  auto [name, suffix] = getUKernelNameAndSuffix(op);
  if (name.empty()) {
    return {};
  }
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (!hasUkernel(target, name)) {
    return {};
  }
  if (isROCMBackend(target)) {
    auto nameAttr = StringAttr::get(
        context, llvm::formatv("iree_uk_amdgpu_{}_{}", name, suffix));
    auto defsAttr = DictionaryAttr::get(
        context, {{StringAttr::get(context, "vm.import.module"),
                   StringAttr::get(context, "rocm")}});
    return IREE::GPU::UKernelConfigAttr::get(context, nameAttr, defsAttr);
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
                  ArrayAttr sourceExecutableObjects, StringRef ukernelName) {
  IREE::GPU::TargetAttr gpuTarget = getGPUTargetAttr(execTarget);
  if (!gpuTarget) {
    return {};
  }
  StringRef gpuArch = gpuTarget.getArch();
  std::string bitcodeFilename = llvm::formatv("{}.{}.bc", ukernelName, gpuArch);

  // Early-return if the source executable.objects already contain an object
  // with the expected file name. This happens with user-provided bitcode in the
  // source IR.
  if (sourceExecutableObjects) {
    for (Attribute a : sourceExecutableObjects) {
      if (auto object = dyn_cast<IREE::HAL::ExecutableObjectAttr>(a)) {
        if (object.getPath() == bitcodeFilename) {
          return object;
        }
      }
    }
  }

  // No user-provided bitcode, so we search our embedded bitcode files in the
  // EmbeddedDataDirectory singleton.
  std::optional<StringRef> bitcode;
  EmbeddedDataDirectory::withGlobal([&](EmbeddedDataDirectory &dir) {
    bitcode = dir.getFile(bitcodeFilename);
  });
  if (!bitcode) {
    return {};
  }
  auto blob = HeapAsmResourceBlob::allocateAndCopyInferAlign(
      ArrayRef<char>(bitcode->data(), bitcode->size()));
  auto bitcodeDenseAttr = DenseI8ResourceElementsAttr::get(
      VectorType::get({static_cast<int64_t>(bitcode->size())},
                      IntegerType::get(context, 8)),
      bitcodeFilename, std::move(blob));
  return IREE::HAL::ExecutableObjectAttr::get(
      context, StringAttr::get(context, bitcodeFilename),
      cast<IREE::Util::SerializableAttrInterface>(bitcodeDenseAttr));
}

// Walks parents ops from `op` to return the nearest hal.executable.objects
// array attribute. If the parent hal.executable.variant is reached, its objects
// attribute is returned.
// Adapted from ExecutableTargetAttr::lookup.
static ArrayAttr lookUpExecutableObjects(Operation *op,
                                         StringRef executableObjectsAttrName) {
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

// Ensures that the op has ukernel bitcode as a hal.executable.object, stored
// as a hal.executable.objects attribute on the op itself, ready to be hoisted
// by the HoistExecutableObjects pass.
// Returns failure if no bitcode was found for the configured ukernel.
static LogicalResult
ensureUKernelBitcode(Operation *op,
                     IREE::GPU::UKernelConfigAttr ukernelConfig) {
  constexpr StringLiteral executableObjectsAttrName = "hal.executable.objects";
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(op);
  ArrayAttr sourceExecutableObjects =
      lookUpExecutableObjects(op, executableObjectsAttrName);
  MLIRContext *context = op->getContext();
  IREE::HAL::ExecutableObjectAttr bitcodeObject = getUKernelBitcode(
      context, target, sourceExecutableObjects, ukernelConfig.getName());
  if (!bitcodeObject) {
    return failure();
  }
  op->setAttr(executableObjectsAttrName,
              ArrayAttr::get(context, bitcodeObject));
  return success();
}

} // namespace

IREE::GPU::UKernelConfigAttr selectUKernel(Operation *op) {
  IREE::GPU::UKernelConfigAttr ukernelConfig = getUKernelConfig(op);
  if (!ukernelConfig) {
    return {};
  }
  if (failed(ensureUKernelBitcode(op, ukernelConfig))) {
    return {};
  }
  return ukernelConfig;
}

} // namespace mlir::iree_compiler
