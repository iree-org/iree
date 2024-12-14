// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUSelectUKernels.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
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

static StringLiteral executableObjectsAttrName = "hal.executable.objects";

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

/// Returns the function name and attributes to use for a ukernel with given
/// `name` and `suffix` on the target described by `targetAttr`.
static IREE::GPU::UKernelSpecAttr
getUKernelSpec(StringRef name, StringRef suffix, MLIRContext *context,
               IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (isROCMBackend(targetAttr)) {
    auto nameAttr = StringAttr::get(
        context, llvm::formatv("iree_uk_amdgpu_{}_{}", name, suffix));
    auto defsAttr = DictionaryAttr::get(
        context, {{StringAttr::get(context, "vm.import.module"),
                   StringAttr::get(context, "rocm")}});
    return IREE::GPU::UKernelSpecAttr::get(context, nameAttr, defsAttr);
  }
  return {};
}

} // namespace

IREE::GPU::UKernelSpecAttr selectUKernelForArgmax(linalg::GenericOp op) {
  if (failed(isArgmaxOp(op))) {
    return {};
  }
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  const char ukernelName[] = "argmax";
  if (!hasUkernel(targetAttr, ukernelName)) {
    return {};
  }
  Value input = op.getDpsInputOperand(0)->get();
  auto inputType = cast<ShapedType>(input.getType());
  Value index = op.getDpsInitOperand(1)->get();
  auto indexType = cast<ShapedType>(index.getType());
  std::string suffix;
  llvm::raw_string_ostream(suffix)
      << inputType.getElementType() << indexType.getElementType();
  MLIRContext *context = op->getContext();
  IREE::GPU::UKernelSpecAttr ukernelSpec =
      getUKernelSpec(ukernelName, suffix, context, targetAttr);
  if (!ukernelSpec) {
    return {};
  }
  auto execTarget = IREE::HAL::ExecutableTargetAttr::lookup(op);
  ArrayAttr sourceExecutableObjects = lookUpExecutableObjects(op);
  IREE::HAL::ExecutableObjectAttr bitcodeObject = getUKernelBitcode(
      context, execTarget, sourceExecutableObjects, ukernelSpec.getName());
  if (!bitcodeObject) {
    return {};
  }
  op->setAttr(executableObjectsAttrName,
              ArrayAttr::get(context, bitcodeObject));
  return ukernelSpec;
}

} // namespace mlir::iree_compiler
