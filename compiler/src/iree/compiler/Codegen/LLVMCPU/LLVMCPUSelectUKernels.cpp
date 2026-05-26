// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/LLVMCPUSelectUKernels.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

namespace mlir::iree_compiler {

namespace {

// Maps an `IREE::CPU::MMAIntrinsic` to the ukernel function/file name that
// implements it. The convention (see the README under
// compiler/plugins/target/LLVMCPU/builtins/ukernel/) is: the enum value,
// lowercased, with the `iree_uk_` prefix (replacing the `MMA_` in the
// enum). Returns empty string for `None` and for the type-polymorphic
// generic-scalar family.
static std::string getUkernelNameForIntrinsic(IREE::CPU::MMAIntrinsic intr) {
  StringRef enumName = IREE::CPU::stringifyMMAIntrinsic(intr);
  // Reject the `None` marker and the generic-scalar family — neither has
  // a corresponding ukernel.
  if (enumName == "None" || enumName.starts_with("MMA_GENERIC_")) {
    return {};
  }
  // Strip the `MMA_` prefix and lowercase the rest.
  if (!enumName.consume_front("MMA_")) {
    return {};
  }
  std::string out = "iree_uk_mma_";
  out.reserve(out.size() + enumName.size());
  for (char c : enumName) {
    out.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

} // namespace

IREE::Codegen::UKernelDescriptorAttr selectCPUUKernel(Operation *op) {
  auto innerTiled = dyn_cast<IREE::Codegen::InnerTiledOp>(op);
  if (!innerTiled) {
    return {};
  }
  auto mma = dyn_cast<IREE::CPU::DataTiledMMAAttr>(innerTiled.getKind());
  if (!mma) {
    return {};
  }

  auto execTarget = IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (!execTarget) {
    return {};
  }
  DictionaryAttr config = execTarget.getConfiguration();
  if (!hasLlvmUkernel(config, "inner_tiled")) {
    return {};
  }

  std::string name = getUkernelNameForIntrinsic(mma.getIntrinsic());
  if (name.empty()) {
    return {};
  }

  // Resolve and attach the bitcode on the op now (at kernel-config time),
  // so it survives all the way down to `LowerUKernelOpsToCalls` as a
  // discardable attribute. Mirrors the GPU side
  // (`ensureUKernelBitcodeAndFinalizeConfig` in
  // `compiler/src/iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUSelectUKernels.cpp`)
  // and makes the configuration-pass output self-contained for lit tests.
  IREE::CPU::attachUKernelBitcodeOnOp(op, name);

  MLIRContext *context = op->getContext();
  return IREE::Codegen::UKernelDescriptorAttr::get(
      context, StringAttr::get(context, name),
      IREE::Codegen::UKernelArgumentKind::Bitcode);
}

} // namespace mlir::iree_compiler
