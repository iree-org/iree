// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <type_traits>
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/dialects/iree_codegen.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

using mlir::iree_compiler::IREE::Codegen::DispatchLoweringPassPipeline;
using mlir::iree_compiler::IREE::Codegen::DispatchLoweringPassPipelineAttr;

bool ireeAttributeIsACodegenDispatchLoweringPassPipelineAttr(
    MlirAttribute attr) {
  return llvm::isa<DispatchLoweringPassPipelineAttr>(unwrap(attr));
}

MlirTypeID ireeCodegenDispatchLoweringPassPipelineAttrGetTypeID() {
  return wrap(DispatchLoweringPassPipelineAttr::getTypeID());
}

static_assert(
    std::is_same_v<uint32_t,
                   std::underlying_type_t<DispatchLoweringPassPipeline>>,
    "Enum type changed");

MlirAttribute
ireeCodegenDispatchLoweringPassPipelineAttrGet(MlirContext mlirCtx,
                                               uint32_t value) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(DispatchLoweringPassPipelineAttr::get(
      ctx, static_cast<DispatchLoweringPassPipeline>(value)));
}

uint32_t
ireeCodegenDispatchLoweringPassPipelineAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<DispatchLoweringPassPipelineAttr>(unwrap(attr)).getValue());
}
