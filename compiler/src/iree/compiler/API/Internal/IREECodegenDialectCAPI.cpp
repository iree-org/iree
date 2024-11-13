// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include <cstdint>
#include <optional>
#include <type_traits>
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/dialects/iree_codegen.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

using mlir::iree_compiler::IREE::Codegen::CompilationInfoAttr;
using mlir::iree_compiler::IREE::Codegen::DispatchLoweringPassPipeline;
using mlir::iree_compiler::IREE::Codegen::DispatchLoweringPassPipelineAttr;
using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttrInterface;
using mlir::iree_compiler::IREE::Codegen::TranslationInfoAttr;

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

bool ireeAttributeIsACodegenTranslationInfoAttr(MlirAttribute attr) {
  return llvm::isa<TranslationInfoAttr>(unwrap(attr));
}

MlirTypeID ireeCodegenTranslationInfoAttrGetTypeID() {
  return wrap(TranslationInfoAttr::getTypeID());
}

MlirAttribute ireeCodegenTranslationInfoAttrGet(
    MlirContext mlirCtx, ireeCodegenTranslationInfoParameters parameters) {
  assert(!mlirAttributeIsNull(parameters.passPipeline) &&
         ireeAttributeIsACodegenDispatchLoweringPassPipelineAttr(
             parameters.passPipeline) &&
         "Invalid pass pipeline attr");

  assert((mlirAttributeIsNull(parameters.codegenSpec) ||
          mlirAttributeIsASymbolRef(parameters.codegenSpec)) &&
         "Invalid codegen spec attr");

  assert((mlirAttributeIsNull(parameters.configuration) ||
          mlirAttributeIsADictionary(parameters.configuration)) &&
         "Invalid configuration attr");

  DispatchLoweringPassPipeline passPipeline =
      llvm::cast<DispatchLoweringPassPipelineAttr>(
          unwrap(parameters.passPipeline))
          .getValue();
  auto codegenSpec = llvm::cast_if_present<mlir::SymbolRefAttr>(
      unwrap(parameters.codegenSpec));

  llvm::ArrayRef<int64_t> workgroupSize;
  if (parameters.workgroupSize) {
    workgroupSize = {parameters.workgroupSize,
                     parameters.numWorkgroupSizeElements};
  }

  std::optional<int64_t> subgroupSize = parameters.subgroupSize;
  auto configuration = llvm::cast_if_present<mlir::DictionaryAttr>(
      unwrap(parameters.configuration));

  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(TranslationInfoAttr::get(ctx, passPipeline, codegenSpec,
                                       workgroupSize, subgroupSize,
                                       configuration));
}

ireeCodegenTranslationInfoParameters
ireeCodegenTranslationInfoAttrGetParameters(MlirAttribute attr) {
  auto translationInfo = llvm::cast<TranslationInfoAttr>(unwrap(attr));

  ireeCodegenTranslationInfoParameters parameters = {};
  parameters.passPipeline = wrap(translationInfo.getPassPipeline());
  parameters.codegenSpec = wrap(translationInfo.getCodegenSpec());
  llvm::ArrayRef<int64_t> workgroupSize = translationInfo.getWorkgroupSize();
  parameters.workgroupSize = workgroupSize.data();
  parameters.numWorkgroupSizeElements = workgroupSize.size();
  parameters.subgroupSize = translationInfo.getSubgroupSize();
  parameters.configuration = wrap(translationInfo.getConfiguration());

  return parameters;
}

bool ireeAttributeIsACodegenCompilationInfoAttr(MlirAttribute attr) {
  return llvm::isa<CompilationInfoAttr>(unwrap(attr));
}

MlirTypeID ireeCodegenCompilationInfoAttrGetTypeID() {
  return wrap(CompilationInfoAttr::getTypeID());
}

MlirAttribute ireeCodegenCompilationInfoAttrGet(
    MlirContext mlirCtx, ireeCodegenCompilationInfoParameters parameters) {
  assert(!mlirAttributeIsNull(parameters.loweringConfig) &&
         "Invalid lowering config attr");
  assert(
      !mlirAttributeIsNull(parameters.translationInfo) &&
      ireeAttributeIsACodegenTranslationInfoAttr(parameters.translationInfo) &&
      "Invalid translation info attr");

  auto loweringConfig = llvm::cast<LoweringConfigAttrInterface>(
      unwrap(parameters.loweringConfig));
  auto translationInfo =
      llvm::cast<TranslationInfoAttr>(unwrap(parameters.translationInfo));

  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(CompilationInfoAttr::get(ctx, loweringConfig, translationInfo));
}

ireeCodegenCompilationInfoParameters
ireeCodegenCompilationInfoAttrGetParameters(MlirAttribute attr) {
  auto compilationInfo = llvm::cast<CompilationInfoAttr>(unwrap(attr));
  ireeCodegenCompilationInfoParameters parameters = {};
  parameters.loweringConfig = wrap(compilationInfo.getLoweringConfig());
  parameters.translationInfo = wrap(compilationInfo.getTranslationInfo());
  return parameters;
}
