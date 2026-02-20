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
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/dialects/iree_codegen.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

using mlir::iree_compiler::IREE::Codegen::CompilationInfoAttr;
using mlir::iree_compiler::IREE::Codegen::DispatchLoweringPassPipeline;
using mlir::iree_compiler::IREE::Codegen::DispatchLoweringPassPipelineAttr;
using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttrInterface;
using mlir::iree_compiler::IREE::Codegen::RootOpAttr;
using mlir::iree_compiler::IREE::Codegen::TranslationInfoAttr;
using mlir::iree_compiler::IREE::HAL::ExecutableVariantOp;

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

bool ireeAttributeIsACodegenRootOpAttr(MlirAttribute attr) {
  return llvm::isa<RootOpAttr>(unwrap(attr));
}

MlirTypeID ireeCodegenRootOpAttrGetTypeID() {
  return wrap(RootOpAttr::getTypeID());
}

MlirAttribute ireeCodegenRootOpAttrGet(MlirContext mlirCtx, int64_t set) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  return wrap(RootOpAttr::get(ctx, set));
}

int64_t ireeCodegenRootOpAttrGetSet(MlirAttribute attr) {
  return llvm::cast<RootOpAttr>(unwrap(attr)).getSet();
}

void ireeCodegenGetExecutableVariantOps(MlirModule module, size_t *numOps,
                                        MlirOperation *executableOps) {
  assert(!mlirModuleIsNull(module) && "module cannot be nullptr");
  assert(numOps && "numOps cannot be nullptr");

  mlir::ModuleOp moduleOp = unwrap(module);
  llvm::SmallVector<ExecutableVariantOp> executableVariantOps =
      mlir::iree_compiler::getExecutableVariantOps(moduleOp);

  if (!executableOps) {
    *numOps = executableVariantOps.size();
    return;
  }

  assert(
      *numOps == executableVariantOps.size() &&
      "*numOps must match the number of elements in the executableVariantOps");

  for (size_t i = 0, e = executableVariantOps.size(); i < e; ++i) {
    executableOps[i] = wrap(executableVariantOps[i]);
  }
}

void ireeCodegenGetTunerRootOps(MlirModule module, size_t *numOps,
                                MlirOperation *rootOps) {
  assert(!mlirModuleIsNull(module) && "module cannot be nullptr");
  assert(numOps && "numOps cannot be nullptr");

  mlir::ModuleOp moduleOp = unwrap(module);
  llvm::SmallVector<mlir::Operation *> tunerRootOps =
      mlir::iree_compiler::getTunerRootOps(moduleOp);

  if (!rootOps) {
    *numOps = tunerRootOps.size();
    return;
  }

  assert(*numOps == tunerRootOps.size() &&
         "*numOps must match the number of elements in the rootOps");

  for (size_t i = 0, e = tunerRootOps.size(); i < e; ++i) {
    rootOps[i] = wrap(tunerRootOps[i]);
  }
}

ireeCodegenAttentionOpDetail
ireeCodegenGetAttentionOpDetail(MlirAffineMap qMap, MlirAffineMap kMap,
                                MlirAffineMap vMap, MlirAffineMap oMap) {
  mlir::AffineMap QMap = unwrap(qMap);
  mlir::AffineMap KMap = unwrap(kMap);
  mlir::AffineMap VMap = unwrap(vMap);
  mlir::AffineMap OMap = unwrap(oMap);

  llvm::FailureOr<mlir::iree_compiler::IREE::LinalgExt::AttentionOpDetail>
      maybeDetail =
          mlir::iree_compiler::IREE::LinalgExt::AttentionOpDetail::get(
              QMap, KMap, VMap, OMap);

  if (failed(maybeDetail)) {
    return ireeCodegenAttentionOpDetail{/*batch=*/wrap(mlir::Attribute()),
                                        /*m=*/wrap(mlir::Attribute()),
                                        /*k1=*/wrap(mlir::Attribute()),
                                        /*k2=*/wrap(mlir::Attribute()),
                                        /*n=*/wrap(mlir::Attribute()),
                                        /*domainRank=*/-1};
  }

  const mlir::iree_compiler::IREE::LinalgExt::AttentionOpDetail &opInfo =
      *maybeDetail;

  mlir::Builder builder(QMap.getContext());

  ireeCodegenAttentionOpDetail result;
  result.batch = wrap(builder.getI64ArrayAttr(opInfo.getBatchDims()));
  result.m = wrap(builder.getI64ArrayAttr(opInfo.getMDims()));
  result.k1 = wrap(builder.getI64ArrayAttr(opInfo.getK1Dims()));
  result.k2 = wrap(builder.getI64ArrayAttr(opInfo.getK2Dims()));
  result.n = wrap(builder.getI64ArrayAttr(opInfo.getNDims()));
  result.domainRank = opInfo.getDomainRank();

  return result;
}

bool ireeCodegenMlirOperationIsACodegenAttentionOp(MlirOperation op) {
  return llvm::isa<mlir::iree_compiler::IREE::LinalgExt::AttentionOp>(
      unwrap(op));
}

bool ireeCodegenHasIGEMMGenericConvDetails(MlirOperation op) {
  auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(unwrap(op));
  if (!linalgOp) {
    return false;
  }

  return succeeded(
      mlir::iree_compiler::IREE::LinalgExt::getIGEMMGenericConvDetails(
          linalgOp));
}

ireeCodegenIGEMMGenericConvDetails
ireeCodegenGetIGEMMGenericConvDetails(MlirOperation op) {
  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(unwrap(op));

  llvm::FailureOr<mlir::iree_compiler::IREE::LinalgExt::IGEMMGenericConvDetails>
      maybeDetails =
          mlir::iree_compiler::IREE::LinalgExt::getIGEMMGenericConvDetails(
              linalgOp);
  assert(succeeded(maybeDetails) &&
         "Failed to get IGEMM details; must check with "
         "ireeCodegenHasIGEMMGenericConvDetails first");

  const mlir::iree_compiler::IREE::LinalgExt::IGEMMGenericConvDetails &details =
      *maybeDetails;

  mlir::Builder builder(linalgOp.getContext());

  ireeCodegenIGEMMGenericConvDetails result;

  result.igemmContractionMaps = wrap(builder.getArrayAttr(llvm::map_to_vector(
      details.igemmContractionMaps, [](auto map) -> mlir::Attribute {
        return mlir::AffineMapAttr::get(map);
      })));

  result.igemmLoopBounds =
      wrap(builder.getI64ArrayAttr(details.igemmLoopBounds));

  llvm::SmallVector<mlir::Attribute> iteratorAttrs;
  for (mlir::utils::IteratorType iterType : details.igemmLoopIterators) {
    iteratorAttrs.push_back(
        builder.getStringAttr(mlir::utils::stringifyIteratorType(iterType)));
  }
  result.igemmLoopIterators = wrap(builder.getArrayAttr(iteratorAttrs));

  result.im2colOutputPerm =
      wrap(builder.getI64ArrayAttr(details.im2colOutputPerm));

  llvm::SmallVector<mlir::Attribute> reassocAttrs;
  for (const mlir::ReassociationIndices &indices :
       details.filterReassocIndices) {
    reassocAttrs.push_back(builder.getI64ArrayAttr(
        llvm::map_to_vector(indices, llvm::StaticCastTo<int64_t>)));
  }
  result.filterReassocIndices = wrap(builder.getArrayAttr(reassocAttrs));

  result.isOutputChannelFirst = details.isOutputChannelFirst;

  // Mapping from conv dimensions to IGEMM dimensions.
  // Encode as ArrayAttr of [conv_dim, igemm_dim] pairs.
  llvm::SmallVector<mlir::Attribute> dimMapAttrs;
  for (const auto &[convDim, igemmExpr] : details.convToIgemmDimMap) {
    // All entries in convToIgemmDimMap must be AffineDimExpr.
    auto dimExpr = llvm::cast<mlir::AffineDimExpr>(igemmExpr);
    llvm::SmallVector<mlir::Attribute> pairAttrs;
    pairAttrs.push_back(builder.getI64IntegerAttr(convDim));
    pairAttrs.push_back(builder.getI64IntegerAttr(dimExpr.getPosition()));
    dimMapAttrs.push_back(builder.getArrayAttr(pairAttrs));
  }
  result.convToIgemmDimMap = wrap(builder.getArrayAttr(dimMapAttrs));

  return result;
}

bool ireeCodegenMlirOperationIsAScaledContractionOp(MlirOperation op) {
  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(unwrap(op));
  return mlir::iree_compiler::IREE::LinalgExt::isaScaledContractionOpInterface(
      linalgOp);
}

ireeCodegenScaledContractionDimensions
ireeCodegenInferScaledContractionDimensions(MlirOperation op) {
  ireeCodegenScaledContractionDimensions result{};
  auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(unwrap(op));
  if (!linalgOp) {
    return result;
  }

  llvm::FailureOr<
      mlir::iree_compiler::IREE::LinalgExt::ScaledContractionDimensions>
      maybeDims =
          mlir::iree_compiler::IREE::LinalgExt::inferScaledContractionDims(
              linalgOp);
  if (failed(maybeDims)) {
    return result;
  }

  const mlir::iree_compiler::IREE::LinalgExt::ScaledContractionDimensions
      &scaledContractionDims = *maybeDims;
  mlir::MLIRContext *ctx = linalgOp.getContext();
  mlir::Builder b(ctx);
  auto toAttr = [&b](llvm::ArrayRef<unsigned> vals) -> MlirAttribute {
    llvm::SmallVector<mlir::Attribute, 2> attrs =
        llvm::map_to_vector(vals, [&b](unsigned val) -> mlir::Attribute {
          return b.getI32IntegerAttr(val);
        });
    return wrap(b.getArrayAttr(attrs));
  };

  result.batch = toAttr(scaledContractionDims.batch);
  result.m = toAttr(scaledContractionDims.m);
  result.n = toAttr(scaledContractionDims.n);
  result.k = toAttr(scaledContractionDims.k);
  result.kB = toAttr(scaledContractionDims.kB);
  return result;
}
