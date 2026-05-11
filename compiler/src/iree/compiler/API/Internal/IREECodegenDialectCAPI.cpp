// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include <cstdint>
#include <optional>

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/dialects/iree_codegen.h"
#include "iree/compiler/dialects/iree_gpu.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
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
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"

using mlir::iree_compiler::IREE::Codegen::CompilationInfoAttr;
using mlir::iree_compiler::IREE::Codegen::ConstraintsOp;
using mlir::iree_compiler::IREE::Codegen::IntKnobAttr;
using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttrInterface;
using mlir::iree_compiler::IREE::Codegen::NoPipelineAttr;
using mlir::iree_compiler::IREE::Codegen::OneOfKnobAttr;
using mlir::iree_compiler::IREE::Codegen::PassPipelineAttr;
using mlir::iree_compiler::IREE::Codegen::PipelineAttrInterface;
using mlir::iree_compiler::IREE::Codegen::RootOpAttr;
using mlir::iree_compiler::IREE::Codegen::TransformDialectCodegenPipelineAttr;
using mlir::iree_compiler::IREE::Codegen::TranslationInfoAttr;
using mlir::iree_compiler::IREE::Codegen::VMVXPipelineAttr;
using mlir::iree_compiler::IREE::HAL::ExecutableVariantOp;

bool ireeAttributeIsACodegenVMVXPipelineAttr(MlirAttribute attr) {
  return llvm::isa<VMVXPipelineAttr>(unwrap(attr));
}

MlirTypeID ireeCodegenVMVXPipelineAttrGetTypeID() {
  return wrap(VMVXPipelineAttr::getTypeID());
}

MlirAttribute ireeCodegenVMVXPipelineAttrGet(MlirContext mlirCtx) {
  return wrap(VMVXPipelineAttr::get(unwrap(mlirCtx)));
}

bool ireeAttributeIsACodegenTransformDialectCodegenPipelineAttr(
    MlirAttribute attr) {
  return llvm::isa<TransformDialectCodegenPipelineAttr>(unwrap(attr));
}

MlirTypeID ireeCodegenTransformDialectCodegenPipelineAttrGetTypeID() {
  return wrap(TransformDialectCodegenPipelineAttr::getTypeID());
}

MlirAttribute
ireeCodegenTransformDialectCodegenPipelineAttrGet(MlirContext mlirCtx) {
  return wrap(TransformDialectCodegenPipelineAttr::get(unwrap(mlirCtx)));
}

bool ireeAttributeIsACodegenNoPipelineAttr(MlirAttribute attr) {
  return llvm::isa<NoPipelineAttr>(unwrap(attr));
}

MlirTypeID ireeCodegenNoPipelineAttrGetTypeID() {
  return wrap(NoPipelineAttr::getTypeID());
}

MlirAttribute ireeCodegenNoPipelineAttrGet(MlirContext mlirCtx) {
  return wrap(NoPipelineAttr::get(unwrap(mlirCtx)));
}

bool ireeAttributeIsACodegenPassPipelineAttr(MlirAttribute attr) {
  return llvm::isa<PassPipelineAttr>(unwrap(attr));
}

MlirTypeID ireeCodegenPassPipelineAttrGetTypeID() {
  return wrap(PassPipelineAttr::getTypeID());
}

MlirAttribute ireeCodegenPassPipelineAttrGet(MlirContext mlirCtx,
                                             MlirStringRef pipeline) {
  return wrap(PassPipelineAttr::get(unwrap(mlirCtx), unwrap(pipeline)));
}

MlirStringRef ireeCodegenPassPipelineAttrGetPipeline(MlirAttribute attr) {
  return wrap(llvm::cast<PassPipelineAttr>(unwrap(attr)).getPipeline());
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
         "Invalid pass pipeline attr: cannot be null");

  mlir::Attribute pipelineAttr = unwrap(parameters.passPipeline);
  assert(
      pipelineAttr.hasPromiseOrImplementsInterface<PipelineAttrInterface>() &&
      "passPipeline must implement PipelineAttrInterface");

  assert((mlirAttributeIsNull(parameters.codegenSpec) ||
          mlirAttributeIsASymbolRef(parameters.codegenSpec)) &&
         "Invalid codegen spec attr");

  assert((mlirAttributeIsNull(parameters.configuration) ||
          mlirAttributeIsADictionary(parameters.configuration)) &&
         "Invalid configuration attr");

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
  return wrap(TranslationInfoAttr::get(ctx, pipelineAttr, codegenSpec,
                                       workgroupSize, subgroupSize.value_or(0),
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

MlirAttribute ireeCodegenConvertConstraintsOpToSMTLIB(MlirOperation op,
                                                      bool emitReset) {
  auto constraintsOp = llvm::cast<ConstraintsOp>(unwrap(op));
  mlir::OwningOpRef<mlir::ModuleOp> smtModule =
      mlir::iree_compiler::convertConstraintsToSMTModule(constraintsOp);

  llvm::SmallString<0> smtlib;
  llvm::raw_svector_ostream os(smtlib);
  mlir::smt::SMTEmissionOptions options;
  options.emitReset = emitReset;
  if (failed(mlir::smt::exportSMTLIB(*smtModule, os, options))) {
    return wrap(mlir::Attribute());
  }
  mlir::Attribute attr =
      mlir::StringAttr::get(constraintsOp->getContext(), os.str());
  return wrap(attr);
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

bool ireeAttributeIsACodegenIntKnobAttr(MlirAttribute attr) {
  return llvm::isa<IntKnobAttr>(unwrap(attr));
}

MlirTypeID ireeCodegenIntKnobAttrGetTypeID() {
  return wrap(IntKnobAttr::getTypeID());
}

MlirAttribute ireeCodegenIntKnobAttrGetName(MlirAttribute attr) {
  return wrap(mlir::Attribute(llvm::cast<IntKnobAttr>(unwrap(attr)).getName()));
}

bool ireeAttributeIsACodegenOneOfKnobAttr(MlirAttribute attr) {
  return llvm::isa<OneOfKnobAttr>(unwrap(attr));
}

MlirTypeID ireeCodegenOneOfKnobAttrGetTypeID() {
  return wrap(OneOfKnobAttr::getTypeID());
}

MlirAttribute ireeCodegenOneOfKnobAttrGetName(MlirAttribute attr) {
  return wrap(
      mlir::Attribute(llvm::cast<OneOfKnobAttr>(unwrap(attr)).getName()));
}

void ireeCodegenOneOfKnobAttrGetOptions(MlirAttribute attr,
                                        intptr_t *numOptions,
                                        MlirAttribute *options) {
  mlir::ArrayAttr opts = llvm::cast<OneOfKnobAttr>(unwrap(attr)).getOptions();
  assert(numOptions && "numOptions cannot be nullptr");
  if (!options) {
    *numOptions = opts.size();
    return;
  }
  assert(static_cast<size_t>(*numOptions) == opts.size() &&
         "*numOptions must match the number of options");
  for (intptr_t i = 0, e = opts.size(); i < e; ++i) {
    options[i] = wrap(opts[i]);
  }
}
