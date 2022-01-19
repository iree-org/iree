// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/LoweringConfig.cpp.inc"
#include "iree/compiler/Codegen/Dialect/LoweringConfigEnums.cpp.inc"

static const char kConfigAttrName[] = "lowering.config";
static const char kTranslationInfoAttrName[] = "translation.info";
static const char kCompilationInfoAttrName[] = "compilation.info";

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility function for common code patterns.
//===----------------------------------------------------------------------===//

static bool checkIntegerArrayAttr(ArrayAttr arrayAttr) {
  return !llvm::any_of(arrayAttr,
                       [](Attribute attr) { return !attr.isa<IntegerAttr>(); });
}

/// Returns an `ArrayAttr` where each element is an `IntegerAttr` of `IndexType`
/// whose values is obtained from `values`.
static ArrayAttr getIndexIntegerArrayAttr(MLIRContext *context,
                                          ArrayRef<int64_t> values) {
  auto attrs = llvm::to_vector<4>(
      llvm::map_range(values, [&context](int64_t value) -> Attribute {
        return IntegerAttr::get(IndexType::get(context), APInt(64, value));
      }));
  return ArrayAttr::get(context, attrs);
}

/// Returns an `ArrayAttr` where each element is an `IntegerAttr` of 64-bit
/// integer type whose values is obtained from `values`.
static ArrayAttr getI64IntegerArrayAttr(MLIRContext *context,
                                        ArrayRef<int64_t> values) {
  auto attrs = llvm::to_vector<4>(
      llvm::map_range(values, [&context](int64_t value) -> Attribute {
        return IntegerAttr::get(IntegerType::get(context, 64),
                                APInt(64, value));
      }));
  return ArrayAttr::get(context, attrs);
}

/// Assumes that `arrayAttr` is a list of `IntegerAttr`s and returns the values
/// in these attributes as a vector.
static SmallVector<int64_t> getIntegerVals(ArrayAttr arrayAttr) {
  if (!arrayAttr) return {};
  SmallVector<int64_t> values(arrayAttr.size());
  for (auto attr : llvm::enumerate(arrayAttr)) {
    values[attr.index()] = attr.value().cast<IntegerAttr>().getInt();
  }
  return values;
}

namespace IREE {
namespace Codegen {

//===----------------------------------------------------------------------===//
// iree_codegen.translation.info
//===----------------------------------------------------------------------===//

TranslationInfoAttr TranslationInfoAttr::get(
    MLIRContext *context, DispatchLoweringPassPipeline passPipeline,
    ArrayRef<int64_t> workloadPerWorkgroup) {
  auto pipelineAttr = StringAttr::get(context, stringifyEnum(passPipeline));
  ArrayAttr workloadPerWorkgroupAttr =
      getI64IntegerArrayAttr(context, workloadPerWorkgroup);
  return get(context, pipelineAttr, workloadPerWorkgroupAttr);
}

DispatchLoweringPassPipeline
TranslationInfoAttr::getDispatchLoweringPassPipeline() {
  Optional<DispatchLoweringPassPipeline> passPipeline =
      symbolizeEnum<DispatchLoweringPassPipeline>(getPassPipeline().getValue());
  return passPipeline.getValue();
}

SmallVector<int64_t> TranslationInfoAttr::getWorkloadPerWorkgroupVals() {
  return getIntegerVals(getWorkloadPerWorkgroup());
}

LogicalResult TranslationInfoAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, StringAttr passPipeline,
    ArrayAttr workloadPerWorkgroup) {
  if (!passPipeline) {
    return emitError() << "missing pass pipeline specification";
  }
  auto passPipelineValue =
      symbolizeEnum<IREE::Codegen::DispatchLoweringPassPipeline>(
          passPipeline.getValue());
  if (!passPipelineValue) {
    return emitError() << "invalid pass pipeline value : "
                       << passPipeline.getValue();
  }
  if (!workloadPerWorkgroup) {
    return emitError() << "expected workload_per_wg to be specified (even if "
                          "specified as empty)";
  }
  if (!checkIntegerArrayAttr(workloadPerWorkgroup)) {
    return emitError() << "expected workload_per_wg to be an IntegerAttr list";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// iree_codegen.lowering.config
//===----------------------------------------------------------------------===//

LoweringConfigAttr LoweringConfigAttr::get(MLIRContext *context,
                                           TileSizesListTypeRef tileSizes,
                                           ArrayRef<int64_t> nativeVectorSize) {
  auto attrList = llvm::to_vector<4>(
      llvm::map_range(tileSizes, [&](ArrayRef<int64_t> sizes) -> Attribute {
        return getI64IntegerArrayAttr(context, sizes);
      }));
  ArrayAttr tileSizesAttr = ArrayAttr::get(context, attrList);
  ArrayAttr nativeVectorSizeAttr =
      getI64IntegerArrayAttr(context, nativeVectorSize);
  return get(context, tileSizesAttr, nativeVectorSizeAttr);
}

TileSizesListType LoweringConfigAttr::getTileSizeVals() {
  auto tileSizesAttr = getTileSizes();
  if (!tileSizesAttr) return {};
  TileSizesListType tileSizes;
  for (auto attr : tileSizesAttr) {
    auto vals = getIntegerVals(attr.cast<ArrayAttr>());
    tileSizes.emplace_back(std::move(vals));
  }
  return tileSizes;
}

SmallVector<int64_t> LoweringConfigAttr::getTileSizeVals(unsigned level) {
  ArrayAttr tileSizesAttr = getTileSizes();
  if (!tileSizesAttr || tileSizesAttr.size() <= level) return {};
  return getIntegerVals(tileSizesAttr[level].cast<ArrayAttr>());
}

SmallVector<int64_t> LoweringConfigAttr::getNativeVectorSizeVals() {
  ArrayAttr nativeVectorSizeAttr = getNativeVectorSize();
  if (!nativeVectorSizeAttr) return {};
  return getIntegerVals(nativeVectorSizeAttr);
}

LogicalResult LoweringConfigAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayAttr tileSizes,
    ArrayAttr nativeVectorSize) {
  if (!tileSizes) {
    return emitError() << "expected tile_sizes to be specified (even is "
                          "specified as empty)";
  }
  if (llvm::any_of(tileSizes, [](Attribute attr) {
        auto arrayAttr = attr.dyn_cast<ArrayAttr>();
        return !arrayAttr || !checkIntegerArrayAttr(arrayAttr);
      })) {
    return emitError()
           << "expected all elements of tile_sizes to be a list of integers";
  }
  if (!nativeVectorSize) {
    return emitError() << "expected native_vector_size to be specified (even "
                          "if specified as empty)";
  }
  if (!checkIntegerArrayAttr(nativeVectorSize)) {
    return emitError()
           << "expected native_vector_size to be a list of integer values";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// iree.compilation.info
//===----------------------------------------------------------------------===//

CompilationInfoAttr CompilationInfoAttr::get(MLIRContext *context,
                                             TileSizesListTypeRef tileSizes,
                                             ArrayRef<int64_t> nativeVectorSize,
                                             ArrayRef<int64_t> workgroupSize) {
  LoweringConfigAttr configAttr =
      LoweringConfigAttr::get(context, tileSizes, nativeVectorSize);
  TranslationInfoAttr translationInfo =
      TranslationInfoAttr::get(context, DispatchLoweringPassPipeline::None);
  ArrayAttr workgroupSizeAttr = getI64IntegerArrayAttr(context, workgroupSize);
  return get(context, configAttr, translationInfo, workgroupSizeAttr);
}

CompilationInfoAttr CompilationInfoAttr::get(
    MLIRContext *context, TileSizesListTypeRef tileSizes,
    ArrayRef<int64_t> nativeVectorSize,
    DispatchLoweringPassPipeline passPipeline,
    ArrayRef<int64_t> workloadPerWorkgroup, ArrayRef<int64_t> workgroupSize) {
  LoweringConfigAttr configAttr =
      LoweringConfigAttr::get(context, tileSizes, nativeVectorSize);
  TranslationInfoAttr translationInfoAttr =
      TranslationInfoAttr::get(context, passPipeline, workloadPerWorkgroup);
  ArrayAttr workgroupSizeAttr = getI64IntegerArrayAttr(context, workgroupSize);
  return get(context, configAttr, translationInfoAttr, workgroupSizeAttr);
}

LogicalResult CompilationInfoAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    LoweringConfigAttr loweringConfig, TranslationInfoAttr translationInfo,
    ArrayAttr workgroupSize) {
  if (!loweringConfig) {
    return emitError() << "missing lowering config";
  }
  if (failed(
          LoweringConfigAttr::verify(emitError, loweringConfig.getTileSizes(),
                                     loweringConfig.getNativeVectorSize()))) {
    return failure();
  }
  if (!translationInfo) {
    return emitError() << "missing translation info";
  }
  if (failed(TranslationInfoAttr::verify(
          emitError, translationInfo.getPassPipeline(),
          translationInfo.getWorkloadPerWorkgroup()))) {
    return failure();
  }
  if (!workgroupSize) {
    return emitError() << "expected workgroup_size to be specified (even if "
                          "specified empty)";
  }
  if (!checkIntegerArrayAttr(workgroupSize)) {
    return emitError() << "expected workgroup_size to be a list of integers";
  }
  return success();
}

SmallVector<int64_t> CompilationInfoAttr::getWorkgroupSizeVals() {
  ArrayAttr workgroupSizeAttr = getWorkgroupSize();
  if (!workgroupSizeAttr) return {};
  return getIntegerVals(workgroupSizeAttr);
}

//===----------------------------------------------------------------------===//
// Initialize attributes
//===----------------------------------------------------------------------===//

void IREECodegenDialect::initializeCodegenAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Codegen/Dialect/LoweringConfig.cpp.inc"  // IWYU pragma: keeep
      >();
}

OptionalParseResult IREECodegenDialect::parseCodegenAttrs(
    DialectAsmParser &parser, StringRef mnemonic, Type type,
    Attribute &value) const {
  return generatedAttributeParser(parser, mnemonic, type, value);
}

LogicalResult IREECodegenDialect::printCodegenAttrs(
    Attribute attr, DialectAsmPrinter &p) const {
  return generatedAttributePrinter(attr, p);
}

}  // namespace Codegen
}  // namespace IREE

//===----------------------------------------------------------------------===//
// Helpers for getting/setting iree_codegen.translation.info attribute on the
// `hal.executable.entry_point`
// ===----------------------------------------------------------------------===//

IREE::Codegen::TranslationInfoAttr getTranslationInfo(
    IREE::HAL::ExecutableEntryPointOp entryPointOp) {
  return entryPointOp->getAttrOfType<IREE::Codegen::TranslationInfoAttr>(
      kTranslationInfoAttrName);
}

SmallVector<int64_t> getWorkgroupSize(
    IREE::HAL::ExecutableEntryPointOp entryPointOp) {
  if (Optional<ArrayAttr> workgroupSizeAttrList =
          entryPointOp.workgroup_size()) {
    return getIntegerVals(*workgroupSizeAttrList);
  }
  return {};
}

void setTranslationInfo(IREE::HAL::ExecutableEntryPointOp entryPointOp,
                        IREE::Codegen::TranslationInfoAttr translationInfo,
                        ArrayRef<int64_t> workgroupSize) {
  entryPointOp->setAttr(kTranslationInfoAttrName, translationInfo);
  // The workgroup size is set on the entry point op directly.
  if (!workgroupSize.empty()) {
    MLIRContext *context = entryPointOp->getContext();
    auto attrs = getIndexIntegerArrayAttr(context, workgroupSize);
    entryPointOp.workgroup_sizeAttr(attrs);
  }
}

//===----------------------------------------------------------------------===//
// Helpers for getting/setting `iree_codegen.lowering.config` attribute on root
// operations.
// ===----------------------------------------------------------------------===//

IREE::Codegen::LoweringConfigAttr getLoweringConfig(Operation *op) {
  return op->getAttrOfType<IREE::Codegen::LoweringConfigAttr>(kConfigAttrName);
}

SmallVector<int64_t> getTileSizes(Operation *op, unsigned level) {
  IREE::Codegen::LoweringConfigAttr configAttr = getLoweringConfig(op);
  if (!configAttr) return {};
  return configAttr.getTileSizeVals(level);
}
SmallVector<Value, 4> getTileSizes(OpBuilder &b, Operation *op,
                                   unsigned level) {
  return llvm::to_vector<4>(
      llvm::map_range(getTileSizes(op, level), [&](int64_t t) -> Value {
        return b.create<arith::ConstantIndexOp>(op->getLoc(), t);
      }));
}

void setLoweringConfig(Operation *op,
                       IREE::Codegen::LoweringConfigAttr config) {
  op->setAttr(kConfigAttrName, config);
}

LogicalResult setOpConfigAndEntryPointFnTranslation(
    FuncOp entryPointFn, Operation *op,
    IREE::Codegen::LoweringConfigAttr config,
    IREE::Codegen::DispatchLoweringPassPipeline passPipeline,
    ArrayRef<int64_t> workgroupSize) {
  auto interfaceOp = cast<IREE::Flow::PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  SmallVector<int64_t, 3> workloadPerWorkgroup;
  auto tileSizes = config.getTileSizeVals(0);
  if (!tileSizes.empty() && !partitionedLoops.empty()) {
    for (unsigned depth : partitionedLoops) {
      if (depth >= tileSizes.size()) {
        return op->emitOpError(
                   "illegal configuration for lowering op, expect first level "
                   "tile size to contain at least ")
               << partitionedLoops.back() << " elements";
      }
      if (tileSizes[depth] == 0) {
        return op->emitOpError("illegal to set tilesize of loop ")
               << depth
               << " to zero since it is set to be partitioned at the flow "
                  "level";
      }
      workloadPerWorkgroup.push_back(tileSizes[depth]);
    }
    if (!workloadPerWorkgroup.empty()) {
      workloadPerWorkgroup =
          llvm::to_vector<3>(llvm::reverse(workloadPerWorkgroup));
    }
  }
  auto entryPointOp = getEntryPoint(entryPointFn);
  if (!entryPointOp) {
    return entryPointFn.emitOpError(
        "unable to find entry point op for entry point function");
  }
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      entryPointOp->getContext(), passPipeline, workloadPerWorkgroup);
  setTranslationInfo(entryPointOp, translationInfo, workgroupSize);
  return success();
}

//===----------------------------------------------------------------------===//
// Helpers for getting/setting `iree_codegen.compilation.info` attribute on root
// operations to override IREEs default compilation.
// ===----------------------------------------------------------------------===//

IREE::Codegen::CompilationInfoAttr getCompilationInfo(Operation *op) {
  return op->getAttrOfType<IREE::Codegen::CompilationInfoAttr>(
      kCompilationInfoAttrName);
}

void setCompilationInfo(Operation *op,
                        IREE::Codegen::CompilationInfoAttr config) {
  op->setAttr(kCompilationInfoAttrName, config);
}

void eraseCompilationInfo(Operation *op) {
  op->removeAttr(kCompilationInfoAttrName);
}

}  // namespace iree_compiler
}  // namespace mlir
