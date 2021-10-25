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

namespace {

// TODO(ravishankarm): The IREEFieldParser is part of the patch D111594 (where
// it is called ::mlir::FieldParser). Remove this when the upstream change lands
// in IREE.

//===----------------------------------------------------------------------===//
// Parse Fields
//===----------------------------------------------------------------------===//

/// Provide a template class that can be specialized by users to dispatch to
/// parsers. Auto-generated parsers generate calls to
/// `IREEFieldParser<T>::parse`, where `T` is the parameter storage type, to
/// parse custom types.
template <typename T, typename = T>
struct IREEFieldParser;

/// Parse an attribute.
template <typename AttributeT>
struct IREEFieldParser<
    AttributeT, std::enable_if_t<std::is_base_of<Attribute, AttributeT>::value,
                                 AttributeT>> {
  static FailureOr<AttributeT> parse(DialectAsmParser &parser) {
    AttributeT value;
    if (parser.parseAttribute(value)) return failure();
    return value;
  }
};

/// Parse any integer.
template <typename IntT>
struct IREEFieldParser<IntT,
                       std::enable_if_t<std::is_integral<IntT>::value, IntT>> {
  static FailureOr<IntT> parse(DialectAsmParser &parser) {
    IntT value;
    if (parser.parseInteger(value)) return failure();
    return value;
  }
};

/// Parse a string.
template <>
struct IREEFieldParser<std::string> {
  static FailureOr<std::string> parse(DialectAsmParser &parser) {
    std::string value;
    if (parser.parseString(&value)) return failure();
    return value;
  }
};

/// Parse any container that supports back insertion as a list.
template <typename ContainerT>
struct IREEFieldParser<
    ContainerT, std::enable_if_t<std::is_member_function_pointer<decltype(
                                     &ContainerT::push_back)>::value,
                                 ContainerT>> {
  using ElementT = typename ContainerT::value_type;
  static FailureOr<ContainerT> parse(DialectAsmParser &parser) {
    ContainerT elements;
    auto elementParser = [&]() {
      auto element = IREEFieldParser<ElementT>::parse(parser);
      if (failed(element)) return failure();
      elements.push_back(element.getValue());
      return success();
    };
    if (parser.parseCommaSeparatedList(elementParser)) return failure();
    return elements;
  }
};
}  // namespace

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

::mlir::Attribute TranslationInfoAttr::parse(::mlir::DialectAsmParser &parser,
                                             ::mlir::Type attrType) {
  ::mlir::FailureOr<StringAttr> _result_passPipeline;
  ::mlir::FailureOr<ArrayAttr> _result_workloadPerWorkgroup;
  // Parse literal '<'
  if (parser.parseLess()) return {};
  // Parse variable 'passPipeline'
  _result_passPipeline = IREEFieldParser<StringAttr>::parse(parser);
  if (failed(_result_passPipeline)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse IREECodegen_TranslationInfoAttr "
                     "parameter 'passPipeline' which is to be a `StringAttr`");
    return {};
  }
  // Parse literal ','
  if (parser.parseComma()) return {};
  // Parse literal 'workload_per_wg'
  if (parser.parseKeyword("workload_per_wg")) return {};
  // Parse literal '='
  if (parser.parseEqual()) return {};
  // Parse variable 'workloadPerWorkgroup'
  _result_workloadPerWorkgroup = IREEFieldParser<ArrayAttr>::parse(parser);
  if (failed(_result_workloadPerWorkgroup)) {
    parser.emitError(
        parser.getCurrentLocation(),
        "failed to parse IREECodegen_TranslationInfoAttr parameter "
        "'workloadPerWorkgroup' which is to be a `ArrayAttr`");
    return {};
  }
  // Parse literal '>'
  if (parser.parseGreater()) return {};
  return TranslationInfoAttr::get(parser.getContext(),
                                  _result_passPipeline.getValue(),
                                  _result_workloadPerWorkgroup.getValue());
}

void TranslationInfoAttr::print(::mlir::DialectAsmPrinter &printer) const {
  printer << "translation.info";
  printer << "<";
  printer << getPassPipeline();
  printer << ",";
  printer << ' ' << "workload_per_wg";
  printer << ' ' << "=";
  printer << ' ';
  printer << getWorkloadPerWorkgroup();
  printer << ">";
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

::mlir::Attribute LoweringConfigAttr::parse(::mlir::DialectAsmParser &parser,
                                            ::mlir::Type attrType) {
  ::mlir::FailureOr<ArrayAttr> _result_tileSizes;
  ::mlir::FailureOr<ArrayAttr> _result_nativeVectorSize;
  // Parse literal '<'
  if (parser.parseLess()) return {};
  // Parse literal 'tile_sizes'
  if (parser.parseKeyword("tile_sizes")) return {};
  // Parse literal '='
  if (parser.parseEqual()) return {};
  // Parse variable 'tileSizes'
  _result_tileSizes = IREEFieldParser<ArrayAttr>::parse(parser);
  if (failed(_result_tileSizes)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse IREECodegen_LoweringConfigAttr parameter "
                     "'tileSizes' which is to be a `ArrayAttr`");
    return {};
  }
  // Parse literal ','
  if (parser.parseComma()) return {};
  // Parse literal 'native_vector_size'
  if (parser.parseKeyword("native_vector_size")) return {};
  // Parse literal '='
  if (parser.parseEqual()) return {};
  // Parse variable 'nativeVectorSize'
  _result_nativeVectorSize = IREEFieldParser<ArrayAttr>::parse(parser);
  if (failed(_result_nativeVectorSize)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse IREECodegen_LoweringConfigAttr parameter "
                     "'nativeVectorSize' which is to be a `ArrayAttr`");
    return {};
  }
  // Parse literal '>'
  if (parser.parseGreater()) return {};
  return LoweringConfigAttr::get(parser.getContext(),
                                 _result_tileSizes.getValue(),
                                 _result_nativeVectorSize.getValue());
}

void LoweringConfigAttr::print(::mlir::DialectAsmPrinter &printer) const {
  printer << "lowering.config";
  printer << "<";
  printer << "tile_sizes";
  printer << ' ' << "=";
  printer << ' ';
  printer << getTileSizes();
  printer << ",";
  printer << ' ' << "native_vector_size";
  printer << ' ' << "=";
  printer << ' ';
  printer << getNativeVectorSize();
  printer << ">";
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

/// Parser method that is copied from the auto-generated using `assemblyFormat`
/// available with patch D111594. Replace after that change is in IREE.
::mlir::Attribute CompilationInfoAttr::parse(::mlir::DialectAsmParser &parser,
                                             ::mlir::Type attrType) {
  ::mlir::FailureOr<LoweringConfigAttr> _result_loweringConfig;
  ::mlir::FailureOr<TranslationInfoAttr> _result_translationInfo;
  ::mlir::FailureOr<ArrayAttr> _result_workgroupSize;
  // Parse literal '<'
  if (parser.parseLess()) return {};
  // Parse variable 'loweringConfig'
  _result_loweringConfig = IREEFieldParser<LoweringConfigAttr>::parse(parser);
  if (failed(_result_loweringConfig)) {
    parser.emitError(
        parser.getCurrentLocation(),
        "failed to parse IREECodegen_CompilationInfoAttr parameter "
        "'loweringConfig' which is to be a `LoweringConfigAttr`");
    return {};
  }
  // Parse literal ','
  if (parser.parseComma()) return {};
  // Parse variable 'translationInfo'
  _result_translationInfo = IREEFieldParser<TranslationInfoAttr>::parse(parser);
  if (failed(_result_translationInfo)) {
    parser.emitError(
        parser.getCurrentLocation(),
        "failed to parse IREECodegen_CompilationInfoAttr parameter "
        "'translationInfo' which is to be a `TranslationInfoAttr`");
    return {};
  }
  // Parse literal ','
  if (parser.parseComma()) return {};
  // Parse literal 'workgroup_size'
  if (parser.parseKeyword("workgroup_size")) return {};
  // Parse literal '='
  if (parser.parseEqual()) return {};
  // Parse variable 'workgroupSize'
  _result_workgroupSize = IREEFieldParser<ArrayAttr>::parse(parser);
  if (failed(_result_workgroupSize)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse IREECodegen_CompilationInfoAttr "
                     "parameter 'workgroupSize' which is to be a `ArrayAttr`");
    return {};
  }
  // Parse literal '>'
  if (parser.parseGreater()) return {};
  return CompilationInfoAttr::get(
      parser.getContext(), _result_loweringConfig.getValue(),
      _result_translationInfo.getValue(), _result_workgroupSize.getValue());
}

/// Printer method that is copied from the auto-generated using `assemblyFormat`
/// available with patch D111594. Replace after that change is in IREE.
void CompilationInfoAttr::print(::mlir::DialectAsmPrinter &printer) const {
  printer << "compilation.info";
  printer << "<";
  printer << getLoweringConfig();
  printer << ",";
  printer << ' ';
  printer << getTranslationInfo();
  printer << ",";
  printer << ' ' << "workgroup_size";
  printer << ' ' << "=";
  printer << ' ';
  printer << getWorkgroupSize();
  printer << ">";
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
  auto partitionedLoops = getPartitionedLoops(op);
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
