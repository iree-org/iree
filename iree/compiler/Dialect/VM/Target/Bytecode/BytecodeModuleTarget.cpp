// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"

#include <algorithm>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/minireflect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/IREE/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"
#include "iree/compiler/Dialect/VM/Analysis/ValueLiveness.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeEncoder.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/ConstantEncoder.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/schemas/bytecode_module_def_generated.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Translation.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

namespace {

using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;
using flatbuffers::Vector;

struct ModuleCounts {
  int importFuncs = 0;
  int exportFuncs = 0;
  int internalFuncs = 0;
  size_t globalBytes = 0;
  int globalRefs = 0;
  int rodatas = 0;
  int rwdatas = 0;
};

struct TypeDef {
  Type type;
  std::string full_name;
};

}  // namespace

// Computes symbol counts within the given |moduleOp|.
// These counts, including the global byte reservation count, are expected to
// match the actual values during serialization.
//
// Preconditions:
//  - OrdinalAllocationPass has run on the module
//  - All ordinals start from 0 and are contiguous
static ModuleCounts computeModuleSymbolCounts(IREE::VM::ModuleOp moduleOp) {
  ModuleCounts counts;
  for (auto &op : moduleOp.getBlock().getOperations()) {
    if (auto funcOp = dyn_cast<IREE::VM::FuncOp>(op)) {
      ++counts.internalFuncs;
    } else if (isa<IREE::VM::ExportOp>(op)) {
      ++counts.exportFuncs;
    } else if (isa<IREE::VM::ImportOp>(op)) {
      ++counts.importFuncs;
    } else if (isa<IREE::VM::RodataOp>(op)) {
      ++counts.rodatas;
    } else if (isa<IREE::VM::GlobalRefOp>(op)) {
      ++counts.globalRefs;
    } else if (auto globalOp = dyn_cast<VMGlobalOp>(op)) {
      counts.globalBytes =
          std::max(counts.globalBytes,
                   globalOp.getOrdinal() + globalOp.getStorageSize());
    }
  }
  return counts;
}

// Finds all types in the module and builds a type table mapping the index in
// the vector to the type represented by the type ordinal.
static std::vector<TypeDef> buildTypeTable(IREE::VM::ModuleOp moduleOp) {
  llvm::DenseMap<Type, std::string> typeMap;
  std::function<void(Type)> tryInsertType;
  tryInsertType = [&](Type type) {
    if (auto refPtrType = type.dyn_cast<IREE::VM::RefType>()) {
      type = refPtrType.getObjectType();
    }
    if (typeMap.count(type)) return;
    std::string str;
    llvm::raw_string_ostream sstream(str);
    type.print(sstream);
    sstream.flush();
    typeMap.try_emplace(type, str);
    if (auto listType = type.dyn_cast<IREE::VM::ListType>()) {
      if (listType.getElementType()) {
        tryInsertType(listType.getElementType());
      }
    }
  };
  for (auto funcOp : moduleOp.getBlock().getOps<IREE::VM::FuncOp>()) {
    funcOp.walk([&](Operation *op) {
      for (auto type : op->getOperandTypes()) tryInsertType(type);
      for (auto type : op->getResultTypes()) tryInsertType(type);
    });
  }

  std::vector<TypeDef> table;
  table.reserve(typeMap.size());
  for (const auto &typeString : typeMap) {
    table.push_back(TypeDef{typeString.first, typeString.second});
  }
  llvm::sort(
      table, +[](const TypeDef &lhs, const TypeDef &rhs) {
        // Always sort builtins above custom types.
        if (lhs.full_name[0] != '!' && rhs.full_name[0] == '!') {
          return true;
        } else if (lhs.full_name[0] == '!' && rhs.full_name[0] != '!') {
          return false;
        }
        return lhs.full_name.compare(rhs.full_name) < 0;
      });
  return table;
}

// Canonicalizes the module to its final form prior to emission.
// This verifies that we only have ops we can serialize and performs any of the
// required transformations (such as debug op stripping).
static LogicalResult canonicalizeModule(BytecodeTargetOptions targetOptions,
                                        IREE::VM::ModuleOp moduleOp) {
  OwningRewritePatternList patterns;
  ConversionTarget target(*moduleOp.getContext());
  target.addLegalDialect<IREE::VM::VMDialect>();
  target.addLegalOp<IREE::DoNotOptimizeOp>();

  // Add all VM canonicalization patterns and mark pseudo-ops illegal.
  auto *context = moduleOp.getContext();
  for (auto *op : context->getRegisteredOperations()) {
    // Non-serializable ops must be removed prior to serialization.
    if (op->hasTrait<OpTrait::IREE::VM::PseudoOp>()) {
      op->getCanonicalizationPatterns(patterns, context);
      target.setOpAction(OperationName(op->name, context),
                         ConversionTarget::LegalizationAction::Illegal);
    }

    // Debug ops must not be present when stripping.
    // TODO(benvanik): add RemoveDisabledDebugOp pattern.
    if (op->hasTrait<OpTrait::IREE::VM::DebugOnly>() &&
        targetOptions.stripDebugOps) {
      target.setOpAction(OperationName(op->name, context),
                         ConversionTarget::LegalizationAction::Illegal);
    }
  }

  if (failed(applyFullConversion(moduleOp, target, patterns))) {
    return moduleOp.emitError() << "unable to fully apply conversion to module";
  }

  PassManager passManager(context);
  mlir::applyPassManagerCLOptions(passManager);
  auto &modulePasses = passManager.nest<IREE::VM::ModuleOp>();

  if (targetOptions.optimize) {
    // TODO(benvanik): does this run until it quiesces?
    modulePasses.addPass(mlir::createInlinerPass());
    modulePasses.addPass(mlir::createCSEPass());
    modulePasses.addPass(mlir::createCanonicalizerPass());
  }

  modulePasses.addPass(createDropCompilerHintsPass());

  // Mark up the module with ordinals for each top-level op (func, etc).
  // This will make it easier to correlate the MLIR textual output to the
  // binary output.
  // We don't want any more modifications after this point as they could
  // invalidate the ordinals.
  modulePasses.addPass(IREE::VM::createOrdinalAllocationPass());

  if (failed(passManager.run(moduleOp.getParentOfType<mlir::ModuleOp>()))) {
    return moduleOp.emitError() << "failed during transform passes";
  }

  return success();
}

// Returns a vector of tables of type T or None if |contents| is empty.
template <typename T>
static Optional<Offset<Vector<Offset<T>>>> createOptionalVector(
    const std::vector<Offset<T>> &contents, FlatBufferBuilder &fbb) {
  if (contents.empty()) return llvm::None;
  return fbb.CreateVector(contents);
}
template <typename T>
static Optional<Offset<Vector<T>>> createOptionalVector(
    const std::vector<T> &contents, FlatBufferBuilder &fbb) {
  if (contents.empty()) return llvm::None;
  return fbb.CreateVector(contents);
}

// Returns a serialized function signature.
static Offset<iree::vm::FunctionSignatureDef> makeFunctionSignatureDef(
    FunctionType functionType, llvm::DenseMap<Type, int> &typeTable,
    DictionaryAttr reflectionAttrs, FlatBufferBuilder &fbb) {
  // Argument types.
  std::vector<int32_t> argumentTypes(functionType.getNumInputs());
  for (int i = 0; i < argumentTypes.size(); ++i) {
    Type type = functionType.getInput(i);
    if (auto refPtrType = type.dyn_cast<IREE::VM::RefType>()) {
      type = refPtrType.getObjectType();
    }
    argumentTypes[i] = typeTable.lookup(type);
  }
  auto argumentTypesOffset = createOptionalVector(argumentTypes, fbb);

  // Result types.
  std::vector<int32_t> resultTypes(functionType.getNumResults());
  for (int i = 0; i < resultTypes.size(); ++i) {
    Type type = functionType.getResult(i);
    if (auto refPtrType = type.dyn_cast<IREE::VM::RefType>()) {
      type = refPtrType.getObjectType();
    }
    resultTypes[i] = typeTable.lookup(type);
  }
  auto resultTypesOffset = createOptionalVector(resultTypes, fbb);

  // Reflection attrs.
  Optional<Offset<Vector<Offset<iree::vm::ReflectionAttrDef>>>>
      reflectionAttrsOffset;
  if (reflectionAttrs) {
    llvm::SmallVector<Offset<iree::vm::ReflectionAttrDef>, 4>
        reflectionAttrItems;
    for (auto reflectionAttr : reflectionAttrs) {
      auto key = reflectionAttr.first.strref();
      auto value = reflectionAttr.second.dyn_cast<StringAttr>();
      if (!value || key.empty()) continue;
      auto keyOffset = fbb.CreateString(key.data(), key.size());
      auto valueOffset =
          fbb.CreateString(value.getValue().data(), value.getValue().size());
      iree::vm::ReflectionAttrDefBuilder rattr(fbb);
      rattr.add_key(keyOffset);
      rattr.add_value(valueOffset);
      reflectionAttrItems.push_back(rattr.Finish());
    }
    reflectionAttrsOffset = fbb.CreateVector(reflectionAttrItems.data(),
                                             reflectionAttrItems.size());
  }

  iree::vm::FunctionSignatureDefBuilder fsd(fbb);
  if (argumentTypesOffset) {
    fsd.add_argument_types(argumentTypesOffset.getValue());
  }
  if (resultTypesOffset) {
    fsd.add_result_types(resultTypesOffset.getValue());
  }
  if (reflectionAttrsOffset) {
    fsd.add_reflection_attrs(reflectionAttrsOffset.getValue());
  }
  return fsd.Finish();
}

// Builds a complete BytecodeModuleDef FlatBuffer object in |fbb|.
// The order of the encoding is ordered to ensure that all metadata is at the
// front of the resulting buffer. Large read-only data and bytecode blobs always
// fill the end of the file meaning that when memory-mapping the file most will
// not need to be paged in to do the initial module preparation.
//
// To keep the actual BytecodeModuleDef and resulting parsing code simple a lot
// has been packed into the top-level table. This results in a messier function
// here during serialization but a much more trivial (and cache-friendly)
// representation at runtime.
static Offset<iree::vm::BytecodeModuleDef> buildFlatBufferModule(
    BytecodeTargetOptions targetOptions, IREE::VM::ModuleOp moduleOp,
    FlatBufferBuilder &fbb) {
  SymbolTable symbolTable(moduleOp);
  auto symbolCounts = computeModuleSymbolCounts(moduleOp);

  // Find all structural ops in the module.
  std::vector<IREE::VM::ImportOp> importFuncOps;
  std::vector<IREE::VM::ExportOp> exportFuncOps;
  std::vector<IREE::VM::FuncOp> internalFuncOps;
  std::vector<IREE::VM::RodataOp> rodataOps;
  importFuncOps.resize(symbolCounts.importFuncs);
  exportFuncOps.resize(symbolCounts.exportFuncs);
  internalFuncOps.resize(symbolCounts.internalFuncs);
  rodataOps.resize(symbolCounts.rodatas);
  for (auto &op : moduleOp.getBlock().getOperations()) {
    if (auto funcOp = dyn_cast<IREE::VM::FuncOp>(op)) {
      internalFuncOps[funcOp.ordinal().getValue().getLimitedValue()] = funcOp;
    } else if (auto exportOp = dyn_cast<IREE::VM::ExportOp>(op)) {
      exportFuncOps[exportOp.ordinal().getValue().getLimitedValue()] = exportOp;
    } else if (auto importOp = dyn_cast<IREE::VM::ImportOp>(op)) {
      importFuncOps[importOp.ordinal().getValue().getLimitedValue()] = importOp;
    } else if (auto rodataOp = dyn_cast<IREE::VM::RodataOp>(op)) {
      rodataOps[rodataOp.ordinal().getValue().getLimitedValue()] = rodataOp;
    }
  }

  // Serialize read-only data first so that it ends up at the end of the file.
  // This is where large things like parameters live and we don't want that to
  // get paged in until it is needed.
  std::vector<Offset<Vector<uint8_t>>> rodataContentOffsets;
  rodataContentOffsets.reserve(rodataOps.size());
  for (auto rodataOp : rodataOps) {
    auto dataOffset =
        serializeConstant(rodataOp.getLoc(), rodataOp.value(), fbb);
    if (dataOffset.IsNull()) {
      rodataOp.emitOpError() << "failed to encode";
      return {};
    }
    rodataContentOffsets.push_back(dataOffset);
  }

  // Find all types in the module to build the type table.
  // Note that we don't emit it yet as we want to keep it near the top of the
  // file (which, in FlatBuffers, is written last).
  auto typeTable = buildTypeTable(moduleOp);
  llvm::DenseMap<Type, int> typeOrdinalMap;
  for (auto typeDef : llvm::enumerate(typeTable)) {
    typeOrdinalMap[typeDef.value().type] = typeDef.index();
  }

  // Serialize function bytecode one at a time and then merge at the end.
  std::vector<std::vector<uint8_t>> bytecodeDataParts;
  std::vector<iree::vm::FunctionDescriptor> functionDescriptors;
  bytecodeDataParts.reserve(internalFuncOps.size());
  functionDescriptors.reserve(internalFuncOps.size());
  size_t totalBytecodeLength = 0;
  for (auto funcOp : internalFuncOps) {
    auto encodedFunction =
        BytecodeEncoder::encodeFunction(funcOp, typeOrdinalMap, symbolTable);
    if (!encodedFunction) {
      funcOp.emitError() << "failed to encode function bytecode";
      return {};
    }
    functionDescriptors.push_back(iree::vm::FunctionDescriptor(
        totalBytecodeLength, encodedFunction->bytecodeData.size(),
        encodedFunction->i32RegisterCount, encodedFunction->refRegisterCount));
    totalBytecodeLength += encodedFunction->bytecodeData.size();
    bytecodeDataParts.push_back(std::move(encodedFunction->bytecodeData));
  }
  // TODO(benvanik): compression? deduping?
  uint8_t *bytecodeDataPtr = nullptr;
  auto bytecodeDataOffset = fbb.CreateUninitializedVector<uint8_t>(
      totalBytecodeLength, &bytecodeDataPtr);
  size_t currentBytecodeOffset = 0;
  for (const auto &it : llvm::enumerate(internalFuncOps)) {
    int ordinal = it.index();
    auto data = std::move(bytecodeDataParts[ordinal]);
    std::memcpy(bytecodeDataPtr + currentBytecodeOffset, data.data(),
                data.size());
    currentBytecodeOffset += data.size();
  }

  // Serialize metadata that should be near the front of the file.
  std::vector<Offset<iree::vm::RodataSegmentDef>> rodataSegmentOffsets;
  rodataSegmentOffsets.reserve(rodataOps.size());
  for (auto rodataContentOffset : rodataContentOffsets) {
    iree::vm::RodataSegmentDefBuilder rsd(fbb);
    rsd.add_data(rodataContentOffset);
    rodataSegmentOffsets.push_back(rsd.Finish());
  }
  std::vector<Offset<iree::vm::RwdataSegmentDef>> rwdataSegmentOffsets;
  std::vector<Offset<iree::vm::TypeDef>> typeOffsets;
  typeOffsets.reserve(typeTable.size());
  for (auto &typeDef : typeTable) {
    auto nameOffset = fbb.CreateString(typeDef.full_name);
    iree::vm::TypeDefBuilder tdb(fbb);
    tdb.add_full_name(nameOffset);
    typeOffsets.push_back(tdb.Finish());
  }
  std::vector<Offset<iree::vm::ImportFunctionDef>> importFuncOffsets;
  importFuncOffsets.reserve(importFuncOps.size());
  for (auto importOp : importFuncOps) {
    auto nameOffset = fbb.CreateString(importOp.getName().str());
    auto signatureOffset =
        makeFunctionSignatureDef(importOp.getType(), typeOrdinalMap,
                                 nullptr /* no reflection for imports */, fbb);
    iree::vm::ImportFunctionDefBuilder ifd(fbb);
    ifd.add_full_name(nameOffset);
    ifd.add_signature(signatureOffset);
    importFuncOffsets.push_back(ifd.Finish());
  }
  std::vector<Offset<iree::vm::ExportFunctionDef>> exportFuncOffsets;
  exportFuncOffsets.reserve(exportFuncOps.size());
  for (auto exportOp : exportFuncOps) {
    auto nameOffset = fbb.CreateString(exportOp.export_name().str());
    auto funcOp = symbolTable.lookup<IREE::VM::FuncOp>(exportOp.function_ref());
    auto signatureOffset =
        makeFunctionSignatureDef(funcOp.getType(), typeOrdinalMap,
                                 nullptr /* no reflection for internal */, fbb);
    iree::vm::ExportFunctionDefBuilder efd(fbb);
    efd.add_local_name(nameOffset);
    efd.add_signature(signatureOffset);
    efd.add_internal_ordinal(funcOp.ordinal().getValue().getLimitedValue());
    exportFuncOffsets.push_back(efd.Finish());
  }
  std::vector<Offset<iree::vm::InternalFunctionDef>> internalFuncOffsets;
  if (!targetOptions.stripSymbols) {
    internalFuncOffsets.reserve(internalFuncOps.size());
    for (auto funcOp : internalFuncOps) {
      auto nameOffset = fbb.CreateString(funcOp.getName().str());
      auto signatureOffset = makeFunctionSignatureDef(
          funcOp.getType(), typeOrdinalMap,
          funcOp.getAttrOfType<DictionaryAttr>("iree.reflection"), fbb);
      iree::vm::InternalFunctionDefBuilder ifd(fbb);
      ifd.add_local_name(nameOffset);
      ifd.add_signature(signatureOffset);
      internalFuncOffsets.push_back(ifd.Finish());
    }
  }

  auto functionDescriptorsOffset =
      fbb.CreateVectorOfStructs(functionDescriptors);
  auto rodataSegmentsOffset = createOptionalVector(rodataSegmentOffsets, fbb);
  auto rwdataSegmentsOffset = createOptionalVector(rwdataSegmentOffsets, fbb);
  auto internalFuncsOffset = fbb.CreateVector(internalFuncOffsets);
  auto exportFuncsOffset = fbb.CreateVector(exportFuncOffsets);
  auto importFuncsOffset = createOptionalVector(importFuncOffsets, fbb);
  auto typesOffset = fbb.CreateVector(typeOffsets);

  Optional<Offset<iree::vm::ModuleStateDef>> moduleStateDef;
  if (symbolCounts.globalBytes || symbolCounts.globalRefs) {
    iree::vm::ModuleStateDefBuilder msd(fbb);
    msd.add_global_bytes_capacity(symbolCounts.globalBytes);
    msd.add_global_ref_count(symbolCounts.globalRefs);
    moduleStateDef = msd.Finish();
  }

  auto nameOffset = fbb.CreateString(
      moduleOp.sym_name().empty() ? "module" : moduleOp.sym_name().str());

  iree::vm::BytecodeModuleDefBuilder bmd(fbb);
  bmd.add_name(nameOffset);
  bmd.add_types(typesOffset);
  if (importFuncsOffset) {
    bmd.add_imported_functions(importFuncsOffset.getValue());
  }
  bmd.add_exported_functions(exportFuncsOffset);
  bmd.add_internal_functions(internalFuncsOffset);
  if (moduleStateDef) {
    bmd.add_module_state(moduleStateDef.getValue());
  }
  if (rwdataSegmentsOffset) {
    bmd.add_rwdata_segments(rwdataSegmentsOffset.getValue());
  }
  if (rodataSegmentsOffset) {
    bmd.add_rodata_segments(rodataSegmentsOffset.getValue());
  }
  bmd.add_function_descriptors(functionDescriptorsOffset);
  bmd.add_bytecode_data(bytecodeDataOffset);
  return bmd.Finish();
}

LogicalResult translateModuleToBytecode(IREE::VM::ModuleOp moduleOp,
                                        BytecodeTargetOptions targetOptions,
                                        llvm::raw_ostream &output) {
  if (failed(canonicalizeModule(targetOptions, moduleOp))) {
    return moduleOp.emitError()
           << "failed to canonicalize vm.module to a serializable form";
  }

  if (targetOptions.outputFormat == BytecodeOutputFormat::kAnnotatedMlirText) {
    // Run register allocation now and put the info in the IR so it's printed.
    for (auto funcOp : moduleOp.getBlock().getOps<IREE::VM::FuncOp>()) {
      if (!funcOp.empty()) {
        if (failed(ValueLiveness::annotateIR(funcOp))) {
          return funcOp.emitError() << "liveness analysis failed";
        } else if (failed(RegisterAllocation::annotateIR(funcOp))) {
          return funcOp.emitError() << "register allocation failed";
        }
      }
    }
  }

  if (targetOptions.outputFormat == BytecodeOutputFormat::kMlirText ||
      targetOptions.outputFormat == BytecodeOutputFormat::kAnnotatedMlirText) {
    // Use the standard MLIR text printer.
    moduleOp.getOperation()->print(output);
    output << "\n";
    return success();
  }

  // NOTE: we order things so that all of the metadata is close to the start of
  // the module header in memory. This ensures that when we map the file only
  // the first few pages need to be accessed to get the metadata and the rest
  // can be large bulk data.
  FlatBufferBuilder fbb;
  auto moduleDef = buildFlatBufferModule(targetOptions, moduleOp, fbb);
  if (moduleDef.IsNull()) {
    return moduleOp.emitError()
           << "failed to build FlatBuffer BytecodeModuleDef";
  }

  iree::vm::FinishBytecodeModuleDefBuffer(fbb, moduleDef);
  const uint8_t *flatbufferBytes = fbb.GetBufferPointer();
  size_t flatbufferByteSize = fbb.GetSize();

  switch (targetOptions.outputFormat) {
    case BytecodeOutputFormat::kFlatBufferBinary:
      output.write(reinterpret_cast<const char *>(flatbufferBytes),
                   flatbufferByteSize);
      break;
    case BytecodeOutputFormat::kFlatBufferText: {
      flatbuffers::ToStringVisitor toStringVisitor("\n", false, "  ", false);
      flatbuffers::IterateFlatBuffer(flatbufferBytes,
                                     iree::vm::BytecodeModuleDefTypeTable(),
                                     &toStringVisitor);
      output << toStringVisitor.s << "\n";
      break;
    }
    default:
      llvm_unreachable("unimplemented output format");
  }

  output.flush();
  return success();
}

LogicalResult translateModuleToBytecode(mlir::ModuleOp outerModuleOp,
                                        BytecodeTargetOptions targetOptions,
                                        llvm::raw_ostream &output) {
  auto moduleOps = outerModuleOp.getOps<IREE::VM::ModuleOp>();
  if (moduleOps.empty()) {
    return outerModuleOp.emitError()
           << "outer module does not contain a vm.module op";
  }
  return translateModuleToBytecode(*moduleOps.begin(), targetOptions, output);
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
