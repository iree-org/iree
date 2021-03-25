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

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/IREE/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"
#include "iree/compiler/Dialect/VM/Analysis/ValueLiveness.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeEncoder.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/ConstantEncoder.h"
#include "iree/compiler/Dialect/VM/Target/CallingConventionUtils.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/TracingUtils.h"
#include "iree/schemas/bytecode_module_def_builder.h"
#include "iree/schemas/bytecode_module_def_json_printer.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
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

struct TypeDef {
  Type type;
  std::string full_name;
};

}  // namespace

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
  OwningRewritePatternList patterns(moduleOp.getContext());
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

  if (failed(applyFullConversion(moduleOp, target, std::move(patterns)))) {
    return moduleOp.emitError() << "unable to fully apply conversion to module";
  }

  PassManager passManager(context);
  mlir::applyPassManagerCLOptions(passManager);
  passManager.addInstrumentation(std::make_unique<PassTracing>());
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

  if (failed(passManager.run(moduleOp->getParentOfType<mlir::ModuleOp>()))) {
    return moduleOp.emitError() << "failed during transform passes";
  }

  return success();
}

// Creates a FunctionSignatureDef based on the given function metadata.
// Some fields are not used on all signature defs and added only when present on
// the argument objects/attrs.
static iree_vm_FunctionSignatureDef_ref_t createFunctionSignatureDef(
    FunctionType functionType, llvm::DenseMap<Type, int> &typeTable,
    StringRef callingConvention,
    iree_vm_ReflectionAttrDef_vec_ref_t reflectionAttrsRef,
    FlatbufferBuilder &fbb) {
  auto resultTypesRef = fbb.createInt32Vec(
      llvm::map_range(functionType.getResults(), [&](Type type) {
        if (auto refPtrType = type.dyn_cast<IREE::VM::RefType>()) {
          type = refPtrType.getObjectType();
        }
        return typeTable.lookup(type);
      }));
  auto argumentTypesRef = fbb.createInt32Vec(
      llvm::map_range(functionType.getInputs(), [&](Type type) {
        if (auto refPtrType = type.dyn_cast<IREE::VM::RefType>()) {
          type = refPtrType.getObjectType();
        }
        return typeTable.lookup(type);
      }));

  auto callingConventionRef = fbb.createString(callingConvention);

  // If the signature would be empty then let's avoid writing the empty table.
  if (!argumentTypesRef && !resultTypesRef && !callingConventionRef &&
      !reflectionAttrsRef) {
    return 0;
  }

  iree_vm_FunctionSignatureDef_start(fbb);
  iree_vm_FunctionSignatureDef_argument_types_add(fbb, argumentTypesRef);
  iree_vm_FunctionSignatureDef_result_types_add(fbb, resultTypesRef);
  iree_vm_FunctionSignatureDef_calling_convention_add(fbb,
                                                      callingConventionRef);
  iree_vm_FunctionSignatureDef_reflection_attrs_add(fbb, reflectionAttrsRef);
  return iree_vm_FunctionSignatureDef_end(fbb);
}

// Returns a serialized function signature.
static iree_vm_FunctionSignatureDef_ref_t makeImportFunctionSignatureDef(
    IREE::VM::ImportOp importOp, llvm::DenseMap<Type, int> &typeTable,
    FlatbufferBuilder &fbb) {
  // Generate the signature calling convention string based on types.
  auto cconv = makeImportCallingConventionString(importOp);
  if (!cconv.hasValue()) return {};
  return createFunctionSignatureDef(importOp.getType(), typeTable,
                                    cconv.getValue(), /*reflectionAttrsRef=*/0,
                                    fbb);
}

// Returns a serialized function signature.
static iree_vm_FunctionSignatureDef_ref_t makeExportFunctionSignatureDef(
    IREE::VM::ExportOp exportOp, IREE::VM::FuncOp funcOp,
    llvm::DenseMap<Type, int> &typeTable, FlatbufferBuilder &fbb) {
  // Generate the signature calling convention string based on types.
  auto cconv = makeCallingConventionString(funcOp);
  if (!cconv.hasValue()) return {};
  return createFunctionSignatureDef(funcOp.getType(), typeTable,
                                    cconv.getValue(), /*reflectionAttrsRef=*/0,
                                    fbb);
}

// Returns a serialized function signature.
static iree_vm_FunctionSignatureDef_ref_t makeInternalFunctionSignatureDef(
    IREE::VM::FuncOp funcOp, llvm::DenseMap<Type, int> &typeTable,
    FlatbufferBuilder &fbb) {
  // Generate the signature calling convention string based on types.
  // TODO(benvanik): only do this on exports. The runtime currently looks on
  // internal functions, though, so we have to have it here.
  auto cconv = makeCallingConventionString(funcOp);
  if (!cconv.hasValue()) return {};

  // Reflection attributes.
  // TODO(benvanik): move these to exports (or remove entirely).
  iree_vm_ReflectionAttrDef_vec_ref_t reflectionAttrsRef = 0;
  if (auto reflectionAttrs =
          funcOp->getAttrOfType<DictionaryAttr>("iree.reflection")) {
    SmallVector<iree_vm_ReflectionAttrDef_ref_t, 4> reflectionAttrRefs;
    for (auto reflectionAttr : reflectionAttrs) {
      auto key = reflectionAttr.first.strref();
      auto value = reflectionAttr.second.dyn_cast<StringAttr>();
      if (!value || key.empty()) continue;
      // NOTE: if we actually want to keep these we should dedupe them (as the
      // keys and likely several of the values are shared across all functions).
      auto valueRef = fbb.createString(value.getValue());
      auto keyRef = fbb.createString(key);
      reflectionAttrRefs.push_back(
          iree_vm_ReflectionAttrDef_create(fbb, keyRef, valueRef));
    }
    reflectionAttrsRef = iree_vm_ReflectionAttrDef_vec_create(
        fbb, reflectionAttrRefs.data(), reflectionAttrRefs.size());
  }

  return createFunctionSignatureDef(funcOp.getType(), typeTable,
                                    cconv.getValue(), reflectionAttrsRef, fbb);
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
static LogicalResult buildFlatBufferModule(BytecodeTargetOptions targetOptions,
                                           IREE::VM::ModuleOp moduleOp,
                                           FlatbufferBuilder &fbb) {
  SymbolTable symbolTable(moduleOp);
  if (!moduleOp.ordinal_counts().hasValue()) {
    return moduleOp.emitError() << "ordinal_counts attribute not found. The "
                                   "OrdinalAllocationPass must be run before.";
  }
  OrdinalCountsAttr ordinalCounts = moduleOp.ordinal_counts().getValue();

  // Find all structural ops in the module.
  std::vector<IREE::VM::ImportOp> importFuncOps;
  std::vector<IREE::VM::ExportOp> exportFuncOps;
  std::vector<IREE::VM::FuncOp> internalFuncOps;
  std::vector<IREE::VM::RodataOp> rodataOps;
  importFuncOps.resize(ordinalCounts.import_funcs());
  exportFuncOps.resize(ordinalCounts.export_funcs());
  internalFuncOps.resize(ordinalCounts.internal_funcs());
  rodataOps.resize(ordinalCounts.rodatas());

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
  //
  // NOTE: flatbuffers are built bottom-up; after each rodata we serialize we
  // move *backward* in the file and prepend the next, meaning that if we
  // were to serialize all rodata we'd have it in the opposite order as we do
  // in the IR. Though this it isn't required for correctness, enabling file
  // layout planning by preserving the order in the IR is useful.
  SmallVector<flatbuffers_uint8_vec_ref_t, 8> rodataContentRefs;
  rodataContentRefs.reserve(rodataOps.size());
  for (auto rodataOp : llvm::reverse(rodataOps)) {
    auto rodataRef =
        serializeConstant(rodataOp.getLoc(), rodataOp.value(), fbb);
    if (!rodataRef) {
      return rodataOp.emitOpError() << "failed to encode";
    }
    rodataContentRefs.push_back(rodataRef);
  }
  // List of references needs to be swapped forward (we wrote backward).
  std::reverse(rodataContentRefs.begin(), rodataContentRefs.end());

  // Find all types in the module to build the type table.
  // Note that we don't emit it yet as we want to keep it near the top of the
  // file (which, in FlatBuffers, is written last).
  auto typeTable = buildTypeTable(moduleOp);
  llvm::DenseMap<Type, int> typeOrdinalMap;
  for (auto typeDef : llvm::enumerate(typeTable)) {
    typeOrdinalMap[typeDef.value().type] = typeDef.index();
  }

  // Serialize function bytecode one at a time and then merge at the end.
  SmallVector<std::vector<uint8_t>, 8> bytecodeDataParts;
  SmallVector<iree_vm_FunctionDescriptor_t, 8> functionDescriptors;
  bytecodeDataParts.resize(internalFuncOps.size());
  functionDescriptors.resize(internalFuncOps.size());
  size_t totalBytecodeLength = 0;
  for (auto funcOp : llvm::enumerate(internalFuncOps)) {
    auto encodedFunction = BytecodeEncoder::encodeFunction(
        funcOp.value(), typeOrdinalMap, symbolTable);
    if (!encodedFunction) {
      return funcOp.value().emitError() << "failed to encode function bytecode";
    }
    iree_vm_FunctionDescriptor_assign(
        &functionDescriptors[funcOp.index()], totalBytecodeLength,
        encodedFunction->bytecodeData.size(), encodedFunction->i32RegisterCount,
        encodedFunction->refRegisterCount);
    totalBytecodeLength += encodedFunction->bytecodeData.size();
    bytecodeDataParts[funcOp.index()] =
        std::move(encodedFunction->bytecodeData);
  }
  flatbuffers_uint8_vec_start(fbb);
  uint8_t *bytecodeDataPtr =
      flatbuffers_uint8_vec_extend(fbb, totalBytecodeLength);
  // NOTE: we need to ensure we clear the output data in case we have gaps for
  // alignment (where otherwise uninitialized memory might sneak in and be bad
  // for both security and determinism).
  memset(bytecodeDataPtr, 0, totalBytecodeLength);
  size_t currentBytecodeOffset = 0;
  for (const auto &it : llvm::enumerate(internalFuncOps)) {
    int ordinal = it.index();
    auto data = std::move(bytecodeDataParts[ordinal]);
    std::memcpy(bytecodeDataPtr + currentBytecodeOffset, data.data(),
                data.size());
    currentBytecodeOffset += data.size();
  }
  auto bytecodeDataRef = flatbuffers_uint8_vec_end(fbb);

  // Encode the function descriptors adjacent to the bytcode data; they are
  // always accessed together. Descriptor 0 is likely within a few hundred bytes
  // of the referenced bytecode data offset 0, and from there we are at least
  // able to hope sequential readahead caching helps; if not, at least we
  // hopefully don't fault on the first function call every time.
  auto functionDescriptorsRef = iree_vm_FunctionDescriptor_vec_create(
      fbb, functionDescriptors.data(), functionDescriptors.size());

  // Serialize metadata that should be near the front of the file.
  auto rodataSegmentRefs = llvm::to_vector<8>(
      llvm::map_range(rodataContentRefs, [&](auto rodataContentRef) {
        iree_vm_RodataSegmentDef_start(fbb);
        iree_vm_RodataSegmentDef_data_add(fbb, rodataContentRef);
        return iree_vm_RodataSegmentDef_end(fbb);
      }));
  SmallVector<iree_vm_RwdataSegmentDef_ref_t, 8> rwdataSegmentRefs;
  // NOTE: rwdata current unused.
  auto typeRefs =
      llvm::to_vector<8>(llvm::map_range(typeTable, [&](auto typeDef) {
        auto fullNameRef = fbb.createString(typeDef.full_name);
        iree_vm_TypeDef_start(fbb);
        iree_vm_TypeDef_full_name_add(fbb, fullNameRef);
        return iree_vm_TypeDef_end(fbb);
      }));
  auto importFuncRefs =
      llvm::to_vector<8>(llvm::map_range(importFuncOps, [&](auto importOp) {
        auto fullNameRef = fbb.createString(importOp.getName());
        auto signatureRef =
            makeImportFunctionSignatureDef(importOp, typeOrdinalMap, fbb);
        iree_vm_ImportFunctionDef_start(fbb);
        iree_vm_ImportFunctionDef_full_name_add(fbb, fullNameRef);
        iree_vm_ImportFunctionDef_signature_add(fbb, signatureRef);
        return iree_vm_ImportFunctionDef_end(fbb);
      }));
  auto exportFuncRefs =
      llvm::to_vector<8>(llvm::map_range(exportFuncOps, [&](auto exportOp) {
        auto localNameRef = fbb.createString(exportOp.export_name());
        auto funcOp =
            symbolTable.lookup<IREE::VM::FuncOp>(exportOp.function_ref());
        auto signatureRef = makeExportFunctionSignatureDef(exportOp, funcOp,
                                                           typeOrdinalMap, fbb);
        iree_vm_ExportFunctionDef_start(fbb);
        iree_vm_ExportFunctionDef_local_name_add(fbb, localNameRef);
        iree_vm_ExportFunctionDef_signature_add(fbb, signatureRef);
        iree_vm_ExportFunctionDef_internal_ordinal_add(
            fbb, funcOp.ordinal().getValue().getLimitedValue());
        return iree_vm_ExportFunctionDef_end(fbb);
      }));
  SmallVector<iree_vm_InternalFunctionDef_ref_t, 8> internalFuncRefs;
  if (!targetOptions.stripSymbols) {
    internalFuncRefs.reserve(internalFuncOps.size());
    for (auto funcOp : internalFuncOps) {
      auto localNameRef = fbb.createString(funcOp.getName());
      auto signatureRef =
          makeInternalFunctionSignatureDef(funcOp, typeOrdinalMap, fbb);
      iree_vm_InternalFunctionDef_start(fbb);
      iree_vm_InternalFunctionDef_local_name_add(fbb, localNameRef);
      iree_vm_InternalFunctionDef_signature_add(fbb, signatureRef);
      internalFuncRefs.push_back(iree_vm_InternalFunctionDef_end(fbb));
    }
  }

  // NOTE: we keep the vectors clustered here so that we can hopefully keep the
  // pages mapped at runtime; vector dereferences in flatbuffers require
  // touching these structs to get length/etc and as such we don't want to be
  // gathering from all over the file (with giant rodata chunks and such
  // inbetween) just to perform a bounds check and deference into another part
  // of the file.
  auto rodataSegmentsRef = fbb.createOffsetVecDestructive(rodataSegmentRefs);
  auto rwdataSegmentsRef = fbb.createOffsetVecDestructive(rwdataSegmentRefs);
  auto internalFuncsRef = fbb.createOffsetVecDestructive(internalFuncRefs);
  auto exportFuncsOffset = fbb.createOffsetVecDestructive(exportFuncRefs);
  auto importFuncsRef = fbb.createOffsetVecDestructive(importFuncRefs);
  auto typesRef = fbb.createOffsetVecDestructive(typeRefs);

  int32_t globalRefs = ordinalCounts.global_refs();
  int32_t globalBytes = ordinalCounts.global_bytes();

  iree_vm_ModuleStateDef_ref_t moduleStateDef = 0;
  if (globalBytes || globalRefs) {
    iree_vm_ModuleStateDef_start(fbb);
    iree_vm_ModuleStateDef_global_bytes_capacity_add(fbb, globalBytes);
    iree_vm_ModuleStateDef_global_ref_count_add(fbb, globalRefs);
    moduleStateDef = iree_vm_ModuleStateDef_end(fbb);
  }

  auto moduleNameRef = fbb.createString(
      moduleOp.sym_name().empty() ? "module" : moduleOp.sym_name());

  iree_vm_BytecodeModuleDef_start_as_root(fbb);
  iree_vm_BytecodeModuleDef_name_add(fbb, moduleNameRef);
  iree_vm_BytecodeModuleDef_types_add(fbb, typesRef);
  iree_vm_BytecodeModuleDef_imported_functions_add(fbb, importFuncsRef);
  iree_vm_BytecodeModuleDef_exported_functions_add(fbb, exportFuncsOffset);
  iree_vm_BytecodeModuleDef_internal_functions_add(fbb, internalFuncsRef);
  iree_vm_BytecodeModuleDef_module_state_add(fbb, moduleStateDef);
  iree_vm_BytecodeModuleDef_rodata_segments_add(fbb, rodataSegmentsRef);
  iree_vm_BytecodeModuleDef_rwdata_segments_add(fbb, rwdataSegmentsRef);
  iree_vm_BytecodeModuleDef_function_descriptors_add(fbb,
                                                     functionDescriptorsRef);
  iree_vm_BytecodeModuleDef_bytecode_data_add(fbb, bytecodeDataRef);
  iree_vm_BytecodeModuleDef_end_as_root(fbb);
  return success();
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
  FlatbufferBuilder fbb;
  if (failed(buildFlatBufferModule(targetOptions, moduleOp, fbb))) {
    return moduleOp.emitError()
           << "failed to build FlatBuffer BytecodeModuleDef";
  }

  switch (targetOptions.outputFormat) {
    case BytecodeOutputFormat::kFlatBufferBinary:
      if (failed(fbb.copyToStream(output))) {
        return moduleOp.emitError()
               << "failed to copy flatbuffer emitter contents to output stream "
                  "- possibly out of memory";
      }
      break;
    case BytecodeOutputFormat::kFlatBufferText: {
      if (failed(fbb.printJsonToStream(/*pretty=*/true,
                                       /*includeDefaults=*/false,
                                       bytecode_module_def_print_json,
                                       output))) {
        return moduleOp.emitError()
               << "failed to print flatbuffer emitter contents to output "
                  "stream - possibly out of memory, possibly unprintable "
                  "structure";
      }
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
