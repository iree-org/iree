// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"

#include <algorithm>

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"
#include "iree/compiler/Dialect/VM/Analysis/ValueLiveness.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/ArchiveWriter.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeEncoder.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Utils/CallingConvention.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/TracingUtils.h"
#include "iree/schemas/bytecode_module_def_builder.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CRC.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(
    mlir::iree_compiler::IREE::VM::BytecodeTargetOptions);

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

namespace {

using namespace llvm::support;

// All constants are defaulted to 16-byte aligned as that is the maximum
// (reasonable) alignment of all data types on all platforms. This can be
// overridden by creators of the rodata with the `alignment` attribute.
static constexpr int kDefaultRodataAlignment = 16;

// Anything over a few KB should be split out of the FlatBuffer.
// This limit is rather arbitrary - we could support hundreds of MB of embedded
// data at the risk of tripping the 31-bit FlatBuffer offset values.
static constexpr int kMaxEmbeddedDataSize = 4 * 1024;

struct TypeDef {
  Type type;
  std::string full_name;
};

// A rodata reference.
// The archive file is empty if the data is to be embedded in the FlatBuffer.
struct RodataRef {
  // Source op.
  IREE::VM::RodataOp rodataOp;
  // Required alignment computed from the rodata or defaults.
  uint64_t alignment = kDefaultRodataAlignment;
  // Total size of the serialized data in bytes.
  uint64_t totalSize = 0;
  // Optional reference to the rodata in the file.
  std::optional<ArchiveWriter::File> archiveFile;
};

}  // namespace

// Gets a file extension based on the given |mimeType| that can be used to help
// applications guess the file type of embedded data.
static StringRef mimeTypeToFileExtension(StringRef mimeType) {
  return StringSwitch<StringRef>(mimeType)
      .Case("application/x-flatbuffers", ".fb")
      .Case("application/octet-stream", ".bin")
      .Case("application/x-elf", ".so")
      .Case("application/x-msdownload", ".dll")
      .Case("application/x-dylib", ".dylib")
      .Case("application/wasm", ".wasm")
      .Case("application/json", ".json")
      .Case("application/x-yaml", ".yaml")
      .Case("text/plain", ".txt")
      .Default(".bin");
}

// Serializes a constant attribute to the FlatBuffer as a binary blob.
// Returns the size in bytes of the serialized value and the FlatBuffers offset
// to the uint8 vec containing the data.
static flatbuffers_uint8_vec_ref_t serializeEmbeddedData(
    Location loc, Attribute valueAttr, uint64_t alignment, uint64_t totalSize,
    FlatbufferBuilder &fbb) {
  flatcc_builder_start_vector(fbb, 1, alignment, FLATBUFFERS_COUNT_MAX(1));

  if (totalSize > SIZE_MAX) {
    mlir::emitError(loc) << "constant size " << totalSize
                         << " exceeds native size_t; unable to serialize";
    return {};
  }

  auto value = valueAttr.dyn_cast<IREE::Util::SerializableAttrInterface>();
  assert(value && "expected a serializable rodata value");

  // Reserve memory in the FlatBuffer for the data.
  uint8_t *bytePtr =
      flatbuffers_uint8_vec_extend(fbb, static_cast<size_t>(totalSize));

  // Serialize the constant into the reserved memory.
  if (failed(value.serializeToBuffer(
          llvm::support::endianness::little,
          ArrayRef<char>(reinterpret_cast<char *>(bytePtr),
                         static_cast<size_t>(totalSize))))) {
    mlir::emitError(loc) << "constant attribute failed to serialize: "
                            "unsupported format or encoding";
    return {};
  }

  return flatbuffers_uint8_vec_end(fbb);
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
      assert(listType.getElementType());
      tryInsertType(listType.getElementType());
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
  llvm::stable_sort(
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
static LogicalResult canonicalizeModule(
    IREE::VM::BytecodeTargetOptions bytecodeOptions,
    IREE::VM::ModuleOp moduleOp) {
  RewritePatternSet patterns(moduleOp.getContext());
  ConversionTarget target(*moduleOp.getContext());
  target.addLegalDialect<IREE::VM::VMDialect>();
  target.addLegalOp<IREE::Util::OptimizationBarrierOp>();

  // Add all VM canonicalization patterns and mark pseudo-ops illegal.
  auto *context = moduleOp.getContext();
  for (auto op : context->getRegisteredOperations()) {
    // Non-serializable ops must be removed prior to serialization.
    if (op.hasTrait<OpTrait::IREE::VM::PseudoOp>()) {
      op.getCanonicalizationPatterns(patterns, context);
      target.setOpAction(op, ConversionTarget::LegalizationAction::Illegal);
    }

    // Debug ops must not be present when stripping.
    // TODO(benvanik): add RemoveDisabledDebugOp pattern.
    if (op.hasTrait<OpTrait::IREE::VM::DebugOnly>() &&
        bytecodeOptions.stripDebugOps) {
      target.setOpAction(op, ConversionTarget::LegalizationAction::Illegal);
    }
  }

  if (failed(applyFullConversion(moduleOp, target, std::move(patterns)))) {
    return moduleOp.emitError() << "unable to fully apply conversion to module";
  }

  PassManager passManager(context);
  // TODO(12938): Handle or investigate failure result.
  auto logicalRes = mlir::applyPassManagerCLOptions(passManager);
  (void)logicalRes;
  mlir::applyDefaultTimingPassManagerCLOptions(passManager);
  passManager.addInstrumentation(std::make_unique<PassTracing>());
  auto &modulePasses = passManager.nest<IREE::VM::ModuleOp>();

  // TODO(benvanik): these ideally happen beforehand but when performing
  // serialization the input IR often has some of these low-level VM ops. In
  // real workflows these have already run earlier and are no-ops.
  modulePasses.addPass(IREE::VM::createGlobalInitializationPass());
  modulePasses.addPass(IREE::VM::createDropEmptyModuleInitializersPass());

  if (bytecodeOptions.optimize) {
    // TODO(benvanik): run this as part of a fixed-point iteration.
    modulePasses.addPass(mlir::createInlinerPass());
    modulePasses.addPass(mlir::createCSEPass());
    modulePasses.addPass(mlir::createCanonicalizerPass());
  }

  modulePasses.addPass(IREE::Util::createDropCompilerHintsPass());

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
    StringRef callingConvention, iree_vm_AttrDef_vec_ref_t attrsRef,
    FlatbufferBuilder &fbb) {
  // If the signature would be empty then let's avoid writing the empty table.
  auto callingConventionRef = fbb.createString(callingConvention);
  if (!callingConventionRef && !attrsRef) {
    return 0;
  }

  iree_vm_FunctionSignatureDef_start(fbb);
  iree_vm_FunctionSignatureDef_calling_convention_add(fbb,
                                                      callingConventionRef);
  iree_vm_FunctionSignatureDef_attrs_add(fbb, attrsRef);
  return iree_vm_FunctionSignatureDef_end(fbb);
}

// Returns a serialized function signature.
static iree_vm_FunctionSignatureDef_ref_t makeImportFunctionSignatureDef(
    IREE::VM::ImportOp importOp, llvm::DenseMap<Type, int> &typeTable,
    FlatbufferBuilder &fbb) {
  // Generate the signature calling convention string based on types.
  auto cconv = makeImportCallingConventionString(importOp);
  if (!cconv.has_value()) return {};
  return createFunctionSignatureDef(importOp.getFunctionType(), typeTable,
                                    cconv.value(), /*attrsRef=*/0, fbb);
}

// Returns a serialized function signature.
static iree_vm_FunctionSignatureDef_ref_t makeFunctionSignatureDef(
    IREE::VM::FuncOp funcOp, llvm::DenseMap<Type, int> &typeTable,
    FlatbufferBuilder &fbb) {
  // Generate the signature calling convention string based on types.
  auto cconv = makeCallingConventionString(funcOp);
  if (!cconv.has_value()) return {};

  // Reflection attributes.
  iree_vm_AttrDef_vec_ref_t attrsRef = 0;
  if (auto attrs = funcOp->getAttrOfType<DictionaryAttr>("iree.reflection")) {
    SmallVector<iree_vm_AttrDef_ref_t, 4> attrRefs;
    for (auto attr : attrs) {
      auto key = attr.getName().strref();
      auto value = attr.getValue().dyn_cast<StringAttr>();
      if (!value || key.empty()) continue;
      // NOTE: if we actually want to keep these we should dedupe them (as the
      // keys and likely several of the values are shared across all functions).
      auto valueRef = fbb.createString(value.getValue());
      auto keyRef = fbb.createString(key);
      attrRefs.push_back(iree_vm_AttrDef_create(fbb, keyRef, valueRef));
    }
    attrsRef =
        iree_vm_AttrDef_vec_create(fbb, attrRefs.data(), attrRefs.size());
  }

  return createFunctionSignatureDef(funcOp.getFunctionType(), typeTable,
                                    cconv.value(), attrsRef, fbb);
}

// Returns a serialized function signature.
static iree_vm_FunctionSignatureDef_ref_t makeInternalFunctionSignatureDef(
    IREE::VM::FuncOp funcOp, llvm::DenseMap<Type, int> &typeTable,
    FlatbufferBuilder &fbb) {
  // Generate the signature calling convention string based on types.
  auto cconv = makeCallingConventionString(funcOp);
  if (!cconv.has_value()) return {};
  return createFunctionSignatureDef(funcOp.getFunctionType(), typeTable,
                                    cconv.value(), /*attrsRef=*/0, fbb);
}

// Walks |rootOp| to find all VM features required by it and its children.
static iree_vm_FeatureBits_enum_t findRequiredFeatures(Operation *rootOp) {
  iree_vm_FeatureBits_enum_t result = 0;
  rootOp->walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::IREE::VM::ExtF32>()) {
      result |= iree_vm_FeatureBits_EXT_F32;
    }
    if (op->hasTrait<OpTrait::IREE::VM::ExtF64>()) {
      result |= iree_vm_FeatureBits_EXT_F64;
    }
  });
  return result;
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
static LogicalResult buildFlatBufferModule(
    IREE::VM::TargetOptions vmOptions,
    IREE::VM::BytecodeTargetOptions bytecodeOptions,
    IREE::VM::ModuleOp moduleOp, MutableArrayRef<RodataRef> rodataRefs,
    FlatbufferBuilder &fbb) {
  // Start the buffer so that we can begin recording data prior to the root
  // table (which we do at the very end). This does not change the layout of the
  // file and is only used to prime the flatcc builder.
  iree_vm_BytecodeModuleDef_start_as_root_with_size(fbb);

  // Debug database is always populated but conditionally written.
  // This allows us to emit the database to a separate file if we want to strip
  // the module but still allow debugging later.
  DebugDatabaseBuilder debugDatabase;

  SymbolTable symbolTable(moduleOp);
  OrdinalCountsAttr ordinalCounts = moduleOp.getOrdinalCountsAttr();
  if (!ordinalCounts) {
    return moduleOp.emitError() << "ordinal_counts attribute not found. The "
                                   "OrdinalAllocationPass must be run before.";
  }

  // Find all structural ops in the module.
  std::vector<IREE::VM::ImportOp> importFuncOps;
  std::vector<IREE::VM::ExportOp> exportFuncOps;
  std::vector<IREE::VM::FuncOp> internalFuncOps;
  importFuncOps.resize(ordinalCounts.getImportFuncs());
  exportFuncOps.resize(ordinalCounts.getExportFuncs());
  internalFuncOps.resize(ordinalCounts.getInternalFuncs());

  for (auto &op : moduleOp.getBlock().getOperations()) {
    if (auto funcOp = dyn_cast<IREE::VM::FuncOp>(op)) {
      internalFuncOps[funcOp.getOrdinal()->getLimitedValue()] = funcOp;
    } else if (auto exportOp = dyn_cast<IREE::VM::ExportOp>(op)) {
      exportFuncOps[exportOp.getOrdinal()->getLimitedValue()] = exportOp;
    } else if (auto importOp = dyn_cast<IREE::VM::ImportOp>(op)) {
      importFuncOps[importOp.getOrdinal()->getLimitedValue()] = importOp;
    }
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
  SmallVector<std::vector<uint8_t>, 8> bytecodeDataParts;
  SmallVector<iree_vm_FunctionDescriptor_t, 8> functionDescriptors;
  bytecodeDataParts.resize(internalFuncOps.size());
  functionDescriptors.resize(internalFuncOps.size());
  iree_vm_FeatureBits_enum_t moduleRequirements = 0;
  size_t totalBytecodeLength = 0;
  for (auto [i, funcOp] : llvm::enumerate(internalFuncOps)) {
    auto encodedFunction = BytecodeEncoder::encodeFunction(
        funcOp, typeOrdinalMap, symbolTable, debugDatabase);
    if (!encodedFunction) {
      return funcOp.emitError() << "failed to encode function bytecode";
    }
    auto funcRequirements = findRequiredFeatures(funcOp);
    moduleRequirements |= funcRequirements;
    iree_vm_FunctionDescriptor_assign(
        &functionDescriptors[i], totalBytecodeLength,
        encodedFunction->bytecodeLength, funcRequirements,
        /*reserved=*/0u, encodedFunction->blockCount,
        encodedFunction->i32RegisterCount, encodedFunction->refRegisterCount);
    totalBytecodeLength += encodedFunction->bytecodeData.size();
    bytecodeDataParts[i] = std::move(encodedFunction->bytecodeData);
  }
  flatbuffers_uint8_vec_start(fbb);
  uint8_t *bytecodeDataPtr =
      flatbuffers_uint8_vec_extend(fbb, totalBytecodeLength);
  // NOTE: we need to ensure we clear the output data in case we have gaps for
  // alignment (where otherwise uninitialized memory might sneak in and be bad
  // for both security and determinism).
  memset(bytecodeDataPtr, 0, totalBytecodeLength);
  size_t currentBytecodeOffset = 0;
  for (const auto &[ordinal, _] : llvm::enumerate(internalFuncOps)) {
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

  // Serialize embedded read-only data and build the rodata references.
  //
  // NOTE: FlatBuffers are built bottom-up; after each rodata we serialize we
  // move *backward* in the file and prepend the next, meaning that if we
  // were to serialize all rodata we'd have it in the opposite order as we do
  // in the IR. Though this it isn't required for correctness, enabling file
  // layout planning by preserving the order in the IR is useful.
  SmallVector<iree_vm_RodataSegmentDef_ref_t, 8> rodataSegmentRefs;
  for (auto &rodataRef : llvm::reverse(rodataRefs)) {
    if (rodataRef.archiveFile.has_value()) {
      // Data is already in the file at a calculated offset.
      iree_vm_RodataSegmentDef_start(fbb);
      iree_vm_RodataSegmentDef_external_data_offset_add(
          fbb, rodataRef.archiveFile->relativeOffset +
                   rodataRef.archiveFile->prefixLength);
      iree_vm_RodataSegmentDef_external_data_length_add(
          fbb, rodataRef.archiveFile->fileLength);
      rodataSegmentRefs.push_back(iree_vm_RodataSegmentDef_end(fbb));
    } else {
      // Serialize the embedded data first so that we can reference it.
      flatbuffers_uint8_vec_ref_t embeddedRef = serializeEmbeddedData(
          rodataRef.rodataOp.getLoc(), rodataRef.rodataOp.getValue(),
          rodataRef.alignment, rodataRef.totalSize, fbb);
      if (!embeddedRef) return failure();
      iree_vm_RodataSegmentDef_start(fbb);
      iree_vm_RodataSegmentDef_embedded_data_add(fbb, embeddedRef);
      rodataSegmentRefs.push_back(iree_vm_RodataSegmentDef_end(fbb));
    }
  }
  std::reverse(rodataSegmentRefs.begin(), rodataSegmentRefs.end());

  // NOTE: rwdata is currently unused.
  SmallVector<iree_vm_RwdataSegmentDef_ref_t, 8> rwdataSegmentRefs;

  auto signatureRefs =
      llvm::to_vector<8>(llvm::map_range(internalFuncOps, [&](auto funcOp) {
        return makeFunctionSignatureDef(funcOp, typeOrdinalMap, fbb);
      }));

  auto exportFuncRefs =
      llvm::to_vector<8>(llvm::map_range(exportFuncOps, [&](auto exportOp) {
        auto localNameRef = fbb.createString(exportOp.getExportName());
        auto funcOp =
            symbolTable.lookup<IREE::VM::FuncOp>(exportOp.getFunctionRef());
        iree_vm_ExportFunctionDef_start(fbb);
        iree_vm_ExportFunctionDef_local_name_add(fbb, localNameRef);
        iree_vm_ExportFunctionDef_internal_ordinal_add(
            fbb, funcOp.getOrdinal()->getLimitedValue());
        return iree_vm_ExportFunctionDef_end(fbb);
      }));

  auto importFuncRefs =
      llvm::to_vector<8>(llvm::map_range(importFuncOps, [&](auto importOp) {
        auto fullNameRef = fbb.createString(importOp.getName());
        auto signatureRef =
            makeImportFunctionSignatureDef(importOp, typeOrdinalMap, fbb);
        iree_vm_ImportFlagBits_enum_t flags =
            importOp.getIsOptional() ? iree_vm_ImportFlagBits_OPTIONAL
                                     : iree_vm_ImportFlagBits_REQUIRED;
        iree_vm_ImportFunctionDef_start(fbb);
        iree_vm_ImportFunctionDef_full_name_add(fbb, fullNameRef);
        iree_vm_ImportFunctionDef_signature_add(fbb, signatureRef);
        iree_vm_ImportFunctionDef_flags_add(fbb, flags);
        return iree_vm_ImportFunctionDef_end(fbb);
      }));

  auto dependencies = moduleOp.getDependencies();
  auto dependencyRefs = llvm::to_vector<8>(
      llvm::map_range(llvm::reverse(dependencies), [&](const auto &dependency) {
        auto nameRef = fbb.createString(dependency.name);
        iree_vm_ModuleDependencyFlagBits_enum_t flags = 0;
        if (dependency.isOptional) {
          // All imported methods are optional and the module is not required.
          flags |= iree_vm_ModuleDependencyFlagBits_OPTIONAL;
        } else {
          // At least one method is required and thus the module is required.
          flags |= iree_vm_ModuleDependencyFlagBits_REQUIRED;
        }
        iree_vm_ModuleDependencyDef_start(fbb);
        iree_vm_ModuleDependencyDef_name_add(fbb, nameRef);
        iree_vm_ModuleDependencyDef_minimum_version_add(
            fbb, dependency.minimumVersion);
        iree_vm_ModuleDependencyDef_flags_add(fbb, flags);
        return iree_vm_ModuleDependencyDef_end(fbb);
      }));

  auto typeRefs =
      llvm::to_vector<8>(llvm::map_range(typeTable, [&](auto typeDef) {
        auto fullNameRef = fbb.createString(typeDef.full_name);
        iree_vm_TypeDef_start(fbb);
        iree_vm_TypeDef_full_name_add(fbb, fullNameRef);
        return iree_vm_TypeDef_end(fbb);
      }));

  // NOTE: we keep the vectors clustered here so that we can hopefully keep the
  // pages mapped at runtime; vector dereferences in FlatBuffers require
  // touching these structs to get length/etc and as such we don't want to be
  // gathering from all over the file (with giant rodata chunks and such
  // inbetween) just to perform a bounds check and deference into another part
  // of the file.
  auto rodataSegmentsRef = fbb.createOffsetVecDestructive(rodataSegmentRefs);
  auto rwdataSegmentsRef = fbb.createOffsetVecDestructive(rwdataSegmentRefs);
  auto signaturesRef = fbb.createOffsetVecDestructive(signatureRefs);
  auto exportFuncsRef = fbb.createOffsetVecDestructive(exportFuncRefs);
  auto importFuncsRef = fbb.createOffsetVecDestructive(importFuncRefs);
  auto dependenciesRef = fbb.createOffsetVecDestructive(dependencyRefs);
  auto typesRef = fbb.createOffsetVecDestructive(typeRefs);

  int32_t globalRefs = ordinalCounts.getGlobalRefs();
  int32_t globalBytes = ordinalCounts.getGlobalBytes();

  iree_vm_ModuleStateDef_ref_t moduleStateDef = 0;
  if (globalBytes || globalRefs) {
    iree_vm_ModuleStateDef_start(fbb);
    iree_vm_ModuleStateDef_global_bytes_capacity_add(fbb, globalBytes);
    iree_vm_ModuleStateDef_global_ref_count_add(fbb, globalRefs);
    moduleStateDef = iree_vm_ModuleStateDef_end(fbb);
  }

  iree_vm_DebugDatabaseDef_ref_t debugDatabaseRef = 0;
  if (!bytecodeOptions.stripSourceMap) {
    debugDatabaseRef = debugDatabase.build(fbb);
  }

  auto moduleNameRef = fbb.createString(
      moduleOp.getSymName().empty() ? "module" : moduleOp.getSymName());

  // TODO(benvanik): let moduleRequirements be a subset of function requirements
  // so that we can multi-version. For now the moduleRequirements will be the OR
  // of all functions.
  iree_vm_FeatureBits_enum_t allowedFeatures = 0;
  if (vmOptions.f32Extension) allowedFeatures |= iree_vm_FeatureBits_EXT_F32;
  if (vmOptions.f64Extension) allowedFeatures |= iree_vm_FeatureBits_EXT_F64;
  if ((moduleRequirements & allowedFeatures) != moduleRequirements) {
    return moduleOp.emitError()
           << "module uses features not allowed by flags (requires "
           << moduleRequirements << ", allowed " << allowedFeatures << ")";
  }

  iree_vm_BytecodeModuleDef_name_add(fbb, moduleNameRef);
  iree_vm_BytecodeModuleDef_version_add(fbb,
                                        moduleOp.getVersion().value_or(0u));
  iree_vm_BytecodeModuleDef_requirements_add(fbb, moduleRequirements);
  // TODO(benvanik): iree_vm_BytecodeModuleDef_attrs_add
  iree_vm_BytecodeModuleDef_types_add(fbb, typesRef);
  iree_vm_BytecodeModuleDef_dependencies_add(fbb, dependenciesRef);
  iree_vm_BytecodeModuleDef_imported_functions_add(fbb, importFuncsRef);
  iree_vm_BytecodeModuleDef_exported_functions_add(fbb, exportFuncsRef);
  iree_vm_BytecodeModuleDef_function_signatures_add(fbb, signaturesRef);
  iree_vm_BytecodeModuleDef_module_state_add(fbb, moduleStateDef);
  iree_vm_BytecodeModuleDef_rodata_segments_add(fbb, rodataSegmentsRef);
  iree_vm_BytecodeModuleDef_rwdata_segments_add(fbb, rwdataSegmentsRef);
  iree_vm_BytecodeModuleDef_function_descriptors_add(fbb,
                                                     functionDescriptorsRef);
  iree_vm_BytecodeModuleDef_bytecode_version_add(fbb,
                                                 BytecodeEncoder::kVersion);
  iree_vm_BytecodeModuleDef_bytecode_data_add(fbb, bytecodeDataRef);
  iree_vm_BytecodeModuleDef_debug_database_add(fbb, debugDatabaseRef);
  iree_vm_BytecodeModuleDef_end_as_root(fbb);

  return success();
}

LogicalResult translateModuleToBytecode(
    IREE::VM::ModuleOp moduleOp, IREE::VM::TargetOptions vmOptions,
    IREE::VM::BytecodeTargetOptions bytecodeOptions,
    llvm::raw_ostream &output) {
  moduleOp.getContext()->getOrLoadDialect<IREE::Util::UtilDialect>();

  if (failed(canonicalizeModule(bytecodeOptions, moduleOp))) {
    return moduleOp.emitError()
           << "failed to canonicalize vm.module to a serializable form";
  }

  // Dump VM assembly source listing to a file and annotate IR locations.
  if (!bytecodeOptions.sourceListing.empty()) {
    OpPrintingFlags printFlags;
    printFlags.elideLargeElementsAttrs(8192);
    if (failed(mlir::generateLocationsFromIR(bytecodeOptions.sourceListing,
                                             "vm", moduleOp, printFlags))) {
      return moduleOp.emitError() << "failed to write source listing to '"
                                  << bytecodeOptions.sourceListing << "'";
    }
  }

  if (bytecodeOptions.outputFormat ==
      BytecodeOutputFormat::kAnnotatedMlirText) {
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

  // Debug-only formats:
  if (bytecodeOptions.outputFormat == BytecodeOutputFormat::kMlirText ||
      bytecodeOptions.outputFormat ==
          BytecodeOutputFormat::kAnnotatedMlirText) {
    // Use the standard MLIR text printer.
    moduleOp.getOperation()->print(output);
    output << "\n";
    return success();
  }

  // Set up the output archive builder based on output format.
  std::unique_ptr<ArchiveWriter> archiveWriter;
  if (bytecodeOptions.emitPolyglotZip &&
      bytecodeOptions.outputFormat == BytecodeOutputFormat::kFlatBufferBinary) {
    archiveWriter =
        std::make_unique<ZIPArchiveWriter>(moduleOp.getLoc(), output);
  } else if (bytecodeOptions.outputFormat ==
             BytecodeOutputFormat::kFlatBufferBinary) {
    archiveWriter =
        std::make_unique<FlatArchiveWriter>(moduleOp.getLoc(), output);
  } else if (bytecodeOptions.outputFormat ==
             BytecodeOutputFormat::kFlatBufferText) {
    archiveWriter =
        std::make_unique<JSONArchiveWriter>(moduleOp.getLoc(), output);
  } else {
    assert(false && "unhandled output format combination");
  }

  // Declare all rodata entries we want to end up as external data first. This
  // allows us to compute offsets if needed without having had to perform
  // serialization yet. Note that not all rodata ends up as external data: if
  // it's small (like strings) we can avoid the extra seeks and keep it more
  // local by embedding it in the FlatBuffer.
  std::vector<IREE::VM::RodataOp> rodataOps;
  rodataOps.resize(moduleOp.getOrdinalCountsAttr().getRodatas());
  for (auto rodataOp : moduleOp.getOps<IREE::VM::RodataOp>()) {
    rodataOps[rodataOp.getOrdinal()->getLimitedValue()] = rodataOp;
  }
  SmallVector<RodataRef> rodataRefs;
  rodataRefs.resize(rodataOps.size());
  for (auto &rodataOp : rodataOps) {
    auto rodataValue =
        rodataOp.getValue().dyn_cast<IREE::Util::SerializableAttrInterface>();
    assert(rodataValue && "expected a serializable rodata value");

    // Split large rodata out of the FlatBuffer to avoid going over 2GB.
    // We also route any rodata that has a mime type defined so that it's
    // easier to work with as a user.
    uint64_t actualSize = rodataValue.getStorageSize();
    bool storeExternal =
        archiveWriter->supportsFiles() && (rodataOp.getMimeType().has_value() ||
                                           actualSize >= kMaxEmbeddedDataSize);

    RodataRef rodataRef;
    rodataRef.rodataOp = rodataOp;
    rodataRef.alignment =
        rodataOp.getAlignment().value_or(kDefaultRodataAlignment);
    rodataRef.totalSize = static_cast<uint64_t>(actualSize);
    if (storeExternal) {
      std::string fileName =
          (rodataOp.getName() +
           mimeTypeToFileExtension(rodataOp.getMimeType().value_or("")))
              .str();
      rodataRef.archiveFile = archiveWriter->declareFile(
          fileName, rodataRef.alignment, rodataRef.totalSize,
          [=](llvm::raw_ostream &os) {
            return rodataValue.serializeToStream(
                llvm::support::endianness::little, os);
          });
    }
    rodataRefs[rodataOp.getOrdinal()->getLimitedValue()] = rodataRef;
  }

  // NOTE: we order things so that all of the metadata is close to the start of
  // the module header in memory. This ensures that when we map the file only
  // the first few pages need to be accessed to get the metadata and the rest
  // can be large bulk data.
  FlatbufferBuilder fbb;
  if (failed(buildFlatBufferModule(vmOptions, bytecodeOptions, moduleOp,
                                   rodataRefs, fbb))) {
    return failure();
  }
  if (failed(archiveWriter->flush(fbb))) {
    return failure();
  }
  archiveWriter.reset();

  return success();
}

LogicalResult translateModuleToBytecode(
    mlir::ModuleOp outerModuleOp, IREE::VM::TargetOptions vmOptions,
    IREE::VM::BytecodeTargetOptions bytecodeOptions,
    llvm::raw_ostream &output) {
  auto moduleOps = outerModuleOp.getOps<IREE::VM::ModuleOp>();
  if (moduleOps.empty()) {
    return outerModuleOp.emitError()
           << "outer module does not contain a vm.module op";
  }
  return translateModuleToBytecode(*moduleOps.begin(), vmOptions,
                                   bytecodeOptions, output);
}

void BytecodeTargetOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory vmBytecodeOptionsCategory(
      "IREE VM bytecode options");

  binder.opt<BytecodeOutputFormat>(
      "iree-vm-bytecode-module-output-format", outputFormat,
      llvm::cl::cat(vmBytecodeOptionsCategory),
      llvm::cl::desc("Output format the bytecode module is written in"),
      llvm::cl::values(
          clEnumValN(BytecodeOutputFormat::kFlatBufferBinary,
                     "flatbuffer-binary", "Binary FlatBuffer file"),
          clEnumValN(BytecodeOutputFormat::kFlatBufferText, "flatbuffer-text",
                     "Text FlatBuffer file, debug-only"),
          clEnumValN(BytecodeOutputFormat::kMlirText, "mlir-text",
                     "MLIR module file in the VM dialect"),
          clEnumValN(BytecodeOutputFormat::kAnnotatedMlirText,
                     "annotated-mlir-text",
                     "MLIR module file in the VM dialect with annotations")));
  binder.opt<bool>(
      "iree-vm-bytecode-module-optimize", optimize,
      llvm::cl::cat(vmBytecodeOptionsCategory),
      llvm::cl::desc("Optimizes the VM module with CSE/inlining/etc prior to "
                     "serialization"));
  binder.opt<std::string>(
      "iree-vm-bytecode-source-listing", sourceListing,
      llvm::cl::cat(vmBytecodeOptionsCategory),
      llvm::cl::desc(
          "Dump a VM MLIR file and annotate source locations with it"));
  binder.opt<bool>("iree-vm-bytecode-module-strip-source-map", stripSourceMap,
                   llvm::cl::cat(vmBytecodeOptionsCategory),
                   llvm::cl::desc("Strips the source map from the module"));
  binder.opt<bool>("iree-vm-bytecode-module-strip-debug-ops", stripDebugOps,
                   llvm::cl::cat(vmBytecodeOptionsCategory),
                   llvm::cl::desc("Strips debug-only ops from the module"));
  binder.opt<bool>(
      "iree-vm-emit-polyglot-zip", emitPolyglotZip,
      llvm::cl::cat(vmBytecodeOptionsCategory),
      llvm::cl::desc(
          "Enables output files to be viewed as zip files for debugging "
          "(only applies to binary targets)"));
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
