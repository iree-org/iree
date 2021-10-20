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
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeEncoder.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Utils/CallingConvention.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/TracingUtils.h"
#include "iree/schemas/bytecode_module_def_builder.h"
#include "iree/schemas/bytecode_module_def_json_printer.h"
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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Translation.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

namespace {

using namespace llvm::support;

struct TypeDef {
  Type type;
  std::string full_name;
};

struct SerializedConstantRef {
  flatbuffers_uint8_vec_ref_t ref = 0;
  int64_t totalSize = 0;
  uint32_t crc32 = 0;
};

// Serializes a constant attribute to the FlatBuffer as a binary blob.
// Returns the size in bytes of the serialized value and the flatbuffers offset
// to the uint8 vec containing the data. If |calculateCRC32| is provided then a
// CRC32 of the data will be computed and returned as well.
SerializedConstantRef serializeConstant(Location loc, Attribute valueAttr,
                                        size_t alignment, bool calculateCRC32,
                                        FlatbufferBuilder &fbb) {
  flatcc_builder_start_vector(fbb, 1, alignment, FLATBUFFERS_COUNT_MAX(1));

  auto value = valueAttr.dyn_cast<IREE::Util::SerializableAttrInterface>();
  assert(value && "expected a serializable rodata value");

  // TODO(benvanik): use fbb.streamUint8Vec + value.serializeToStream.
  // Right now this will allocate a single slab of the entire storage size and
  // write the contents into it. streamUint8Vec also does the same thing but
  // we could extend it with custom fbb storage such that we could reserve the
  // size in the file and then fix it up after we write it. The complication is
  // that we need the CRC below and thus have to have the bytes in memory at
  // some point. An interface member for computeCRC() could be useful as even
  // though slow it would avoid the need to malloc everything. We could also
  // switch implementations based on calculateCRC32 - models with GB of params
  // are probably fine not to have nice hackability :)
  uint64_t actualSize = value.getStorageSize();
  if (actualSize > SIZE_MAX) {
    mlir::emitError(loc) << "constant size " << actualSize
                         << " exceeds native size_t; unable to serialize";
    return {};
  }
  size_t size = static_cast<size_t>(value.getStorageSize());
  uint8_t *bytePtr = flatbuffers_uint8_vec_extend(fbb, size);
  if (failed(value.serializeToBuffer(llvm::support::endianness::little,
                                     ArrayRef<char>((char *)bytePtr, size)))) {
    return {};
  }

  uint8_t *dataPtr =
      reinterpret_cast<uint8_t *>(flatcc_builder_vector_edit(fbb));
  size_t totalSize = flatcc_builder_vector_count(fbb);
  uint32_t crc32Value = 0;
  if (calculateCRC32) {
    crc32Value = llvm::crc32(0u, ArrayRef<uint8_t>(dataPtr, totalSize));
  }
  return SerializedConstantRef{
      flatbuffers_uint8_vec_end(fbb),
      static_cast<int64_t>(totalSize),
      crc32Value,
  };
}

LLVM_PACKED_START
struct ZIPEndOfCentralDirectoryRecord {
  ulittle32_t signature;  // 0x06054B50
  ulittle16_t diskNumber;
  ulittle16_t startDiskNumber;
  ulittle16_t entriesOnDisk;
  ulittle16_t entryCount;
  ulittle32_t directorySize;
  ulittle32_t directoryOffset;
  ulittle16_t commentLength;
  // comment (variable size)
};
static_assert(sizeof(ZIPEndOfCentralDirectoryRecord) == 22, "bad packing");
struct ZIPCentralDirectoryRecord {
  ulittle32_t signature;  // 0x02014B50
  ulittle16_t versionMadeBy;
  ulittle16_t versionToExtract;
  ulittle16_t generalPurposeFlags;
  ulittle16_t compressionMethod;
  ulittle16_t lastModifiedTime;
  ulittle16_t lastModifiedDate;
  ulittle32_t crc32;
  ulittle32_t compressedSize;
  ulittle32_t uncompressedSize;
  ulittle16_t fileNameLength;
  ulittle16_t extraFieldLength;
  ulittle16_t fileCommentLength;
  ulittle16_t diskStartNumber;
  ulittle16_t internalFileAttributes;
  ulittle32_t externalFileAttributes;
  ulittle32_t localHeaderOffset;
  // file name (variable size)
  // extra field (variable size)
  // file comment (variable size)
};
static_assert(sizeof(ZIPCentralDirectoryRecord) == 46, "bad packing");
struct ZIPLocalFileHeader {
  ulittle32_t signature;  // 0x04034B50
  ulittle16_t versionToExtract;
  ulittle16_t generalPurposeFlag;
  ulittle16_t compressionMethod;
  ulittle16_t lastModifiedTime;
  ulittle16_t lastModifiedDate;
  ulittle32_t crc32;
  ulittle32_t compressedSize;
  ulittle32_t uncompressedSize;
  ulittle16_t fileNameLength;
  ulittle16_t extraFieldLength;
  // file name (variable size)
  // extra field (variable size)
};
static_assert(sizeof(ZIPLocalFileHeader) == 30, "bad packing");
struct ZIPExtraFieldHeader {
  ulittle16_t id;
  ulittle16_t size;
};
static_assert(sizeof(ZIPExtraFieldHeader) == 4, "bad packing");
LLVM_PACKED_END

// A ZIP file reference into the flatbuffer output data.
struct ZIPFileRef {
  // Offset of the local file header in the flatbuffer. Relative to the end of
  // the file.
  flatcc_builder_ref_t localHeaderOffset;
  // Name of the file used within the ZIP archive.
  std::string fileName;
  // Total size, in bytes, of the file uncompressed.
  uint32_t totalSize;
  // CRC32 of the file.
  uint32_t crc32;
  // Extra field padding (total).
  uint16_t paddingLength;
};

// TODO(benvanik): figure out why we need to offset all flatbuffer refs by this
// value in order to get proper absolute file offsets. The current value used
// here was derived empirically and is like a combination of the flatbuffer
// file prefix and some alignment.
static constexpr int kZIPMagicLocalOffset = 90;

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

// Appends a ZIP local file header at the current location.
// The header is a prefix to the actual rodata contents. ZIP requires that the
// payload start immediately after the header but we have the flatbuffer header
// there. To skip over the flatbuffer data we pad out the header with a dummy
// extra data field that lets us control the length.
//
//  [zip local file header] + 4 byte suffix length
//  [flatbuffer vector header] (4 bytes)
//  [payload]
static ZIPFileRef appendZIPLocalFileHeader(IREE::VM::RodataOp rodataOp,
                                           size_t rodataSize, uint32_t crc32,
                                           FlatbufferBuilder &fbb) {
  // Use the mime type to map to a file extension.
  std::string fileName =
      (rodataOp.getName() +
       mimeTypeToFileExtension(rodataOp.mime_type().getValueOr("")))
          .str();

  // The data is stored in the flatbuffer prefixed with a vector header of
  // a 32-bit byte count. We need to ignore this when computing the CRC as we
  // want only the payload to be visible in the ZIP.
  size_t vectorPrefixLength = sizeof(uint32_t);

  // header + file name + extra field header
  size_t totalHeaderLength = sizeof(ZIPLocalFileHeader) + fileName.size() +
                             sizeof(ZIPExtraFieldHeader);

  // Append local file header.
  auto *header = reinterpret_cast<ZIPLocalFileHeader *>(
      flatcc_builder_start_struct(fbb, totalHeaderLength, 1));
  header->signature = 0x04034B50u;
  header->versionToExtract = 0;
  header->generalPurposeFlag = 0;
  header->compressionMethod = 0;  // COMP_STORED
  header->lastModifiedTime = 0;
  header->lastModifiedDate = 0;
  header->crc32 = crc32;
  header->compressedSize = static_cast<uint32_t>(rodataSize);
  header->uncompressedSize = static_cast<uint32_t>(rodataSize);
  header->fileNameLength = static_cast<uint16_t>(fileName.size());
  header->extraFieldLength = sizeof(ZIPExtraFieldHeader) + vectorPrefixLength;
  char *fileNamePtr = reinterpret_cast<char *>(header + 1);
  memcpy(fileNamePtr, fileName.data(), fileName.size());
  auto *extraField =
      reinterpret_cast<ZIPExtraFieldHeader *>(fileNamePtr + fileName.size());
  extraField->id = 0xFECAu;
  extraField->size = static_cast<uint16_t>(vectorPrefixLength);
  flatcc_builder_ref_t relativeHeaderOffset = flatcc_builder_end_struct(fbb);

  ZIPFileRef fileRef;
  fileRef.localHeaderOffset = relativeHeaderOffset;
  fileRef.fileName = std::move(fileName);
  fileRef.totalSize = static_cast<uint32_t>(rodataSize);
  fileRef.crc32 = crc32;
  fileRef.paddingLength = static_cast<uint16_t>(vectorPrefixLength);
  return fileRef;
}

// Appends a ZIP central directory to |output| with the references to all of
// |zipFileRefs| with offsets applied. |startOffset| and |endOffset| define the
// absolute offsets into |output| of the flatbuffer data.
//
// The technique used here is the same as that used in self-extracting archives:
// byte offset 0 of the file will contain the native format header (like the
// flatbuffers file identifier) and a ZIP application will need to scan from the
// back of the file to find the ZIP central directory. This often means that
// naming the file .zip will not work: most ZIP applications will try to find
// a PK header at byte 0.
static void appendZIPCentralDirectory(ArrayRef<ZIPFileRef> zipFileRefs,
                                      uint64_t startOffset, uint64_t endOffset,
                                      llvm::raw_ostream &output) {
  // Append the central directory, which contains the local file headers with
  // some extra junk and references back to where the local headers are in the
  // file.
  uint64_t centralDirectoryStartOffset = output.tell();
  for (auto zipFileRef : zipFileRefs) {
    // Fixed-size header.
    ZIPCentralDirectoryRecord cdr;
    cdr.signature = 0x02014B50u;
    cdr.versionMadeBy = 798;
    cdr.versionToExtract = 20;
    cdr.generalPurposeFlags = 0;
    cdr.compressionMethod = 0;  // COMP_STORED
    cdr.lastModifiedTime = 0;
    cdr.lastModifiedDate = 0;
    cdr.crc32 = zipFileRef.crc32;
    cdr.compressedSize = zipFileRef.totalSize;
    cdr.uncompressedSize = zipFileRef.totalSize;
    cdr.fileNameLength = static_cast<uint16_t>(zipFileRef.fileName.size());
    cdr.extraFieldLength =
        sizeof(ZIPExtraFieldHeader) + zipFileRef.paddingLength;
    cdr.fileCommentLength = 0;
    cdr.diskStartNumber = 0;
    cdr.internalFileAttributes = 0;
    cdr.externalFileAttributes = 0;
    cdr.localHeaderOffset = static_cast<uint32_t>(
        endOffset + zipFileRef.localHeaderOffset - kZIPMagicLocalOffset);
    output.write(reinterpret_cast<const char *>(&cdr), sizeof(cdr));
    output.write(zipFileRef.fileName.data(), zipFileRef.fileName.size());
    ZIPExtraFieldHeader extraField;
    extraField.id = 0xFECAu;
    extraField.size = zipFileRef.paddingLength;
    output.write(reinterpret_cast<const char *>(&extraField),
                 sizeof(extraField));
    output.write_zeros(extraField.size);
  }
  uint64_t centralDirectoryEndOffset = output.tell();

  // Append the final ZIP file footer.
  // NOTE: this must come at the very end of the file.
  ZIPEndOfCentralDirectoryRecord endOfCDR;
  endOfCDR.signature = 0x06054B50u;
  endOfCDR.diskNumber = 0;
  endOfCDR.startDiskNumber = 0;
  endOfCDR.entriesOnDisk = static_cast<uint16_t>(zipFileRefs.size());
  endOfCDR.entryCount = static_cast<uint16_t>(zipFileRefs.size());
  endOfCDR.directorySize = static_cast<uint32_t>(centralDirectoryEndOffset -
                                                 centralDirectoryStartOffset);
  endOfCDR.directoryOffset = static_cast<uint32_t>(centralDirectoryStartOffset);
  endOfCDR.commentLength = 0;
  output.write(reinterpret_cast<const char *>(&endOfCDR), sizeof(endOfCDR));
}

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
static LogicalResult canonicalizeModule(BytecodeTargetOptions targetOptions,
                                        IREE::VM::ModuleOp moduleOp) {
  OwningRewritePatternList patterns(moduleOp.getContext());
  ConversionTarget target(*moduleOp.getContext());
  target.addLegalDialect<IREE::VM::VMDialect>();
  target.addLegalOp<IREE::Util::DoNotOptimizeOp>();

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
  mlir::applyDefaultTimingPassManagerCLOptions(passManager);
  passManager.addInstrumentation(std::make_unique<PassTracing>());
  auto &modulePasses = passManager.nest<IREE::VM::ModuleOp>();

  if (targetOptions.optimize) {
    // TODO(benvanik): does this run until it quiesces?
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

  // Reflection attributes.
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

// Returns a serialized function signature.
static iree_vm_FunctionSignatureDef_ref_t makeInternalFunctionSignatureDef(
    IREE::VM::FuncOp funcOp, llvm::DenseMap<Type, int> &typeTable,
    FlatbufferBuilder &fbb) {
  // Generate the signature calling convention string based on types.
  auto cconv = makeCallingConventionString(funcOp);
  if (!cconv.hasValue()) return {};
  return createFunctionSignatureDef(funcOp.getType(), typeTable,
                                    cconv.getValue(), /*reflectionAttrsRef=*/0,
                                    fbb);
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
                                           SmallVector<ZIPFileRef> &zipFileRefs,
                                           FlatbufferBuilder &fbb) {
  // Start the buffer so that we can begin recording data prior to the root
  // table (which we do at the very end). This does not change the layout of the
  // file and is only used to prime the flatcc builder.
  iree_vm_BytecodeModuleDef_start_as_root(fbb);

  // Debug database is always populated but conditionally written.
  // This allows us to emit the database to a separate file if we want to strip
  // the module but still allow debugging later.
  DebugDatabaseBuilder debugDatabase;

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

  // All constants are defaulted to 16-byte aligned as that is the maximum
  // (reasonable) alignment of all data types on all platforms. This can be
  // overridden by creators of the rodata with the `alignment` attribute.
  static constexpr int kDefaultRodataAlignment = 16;

  for (auto rodataOp : llvm::reverse(rodataOps)) {
    // Only include rodata entries in the ZIP if they are file-like. This
    // prevents all of our string tables from getting included.
    bool includeInZIP =
        targetOptions.emitPolyglotZip && rodataOp.mime_type().hasValue();

    // Embed the rodata contents.
    size_t alignment =
        rodataOp.alignment()
            ? static_cast<size_t>(rodataOp.alignment().getValue())
            : 0;
    if (alignment == 0) alignment = kDefaultRodataAlignment;
    auto constantRef =
        serializeConstant(rodataOp.getLoc(), rodataOp.value(), alignment,
                          /*calculateCRC32=*/includeInZIP, fbb);
    if (!constantRef.ref) {
      return rodataOp.emitOpError() << "failed to encode";
    }
    rodataContentRefs.push_back(constantRef.ref);

    // Add the ZIP per-file header.
    if (includeInZIP) {
      zipFileRefs.push_back(appendZIPLocalFileHeader(
          rodataOp, constantRef.totalSize, constantRef.crc32, fbb));
    }
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
        funcOp.value(), typeOrdinalMap, symbolTable, debugDatabase);
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

  // NOTE: we keep the vectors clustered here so that we can hopefully keep the
  // pages mapped at runtime; vector dereferences in flatbuffers require
  // touching these structs to get length/etc and as such we don't want to be
  // gathering from all over the file (with giant rodata chunks and such
  // inbetween) just to perform a bounds check and deference into another part
  // of the file.
  auto rodataSegmentsRef = fbb.createOffsetVecDestructive(rodataSegmentRefs);
  auto rwdataSegmentsRef = fbb.createOffsetVecDestructive(rwdataSegmentRefs);
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

  iree_vm_DebugDatabaseDef_ref_t debugDatabaseRef = 0;
  if (!targetOptions.stripSourceMap) {
    debugDatabaseRef = debugDatabase.build(fbb);
  }

  auto moduleNameRef = fbb.createString(
      moduleOp.sym_name().empty() ? "module" : moduleOp.sym_name());

  iree_vm_BytecodeModuleDef_name_add(fbb, moduleNameRef);
  iree_vm_BytecodeModuleDef_types_add(fbb, typesRef);
  iree_vm_BytecodeModuleDef_imported_functions_add(fbb, importFuncsRef);
  iree_vm_BytecodeModuleDef_exported_functions_add(fbb, exportFuncsOffset);
  iree_vm_BytecodeModuleDef_module_state_add(fbb, moduleStateDef);
  iree_vm_BytecodeModuleDef_rodata_segments_add(fbb, rodataSegmentsRef);
  iree_vm_BytecodeModuleDef_rwdata_segments_add(fbb, rwdataSegmentsRef);
  iree_vm_BytecodeModuleDef_function_descriptors_add(fbb,
                                                     functionDescriptorsRef);
  iree_vm_BytecodeModuleDef_bytecode_data_add(fbb, bytecodeDataRef);
  iree_vm_BytecodeModuleDef_debug_database_add(fbb, debugDatabaseRef);
  iree_vm_BytecodeModuleDef_end_as_root(fbb);

  return success();
}

LogicalResult translateModuleToBytecode(IREE::VM::ModuleOp moduleOp,
                                        BytecodeTargetOptions targetOptions,
                                        llvm::raw_ostream &output) {
  moduleOp.getContext()->getOrLoadDialect<IREE::Util::UtilDialect>();

  uint64_t startOffset = output.tell();

  if (failed(canonicalizeModule(targetOptions, moduleOp))) {
    return moduleOp.emitError()
           << "failed to canonicalize vm.module to a serializable form";
  }

  // Dump VM assembly source listing to a file and annotate IR locations.
  if (!targetOptions.sourceListing.empty()) {
    OpPrintingFlags printFlags;
    printFlags.elideLargeElementsAttrs(8192);
    if (failed(mlir::generateLocationsFromIR(targetOptions.sourceListing, "vm",
                                             moduleOp, printFlags))) {
      return moduleOp.emitError() << "failed to write source listing to '"
                                  << targetOptions.sourceListing << "'";
    }
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
  SmallVector<ZIPFileRef> zipFileRefs;
  if (failed(
          buildFlatBufferModule(targetOptions, moduleOp, zipFileRefs, fbb))) {
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

  if (targetOptions.emitPolyglotZip) {
    // Append the ZIP central directory to the end of the output.
    // We have to do this here as we need to have flushed the flatbuffer
    // contents to the output so that we have their final absolute addresses.
    uint64_t endOffset = output.tell();
    appendZIPCentralDirectory(zipFileRefs, startOffset, endOffset, output);
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
