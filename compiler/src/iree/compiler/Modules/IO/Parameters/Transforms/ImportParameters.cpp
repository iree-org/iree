// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Modules/IO/Parameters/Transforms/ArchiveUtils.h"
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/FileUtilities.h"

#include "iree/base/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/formats/gguf/gguf_parser.h"
#include "iree/io/formats/irpa/irpa_parser.h"
#include "iree/io/formats/safetensors/safetensors_parser.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

#define GEN_PASS_DEF_IMPORTPARAMETERSPASS
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h.inc"

namespace {

using FileHandle =
    std::unique_ptr<iree_io_file_handle_t, void (*)(iree_io_file_handle_t *)>;
using ParameterIndex = std::unique_ptr<iree_io_parameter_index_t,
                                       void (*)(iree_io_parameter_index_t *)>;

static FailureOr<FileHandle> openArchiveFile(ModuleOp moduleOp,
                                             StringRef archivePath) {
  iree_allocator_t hostAllocator = iree_allocator_system();

  // Open the archive (hopefully mapped).
  auto fileOrErr = llvm::MemoryBuffer::getFile(
      archivePath, /*IsText=*/false, /*RequiresNullTerminator=*/false,
      /*IsVolatile=*/false, /*Alignment=*/std::nullopt);
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << "cannot open archive input file '" + archivePath +
                        "': " + error.message();
    return failure();
  }
  auto file = std::move(fileOrErr.get());

  // A callback issued when a file is released to destroy the file.
  iree_io_file_handle_release_callback_t fileReleaseCallback;
  fileReleaseCallback.fn =
      +[](void *user_data, iree_io_file_handle_primitive_t handle_primitive) {
        delete reinterpret_cast<llvm::MemoryBuffer *>(user_data);
      };
  fileReleaseCallback.user_data = file.get();

  // Wrap the archive in a file handle.
  iree_io_file_handle_t *fileHandle = nullptr;
  if (failed(handleRuntimeError(
          moduleOp,
          iree_io_file_handle_wrap_host_allocation(
              IREE_IO_FILE_ACCESS_READ,
              iree_make_byte_span(const_cast<char *>(file->getBufferStart()),
                                  file->getBufferSize()),
              fileReleaseCallback, hostAllocator, &fileHandle),
          "unable to wrap archive memory buffer"))) {
    return failure();
  }
  file.release(); // now owned by the fileHandle

  return FileHandle(fileHandle, iree_io_file_handle_release);
}

static LogicalResult
loadParameterIndex(ModuleOp moduleOp, StringRef path,
                   iree_io_parameter_index_t *parameterIndex) {
  // Open the archive file (hopefully mapping it).
  auto fileHandle = openArchiveFile(moduleOp, path);
  if (failed(fileHandle))
    return failure();

  // Parse the archive as a particular format.
  // TODO(benvanik): centralize this type selection logic in iree/io/.
  StringRef format = llvm::sys::path::extension(path);
  if (format == ".gguf") {
    return handleRuntimeError(
        moduleOp, iree_io_parse_gguf_index(fileHandle->get(), parameterIndex),
        "parsing gguf file");
  } else if (format == ".irpa") {
    return handleRuntimeError(
        moduleOp, iree_io_parse_irpa_index(fileHandle->get(), parameterIndex),
        "parsing irpa file");
  } else if (format == ".safetensors") {
    return handleRuntimeError(
        moduleOp,
        iree_io_parse_safetensors_index(fileHandle->get(), parameterIndex),
        "parsing safetensors file");
  } else {
    llvm::errs() << "unsupported archive file format: '" << path << "'\n";
    return failure();
  }
}

class ParameterIndices {
public:
  bool contains(StringRef scope) const {
    return indicesByScope.contains(scope);
  }

  iree_io_parameter_index_t *lookup(StringRef scope) const {
    auto it = indicesByScope.find(scope);
    return it == indicesByScope.end() ? nullptr : it->second;
  }

  iree_io_parameter_index_t *lookupOrCreate(ModuleOp moduleOp,
                                            StringRef scope) {
    iree_allocator_t hostAllocator = iree_allocator_system();
    if (iree_io_parameter_index_t *existing = lookup(scope))
      return existing;
    iree_io_parameter_index_t *parameterIndexPtr = nullptr;
    if (failed(handleRuntimeError(
            moduleOp,
            iree_io_parameter_index_create(hostAllocator, &parameterIndexPtr),
            "unable to allocate empty parameter index"))) {
      return nullptr;
    }
    auto parameterIndex =
        ParameterIndex(parameterIndexPtr, iree_io_parameter_index_release);
    auto *ptr = parameterIndex.get();
    indices.push_back(std::move(parameterIndex));
    indicesByScope.try_emplace(scope, ptr);
    return ptr;
  }

private:
  SmallVector<ParameterIndex> indices;
  DenseMap<StringRef, iree_io_parameter_index_t *> indicesByScope;
};

using ScopePath = std::pair<StringRef, StringRef>;
static ScopePath splitScopePath(StringRef scopePath) {
  size_t i = scopePath.find_first_of('=');
  if (i == StringRef::npos)
    return ScopePath("", scopePath);
  else
    return ScopePath(scopePath.substr(0, i), scopePath.substr(i + 1));
}

// Allocates one parameter index per scope (possibly an empty string) and
// merges in parameters from the referenced files.
static FailureOr<ParameterIndices>
loadParameterArchives(ModuleOp moduleOp, ArrayRef<std::string> scopePaths) {
  ParameterIndices parameterIndices;
  for (auto &scopePath : scopePaths) {
    auto [scope, path] = splitScopePath(scopePath);
    auto *parameterIndex = parameterIndices.lookupOrCreate(moduleOp, scope);
    if (failed(loadParameterIndex(moduleOp, path, parameterIndex)))
      return failure();
  }
  return parameterIndices;
}

// Today only shaped types of elements where we know we can directly access the
// data as stored in the file.
static bool isTypeSupported(Type type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType)
    return false;
  auto elementType = shapedType.getElementType();
  // NOTE: packed types not yet supported.
  if (!elementType.isIntOrFloat())
    return false;
  const unsigned logicalBitWidth = elementType.getIntOrFloatBitWidth();
  switch (logicalBitWidth) {
  case 8:
  case 16:
  case 32:
  case 64:
    return true;
  default:
    return false;
  }
}

static FailureOr<TypedAttr>
importParameterFromSplat(StringRef fullName, ShapedType globalType,
                         const iree_io_parameter_index_entry_t *entry) {
  // Ensure we have the right bit count.
  // NOTE: this will need to change for packed types.
  auto elementType = globalType.getElementType();
  if (elementType.getIntOrFloatBitWidth() !=
      entry->storage.splat.pattern_length * 8) {
    llvm::errs() << "splat pattern has insufficient bits for type "
                 << globalType << "\n";
    return failure();
  }

  // Map the splat pattern into an attribute.
  Attribute valueAttr;
  if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    uint64_t value = 0;
    switch (integerType.getWidth()) {
    case 8:
      value = entry->storage.splat.pattern[0];
      break;
    case 16:
      value = llvm::support::endian::read16le(entry->storage.splat.pattern);
      break;
    case 32:
      value = llvm::support::endian::read32le(entry->storage.splat.pattern);
      break;
    case 64:
      value = llvm::support::endian::read64le(entry->storage.splat.pattern);
      break;
    default:
      assert(false && "integer width not supported");
      return failure();
    }
    return TypedAttr(
        IntegerAttr::get(globalType, APInt(integerType.getWidth(), value)));
  } else if (auto floatType = dyn_cast<FloatType>(elementType)) {
    uint64_t value = 0;
    switch (floatType.getWidth()) {
    case 8:
      value = entry->storage.splat.pattern[0];
      break;
    case 16:
      value = llvm::support::endian::read16le(entry->storage.splat.pattern);
      break;
    case 32:
      value = llvm::support::endian::read32le(entry->storage.splat.pattern);
      break;
    case 64:
      value = llvm::support::endian::read64le(entry->storage.splat.pattern);
      break;
    default:
      assert(false && "integer width not supported");
      return failure();
    }
    return TypedAttr(FloatAttr::get(
        globalType, APFloat(floatType.getFloatSemantics(),
                            APInt(integerType.getWidth(), value))));
  }
  if (!valueAttr) {
    llvm::errs() << "unsupported splat type: " << elementType << "\n";
    return failure();
  }

  // Create a splat with the given element value.
  return TypedAttr(SplatElementsAttr::get(globalType, valueAttr));
}

// TODO(benvanik): replace with resources, maybe? there's no FileAsmResourceBlob
// yet but we could use that to point back to the file on disk. For now we just
// import as a raw attr to ensure that imported parameters behave exactly as
// constants would everywhere and can be serialized/deserialized across
// reproducers/etc.
static FailureOr<TypedAttr>
importParameterFromFile(StringRef fullName, ShapedType globalType,
                        const iree_io_parameter_index_entry_t *entry) {
  // We currently only support mapped files, but could instead handle file path
  // references and point resource blobs directly at them.
  iree_io_file_handle_primitive_t filePrimitive =
      iree_io_file_handle_primitive(entry->storage.file.handle);
  if (filePrimitive.type != IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    llvm::errs() << "only host allocation file primitives are supported\n";
    return failure();
  }
  const uint8_t *fileData = filePrimitive.value.host_allocation.data;

  // Copy the data from the parameter file into an attribute
  return TypedAttr(DenseElementsAttr::getFromRawBuffer(
      globalType, ArrayRef<char>(reinterpret_cast<const char *>(
                                     fileData + entry->storage.file.offset),
                                 entry->length)));
}

// Import the given |parameterAttr| from |entry|.
static FailureOr<TypedAttr>
importParameter(StringRef fullName, ShapedType globalType,
                IREE::Stream::NamedParameterAttr parameterAttr,
                const iree_io_parameter_index_entry_t *entry) {
  switch (entry->type) {
  case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT:
    return importParameterFromSplat(fullName, globalType, entry);
  case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE:
    return importParameterFromFile(fullName, globalType, entry);
  default:
    // Unsupported type.
    llvm::errs() << "found parameter but type is not supported: "
                 << parameterAttr.getKey().getValue() << "\n";
    return failure();
  }
}

struct ImportParametersPass
    : public IREE::IO::Parameters::impl::ImportParametersPassBase<
          ImportParametersPass> {
  using IREE::IO::Parameters::impl::ImportParametersPassBase<
      ImportParametersPass>::ImportParametersPassBase;

  void runOnOperation() override {
    // Nothing to do if no path specified.
    if (scopePaths.empty())
      return;

    ModuleOp moduleOp = getOperation();
    Explorer explorer(moduleOp, TraversalAction::SHALLOW);
    explorer.setOpInterfaceAction<mlir::FunctionOpInterface>(
        TraversalAction::RECURSE);
    explorer.initialize();

    // Open the archive file (hopefully mapping it) and parse the index.
    auto parameterIndices = loadParameterArchives(moduleOp, scopePaths);
    if (failed(parameterIndices))
      return signalPassFailure();

    // Decide whether to import a particular parameter.
    DenseSet<StringRef> importKeys;
    for (auto &key : keys)
      importKeys.insert(key);
    auto shouldImportParameter =
        [&](IREE::Stream::NamedParameterAttr parameterAttr) -> bool {
      // Always try to import explicitly named parameters.
      if (importKeys.contains(parameterAttr.getKey().getValue()))
        return true; // key match
      // If a maximum size is specified use that to limit what we import
      // (users may want to bring in small parameters but leave the big ones
      // out).
      if (maximumSize && parameterAttr.getStorageSize() <= maximumSize)
        return true; // <= max size
      // Default to not importing.
      return false;
    };

    // Find all parameters and try to import them.
    SmallVector<Operation *> deadOps;
    explorer.forEachGlobal([&](const Explorer::GlobalInfo *globalInfo) {
      auto globalOp = globalInfo->op;

      // Only inspect parameter globals.
      auto parameterAttr =
          dyn_cast_if_present<IREE::Stream::NamedParameterAttr>(
              globalOp.getGlobalInitialValue());
      if (!parameterAttr)
        return;

      // Filter only to globals of types we serialize.
      if (!isTypeSupported(globalOp.getGlobalType())) {
        return;
      }

      // Lookup the parameter index for the scope.
      auto scope = parameterAttr.getScope().getValue();
      auto *parameterIndex = parameterIndices->lookup(scope);
      if (!parameterIndex) {
        // Scope not populated via a provided archive.
        return;
      }

      // See if the parameter is present in the scope (we may have only been
      // provided as partial index).
      auto key = parameterAttr.getKey().getValue();
      const iree_io_parameter_index_entry_t *entry = nullptr;
      iree_status_t lookupStatus = iree_io_parameter_index_lookup(
          parameterIndex, iree_make_string_view(key.data(), key.size()),
          &entry);
      if (!iree_status_is_ok(lookupStatus)) {
        // Parameter not found.
        iree_status_ignore(lookupStatus);
        return;
      }

      // Only import if the parameter meets any filtering criteria we have
      // from the pass options.
      if (shouldImportParameter(parameterAttr)) {
        std::string fullName =
            (StringRef("__import_") + scope + "_" + key).str();
        auto valueOr = importParameter(
            fullName, cast<ShapedType>(globalOp.getGlobalType()), parameterAttr,
            entry);
        if (failed(valueOr))
          return signalPassFailure();

        // If the global is mutable we just change the initializer and otherwise
        // replace the global with inlined constant values. The constants may be
        // hoisted back to globals at some point but making them local allows
        // for more patterns/folders to kick in earlier.
        if (globalOp.isGlobalMutable()) {
          globalOp.setGlobalInitialValue(*valueOr);
        } else {
          deadOps.push_back(globalOp);
          for (auto loadOp : globalInfo->getLoads()) {
            OpBuilder builder(loadOp);
            Value value =
                builder.create<arith::ConstantOp>(loadOp.getLoc(), *valueOr);
            loadOp.getLoadedGlobalValue().replaceAllUsesWith(value);
            deadOps.push_back(loadOp);
          }
        }
      }
    });
    for (auto *deadOp : deadOps)
      deadOp->erase();
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::IO::Parameters
