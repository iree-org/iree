// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/hal/api.h"
#include "iree/io/formats/irpa/irpa_builder.h"
#include "iree/tooling/parameter_util.h"

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

#define GEN_PASS_DEF_EXPORTPARAMETERSPASS
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h.inc"

namespace {

// Hoists all serializable constants with storage size at least |minimumSize|
// into their own globals with initial value equal to the constant value.
static void hoistConstantsIntoGlobals(mlir::ModuleOp moduleOp,
                                      int64_t minimumSize) {
  SymbolTable moduleSymbols(moduleOp);
  IRRewriter rewriter(OpBuilder::atBlockBegin(moduleOp.getBody()));
  llvm::DenseMap<arith::ConstantOp, Util::GlobalOp> hoistedMap;
  moduleOp.walk([&](arith::ConstantOp constant) {
    // Constants part of a different logical program should not be hoisted.
    if (SymbolTable::getNearestSymbolTable(constant) != moduleOp) {
      return;
    }
    TypedAttr initialValueAttr = constant.getValue();
    auto serializableAttr =
        dyn_cast<IREE::Util::SerializableAttrInterface>(initialValueAttr);
    if (!serializableAttr) {
      return;
    }

    // Check that the serialized size of the attribute is at least as big as
    // the pass configured minimum storage size.
    iree_io_physical_size_t storageSize = serializableAttr.getStorageSize();
    if (storageSize < minimumSize) {
      return;
    }

    // Create a new global with initial value equal to the constant.
    Location loc = constant.getLoc();
    Util::GlobalOp globalOp = rewriter.create<Util::GlobalOp>(
        loc, "constant_hoisted", false, constant.getType());
    moduleSymbols.insert(globalOp);
    // Attributes are stored uniqued by their contents so this is not a copy.
    globalOp.setInitialValueAttr(initialValueAttr);
    SymbolTable::setSymbolVisibility(globalOp,
                                     SymbolTable::Visibility::Private);
    hoistedMap[constant] = globalOp;
  });

  // Replace all constants with their associated hoisted globals.
  for (auto it : hoistedMap) {
    arith::ConstantOp originalConstant = it.first;
    Util::GlobalOp globalOp = it.second;
    rewriter.setInsertionPointAfterValue(originalConstant);
    Value load =
        rewriter.create<Util::GlobalLoadOp>(globalOp->getLoc(), globalOp);
    rewriter.replaceOp(originalConstant, load);
  }
}

extern "C" {
static void iree_io_file_handle_release_mapping(
    void *user_data, iree_io_file_handle_primitive_t handle_primitive) {
  iree_file_contents_free((iree_file_contents_t *)user_data);
}

static iree_status_t iree_tooling_open_output_parameter_file(
    iree_io_physical_offset_t archive_offset,
    iree_io_physical_size_t archive_length, iree_allocator_t host_allocator,
    iree_io_file_handle_t **out_file_handle, const char *path) {
  iree_file_contents_t *file_contents = NULL;
  IREE_RETURN_IF_ERROR(iree_file_create_mapped(
      path, archive_offset + archive_length, archive_offset,
      (iree_host_size_t)archive_length, host_allocator, &file_contents));
  iree_io_file_handle_release_callback_t release_callback = {
      iree_io_file_handle_release_mapping, file_contents};
  iree_status_t status = iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_WRITE, file_contents->buffer, release_callback,
      host_allocator, out_file_handle);
  if (!iree_status_is_ok(status)) {
    iree_file_contents_free(file_contents);
  }
  return status;
}
}

static LogicalResult handleRuntimeError(ModuleOp moduleOp, iree_status_t status,
                                        StringRef failureMessage) {
  if (iree_status_is_ok(status))
    return success();
  std::string message;
  message.resize(512);
  iree_host_size_t buffer_length;
  if (!iree_status_format(status, message.size(), &message[0],
                          &buffer_length)) {
    message.resize(buffer_length + 1);
    iree_status_format(status, message.size(), &message[0], &buffer_length);
  }
  message.resize(buffer_length);
  iree_status_ignore(status);
  return moduleOp.emitError() << failureMessage << message;
}

struct ExportParametersPass
    : public IREE::IO::Parameters::impl::ExportParametersPassBase<
          ExportParametersPass> {
  using IREE::IO::Parameters::impl::ExportParametersPassBase<
      ExportParametersPass>::ExportParametersPassBase;

  void runOnOperation() override {
    // Nothing to do if no path specified.
    if (archivePath.empty()) {
      return;
    }

    MLIRContext *context = &getContext();
    ModuleOp moduleOp = getOperation();

    // First hoist all inline constants into their own globals.
    hoistConstantsIntoGlobals(moduleOp, minimumSize);

    iree_allocator_t host_allocator = iree_allocator_system();

    // Create the parameter archive builder.
    iree_io_parameter_archive_builder_t builder;
    iree_io_parameter_archive_builder_initialize(host_allocator, &builder);

    auto deinitializeExit = llvm::make_scope_exit([&]() {
      return iree_io_parameter_archive_builder_deinitialize(&builder);
    });

    SmallVector<IREE::Util::GlobalOp> constantGlobals;
    // Walk the globals in the module.
    for (auto global : moduleOp.getOps<IREE::Util::GlobalOp>()) {
      // TODO: Support exporting mutable globals.
      if (global.getIsMutable()) {
        continue;
      }
      // Only globals initialized with initial values can be parameterized.
      auto initialValueAttr = global.getInitialValueAttr();
      if (!initialValueAttr) {
        continue;
      }

      // The attribute must be serializable to be turned into a parameter.
      auto serializableAttr =
          dyn_cast<IREE::Util::SerializableAttrInterface>(initialValueAttr);
      if (!serializableAttr) {
        continue;
      }

      // Check that the serialized size of the attribute is at least as big as
      // the pass configured minimum storage size.
      iree_io_physical_size_t storageSize = serializableAttr.getStorageSize();
      if (storageSize < minimumSize) {
        continue;
      }
      StringRef name = global.getSymName();

      // Add a data entry to the builder for this global.
      iree_status_t status = iree_io_parameter_archive_builder_add_data_entry(
          &builder,
          iree_string_view_t{name.data(),
                             static_cast<iree_host_size_t>(name.size())},
          /*metadata=*/iree_const_byte_span_empty(),
          /*alignment=*/IREE_IO_PARAMETER_ARCHIVE_DEFAULT_DATA_ALIGNMENT,
          storageSize);
      if (failed(handleRuntimeError(moduleOp, status,
                                    "Failed to add data entry for global"))) {
        return signalPassFailure();
      }

      constantGlobals.push_back(global);
    }

    // Early exit if no parameterizable globals present.
    if (constantGlobals.empty()) {
      return;
    }

    // Open a file of sufficient size (now that we know it) for writing.
    iree_io_physical_offset_t target_file_offset = 0;
    iree_io_physical_offset_t archive_offset = iree_align_uint64(
        target_file_offset, IREE_IO_PARAMETER_ARCHIVE_HEADER_ALIGNMENT);
    iree_io_physical_size_t archive_length =
        iree_io_parameter_archive_builder_total_size(&builder);

    iree_io_file_handle_t *target_file_handle = NULL;
    iree_status_t status = iree_tooling_open_output_parameter_file(
        archive_offset, archive_length, host_allocator, &target_file_handle,
        archivePath.c_str());
    auto releaseFileExit = llvm::make_scope_exit(
        [&]() { return iree_io_file_handle_release(target_file_handle); });
    if (failed(handleRuntimeError(moduleOp, status,
                                  "Failed to open output parameter archive"))) {
      return signalPassFailure();
    }

    // Wrap the target file in a stream.
    iree_io_stream_t *target_stream = NULL;
    status =
        iree_io_stream_open(IREE_IO_STREAM_MODE_WRITABLE, target_file_handle,
                            target_file_offset, host_allocator, &target_stream);
    auto releaseStreamExit = llvm::make_scope_exit(
        [&]() { return iree_io_stream_release(target_stream); });
    if (failed(handleRuntimeError(
            moduleOp, status, "Failed to create I/O stream to output file"))) {
      return signalPassFailure();
    }

    // Allocate an index we'll populate during building to allow us to get the
    // storage ranges of non-metadata parameters.
    iree_io_parameter_index_t *built_index = NULL;
    status = iree_io_parameter_index_create(host_allocator, &built_index);
    auto releaseIndexExit = llvm::make_scope_exit(
        [&]() { return iree_io_parameter_index_release(built_index); });
    if (failed(handleRuntimeError(moduleOp, status,
                                  "Failed to allocate parameter index"))) {
      return signalPassFailure();
    }

    // Commit the archive header to the file and produce an index referencing
    // it. This will allow us to know where to copy file contents.
    status = iree_io_parameter_archive_builder_write(
        &builder, target_file_handle, target_file_offset, target_stream,
        built_index);
    if (failed(handleRuntimeError(
            moduleOp, status,
            "Failed to write parameter index header to output file"))) {
      return signalPassFailure();
    }

    StringAttr scopeAttr = parameterScope.empty()
                               ? StringAttr()
                               : StringAttr::get(context, parameterScope);

    // Write all of the global contents to the appropriate data storage
    // segments.
    for (auto constantGlobal : constantGlobals) {
      StringRef name = constantGlobal.getSymName();

      const iree_io_parameter_index_entry_t *target_entry = NULL;
      status = iree_io_parameter_index_lookup(
          built_index,
          iree_string_view_t{name.data(),
                             static_cast<iree_host_size_t>(name.size())},
          &target_entry);
      if (failed(handleRuntimeError(
              moduleOp, status,
              "Failed to write parameter index header to output file"))) {
        return signalPassFailure();
      }
      status = iree_io_stream_seek(target_stream, IREE_IO_STREAM_SEEK_SET,
                                   target_file_offset +
                                       target_entry->storage.file.offset);
      if (failed(handleRuntimeError(
              moduleOp, status,
              "Failed to seek to location of global in index"))) {
        return signalPassFailure();
      }

      auto initialValueAttr = constantGlobal.getInitialValueAttr();
      auto serializableAttr =
          dyn_cast<IREE::Util::SerializableAttrInterface>(initialValueAttr);

      SmallVector<char> buffer;
      if (failed(serializableAttr.serializeToVector(
              constantGlobal.getLoc(), llvm::endianness::native, buffer))) {
        moduleOp.emitError() << "Failed to serialize global " << constantGlobal;
        return signalPassFailure();
      }

      status =
          iree_io_stream_write(target_stream, buffer.size(),
                               reinterpret_cast<const char *>(buffer.data()));
      if (failed(handleRuntimeError(moduleOp, status,
                                    "Failed to write global to index"))) {
        return signalPassFailure();
      }

      // Now we can just replace the existing initial value with a reference to
      // the parameter.
      auto param = IREE::Stream::NamedParameterAttr::get(
          context, constantGlobal.getType(), scopeAttr,
          StringAttr::get(context, name), DictionaryAttr());
      constantGlobal.setInitialValueAttr(param);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::IO::Parameters
