// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Modules/IO/Parameters/Transforms/ArchiveUtils.h"
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/FileUtilities.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

#define GEN_PASS_DEF_EXPORTPARAMETERSPASS
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h.inc"

namespace {

static LogicalResult
addSplatEntry(IREE::Util::GlobalOpInterface globalOp,
              SplatElementsAttr valueAttr, int64_t storageSize,
              iree_io_parameter_archive_builder_t *builder) {
  SmallVector<char, IREE_IO_PARAMETER_MAX_SPLAT_PATTERN_LENGTH> pattern;
  llvm::raw_svector_ostream os(pattern);
  if (failed(IREE::Util::SerializableAttrInterface::serializeSplatValue(
          globalOp.getLoc(), valueAttr.getSplatValue<Attribute>(),
          /*count=*/1, llvm::endianness::little, os))) {
    return failure();
  }

  StringRef name = globalOp.getGlobalName();
  return handleRuntimeError(
      globalOp,
      iree_io_parameter_archive_builder_add_splat_entry(
          builder, iree_make_string_view(name.data(), name.size()),
          /*metadata=*/iree_const_byte_span_empty(), pattern.data(),
          static_cast<uint8_t>(pattern.size()), storageSize),
      "failed to add splat entry for global");
}

static LogicalResult
addDataEntry(IREE::Util::GlobalOpInterface globalOp,
             IREE::Util::SerializableAttrInterface valueAttr,
             int64_t storageSize,
             iree_io_parameter_archive_builder_t *builder) {
  StringRef name = globalOp.getGlobalName();
  return handleRuntimeError(
      globalOp,
      iree_io_parameter_archive_builder_add_data_entry(
          builder, iree_make_string_view(name.data(), name.size()),
          /*metadata=*/iree_const_byte_span_empty(),
          /*alignment=*/
          IREE_IO_PARAMETER_ARCHIVE_DEFAULT_DATA_ALIGNMENT, storageSize),
      "failed to add data entry for global");
}

// Adds an entry to the parameter archive builder for the given global.
// If the global is mutable we allocate archive storage for the full
// serialized parameter. This allows the parameter to be mapped for
// read/write in the file. If the global is immutable and a splat we can
// add a splat entry instead to save on archive size and startup time.
static LogicalResult addEntry(IREE::Util::GlobalOpInterface globalOp,
                              IREE::Util::SerializableAttrInterface valueAttr,
                              iree_io_parameter_archive_builder_t *builder) {
  if (!globalOp.isGlobalMutable()) {
    if (auto elementsAttr = dyn_cast<SplatElementsAttr>(valueAttr)) {
      return addSplatEntry(globalOp, elementsAttr, valueAttr.getStorageSize(),
                           builder);
    }
  }
  return addDataEntry(globalOp, valueAttr, valueAttr.getStorageSize(), builder);
}

struct ExportParametersPass
    : public IREE::IO::Parameters::impl::ExportParametersPassBase<
          ExportParametersPass> {
  using IREE::IO::Parameters::impl::ExportParametersPassBase<
      ExportParametersPass>::ExportParametersPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    // Nothing to do if no path specified.
    if (scopePath.empty())
      return;
    auto [scope, path] = splitScopePath(scopePath);

    // Create a builder used to accumulate the parameters.
    ModuleOp moduleOp = getOperation();
    auto builder = createArchiveBuilder(moduleOp);
    if (failed(builder))
      return signalPassFailure();

    // Accumulate globals that match the pass options and add them to the index.
    SmallVector<IREE::Util::GlobalOpInterface> constantGlobalOps;
    for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
      // Only globals initialized with serializable initial values can be
      // parameterized.
      auto serializableAttr =
          dyn_cast_if_present<IREE::Util::SerializableAttrInterface>(
              globalOp.getGlobalInitialValue());
      if (!serializableAttr)
        continue;

      // Check that the serialized size of the attribute is at least as big as
      // the pass configured minimum storage size.
      int64_t storageSize = serializableAttr.getStorageSize();
      if (storageSize < minimumSize)
        continue;

      // Add the entry with a type based on its contents.
      if (failed(addEntry(globalOp, serializableAttr, builder->get())))
        return signalPassFailure();

      constantGlobalOps.push_back(globalOp);
    }

    // Early exit if no parameterizable globals are present.
    if (constantGlobalOps.empty())
      return;

    // Create the parameter archive file opened for writing.
    auto fileStreamIndexOr =
        createParameterIndex(moduleOp, std::move(builder.value()), path);
    if (failed(fileStreamIndexOr))
      return signalPassFailure();
    auto [file, stream, index] = *std::move(fileStreamIndexOr);

    // Serialize parameters to the file.
    for (auto globalOp : constantGlobalOps) {
      // Lookup the entry in the index corresponding to the global.
      const iree_io_parameter_index_entry_t *entry = nullptr;
      StringRef name = globalOp.getGlobalName();
      if (failed(handleRuntimeError(
              globalOp,
              iree_io_parameter_index_lookup(
                  index.get(), iree_make_string_view(name.data(), name.size()),
                  &entry),
              "retrieve global from index"))) {
        return signalPassFailure();
      }

      // Only file entries get stored; splats are in the metadata table.
      if (entry->type == IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE) {
        // Seek to where the serialized global begins in the file.
        if (failed(handleRuntimeError(
                globalOp,
                iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET,
                                    entry->storage.file.offset),
                "failed to seek to location of global in archive"))) {
          return signalPassFailure();
        }

        // Serialize the global contents to the stream.
        iree_io_stream_ostream os(stream.get());
        auto serializableAttr = cast<IREE::Util::SerializableAttrInterface>(
            globalOp.getGlobalInitialValue());
        if (failed(serializableAttr.serializeToStream(
                globalOp.getLoc(), llvm::endianness::native, os))) {
          globalOp.emitError() << "failed to serialize global to archive";
          return signalPassFailure();
        }
        os.flush();
      }

      // Change the global to reference the parameter.
      globalOp.setGlobalInitialValue(IREE::Stream::NamedParameterAttr::get(
          context, globalOp.getGlobalType(), StringAttr::get(context, scope),
          StringAttr::get(context, name), DictionaryAttr()));
    }

    // Commit the written file.
    if (llvm::Error maybeCommit = file->commit()) {
      InFlightDiagnostic errorStream =
          moduleOp.emitError() << "failed to commit archive with error: ";
      llvm::handleAllErrors(std::move(maybeCommit),
                            [&](const llvm::ErrorInfoBase &PE) {
                              errorStream << PE.message() << "\n";
                            });
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::IO::Parameters
