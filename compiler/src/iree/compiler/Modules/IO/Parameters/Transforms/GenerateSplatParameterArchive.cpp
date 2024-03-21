// Copyright 2024 The IREE Authors
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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/FileUtilities.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

#define GEN_PASS_DEF_GENERATESPLATPARAMETERARCHIVEPASS
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h.inc"

namespace {

static Attribute getDefaultSplatAttr(Type elementType) {
  // Today we only support basic types where 0 bits represent zeros - that lets
  // us just splat out the right number of bits.
  int64_t storageSize =
      IREE::Util::getRoundedPhysicalStorageSize(1, elementType);
  return IntegerAttr::get(
      IntegerType::get(elementType.getContext(), storageSize * 8), 0);
}

struct GenerateSplatParameterArchivePass
    : public IREE::IO::Parameters::impl::GenerateSplatParameterArchivePassBase<
          GenerateSplatParameterArchivePass> {
  using IREE::IO::Parameters::impl::GenerateSplatParameterArchivePassBase<
      GenerateSplatParameterArchivePass>::GenerateSplatParameterArchivePassBase;

  void runOnOperation() override {
    // Nothing to do if no path specified.
    if (filePath.empty())
      return;

    // Create a builder used to accumulate the parameters.
    ModuleOp moduleOp = getOperation();
    auto builder = createArchiveBuilder(moduleOp);
    if (failed(builder))
      return signalPassFailure();

    // Walk the globals in the module.
    for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
      // Only support types we can meaningfully generate splats for.
      auto shapedType = dyn_cast<ShapedType>(globalOp.getGlobalType());
      if (!shapedType)
        continue;

      // Look for globals backed by parameters.
      auto parameterAttr =
          dyn_cast_if_present<IREE::Stream::NamedParameterAttr>(
              globalOp.getGlobalInitialValue());
      if (!parameterAttr)
        continue;

      // TODO: support other patterns/generators.
      auto elementAttr = getDefaultSplatAttr(shapedType.getElementType());

      // Serialize the splat pattern.
      SmallVector<char, IREE_IO_PARAMETER_MAX_SPLAT_PATTERN_LENGTH> pattern;
      llvm::raw_svector_ostream os(pattern);
      if (failed(IREE::Util::SerializableAttrInterface::serializeSplatValue(
              globalOp.getLoc(), elementAttr,
              /*count=*/1, llvm::endianness::little, os))) {
        return signalPassFailure();
      }

      // Add a zero-splat entry to the builder for this global.
      auto parameterName = parameterAttr.getKey().getValue();
      if (failed(handleRuntimeError(
              moduleOp,
              iree_io_parameter_archive_builder_add_splat_entry(
                  builder->get(),
                  iree_string_view_t{
                      parameterName.data(),
                      static_cast<iree_host_size_t>(parameterName.size())},
                  /*metadata=*/iree_const_byte_span_empty(),
                  /*pattern=*/pattern.data(),
                  /*pattern_length=*/static_cast<uint8_t>(pattern.size()),
                  /*data_length=*/parameterAttr.getStorageSize()),
              "failed to add splat entry for global"))) {
        return signalPassFailure();
      }
    }

    // Early exit if no parameter backed globals present.
    if (iree_io_parameter_archive_builder_is_empty(builder->get()))
      return;

    // Create the parameter archive file.
    auto fileStreamIndexOr =
        createParameterIndex(moduleOp, std::move(builder.value()), filePath);
    if (failed(fileStreamIndexOr))
      return signalPassFailure();
    auto [file, stream, index] = *std::move(fileStreamIndexOr);

    // Commit the written file.
    if (llvm::Error maybeCommit = file->commit()) {
      InFlightDiagnostic errorStream =
          moduleOp.emitError() << "Failed to commit archive with error: ";
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
